"""Shared ONNX inference plumbing: sessions, letterboxing, decoding, NMS.

Kept out of the analyzers because all of Phase 4 and most of Phase 5 need the
same four things, and because this is where the subtle bugs live. Letterbox
coordinate mapping in particular is easy to get *nearly* right — off by the
padding offset, or scaled by the wrong axis — and the symptom is boxes that
drift toward one corner rather than an obvious crash.

Everything here is pure numpy plus onnxruntime. No OpenCV dependency: the
resize is done with Pillow, which is already a hard requirement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

#: Threads per session. Batch mode already runs one process per core, so
#: letting every session spawn its own pool oversubscribes the machine badly.
DEFAULT_THREADS = int(os.environ.get("IMGINTEL_ONNX_THREADS", "1"))


@dataclass(frozen=True, slots=True)
class Letterbox:
    """The transform applied to fit an image into a fixed-size network input.

    ``scale`` and the pad offsets are what map a detection back to source
    coordinates. Keeping them together with the tensor makes it impossible to
    map boxes back using the wrong values, which is the classic version of
    this bug.
    """

    scale: float
    pad_x: int
    pad_y: int
    input_width: int
    input_height: int
    source_width: int
    source_height: int

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map (N, 4) xyxy boxes from network space back to source pixels."""
        if not len(boxes):
            return boxes.reshape(0, 4)
        out = boxes.astype(np.float32).copy()
        out[:, [0, 2]] -= self.pad_x
        out[:, [1, 3]] -= self.pad_y
        out /= self.scale
        out[:, [0, 2]] = out[:, [0, 2]].clip(0, self.source_width)
        out[:, [1, 3]] = out[:, [1, 3]].clip(0, self.source_height)
        return out


def letterbox(
    image: np.ndarray,
    size: tuple[int, int],
    *,
    fill: int = 114,
) -> tuple[np.ndarray, Letterbox]:
    """Resize preserving aspect ratio and pad to ``size`` (width, height).

    Padding goes bottom-right rather than centred, matching YOLOX's own
    preprocessing — centring instead would shift every box by half the pad.
    """
    target_w, target_h = size
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))

    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8
    )
    canvas = np.full((target_h, target_w, image.shape[2]), fill, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    return canvas, Letterbox(
        scale=scale,
        pad_x=0,
        pad_y=0,
        input_width=target_w,
        input_height=target_h,
        source_width=src_w,
        source_height=src_h,
    )


def to_nchw(image: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """HWC uint8 -> 1CHW float32, optionally scaled to 0..1.

    YOLOX consumes raw 0..255 values (no mean/std normalisation); most other
    detectors do not. Getting this backwards produces plausible-looking but
    uniformly wrong detections, so it is an explicit argument.
    """
    tensor = image.astype(np.float32)
    if normalize:
        tensor /= 255.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])


# -- non-maximum suppression ----------------------------------------------


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS over (N, 4) xyxy boxes. Returns kept indices, best first."""
    if not len(boxes):
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = (np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest])).clip(0)
        inter_h = (np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest])).clip(0)
        inter = inter_w * inter_h
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Class-aware NMS.

    Suppressing across classes would delete the person standing in front of
    the car. Offsetting each class into its own coordinate band keeps them
    independent in a single pass.
    """
    if not len(boxes):
        return []
    spread = float(boxes.max()) + 1.0 if len(boxes) else 1.0
    shifted = boxes + (labels.astype(np.float32) * spread)[:, None]
    return nms(shifted, scores, iou_threshold)


# -- YOLOX decoding --------------------------------------------------------

#: YOLOX heads are evaluated at these strides relative to the input.
YOLOX_STRIDES = (8, 16, 32)


@lru_cache(maxsize=8)
def _yolox_grid(width: int, height: int, strides: tuple[int, ...]) -> tuple:
    """Precompute the anchor-point grid — it depends only on input size."""
    grids, expanded = [], []
    for stride in strides:
        gw, gh = width // stride, height // stride
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grids.append(np.stack((xv, yv), axis=2).reshape(1, -1, 2).astype(np.float32))
        expanded.append(np.full((1, gh * gw, 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=1), np.concatenate(expanded, axis=1)


def decode_yolox(
    output: np.ndarray,
    input_size: tuple[int, int],
    *,
    strides: tuple[int, ...] = YOLOX_STRIDES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw YOLOX head into (boxes_xyxy, scores, labels).

    The head emits, per anchor point: cx, cy offsets in grid units, log
    width/height, an objectness logit already sigmoided by the export, then one
    score per class. Final confidence is objectness * class score — using class
    score alone is a common mistake that floods the output with background.
    """
    predictions = output[0] if output.ndim == 3 else output
    grid, stride = _yolox_grid(input_size[0], input_size[1], tuple(strides))
    grid, stride = grid[0], stride[0]

    if predictions.shape[0] != grid.shape[0]:
        raise ValueError(
            f"YOLOX output has {predictions.shape[0]} anchors but a "
            f"{input_size[0]}x{input_size[1]} input implies {grid.shape[0]}"
        )

    centres = (predictions[:, :2] + grid) * stride
    sizes = np.exp(predictions[:, 2:4]) * stride
    objectness = predictions[:, 4]
    class_scores = predictions[:, 5:]

    labels = class_scores.argmax(axis=1)
    scores = objectness * class_scores[np.arange(len(labels)), labels]

    half = sizes / 2.0
    boxes = np.concatenate([centres - half, centres + half], axis=1)
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int32)


# -- sessions --------------------------------------------------------------


@lru_cache(maxsize=8)
def load_session(model_path: str, threads: int = DEFAULT_THREADS) -> Any:
    """Build (and cache) an inference session for a model file.

    Cached per process: constructing a session parses and optimises the whole
    graph, which costs far more than a single inference. The batch runner's
    pool initializer means each worker pays it once.
    """
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3  # warnings and below are noise here

    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=available_providers()
    )


def available_providers() -> list[str]:
    """Preferred execution providers present on this machine.

    CPU always works and is the only guaranteed option; the accelerators are
    opportunistic. A missing GPU must never be an error — this tool runs on
    investigator laptops, not inference servers.
    """
    import onnxruntime as ort  # noqa: PLC0415

    present = set(ort.get_available_providers())
    preferred = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in present] or ["CPUExecutionProvider"]


def session_input_size(session: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    """Read (width, height) from the graph, falling back for dynamic axes."""
    try:
        shape = session.get_inputs()[0].shape
    except (IndexError, AttributeError):
        return fallback
    if len(shape) != 4:
        return fallback
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return fallback  # dynamic axis
    return width, height


__all__ = [
    "DEFAULT_THREADS",
    "YOLOX_STRIDES",
    "Letterbox",
    "available_providers",
    "batched_nms",
    "decode_yolox",
    "letterbox",
    "load_session",
    "nms",
    "session_input_size",
    "to_nchw",
]
