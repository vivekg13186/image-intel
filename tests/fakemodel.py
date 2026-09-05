"""Build a real ONNX model that emits detections we choose.

The published YOLOX weights are ~36 MB and cannot be fetched in every CI or
development environment. But the code that *needs* testing is not the weights
— it is the path around them: session construction, letterboxing, the
anchor-free decode, class-aware NMS, and the coordinate mapping back to
analysis space. That path is exercised exactly as hard by a graph whose output
we control, and unlike real weights the expected boxes are then known to the
pixel.

The graph is a genuine ONNX file executed by onnxruntime — not a mock. It
ignores its input and returns a constant tensor shaped like a YOLOX head.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from imgintel.util.onnx import YOLOX_STRIDES

NUM_CLASSES = 80


def anchor_layout(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """(stride, grid_x, grid_y, flat_index) for every anchor point, in order."""
    layout: list[tuple[int, int, int, int]] = []
    index = 0
    for stride in YOLOX_STRIDES:
        gw, gh = width // stride, height // stride
        for gy in range(gh):
            for gx in range(gw):
                layout.append((stride, gx, gy, index))
                index += 1
    return layout


def total_anchors(width: int, height: int) -> int:
    return sum((width // s) * (height // s) for s in YOLOX_STRIDES)


def encode_detection(
    cx: float, cy: float, w: float, h: float, stride: int, gx: int, gy: int
) -> tuple[float, float, float, float]:
    """Invert the YOLOX decode, so a wanted box becomes raw head values.

    Encoding rather than hand-writing raw numbers is what makes the test
    meaningful: if the decode changes, these expectations break.
    """
    return (
        cx / stride - gx,
        cy / stride - gy,
        math.log(w / stride),
        math.log(h / stride),
    )


def build_head(
    width: int,
    height: int,
    detections: list[dict],
) -> np.ndarray:
    """A (1, anchors, 85) YOLOX-shaped tensor containing exactly these boxes.

    Each detection is {"bbox": (x0,y0,x1,y1), "class_id": int, "score": float,
    "anchor": int} where ``anchor`` selects which grid point emits it.
    """
    layout = anchor_layout(width, height)
    head = np.zeros((1, len(layout), 5 + NUM_CLASSES), dtype=np.float32)
    # Small non-zero background so argmax is well-defined everywhere.
    head[0, :, 5:] = 0.001

    for det in detections:
        stride, gx, gy, _ = layout[det["anchor"]]
        x0, y0, x1, y1 = det["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = x1 - x0, y1 - y0
        head[0, det["anchor"], :4] = encode_detection(cx, cy, bw, bh, stride, gx, gy)
        # Confidence is objectness * class score, so split the target evenly.
        root = math.sqrt(det["score"])
        head[0, det["anchor"], 4] = root
        head[0, det["anchor"], 5 + det["class_id"]] = root
    return head


def write_constant_model(path: Path, head: np.ndarray, input_size: tuple[int, int]) -> Path:
    """Write an ONNX graph that ignores its input and returns ``head``."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    width, height = input_size
    inp = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, height, width]
    )
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(head.shape))

    const = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=numpy_helper.from_array(head.astype(np.float32), name="head"),
    )
    graph = helper.make_graph([const], "fake_yolox", [inp], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="imgintel-tests"
    )
    model.ir_version = 9  # onnxruntime 1.23 rejects newer IR versions
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
