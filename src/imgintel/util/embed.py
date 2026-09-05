"""Embedding backends for the entity database.

Face embeddings use SFace via OpenCV's ``FaceRecognizerSF``. SFace rather than
InsightFace's ArcFace for the licensing reason that runs through this whole
project: InsightFace's code is MIT but its pretrained models are restricted to
non-commercial research. SFace is Apache-2.0 and comes from the same
opencv_zoo collection as the YuNet detector already in use, so detection and
recognition share a lineage and a licence.

Object references use ORB descriptors rather than a learned embedding — see
:mod:`imgintel.analyzers.objectdb` for why.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

#: SFace consumes a 112x112 aligned crop and emits 128 dimensions.
FACE_EMBED_DIMS = 128
FACE_EMBEDDER = "sface-2021dec"


@lru_cache(maxsize=2)
def face_recognizer(model_path: str):
    """One recognizer per process — construction parses the whole graph."""
    import cv2  # noqa: PLC0415

    return cv2.FaceRecognizerSF.create(model=model_path, config="")


def embed_faces(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    model_path: str,
    *,
    scale: float = 1.0,
) -> list[np.ndarray | None]:
    """Embed each detected face, using its landmarks for alignment.

    Alignment matters more than it looks: SFace expects a canonically aligned
    crop, and feeding it a raw bounding box costs a large chunk of accuracy.
    The five landmarks YuNet already produces are exactly what ``alignCrop``
    wants, which is the other reason to pair these two models.

    ``scale`` maps detections from analysis space onto ``image_rgb``. Faces are
    *detected* at analysis resolution but should be *embedded* at full
    resolution: SFace works from a 112x112 crop, and a face 60 px wide in the
    downscaled image has been upsampled from nothing by the time it gets there.
    Identification is the one place where the extra decode is worth it.

    Returns one entry per detection, ``None`` where alignment failed, so the
    caller can keep embeddings aligned with the detections that produced them.
    """
    import cv2  # noqa: PLC0415

    recognizer = face_recognizer(model_path)
    bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    out: list[np.ndarray | None] = []

    for detection in detections:
        row = _detection_row(detection, scale=scale, width=width, height=height)
        if row is None:
            out.append(None)
            continue
        try:
            aligned = recognizer.alignCrop(bgr, row)
            feature = recognizer.feature(aligned)
        except cv2.error:
            out.append(None)
            continue
        out.append(np.asarray(feature, dtype=np.float32).ravel())
    return out


def _detection_row(
    detection: dict[str, Any],
    *,
    scale: float = 1.0,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray | None:
    """Rebuild the 15-column row OpenCV's alignCrop expects.

    The faces analyzer stores a friendlier structure than YuNet's raw output;
    this reverses that so the two models can talk to each other without the
    analyzer having to leak OpenCV's array layout into its findings.
    """
    bbox = detection.get("bbox")
    landmarks = detection.get("landmarks") or {}
    if not bbox or len(bbox) != 4:
        return None

    x0, y0, x1, y1 = (v * scale for v in bbox)
    if width is not None and height is not None:
        x0, x1 = max(0.0, x0), min(float(width), x1)
        y0, y1 = max(0.0, y0), min(float(height), y1)
    if x1 <= x0 or y1 <= y0:
        return None

    row = np.zeros(15, dtype=np.float32)
    row[0:4] = [x0, y0, x1 - x0, y1 - y0]

    order = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")
    for i, name in enumerate(order):
        point = landmarks.get(name)
        if not point or len(point) != 2:
            return None  # alignCrop needs all five
        row[4 + i * 2] = point[0] * scale
        row[5 + i * 2] = point[1] * scale

    row[14] = float(detection.get("confidence", 1.0))
    return row.reshape(1, -1)


def orb_descriptors(image_gray: np.ndarray, n_features: int = 1500):
    """Keypoints and descriptors for object reference matching."""

    orb = _orb(n_features)
    return orb.detectAndCompute(np.ascontiguousarray(image_gray, dtype=np.uint8), None)


@lru_cache(maxsize=2)
def _orb(n_features: int):
    import cv2  # noqa: PLC0415

    return cv2.ORB_create(nfeatures=n_features)


__all__ = [
    "FACE_EMBEDDER",
    "FACE_EMBED_DIMS",
    "embed_faces",
    "face_recognizer",
    "orb_descriptors",
]
