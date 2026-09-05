"""Face **detection** with YuNet (MIT).

Detection only — locating and counting faces. No embeddings, no identity
matching, no comparison against any database. That is Phase 5, it is a
materially different thing legally, and it stays behind an explicit
lawful-basis flag. Keeping the two apart is the point of this file: an
investigator who wants "how many people are in these 10,000 images" or "what
should I blur before sharing" gets it without touching biometric identification.

YuNet over InsightFace because InsightFace's *code* is MIT but its pretrained
models are licensed for non-commercial research only — a trap that catches a
lot of projects. YuNet is MIT throughout and is 232 KB.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module

SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 500
#: Below this many pixels on the long edge a face is too small to be useful
#: for anything downstream, including redaction review.
MIN_FACE_PX = 12

_LANDMARKS = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


class FaceAnalyzer(Analyzer):
    name = "faces"
    version = "1.0.0"
    title = "Face detection"
    description = "Locates and counts faces with YuNet (detection only, no identification)"
    cost = Cost.HEAVY
    needs_models = ("face",)

    def available(self) -> Availability:
        if not has_module("cv2"):
            return Availability.no(
                "opencv is not installed", "pip install 'imgintel[faces]'"
            )
        import cv2  # noqa: PLC0415

        if not hasattr(cv2, "FaceDetectorYN"):
            return Availability.no(
                f"opencv {cv2.__version__} has no FaceDetectorYN",
                "pip install -U opencv-python-headless",
            )
        from imgintel.core.modelstore import available as models_available

        ok, reason = models_available(*self.needs_models)
        if ok:
            return Availability.yes()
        # The hint is what makes an unavailable analyzer actionable.
        return Availability.no(reason, f"imgintel models pull {' '.join(self.needs_models)}")

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:

        from imgintel.core.modelstore import status as model_status

        image = ctx.small
        height, width = image.shape[:2]

        detector = _detector(model_status("face").path, (width, height))
        # OpenCV wants BGR; ctx.small is RGB.
        _, raw = detector.detect(image[:, :, ::-1])

        faces = _build(raw, width, height)
        data = {
            "model": "yunet-2023mar",
            "score_threshold": SCORE_THRESHOLD,
            "analysis_dimensions": [width, height],
            "count": len(faces),
            "faces": faces,
            "identification": False,
            "note": "Detection only. No embeddings computed and no identity matching "
            "performed; that is a separate, explicitly gated capability.",
        }
        return self.ok(data, _findings(faces, width, height))


@lru_cache(maxsize=4)
def _build_detector(model_path: str):
    """Construct the detector once per process — it parses the whole graph."""
    import cv2  # noqa: PLC0415

    return cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(320, 320),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=TOP_K,
    )


def _detector(model_path: str, size: tuple[int, int]):
    """Reuse the cached detector, resized to this image.

    ``setInputSize`` is cheap; construction is not. This is the same
    per-process warm-state rule that the OCR engine and ONNX sessions follow.
    """
    detector = _build_detector(model_path)
    detector.setInputSize(size)
    return detector


def _build(raw: Any, width: int, height: int) -> list[dict[str, Any]]:
    """YuNet rows are [x, y, w, h, 10 landmark coords, score]."""
    if raw is None:
        return []
    faces: list[dict[str, Any]] = []
    for row in np.asarray(raw, dtype=np.float32):
        x, y, w, h = (float(v) for v in row[:4])
        if max(w, h) < MIN_FACE_PX:
            continue
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(width, int(x + w)), min(height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue

        landmarks = {
            name: [int(row[4 + i * 2]), int(row[5 + i * 2])]
            for i, name in enumerate(_LANDMARKS)
            if len(row) >= 6 + i * 2
        }
        faces.append(
            {
                "bbox": [x0, y0, x1, y1],
                "confidence": round(float(row[-1]), 4),
                "width": x1 - x0,
                "height": y1 - y0,
                "area_fraction": round(((x1 - x0) * (y1 - y0)) / max(width * height, 1), 5),
                "landmarks": landmarks,
            }
        )
    return sorted(faces, key=lambda f: -f["confidence"])


def _findings(faces: list[dict], width: int, height: int) -> list[Finding]:
    if not faces:
        return [Finding("people", "No faces detected", 0, Severity.INFO)]

    largest = max(faces, key=lambda f: f["area_fraction"])
    out = [
        Finding(
            "people",
            f"{len(faces)} face(s) detected",
            len(faces),
            Severity.MEDIUM,
            confidence=round(float(np.mean([f["confidence"] for f in faces])), 3),
            bbox=tuple(largest["bbox"]),
            detail="Faces located but not identified. Redact before sharing if the "
            "individuals are not the subject of the investigation",
        )
    ]

    # A face filling much of the frame is a portrait — likely the subject, and
    # likely high enough resolution for identification by other means.
    if largest["area_fraction"] > 0.08:
        out.append(
            Finding(
                "people",
                "Prominent face in frame",
                f"{largest['width']}x{largest['height']} px "
                f"({largest['area_fraction']:.1%} of frame)",
                Severity.MEDIUM,
                confidence=largest["confidence"],
                bbox=tuple(largest["bbox"]),
                detail="Large enough to support identification; treat as personal data",
            )
        )

    if len(faces) >= 5:
        out.append(
            Finding(
                "people",
                "Group of people",
                len(faces),
                Severity.MEDIUM,
                confidence=0.7,
                detail="Multiple identifiable individuals — a gathering, event or public space",
            )
        )

    return out
