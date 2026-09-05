"""Object detection with YOLOX-S (Apache-2.0) on onnxruntime.

YOLOX rather than Ultralytics YOLO deliberately: Ultralytics is AGPL-3.0, so
shipping it inside a distributed tool obliges you to release that tool under
AGPL or buy a commercial licence. YOLOX is Apache-2.0 for both code and
weights, with no non-commercial rider of the kind InsightFace attaches to its
models. This is the single most consequential licensing decision in the
project, and it is why the decode below is implemented by hand rather than
imported from a framework.

Boxes come back in ``ctx.small`` coordinates, like every other analyzer's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module
from imgintel.util.coco import (
    ANIMAL_CLASSES,
    COCO_CLASSES,
    PET_CLASSES,
    SCREEN_CLASSES,
    VEHICLE_CLASSES,
)
from imgintel.util.onnx import (
    batched_nms,
    decode_yolox,
    letterbox,
    load_session,
    session_input_size,
    to_nchw,
)

#: YOLOX-S was trained at 640x640; used when the graph has a dynamic axis.
DEFAULT_INPUT = (640, 640)
SCORE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100


class ObjectAnalyzer(Analyzer):
    name = "objects"
    version = "1.0.0"
    title = "Object detection"
    description = "YOLOX-S detection of 80 COCO classes, with boxes and scores"
    cost = Cost.HEAVY
    needs_models = ("detect",)

    def available(self) -> Availability:
        if not has_module("onnxruntime"):
            return Availability.no(
                "onnxruntime not installed", "pip install 'imgintel[detect]'"
            )
        from imgintel.core.modelstore import available as models_available

        ok, reason = models_available(*self.needs_models)
        if ok:
            return Availability.yes()
        # The hint is what makes an unavailable analyzer actionable.
        return Availability.no(reason, f"imgintel models pull {' '.join(self.needs_models)}")

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        from imgintel.core.modelstore import status as model_status

        model = model_status("detect")
        session = load_session(model.path)
        input_size = session_input_size(session, DEFAULT_INPUT)

        padded, transform = letterbox(ctx.small, input_size)
        # YOLOX consumes raw 0..255 values with no mean/std normalisation.
        tensor = to_nchw(padded, normalize=False)

        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        boxes, scores, labels = decode_yolox(np.asarray(raw), input_size)

        keep = scores >= SCORE_THRESHOLD
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        kept = batched_nms(boxes, scores, labels, IOU_THRESHOLD)[:MAX_DETECTIONS]
        boxes = transform.to_source(boxes[kept])
        scores, labels = scores[kept], labels[kept]

        detections = _build(boxes, scores, labels, ctx.small.shape[:2])
        counts: dict[str, int] = {}
        for det in detections:
            counts[det["label"]] = counts.get(det["label"], 0) + 1

        data = {
            "model": "yolox-s",
            "input_size": list(input_size),
            "score_threshold": SCORE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "analysis_dimensions": [ctx.small.shape[1], ctx.small.shape[0]],
            "count": len(detections),
            "counts_by_label": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "labels": sorted(counts),
            "detections": detections,
        }
        return self.ok(data, _findings(detections, counts))


def _build(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, shape: tuple[int, int]
) -> list[dict[str, Any]]:
    height, width = shape
    out: list[dict[str, Any]] = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        x0, y0, x1, y1 = (float(v) for v in box)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue  # degenerate box; nothing useful downstream
        name = COCO_CLASSES[int(label)] if int(label) < len(COCO_CLASSES) else f"class_{label}"
        out.append(
            {
                "label": name,
                "class_id": int(label),
                "confidence": round(float(score), 4),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "area_fraction": round(((x1 - x0) * (y1 - y0)) / max(width * height, 1), 5),
            }
        )
    return sorted(out, key=lambda d: -d["confidence"])


def _findings(detections: list[dict], counts: dict[str, int]) -> list[Finding]:
    if not detections:
        return [Finding("objects", "No objects detected", None, Severity.INFO)]

    summary = ", ".join(f"{label} x{n}" if n > 1 else label for label, n in counts.items())
    out = [
        Finding(
            "objects",
            f"{len(detections)} object(s) detected",
            summary[:200],
            Severity.INFO,
            confidence=round(float(np.mean([d["confidence"] for d in detections])), 3),
        )
    ]

    people = counts.get("person", 0)
    if people:
        largest = max(
            (d for d in detections if d["label"] == "person"), key=lambda d: d["area_fraction"]
        )
        out.append(
            Finding(
                "people",
                f"{people} {'person' if people == 1 else 'people'} detected",
                people,
                Severity.MEDIUM,
                confidence=largest["confidence"],
                bbox=tuple(largest["bbox"]),
                detail="Identifiable individuals may be present; consider redaction before "
                "sharing this image",
            )
        )

    screens = [d for d in detections if d["label"] in SCREEN_CLASSES]
    if screens:
        out.append(
            Finding(
                "objects",
                "Screen visible in frame",
                ", ".join(sorted({d["label"] for d in screens})),
                Severity.MEDIUM,
                confidence=0.6,
                bbox=tuple(screens[0]["bbox"]),
                detail="Displays may show readable content — run OCR on the region",
            )
        )

    pets = [d for d in detections if d["label"] in PET_CLASSES]
    if pets:
        out.append(
            Finding(
                "objects",
                "Pet present",
                ", ".join(sorted({d["label"] for d in pets})),
                Severity.LOW,
                confidence=pets[0]["confidence"],
                bbox=tuple(pets[0]["bbox"]),
                detail="Distinctive animals are surprisingly identifying in linked imagery",
            )
        )

    other_animals = {d["label"] for d in detections if d["label"] in ANIMAL_CLASSES - PET_CLASSES}
    if other_animals:
        out.append(
            Finding("objects", "Animals present", ", ".join(sorted(other_animals)), Severity.INFO)
        )

    if any(d["label"] in VEHICLE_CLASSES for d in detections):
        # The vehicles analyzer expands on this; noted here so a run without it
        # still surfaces the fact.
        out.append(
            Finding(
                "objects",
                "Vehicles present",
                ", ".join(sorted({d["label"] for d in detections if d["label"] in VEHICLE_CLASSES})),
                Severity.INFO,
            )
        )

    return out
