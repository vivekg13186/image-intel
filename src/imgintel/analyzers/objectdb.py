"""Matching specific enrolled objects — a particular vehicle, bag, or mark.

Not "is there a car", which the detector already answers, but "is this *that*
car". The distinction is what makes the entity database useful: an
investigation cares about one specific thing that recurs across a set of
images.

Matching is ORB keypoints plus a homography check rather than a learned
embedding. That trade is deliberate:

* it needs no model download, so it works on an air-gapped machine;
* it is fully inspectable — a match reports how many keypoints agreed and
  where, rather than a similarity score from an opaque encoder;
* on rigid, textured subjects it is strong.

It is weak on deformable subjects. A specific dog is not reliably matchable
this way, and the finding says so rather than implying otherwise. An
embedding-backed analyzer can be added later under a different name and both
can run side by side.

Object references are stored in the same entity database as identities, but
carry no biometric gate: a photograph of a suitcase is not biometric data.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module
from imgintel.util.orbmatch import Reference, confidence_for, match_all

#: Reference descriptors are stored as bytes in the entity DB under this name.
OBJECT_EMBEDDER = "orb-v1"


class ObjectDatabaseAnalyzer(Analyzer):
    name = "objectdb"
    version = "1.0.0"
    title = "Enrolled objects"
    description = "Matches specific enrolled objects by ORB features and homography"
    # Detections are used to say *which* detected thing was matched, but the
    # match itself is whole-frame, so the detector is optional.
    after = ("objects",)
    cost = Cost.MEDIUM

    def precheck(self, ctx: AnalysisContext) -> str | None:
        if ctx.entity_db is None:
            return "no entity database selected (pass --entity-db)"
        if not ctx.entity_db.exists():
            return f"entity database not found: {ctx.entity_db}"
        return None

    def available(self) -> Availability:
        if not has_module("cv2"):
            return Availability.no("opencv is not installed", "pip install 'imgintel[logo]'")
        return Availability.yes()

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        from imgintel.store.entities import EntityStore

        with EntityStore(ctx.entity_db) as store:
            references = _load_references(store)
            enrolled = store.count("object")
            if references:
                store.log(
                    "search:object",
                    f"{len(references)} reference(s) from {enrolled} enrolled objects",
                    operator=ctx.case.operator,
                    case_id=ctx.case.case_id,
                )

        data: dict[str, Any] = {
            "database": str(ctx.entity_db),
            "enrolled_objects": enrolled,
            "references_loaded": len(references),
            "matches": [],
        }
        if not references:
            return self.ok(data, [])

        gray = ctx.small_gray.astype(np.uint8)
        matches, keypoints = match_all(references, gray)
        data["scene_keypoints"] = keypoints
        data["matches"] = matches

        detections = ctx.data("objects").get("detections", [])
        for match in matches:
            match["overlaps"] = _overlapping_labels(match["bbox"], detections)

        return self.ok(data, [_finding(m) for m in matches])


def _load_references(store) -> list[Reference]:
    """Rebuild ORB references from the descriptors stored at enrolment.

    Descriptors are stored rather than the reference images themselves: it
    keeps the database small, and means an enrolled object does not require
    retaining a copy of every source photograph.
    """
    import cv2  # noqa: PLC0415

    references: list[Reference] = []
    for row in store.raw_references("object", OBJECT_EMBEDDER):
        decoded = _decode_descriptors(row["payload"])
        if decoded is None:
            continue
        descriptors, shape, points = decoded
        keypoints = [cv2.KeyPoint(x=float(x), y=float(y), size=float(s)) for x, y, s in points]
        references.append(
            Reference(
                name=row["name"],
                keypoints=keypoints,
                descriptors=descriptors,
                shape=shape,
                source=row["source_path"],
                entity_id=row["entity_id"],
            )
        )
    return references


def encode_descriptors(keypoints: Any, descriptors: np.ndarray, shape: tuple[int, int]) -> bytes:
    """Pack ORB output for storage in the entity database."""
    points = [[float(k.pt[0]), float(k.pt[1]), float(k.size)] for k in keypoints]
    header = json.dumps(
        {
            "shape": [int(shape[0]), int(shape[1])],
            "points": points,
            "rows": int(descriptors.shape[0]),
            "cols": int(descriptors.shape[1]),
        }
    ).encode("utf-8")
    return len(header).to_bytes(4, "big") + header + descriptors.astype(np.uint8).tobytes()


def _decode_descriptors(blob: bytes):
    try:
        header_len = int.from_bytes(blob[:4], "big")
        header = json.loads(blob[4 : 4 + header_len].decode("utf-8"))
        raw = np.frombuffer(blob[4 + header_len :], dtype=np.uint8)
        descriptors = raw.reshape(header["rows"], header["cols"])
        return descriptors, tuple(header["shape"]), header["points"]
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def _overlapping_labels(bbox: list[int], detections: list[dict]) -> list[str]:
    """Which detected objects the match sits on top of.

    Turns "the reference matched here" into "the reference matched this car",
    which is what someone reading the report actually wants.
    """
    x0, y0, x1, y1 = bbox
    labels = []
    for detection in detections:
        dx0, dy0, dx1, dy1 = detection["bbox"]
        overlap_w = min(x1, dx1) - max(x0, dx0)
        overlap_h = min(y1, dy1) - max(y0, dy0)
        if overlap_w <= 0 or overlap_h <= 0:
            continue
        area = max((x1 - x0) * (y1 - y0), 1)
        if (overlap_w * overlap_h) / area > 0.3:
            labels.append(detection["label"])
    return sorted(set(labels))


def _finding(match: dict) -> Finding:
    where = f" on {', '.join(match['overlaps'])}" if match.get("overlaps") else ""
    return Finding(
        "entity",
        f"Enrolled object matched: {match['name']}",
        f"{match['inliers']} consistent keypoints{where}",
        Severity.HIGH,
        confidence=confidence_for(match["inliers"]),
        bbox=tuple(match["bbox"]),
        detail="Geometrically verified against an enrolled reference. Strong on rigid, "
        "textured subjects; unreliable on deformable ones such as animals or clothing. "
        "Confirm visually before treating it as the same object",
    )
