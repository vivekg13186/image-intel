"""Vehicles, and the plate regions worth reading.

Derived from the object detector rather than running a second network: the
COCO classes already cover car/truck/bus/motorcycle, and a dedicated vehicle
model would add ~40 MB to detect the same things.

Plate *localisation* is done geometrically — number plates sit low and central
on a vehicle and have a distinctive aspect ratio, so a candidate region can be
proposed without a plate detector. Whether that region contains readable text
is a question for OCR, which is the honest division of labour: this analyzer
says "look here", it does not claim to have read anything.

Plate *format validation* is deliberately absent. It is jurisdiction-specific
and belongs in a rules file a user can extend, not hard-coded to one country.
"""

from __future__ import annotations

from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.coco import MOTOR_VEHICLE_CLASSES, VEHICLE_CLASSES

#: Plates are wide and short. Generous because of viewing angle.
PLATE_ASPECT_MIN = 2.0
PLATE_ASPECT_MAX = 6.5
#: A vehicle smaller than this fraction of the frame has no readable plate.
MIN_VEHICLE_AREA = 0.01


class VehicleAnalyzer(Analyzer):
    name = "vehicles"
    version = "1.0.0"
    title = "Vehicles"
    description = "Vehicle inventory and candidate number-plate regions"
    requires = ("objects",)
    # OCR text is matched against plate regions when both ran, but plate
    # localisation works without it.
    after = ("ocr",)
    # CHEAP because it reads other analyzers' output and never touches pixels.
    # It is deep-only regardless, since `objects` is HEAVY and profile
    # selection drops analyzers whose dependencies are out of reach.
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        detections = ctx.data("objects").get("detections", [])
        vehicles = [d for d in detections if d["label"] in VEHICLE_CLASSES]

        if not vehicles:
            return self.ok({"count": 0, "vehicles": [], "plate_candidates": []}, [])

        blocks = ctx.data("ocr").get("blocks", [])
        entries: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        for vehicle in vehicles:
            entry = dict(vehicle)
            region = _plate_region(vehicle)
            entry["plate_region"] = region
            if region is not None:
                text = _text_in(region, blocks)
                candidate = {
                    "vehicle": vehicle["label"],
                    "vehicle_bbox": vehicle["bbox"],
                    "bbox": region,
                    "ocr_text": text,
                }
                candidates.append(candidate)
                entry["plate_text"] = text
            entries.append(entry)

        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry["label"]] = counts.get(entry["label"], 0) + 1

        data = {
            "count": len(entries),
            "counts_by_type": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "motor_vehicle_count": sum(
                n for label, n in counts.items() if label in MOTOR_VEHICLE_CLASSES
            ),
            "vehicles": entries,
            "plate_candidates": candidates,
            "ocr_available": bool(blocks),
        }
        return self.ok(data, _findings(entries, counts, candidates, bool(blocks)))


def _plate_region(vehicle: dict) -> list[int] | None:
    """Propose where a plate would be on this vehicle, if it is big enough.

    Lower-central band of the bounding box, which is where plates sit on
    front and rear views. A side-on vehicle has no visible plate and the
    aspect check below discards it.
    """
    if vehicle.get("area_fraction", 0) < MIN_VEHICLE_AREA:
        return None
    x0, y0, x1, y1 = vehicle["bbox"]
    width, height = x1 - x0, y1 - y0
    if width < 40 or height < 30:
        return None

    band_top = y0 + int(height * 0.60)
    band_bottom = y0 + int(height * 0.95)
    inset = int(width * 0.22)
    region = [x0 + inset, band_top, x1 - inset, band_bottom]

    region_w, region_h = region[2] - region[0], region[3] - region[1]
    if region_w <= 0 or region_h <= 0:
        return None
    aspect = region_w / region_h
    if not PLATE_ASPECT_MIN <= aspect <= PLATE_ASPECT_MAX:
        return None
    return region


def _text_in(region: list[int], blocks: list[dict]) -> str | None:
    """OCR text whose box centre falls inside the candidate region."""
    x0, y0, x1, y1 = region
    hits = []
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            hits.append(block.get("text", "").strip())
    return " ".join(h for h in hits if h) or None


def _findings(
    entries: list[dict], counts: dict[str, int], candidates: list[dict], ocr_ran: bool
) -> list[Finding]:
    summary = ", ".join(f"{label} x{n}" if n > 1 else label for label, n in counts.items())
    largest = max(entries, key=lambda v: v.get("area_fraction", 0))

    out = [
        Finding(
            "vehicles",
            f"{len(entries)} vehicle(s) detected",
            summary,
            Severity.INFO,
            bbox=tuple(largest["bbox"]),
        )
    ]

    read = [c for c in candidates if c["ocr_text"]]
    for candidate in read:
        out.append(
            Finding(
                "vehicles",
                "Text in plate region",
                candidate["ocr_text"][:40],
                Severity.HIGH,
                confidence=0.55,
                bbox=tuple(candidate["bbox"]),
                detail="Text found where a number plate would sit. Not validated against any "
                "national format — confirm visually before treating it as a registration",
            )
        )

    unread = [c for c in candidates if not c["ocr_text"]]
    if unread:
        detail = (
            "Plate may be readable at full resolution — re-run with OCR on a crop"
            if ocr_ran
            else "Run with --profile deep to attempt reading these regions"
        )
        out.append(
            Finding(
                "vehicles",
                f"{len(unread)} candidate plate region(s), no text read",
                len(unread),
                Severity.MEDIUM,
                confidence=0.4,
                bbox=tuple(unread[0]["bbox"]),
                detail=detail,
            )
        )

    return out
