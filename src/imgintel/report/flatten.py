"""One image, one row.

Defines *the* set of fields that matter for triage. The batch summary CSV, the
sqlite index and the dedupe command all read from here, so "what counts as an
important field" is decided in exactly one place.

Fields are pulled defensively: an analyzer that did not run leaves its columns
empty rather than breaking the row. That is what lets a `quick` run and a
`deep` run land in the same table.
"""

from __future__ import annotations

from typing import Any

from imgintel.core.schema import FindingsDocument

#: Column order for every flat export. Append only — downstream tooling and
#: saved spreadsheets depend on positions staying put.
FLAT_COLUMNS: tuple[str, ...] = (
    # identity
    "path", "filename", "size_bytes", "format", "sha256", "pixel_sha256", "phash", "dhash",
    # geometry
    "width", "height", "megapixels", "orientation",
    # device and provenance
    "camera", "lens", "serial", "software", "artist", "captured", "filename_source",
    # location
    "latitude", "longitude", "place", "country", "altitude_m", "bearing_deg", "timezone",
    # pixels
    "jpeg_quality", "sharpness", "blur_verdict", "localized_blur_regions",
    "noise_sigma", "entropy", "dominant_color", "is_grayscale",
    # phase 3 content
    "ocr_text_chars", "ocr_blocks", "languages", "barcodes", "pii_types", "pii_count",
    # phase 4 detection
    "objects", "object_count", "people_count", "face_count", "vehicle_count",
    "plate_text", "scene", "logos",
    # phase 5 entity matching
    "identities", "identity_verdicts", "matched_objects",
    # phase 6 forensics and estimation
    "tamper_indicators", "tamper_indicator_count", "estimated_country",
    "estimated_country_confidence", "summary",
    # verdict
    "findings_high", "findings_medium", "findings_low", "findings_info", "findings_total",
    "top_findings", "analyzers_ok", "analyzers_failed", "duration_ms", "error",
)


def _get(doc: FindingsDocument, analyzer: str, *path: str, default: Any = None) -> Any:
    """Read a nested key from an analyzer's data, tolerating absence."""
    block = doc.analyzers.get(analyzer)
    if block is None or block.status != "ok":
        return default
    node: Any = block.data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return default if node is None else node


def flatten(doc: FindingsDocument, error: str | None = None) -> dict[str, Any]:
    """Project a findings document onto :data:`FLAT_COLUMNS`."""
    sev = doc.summary.by_severity

    # The three or four findings a human would actually want in a summary row.
    notable = [f for f in doc.findings if f.severity in ("high", "medium")][:4]
    top = "; ".join(f"{f.label}={f.value}" if f.value is not None else f.label for f in notable)

    palette = _get(doc, "color", "palette", default=[])
    dominant = palette[0]["hex"] if palette else None

    sources = _get(doc, "fileinfo", "filename_sources", default=[])

    row: dict[str, Any] = {
        "path": doc.target.path,
        "filename": doc.target.filename,
        "size_bytes": doc.target.size_bytes,
        "format": _get(doc, "fileinfo", "format_actual"),
        "sha256": _get(doc, "hashes", "sha256"),
        "pixel_sha256": _get(doc, "perceptual", "pixel_sha256"),
        "phash": _get(doc, "perceptual", "phash"),
        "dhash": _get(doc, "perceptual", "dhash"),
        "width": _get(doc, "fileinfo", "width"),
        "height": _get(doc, "fileinfo", "height"),
        "megapixels": _get(doc, "fileinfo", "megapixels"),
        "orientation": _get(doc, "fileinfo", "orientation"),
        "camera": _camera(doc),
        "lens": _get(doc, "exif", "highlights", "lens"),
        "serial": _get(doc, "exif", "highlights", "serial"),
        "software": _get(doc, "exif", "highlights", "software"),
        "artist": _get(doc, "exif", "highlights", "artist"),
        "captured": _get(doc, "exif", "highlights", "datetime_original"),
        "filename_source": ", ".join(sources) if sources else None,
        "latitude": _get(doc, "gps", "latitude"),
        "longitude": _get(doc, "gps", "longitude"),
        "place": _place(doc),
        "country": _get(doc, "gps", "place", "country_code"),
        "altitude_m": _get(doc, "gps", "altitude_m"),
        "bearing_deg": _get(doc, "gps", "img_direction_deg"),
        "timezone": _get(doc, "gps", "timezone"),
        "jpeg_quality": _get(doc, "quality", "jpeg", "quality_estimate"),
        "sharpness": _get(doc, "blur", "laplacian_variance"),
        "blur_verdict": _get(doc, "blur", "verdict"),
        "localized_blur_regions": len(_get(doc, "blur", "localized_blur", default=[]) or []),
        "noise_sigma": _get(doc, "quality", "noise_sigma"),
        "entropy": _get(doc, "quality", "entropy_bits"),
        "dominant_color": dominant,
        "is_grayscale": _get(doc, "color", "is_grayscale"),
        "ocr_text_chars": _get(doc, "ocr", "char_count"),
        "ocr_blocks": _get(doc, "ocr", "block_count"),
        "languages": ", ".join(_get(doc, "language", "languages", default=[]) or []) or None,
        "barcodes": _barcodes(doc),
        "pii_types": ", ".join(sorted(_get(doc, "sensitive", "types", default=[]) or [])) or None,
        "pii_count": _get(doc, "sensitive", "match_count"),
        "objects": _object_summary(doc),
        "object_count": _get(doc, "objects", "count"),
        "people_count": _get(doc, "objects", "counts_by_label", "person"),
        "face_count": _get(doc, "faces", "count"),
        "vehicle_count": _get(doc, "vehicles", "count"),
        "plate_text": _plate_text(doc),
        "scene": _get(doc, "scene", "primary"),
        "logos": ", ".join(m["name"] for m in _get(doc, "logo", "matches", default=[]) or [])
        or None,
        "identities": _identities(doc, conclusive_only=True),
        # Inconclusive results belong in the summary too: a row showing only
        # confirmed matches would hide every near-miss the analyst should see.
        "identity_verdicts": _identity_verdicts(doc),
        "matched_objects": ", ".join(
            m["name"] for m in _get(doc, "objectdb", "matches", default=[]) or []
        )
        or None,
        "tamper_indicators": ", ".join(
            _get(doc, "tamper", "indicators_raised", default=[]) or []
        )
        or None,
        "tamper_indicator_count": _get(doc, "tamper", "indicator_count"),
        "estimated_country": _estimated_country(doc, "country_name"),
        "estimated_country_confidence": _estimated_country(doc, "confidence"),
        "summary": _get(doc, "summary", "headline"),
        "findings_high": sev.get("high", 0),
        "findings_medium": sev.get("medium", 0),
        "findings_low": sev.get("low", 0),
        "findings_info": sev.get("info", 0),
        "findings_total": doc.summary.findings_total,
        "top_findings": top or None,
        "analyzers_ok": doc.summary.analyzers_ok,
        "analyzers_failed": doc.summary.analyzers_failed,
        "duration_ms": round(doc.run.duration_ms, 1),
        "error": error,
    }
    return {key: row.get(key) for key in FLAT_COLUMNS}


def failed_row(path: str, error: str) -> dict[str, Any]:
    """A row for an image that could not be analyzed at all.

    Batch output must account for every input; silently dropping failures is
    how an investigator ends up believing a file was clean.
    """
    row = dict.fromkeys(FLAT_COLUMNS)
    row["path"] = path
    row["filename"] = path.rsplit("/", 1)[-1]
    row["error"] = error
    return row


def _camera(doc: FindingsDocument) -> str | None:
    for finding in doc.findings:
        if finding.analyzer == "exif" and finding.label == "Camera":
            return str(finding.value)
    return None


def _place(doc: FindingsDocument) -> str | None:
    place = _get(doc, "gps", "place")
    if not isinstance(place, dict):
        return None
    parts = [place.get("name"), place.get("admin1"), place.get("country_code")]
    return ", ".join(str(p) for p in parts if p) or None


def _estimated_country(doc: FindingsDocument, field: str):
    """The leading location hypothesis — from GPS if known, else from evidence."""
    geo = _get(doc, "geoestimate", default={})
    if not isinstance(geo, dict):
        return None
    if geo.get("resolved"):
        # A GPS fix is not a hypothesis; report it as a certainty.
        return geo.get("country") if field == "country_name" else 1.0
    hypotheses = geo.get("hypotheses") or []
    return hypotheses[0].get(field) if hypotheses else None


def _identities(doc: FindingsDocument, *, conclusive_only: bool) -> str | None:
    matches = _get(doc, "facedb", "matches", default=[]) or []
    names = [
        m["entity"]
        for m in matches
        if isinstance(m, dict)
        and m.get("entity")
        and (m.get("verdict") == "match" or not conclusive_only)
    ]
    return ", ".join(dict.fromkeys(names)) or None


def _identity_verdicts(doc: FindingsDocument) -> str | None:
    """Every verdict, so an inconclusive result is visible in a summary row."""
    matches = _get(doc, "facedb", "matches", default=[]) or []
    parts = [
        f"{m['entity']}:{m['verdict']}"
        for m in matches
        if isinstance(m, dict) and m.get("entity") and m.get("verdict")
    ]
    return "; ".join(parts) or None


def _object_summary(doc: FindingsDocument) -> str | None:
    counts = _get(doc, "objects", "counts_by_label", default={})
    if not isinstance(counts, dict) or not counts:
        return None
    return ", ".join(f"{label} x{n}" if n > 1 else label for label, n in counts.items())


def _plate_text(doc: FindingsDocument) -> str | None:
    candidates = _get(doc, "vehicles", "plate_candidates", default=[]) or []
    texts = [c.get("ocr_text") for c in candidates if isinstance(c, dict) and c.get("ocr_text")]
    return " | ".join(texts) or None


def _barcodes(doc: FindingsDocument) -> str | None:
    codes = _get(doc, "codes", "codes", default=[]) or []
    texts = [str(c.get("text", ""))[:80] for c in codes if isinstance(c, dict)]
    return " | ".join(t for t in texts if t) or None
