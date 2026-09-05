"""Embedded metadata: EXIF, XMP, IPTC, PNG text chunks, MakerNotes.

Uses the ``exiftool`` binary when present — it reads MakerNotes (camera serial,
shutter count, lens ID, owner name), which is where most of the identifying
intel actually lives and which Pillow cannot see. Falls back to Pillow so the
analyzer still produces useful output on a bare install.
"""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from PIL import ExifTags, Image

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import binary_path

# Tags that identify a physical device or a person, not just a camera model.
_IDENTIFYING = {
    "SerialNumber", "InternalSerialNumber", "BodySerialNumber", "CameraSerialNumber",
    "LensSerialNumber", "OwnerName", "CameraOwnerName", "Artist", "Creator",
    "Copyright", "By-line", "UserComment", "ImageDescription", "XPAuthor",
    "XPComment", "XPSubject", "HostComputer", "DeviceSettingDescription",
}
# Software strings implying the pixels were edited after capture.
_EDITORS = (
    "photoshop", "lightroom", "gimp", "affinity", "capture one", "luminar",
    "snapseed", "picsart", "pixlr", "canva", "paint.net", "facetune",
    "krita", "darktable", "rawtherapee", "topaz", "remini",
)
# Markers of generative-AI provenance.
_AI_MARKERS = (
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dall_e",
    "dalle", "firefly", "imagen", "flux", "comfyui", "automatic1111",
    "novelai", "leonardo.ai", "ideogram", "sora",
)
_AI_PNG_KEYS = {"parameters", "prompt", "workflow", "negative_prompt", "sd-metadata"}
# Container/encoder bookkeeping — present in every file, carries no intel.
_CONTAINER_NOISE = {
    "exif", "icc_profile", "xmp", "photoshop", "jfif", "jfif_version", "jfif_unit",
    "jfif_density", "dpi", "progression", "progressive", "optimize", "streamtype",
    "adobe", "adobe_transform", "transparency", "gamma", "srgb", "chromaticity",
    "aspect", "background", "loop", "duration", "version", "interlace", "palette",
    "compression", "resolution", "resolution_unit", "signed", "bits", "extrema",
}
#: Groups that constitute genuine capture metadata (as opposed to file chunks).
_REAL_METADATA_GROUPS = ("EXIF", "XMP", "IPTC", "MakerNotes", "Composite", "GPS")


def _clean(value: Any, depth: int = 0) -> Any:
    """Coerce Pillow/exiftool values into JSON-safe primitives."""
    if depth > 4:
        return str(value)[:200]
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8", errors="strict").strip("\x00").strip()
            return text[:500] if text else None
        except UnicodeDecodeError:
            return f"<{len(value)} bytes: {value[:16].hex()}...>"
    if isinstance(value, (list, tuple)):
        return [_clean(v, depth + 1) for v in value][:64]
    if isinstance(value, dict):
        return {str(k): _clean(v, depth + 1) for k, v in list(value.items())[:200]}
    if isinstance(value, str):
        return value.strip("\x00").strip()[:500] or None
    if hasattr(value, "numerator") and hasattr(value, "denominator"):  # IFDRational
        try:
            return float(value) if value.denominator else None
        except (ZeroDivisionError, TypeError):
            return None
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def _read_with_exiftool(path: str) -> dict[str, Any] | None:
    exe = binary_path("exiftool")
    if not exe:
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [exe, "-json", "-G", "-n", "-a", "-u", "-charset", "utf8", path],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0 and not proc.stdout:
            return None
        payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        return {k: _clean(v) for k, v in payload[0].items()} if payload else {}
    except Exception:  # noqa: BLE001 - fall back to Pillow
        return None


def _read_with_pillow(img: Image.Image) -> dict[str, Any]:
    tags: dict[str, Any] = {}

    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        exif = None

    if exif:
        for tag_id, value in exif.items():
            name = ExifTags.TAGS.get(tag_id, f"Tag{tag_id:04X}")
            cleaned = _clean(value)
            if cleaned is not None:
                tags[f"EXIF:{name}"] = cleaned
        # The Exif sub-IFD holds the interesting capture settings.
        for ifd_id, prefix in ((0x8769, "EXIF"), (0xA005, "Interop")):
            try:
                sub = exif.get_ifd(ifd_id)
            except Exception:  # noqa: BLE001
                continue
            for tag_id, value in (sub or {}).items():
                name = ExifTags.TAGS.get(tag_id, f"Tag{tag_id:04X}")
                cleaned = _clean(value)
                if cleaned is not None:
                    tags[f"{prefix}:{name}"] = cleaned
        try:
            gps = exif.get_ifd(0x8825)
        except Exception:  # noqa: BLE001
            gps = None
        for tag_id, value in (gps or {}).items():
            name = ExifTags.GPSTAGS.get(tag_id, f"GPSTag{tag_id:04X}")
            cleaned = _clean(value)
            if cleaned is not None:
                tags[f"GPS:{name}"] = cleaned

    # XMP (Pillow >= 8.2).
    try:
        xmp = img.getxmp()
        if xmp:
            tags["XMP:_raw"] = _clean(xmp)
    except Exception:  # noqa: BLE001
        pass

    # PNG text chunks and other container-level info. Encoder bookkeeping is
    # excluded so that "no embedded metadata" stays a meaningful signal: a JPEG
    # carrying only JFIF density fields has been stripped, and saying so is the
    # whole point.
    for key, value in (img.info or {}).items():
        if key in _CONTAINER_NOISE:
            continue
        cleaned = _clean(value)
        if cleaned not in (None, "", [], {}):
            tags[f"File:{key}"] = cleaned

    return tags


def _camera_label(highlights: dict[str, Any]) -> str:
    """Join Make and Model without repeating the brand.

    Manufacturers routinely store Make="Canon" and Model="Canon EOS R5", so a
    naive join produces "Canon Canon EOS R5".
    """
    make = str(highlights.get("make") or "").strip()
    model = str(highlights.get("model") or "").strip()
    if not make:
        return model
    if not model:
        return make
    if model.lower().startswith(make.lower()):
        return model
    return f"{make} {model}"


def _first(tags: dict[str, Any], *names: str) -> Any:
    """Look up a bare tag name across whatever group prefix it landed under."""
    for name in names:
        for key, value in tags.items():
            if key.split(":", 1)[-1].lower() == name.lower() and value not in (None, ""):
                return value
    return None


class ExifAnalyzer(Analyzer):
    name = "exif"
    version = "1.0.0"
    title = "Embedded metadata"
    description = "EXIF, XMP, IPTC and PNG text chunks; camera, software and authorship"
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        tags = _read_with_exiftool(str(ctx.path))
        source = "exiftool"
        if tags is None:
            source = "pillow"
            # The header carries EXIF/XMP/text chunks; no decode required.
            tags = _read_with_pillow(ctx.header) if ctx.readable else {}

        if source == "exiftool":
            # Drop exiftool's own bookkeeping so the output is all subject data.
            tags = {
                k: v
                for k, v in tags.items()
                if not k.startswith(("ExifTool:", "SourceFile"))
                and k not in ("File:Directory", "File:FileName", "File:FilePath")
            }

        highlights = {
            "make": _first(tags, "Make"),
            "model": _first(tags, "Model"),
            "lens": _first(tags, "LensModel", "LensID", "Lens", "LensInfo"),
            "software": _first(tags, "Software", "ProcessingSoftware", "CreatorTool"),
            "datetime_original": _first(tags, "DateTimeOriginal", "CreateDate"),
            "datetime_modified": _first(tags, "ModifyDate", "DateTime"),
            "offset_time": _first(tags, "OffsetTimeOriginal", "OffsetTime"),
            "orientation": _first(tags, "Orientation"),
            "iso": _first(tags, "ISO", "ISOSpeedRatings"),
            "f_number": _first(tags, "FNumber", "ApertureValue"),
            "exposure_time": _first(tags, "ExposureTime"),
            "focal_length": _first(tags, "FocalLength"),
            "flash": _first(tags, "Flash"),
            "artist": _first(tags, "Artist", "Creator", "By-line"),
            "copyright": _first(tags, "Copyright", "Rights"),
            "owner": _first(tags, "OwnerName", "CameraOwnerName"),
            "serial": _first(tags, "SerialNumber", "BodySerialNumber", "InternalSerialNumber"),
            "user_comment": _first(tags, "UserComment", "ImageDescription"),
        }
        highlights = {k: v for k, v in highlights.items() if v not in (None, "")}

        data = {
            "source": source,
            "tag_count": len(tags),
            "groups": sorted({k.split(":", 1)[0] for k in tags}),
            "highlights": highlights,
            "tags": tags,
            "has_thumbnail": any("ThumbnailImage" in k or "ThumbnailOffset" in k for k in tags),
        }

        findings = self._findings(ctx, tags, highlights)
        return self.ok(data, findings)

    # -- interpretation ----------------------------------------------------

    def _findings(
        self, ctx: AnalysisContext, tags: dict[str, Any], hl: dict[str, Any]
    ) -> list[Finding]:
        out: list[Finding] = []
        # exiftool always reports File:* basics, so emptiness is the wrong test
        # for "stripped" — look for actual capture metadata groups instead.
        has_capture_meta = any(k.split(":", 1)[0] in _REAL_METADATA_GROUPS for k in tags)

        if not has_capture_meta:
            out.append(
                Finding(
                    "metadata",
                    "No capture metadata",
                    None,
                    Severity.LOW,
                    detail="No EXIF/XMP/IPTC present — stripped by a messaging app or social "
                    "platform, re-encoded, screenshotted, or deliberately scrubbed",
                )
            )
            if not tags:
                return out

        if hl.get("make") or hl.get("model"):
            out.append(Finding("device", "Camera", _camera_label(hl), Severity.INFO))
        if hl.get("lens"):
            out.append(Finding("device", "Lens", hl["lens"], Severity.INFO))

        # Device- and person-identifying tags.
        for key, value in tags.items():
            bare = key.split(":", 1)[-1]
            if bare in _IDENTIFYING and value not in (None, "", 0):
                sev = Severity.MEDIUM if "serial" in bare.lower() else Severity.LOW
                out.append(
                    Finding(
                        "attribution",
                        f"Identifying tag: {bare}",
                        value,
                        sev,
                        detail="Ties the image to a specific device or person",
                    )
                )

        # Editing software.
        software = str(hl.get("software") or "")
        if software:
            out.append(Finding("provenance", "Software", software, Severity.INFO))
            match = next((e for e in _EDITORS if e in software.lower()), None)
            if match:
                out.append(
                    Finding(
                        "manipulation",
                        "Edited in image-editing software",
                        software,
                        Severity.MEDIUM,
                        confidence=0.8,
                        detail=f"Software tag names {match}; pixels were processed after capture",
                    )
                )

        # Generative-AI provenance.
        blob = " ".join(f"{k} {v}" for k, v in tags.items()).lower()
        ai_hits = sorted({m for m in _AI_MARKERS if m in blob})
        ai_keys = sorted({k for k in tags if k.split(":", 1)[-1].lower() in _AI_PNG_KEYS})
        if ai_hits or ai_keys:
            out.append(
                Finding(
                    "provenance",
                    "Generative-AI metadata present",
                    ai_hits or ai_keys,
                    Severity.HIGH,
                    confidence=0.9,
                    detail="Metadata names an image-generation tool or carries its parameters",
                )
            )

        # C2PA / Content Credentials.
        head = ctx.raw[:200_000]
        if b"c2pa" in head or b"jumbf" in head or b"urn:uuid:" in head[:4096]:
            if b"c2pa" in head:
                out.append(
                    Finding(
                        "provenance",
                        "C2PA content credentials present",
                        None,
                        Severity.INFO,
                        detail="Signed provenance manifest embedded; verify separately",
                    )
                )

        # Timestamp coherence.
        orig = str(hl.get("datetime_original") or "")
        mod = str(hl.get("datetime_modified") or "")
        if orig and mod and orig != mod:
            out.append(
                Finding(
                    "timeline",
                    "Capture and modification timestamps differ",
                    f"captured {orig}, modified {mod}",
                    Severity.LOW,
                    detail="Normal after any edit or export; note the gap",
                )
            )
        if orig:
            out.append(Finding("timeline", "Captured", orig, Severity.INFO))
        elif has_capture_meta:
            # Only meaningful when real capture metadata exists but the
            # timestamp specifically is missing.
            out.append(
                Finding(
                    "timeline",
                    "No capture timestamp",
                    None,
                    Severity.LOW,
                    detail="Capture metadata present but DateTimeOriginal absent — "
                    "typically re-encoded or partially stripped",
                )
            )

        # The gps analyzer parses these properly; only note them here when it
        # is not part of this run, to avoid duplicating its finding.
        if any(re.search(r"gps", k, re.I) for k in tags) and "gps" not in ctx.planned:
            out.append(
                Finding(
                    "location",
                    "GPS tags present",
                    None,
                    Severity.MEDIUM,
                    detail="Raw GPS tags are in the metadata but were not parsed. Include the "
                    "`gps` analyzer to get coordinates, a place name and the timestamp "
                    "cross-check that tests whether they were edited",
                )
            )

        return out
