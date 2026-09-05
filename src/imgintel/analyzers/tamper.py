"""Manipulation indicators — a panel, not a verdict.

**This analyzer never says an image is fake.** It reports independent
indicators, each with a confidence and an explanation of what could produce it
benignly. That is not hedging; it is the difference between a forensic tool and
a liability. ELA in particular is widely misread as a manipulation detector,
and published evaluations find it unreliable as a standalone test — a bright
region in an ELA map is far more often texture or a recent recompression than
an edit.

Seven indicators run, deliberately measuring different things so that
agreement between them means something:

* **error level** — regions that respond differently to recompression
* **JPEG ghosts** — a dip in the recompression curve reveals a prior save
* **DCT first digits** — double quantisation disturbs a Benford-like law
* **quantization tables** — non-standard tables fingerprint the encoder
* **noise floor** — a splice carries its source camera's noise
* **copy-move** — a region duplicated elsewhere in the same frame
* **metadata contradictions** — editor software, a mismatched thumbnail

Every indicator is gated on ``quality.content_type``: none of them mean
anything on a screenshot or a document scan, where "compressed differently in
different places" is simply how the image was made.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.forensics import (
    copy_move_candidates,
    dct_first_digit_deviation,
    error_level,
    jpeg_ghost_map,
    jpeg_ghost_sweep,
    noise_field,
    outlier_tiles,
    quant_table_signature,
    tiled_stats,
)

#: Software strings implying the pixels were edited after capture.
_EDITORS = (
    "photoshop", "lightroom", "gimp", "affinity", "capture one", "luminar",
    "snapseed", "picsart", "pixlr", "canva", "paint.net", "krita",
    "darktable", "rawtherapee", "topaz", "remini", "facetune",
)
#: Outlier strength before a tile is worth reporting.
ELA_Z = 3.5
NOISE_Z = 3.5
#: Benford divergence above this is unusual for a singly-compressed image.
#: Measured on 1/f test content: single compression sits around 0.22, double
#: around 0.49-0.59. The absolute value is content-dependent, which is why this
#: indicator carries low confidence and is only meaningful as corroboration.
BENFORD_SUSPICIOUS = 0.40
#: Above this share of ghosting tiles it is the whole image's own compression
#: history, not a localised splice.
LOCALISED_GHOST_MAX = 0.40


class TamperAnalyzer(Analyzer):
    name = "tamper"
    version = "1.0.0"
    title = "Manipulation indicators"
    description = "Panel of independent tamper indicators — reports signals, never verdicts"
    requires = ("quality",)
    # Metadata contradictions and the embedded thumbnail come from exif when
    # it ran, but the pixel indicators stand alone.
    after = ("exif",)
    cost = Cost.HEAVY

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        content = ctx.data("quality").get("content_type", "photograph")
        indicators: dict[str, Any] = {}
        findings: list[Finding] = []

        data: dict[str, Any] = {
            "content_type": content,
            "analysis_dimensions": [ctx.small.shape[1], ctx.small.shape[0]],
            "indicators": indicators,
            "disclaimer": (
                "These are indicators, not conclusions. Several have ordinary "
                "explanations; agreement between independent indicators is what "
                "carries weight, not any single one."
            ),
        }

        if content != "photograph":
            data["skipped_reason"] = (
                f"pixel forensics are not meaningful on a {content}: uneven compression "
                "and flat noise are how such images are produced"
            )
            findings.append(
                Finding(
                    "manipulation",
                    "Pixel forensics not applicable",
                    content,
                    Severity.INFO,
                    detail=data["skipped_reason"],
                )
            )
            # Metadata contradictions still apply to any file.
            findings.extend(self._metadata(ctx, indicators))
            return self.ok(data, findings)

        rgb = ctx.small
        gray = ctx.small_gray

        findings.extend(self._error_level(rgb, indicators))
        findings.extend(self._ghosts(ctx, rgb, indicators))
        findings.extend(self._dct(gray, indicators))
        findings.extend(self._quant_tables(ctx, indicators))
        findings.extend(self._noise(gray, indicators))
        findings.extend(self._copy_move(gray, indicators))
        findings.extend(self._metadata(ctx, indicators))

        # A corroboration count, not a score. Explicitly not summed into a
        # single "manipulation probability": these indicators are not
        # independent enough for that arithmetic to mean anything.
        raised = sorted(k for k, v in indicators.items() if v.get("raised"))
        data["indicators_raised"] = raised
        data["indicator_count"] = len(raised)

        if len(raised) >= 3:
            findings.append(
                Finding(
                    "manipulation",
                    "Multiple independent indicators raised",
                    ", ".join(raised),
                    Severity.HIGH,
                    confidence=0.6,
                    detail="Several indicators that measure different things agree. That is "
                    "worth investigating, but is still not proof of manipulation — "
                    "examine each one and the image itself before concluding anything",
                )
            )
        return self.ok(data, findings)

    # -- indicators --------------------------------------------------------

    def _error_level(self, rgb: np.ndarray, indicators: dict) -> list[Finding]:
        field = error_level(rgb)
        values, tiles = tiled_stats(field)
        outliers = outlier_tiles(values, tiles, z=ELA_Z)

        indicators["error_level"] = {
            "raised": bool(outliers),
            "mean": round(float(field.mean()), 4),
            "tiles": tiles,
            "outliers": outliers[:8],
        }
        if not outliers:
            return []
        return [
            Finding(
                "manipulation",
                "Uneven error level",
                f"{len(outliers)} region(s), strongest z={outliers[0]['z_score']}",
                Severity.LOW,
                confidence=0.35,
                bbox=tuple(outliers[0]["bbox"]),
                detail="Regions responding differently to recompression. Sharp edges, text "
                "overlays and areas of flat colour do this routinely — ELA alone is not "
                "evidence of editing and is frequently misread as if it were",
            )
        ]

    def _ghosts(self, ctx: AnalysisContext, rgb: np.ndarray, indicators: dict) -> list[Finding]:
        if ctx.jpeg_quant_tables is None:
            indicators["jpeg_ghost"] = {"raised": False, "reason": "not a JPEG"}
            return []

        ghost_map = jpeg_ghost_map(rgb)
        if not ghost_map.get("available"):
            indicators["jpeg_ghost"] = {"raised": False, "reason": ghost_map.get("reason")}
            return []

        fraction = ghost_map["ghost_fraction"]
        tiles = ghost_map["ghost_tiles"]
        qualities = [t["ghost_quality"] for t in tiles]

        # Every tile ghosting at the same quality is the *whole image's* prior
        # compression, not a splice — informative, but not a manipulation
        # signal. Only a minority of tiles disagreeing localises anything.
        localised = 0 < fraction <= LOCALISED_GHOST_MAX
        indicators["jpeg_ghost"] = {
            "raised": bool(localised),
            "ghost_fraction": fraction,
            "ghost_count": ghost_map["ghost_count"],
            "ghost_tiles": tiles,
            "whole_image": fraction > LOCALISED_GHOST_MAX,
        }

        out: list[Finding] = []
        if fraction > LOCALISED_GHOST_MAX and qualities:
            out.append(
                Finding(
                    "provenance",
                    "Prior compression recovered from pixels",
                    f"q≈{max(set(qualities), key=qualities.count)} across the frame",
                    Severity.INFO,
                    confidence=0.6,
                    detail="The whole image responds this way, so this is its own encoding "
                    "history rather than evidence of an edit",
                )
            )
        elif localised:
            worst = tiles[0]
            out.append(
                Finding(
                    "manipulation",
                    "Localised recompression ghost",
                    f"{ghost_map['ghost_count']} region(s) with a prior quality near "
                    f"q={worst['ghost_quality']}, unlike the rest of the frame",
                    Severity.HIGH,
                    confidence=0.6,
                    bbox=tuple(worst["bbox"]),
                    detail="These regions were compressed at a quality the rest of the image "
                    "was not, which is what a splice from another file looks like. Strong "
                    "when it coincides with a noise-floor anomaly in the same place",
                )
            )
        # Also keep the global curve: it recovers the file's own last-save
        # quality even when nothing is localised.
        sweep = jpeg_ghost_sweep(rgb)
        indicators["jpeg_ghost"]["global_minimum_quality"] = sweep.minimum_quality
        indicators["jpeg_ghost"]["global_curve"] = sweep.curve
        return out

    def _dct(self, gray: np.ndarray, indicators: dict) -> list[Finding]:
        divergence = dct_first_digit_deviation(gray)
        raised = divergence > BENFORD_SUSPICIOUS
        indicators["dct_first_digit"] = {
            "raised": bool(raised),
            "divergence": round(divergence, 5),
            "threshold": BENFORD_SUSPICIOUS,
        }
        if not raised:
            return []
        return [
            Finding(
                "manipulation",
                "DCT coefficients depart from the expected distribution",
                round(divergence, 4),
                Severity.LOW,
                confidence=0.3,
                detail="Consistent with a second quantisation, i.e. the image was compressed, "
                "altered and compressed again. Heavy but honest editing produces this too",
            )
        ]

    def _quant_tables(self, ctx: AnalysisContext, indicators: dict) -> list[Finding]:
        tables = ctx.jpeg_quant_tables
        if not tables:
            indicators["quant_tables"] = {"raised": False, "reason": "not a JPEG"}
            return []

        signatures = {str(k): quant_table_signature(v) for k, v in sorted(tables.items())}
        standard = ctx.data("quality").get("jpeg", {}).get("standard_tables")
        software = str(ctx.data("exif").get("highlights", {}).get("software") or "")
        editor = next((e for e in _EDITORS if e in software.lower()), None)

        # Non-standard tables plus an editor name is the corroborated case:
        # the tables say "not a stock encoder" and the metadata says which.
        raised = standard is False and bool(editor)
        indicators["quant_tables"] = {
            "raised": bool(raised),
            "standard": standard,
            "signatures": signatures,
            "software": software or None,
        }
        if not raised:
            return []
        return [
            Finding(
                "manipulation",
                "Encoder tables match editing software, not a camera",
                f"{software}",
                Severity.MEDIUM,
                confidence=0.6,
                detail="Quantization tables do not follow the libjpeg curve and the metadata "
                "names an image editor. Together these place the last save in that "
                "application rather than in a camera",
            )
        ]

    def _noise(self, gray: np.ndarray, indicators: dict) -> list[Finding]:
        values, tiles = noise_field(gray)
        outliers = outlier_tiles(values, tiles, z=NOISE_Z)
        indicators["noise"] = {
            "raised": bool(outliers),
            "median_sigma": round(float(np.median(values)), 4) if len(values) else None,
            "tiles": tiles,
            "outliers": outliers[:8],
        }
        if not outliers:
            return []

        median = float(np.median(values))
        worst = outliers[0]
        direction = "cleaner" if worst["value"] < median else "noisier"
        return [
            Finding(
                "manipulation",
                f"Noise floor inconsistent ({direction} region)",
                f"{len(outliers)} region(s), z={worst['z_score']}",
                Severity.MEDIUM,
                confidence=0.45,
                bbox=tuple(worst["bbox"]),
                detail="Sensor noise should be broadly uniform across one exposure. A region "
                "that departs from it may be spliced, denoised or upscaled — though sky, "
                "shadow and blurred backgrounds legitimately carry less noise",
            )
        ]

    def _copy_move(self, gray: np.ndarray, indicators: dict) -> list[Finding]:
        candidates = copy_move_candidates(gray.astype(np.uint8))
        indicators["copy_move"] = {
            "raised": bool(candidates),
            "candidates": candidates[:5],
        }
        if not candidates:
            return []

        best = candidates[0]
        return [
            Finding(
                "manipulation",
                "Duplicated region within the image",
                f"{best['pairs']} keypoint pairs, offset {best['offset']}",
                Severity.HIGH,
                confidence=min(0.75, 0.4 + best["pairs"] / 40),
                bbox=tuple(best["target_bbox"]),
                detail="A region appears twice with a consistent displacement — the signature "
                "of cloning to conceal or duplicate something. Repetitive architecture and "
                "patterned fabric can reproduce it, so confirm visually",
            )
        ]

    def _metadata(self, ctx: AnalysisContext, indicators: dict) -> list[Finding]:
        """Contradictions between what the file claims and what it contains."""
        exif = ctx.data("exif")
        highlights = exif.get("highlights", {})
        tags = exif.get("tags", {})
        out: list[Finding] = []
        signals: list[str] = []

        software = str(highlights.get("software") or "")
        editor = next((e for e in _EDITORS if e in software.lower()), None)
        make = highlights.get("make")

        # A file claiming a camera but carrying no MakerNotes has usually been
        # through something that rewrote it.
        has_makernotes = any("makernote" in k.lower() for k in tags)
        if make and not has_makernotes and exif.get("source") == "exiftool":
            signals.append("camera make present but no MakerNotes")
            out.append(
                Finding(
                    "manipulation",
                    "Camera metadata without MakerNotes",
                    str(make),
                    Severity.LOW,
                    confidence=0.4,
                    detail="A camera original normally carries manufacturer MakerNotes. Their "
                    "absence suggests the file was re-saved by software that dropped them",
                )
            )

        if editor and make:
            signals.append(f"editor '{editor}' with camera metadata")

        thumbnail = self._thumbnail_mismatch(ctx)
        if thumbnail is not None:
            indicators["thumbnail"] = thumbnail
            if thumbnail.get("raised"):
                signals.append("embedded thumbnail differs from the image")
                out.append(
                    Finding(
                        "manipulation",
                        "Embedded thumbnail does not match the image",
                        f"pHash distance {thumbnail['distance']}",
                        Severity.HIGH,
                        confidence=0.7,
                        detail="The preview embedded at capture shows something different from "
                        "the main image. Thumbnails often survive edits that change the "
                        "picture, which makes this one of the more telling contradictions",
                    )
                )

        indicators["metadata"] = {"raised": bool(signals), "signals": signals}
        return out

    def _thumbnail_mismatch(self, ctx: AnalysisContext) -> dict | None:
        """Compare the embedded EXIF thumbnail with the image itself."""
        try:
            exif = ctx.header.getexif()
            thumbnail_bytes = exif.get_ifd(0x0201) if exif else None
        except Exception:  # noqa: BLE001
            thumbnail_bytes = None

        raw = _extract_thumbnail(ctx)
        if raw is None:
            return None

        try:
            from PIL import Image

            from imgintel.analyzers.perceptual import perceptual_hash
            from imgintel.util.imgmath import hamming_hex

            with Image.open(io.BytesIO(raw)) as thumb:
                thumb_hash = perceptual_hash(thumb.convert("RGB"))
            main_hash = perceptual_hash(ctx.pil)
            distance = hamming_hex(thumb_hash, main_hash)
        except Exception:  # noqa: BLE001 - a broken thumbnail is not a finding
            return None

        _ = thumbnail_bytes
        return {
            "raised": distance > 18,
            "distance": distance,
            "thumbnail_phash": thumb_hash,
            "image_phash": main_hash,
        }


def _extract_thumbnail(ctx: AnalysisContext) -> bytes | None:
    """Pull the embedded JPEG thumbnail out of the EXIF APP1 segment."""
    try:
        exif = ctx.header.getexif()
        if not exif:
            return None
        ifd1 = exif.get_ifd(0x0201)
        if isinstance(ifd1, bytes) and ifd1[:2] == b"\xff\xd8":
            return ifd1
        # Pillow exposes IFD1 offsets rather than the bytes on some formats.
        offset = exif.get(0x0201) or (ifd1 or {}).get(0x0201)
        length = exif.get(0x0202) or (ifd1 or {}).get(0x0202)
        if offset and length:
            blob = ctx.raw
            start = int(offset)
            candidate = blob[start : start + int(length)]
            if candidate[:2] == b"\xff\xd8":
                return candidate
    except Exception:  # noqa: BLE001
        return None
    return None
