"""Encoding quality, exposure, noise and resolution.

The JPEG quality estimate is the interesting one: it is recovered by inverting
the libjpeg quantization-table scaling, and a table that does *not* fit that
scaling means a non-standard encoder — a fingerprint that Phase 6 tamper
detection builds on directly.
"""

from __future__ import annotations

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.imgmath import estimate_noise_sigma, shannon_entropy

# ITU T.81 Annex K reference tables.
STD_LUMA = np.array(
    [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99],
    dtype=np.float64,
)
STD_CHROMA = np.array(
    [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
     24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99],
    dtype=np.float64,
)

# Dimensions that betray a platform re-encode rather than an original capture.
_PLATFORM_SIZES: dict[tuple[int, int], str] = {
    (1080, 1080): "Instagram square", (1080, 1350): "Instagram portrait",
    (1200, 630): "Open Graph / Facebook link card", (1200, 675): "Twitter/X card",
    (960, 1280): "WhatsApp portrait", (1280, 960): "WhatsApp landscape",
    (1600, 1200): "Common messaging resize", (2048, 1536): "Facebook high-res",
}


#: Entropy below this with hard edges means few distinct tones — a graphic.
GRAPHIC_ENTROPY = 4.0
GRAPHIC_SHARPNESS = 200.0
BLANK_ENTROPY = 1.5
BLANK_SHARPNESS = 50.0


def _classify(entropy_bits: float, sharpness: float) -> str:
    """photograph | graphic | blank.

    Two cheap numbers separate these well: a photograph has a rich tonal
    histogram, a graphic has few tones but very hard edges, and a blank frame
    has neither. Everything downstream that assumes camera output is gated on
    the result.
    """
    if entropy_bits < BLANK_ENTROPY and sharpness < BLANK_SHARPNESS:
        return "blank"
    if entropy_bits < GRAPHIC_ENTROPY and sharpness > GRAPHIC_SHARPNESS:
        return "graphic"
    return "photograph"


def _scale_table(base: np.ndarray, quality: int) -> np.ndarray:
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    table = np.floor((base * scale + 50) / 100)
    return np.clip(table, 1, 255)


def estimate_jpeg_quality(table: list[int], chroma: bool = False) -> tuple[int, float]:
    """Recover the libjpeg quality setting from a quantization table.

    Both tables are sorted before comparison so the result does not depend on
    whether the decoder handed us zigzag or raster order.
    """
    actual = np.sort(np.asarray(table, dtype=np.float64))
    base = STD_CHROMA if chroma else STD_LUMA
    best_q, best_err = 0, float("inf")
    for q in range(1, 101):
        err = float(np.abs(np.sort(_scale_table(base, q)) - actual).mean())
        if err < best_err:
            best_q, best_err = q, err
    return best_q, best_err


class QualityAnalyzer(Analyzer):
    name = "quality"
    version = "1.0.0"
    title = "Image quality"
    description = "JPEG quality estimate, exposure clipping, noise, entropy, resolution class"
    cost = Cost.MEDIUM

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        gray = ctx.small_gray
        w, h = ctx.dimensions
        data: dict = {}
        findings: list[Finding] = []

        # -- resolution ----------------------------------------------------
        mp = ctx.megapixels
        data["width"] = w
        data["height"] = h
        data["megapixels"] = round(mp, 3)
        data["resolution_class"] = (
            "thumbnail" if mp < 0.15
            else "low" if mp < 1.0
            else "web" if mp < 3.0
            else "standard" if mp < 12.0
            else "high" if mp < 40.0
            else "very high"
        )
        if mp < 0.15:
            findings.append(
                Finding("quality", "Thumbnail-sized image", f"{w}x{h}", Severity.LOW,
                        detail="Too small for reliable detection or OCR downstream")
            )
        if (w, h) in _PLATFORM_SIZES or (h, w) in _PLATFORM_SIZES:
            label = _PLATFORM_SIZES.get((w, h)) or _PLATFORM_SIZES.get((h, w))
            findings.append(
                Finding("provenance", "Dimensions match a platform preset", f"{w}x{h} — {label}",
                        Severity.MEDIUM, confidence=0.7,
                        detail="Strongly suggests a re-encode by that platform, not an original")
            )

        # -- JPEG encoding -------------------------------------------------
        tables = ctx.jpeg_quant_tables
        if tables:
            estimates = {}
            for idx, table in sorted(tables.items()):
                q, err = estimate_jpeg_quality(table, chroma=idx > 0)
                estimates[f"table_{idx}"] = {
                    "quality": q,
                    "fit_error": round(err, 3),
                    "sum": int(sum(table)),
                }
            luma = estimates.get("table_0", next(iter(estimates.values())))
            data["jpeg"] = {
                "tables": len(tables),
                "estimates": estimates,
                "quality_estimate": luma["quality"],
                "standard_tables": luma["fit_error"] < 2.0,
                "subsampling": getattr(ctx.pil_raw, "layers", None),
            }
            findings.append(
                Finding("quality", "Estimated JPEG quality", luma["quality"], Severity.INFO,
                        confidence=0.9 if luma["fit_error"] < 2 else 0.5)
            )
            if luma["quality"] < 60:
                findings.append(
                    Finding("quality", "Heavily compressed JPEG", luma["quality"], Severity.LOW,
                            detail="Compression artifacts will degrade OCR and detection accuracy")
                )
            if luma["fit_error"] >= 2.0:
                findings.append(
                    Finding(
                        "provenance",
                        "Non-standard quantization tables",
                        f"fit error {luma['fit_error']:.1f}",
                        Severity.MEDIUM,
                        confidence=0.7,
                        detail="Tables do not follow the libjpeg scaling curve — the encoder was "
                        "a specific device or application, which can be fingerprinted",
                    )
                )
        else:
            data["jpeg"] = None

        # -- what kind of image is this? -----------------------------------
        # Exposure and noise findings only mean anything for camera output.
        # A screenshot has 70% "blown highlights" (it is a white page) and a
        # document has near-zero entropy — reporting those as photographic
        # defects is noise, and the classification itself is useful provenance.
        entropy_bits = shannon_entropy(gray)
        content = _classify(entropy_bits, ctx.sharpness)
        data["content_type"] = content
        if content == "graphic":
            findings.append(
                Finding(
                    "provenance",
                    "Synthetic image, not a camera photograph",
                    f"entropy {entropy_bits:.2f} bits with sharpness {ctx.sharpness:.0f}",
                    Severity.MEDIUM,
                    confidence=0.7,
                    detail="Few distinct tones combined with hard edges — a screenshot, "
                    "document scan, rendering, or generated graphic",
                )
            )

        # -- exposure ------------------------------------------------------
        total = gray.size
        clipped_hi = float((gray >= 254).sum() / total)
        clipped_lo = float((gray <= 1).sum() / total)
        p1, p50, p99 = (float(x) for x in np.percentile(gray, [1, 50, 99]))
        data["exposure"] = {
            "mean": round(float(gray.mean()), 2),
            "median": round(p50, 2),
            "std": round(float(gray.std()), 2),
            "clipped_highlights": round(clipped_hi, 5),
            "clipped_shadows": round(clipped_lo, 5),
            "dynamic_range_p1_p99": round(p99 - p1, 2),
        }
        if content == "photograph":
            if clipped_hi > 0.05:
                findings.append(
                    Finding("quality", "Blown highlights", f"{clipped_hi:.1%} of pixels",
                            Severity.LOW, detail="Detail unrecoverable in those regions")
                )
            if clipped_lo > 0.10:
                findings.append(
                    Finding("quality", "Crushed shadows", f"{clipped_lo:.1%} of pixels",
                            Severity.LOW)
                )
            if p99 - p1 < 40:
                findings.append(
                    Finding("quality", "Very low dynamic range", round(p99 - p1, 1), Severity.LOW,
                            detail="Flat, hazy or heavily faded — may be a photo of a screen")
                )

        # -- noise and information content ---------------------------------
        sigma = estimate_noise_sigma(gray)
        data["noise_sigma"] = round(sigma, 3)
        data["entropy_bits"] = round(entropy_bits, 3)
        data["noise_class"] = "low" if sigma < 2 else "moderate" if sigma < 6 else "high"
        if sigma > 8 and content == "photograph":
            findings.append(
                Finding("quality", "High sensor noise", round(sigma, 1), Severity.LOW,
                        detail="Low light, high ISO, or heavy upscaling")
            )
        if content == "blank":
            findings.append(
                Finding("quality", "Very low information content", round(entropy_bits, 2),
                        Severity.MEDIUM,
                        detail="Nearly uniform and detail-free — blank, solid colour, "
                        "or a failed capture")
            )

        # A near-zero noise floor on a *detailed* image is itself odd. The
        # sharpness gate matters: a blurred photo legitimately has no noise
        # left, and flagging that as suspicious would be a false positive.
        if sigma < 0.4 and mp > 1.0 and ctx.sharpness > 40:
            findings.append(
                Finding(
                    "manipulation",
                    "Implausibly clean noise floor",
                    round(sigma, 3),
                    Severity.LOW,
                    confidence=0.5,
                    detail="Real sensor output carries noise; this suggests rendering, heavy "
                    "denoising, or synthetic generation",
                )
            )

        return self.ok(data, findings)
