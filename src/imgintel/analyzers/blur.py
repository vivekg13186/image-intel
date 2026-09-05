"""Sharpness, blur type, and localized blur.

The global focus measure answers "is this photo any good". The tile map answers
the more interesting question: *which parts* are blurred. A sharp image with a
few soft rectangles is an attempted redaction, not a bad photo — and that is a
finding no single global number will ever surface.
"""

from __future__ import annotations

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.imgmath import gradient_orientation_stats, laplacian_variance

GRID = 8  # tiles per axis
MIN_TILE = 24  # pixels; below this the variance is meaningless

# Absolute focus thresholds, applied to ctx.small (longest edge 1024). They are
# content-dependent and therefore coarse — a flat sky is legitimately "soft".
# The localized check below is deliberately threshold-free and carries more
# weight; treat these two numbers as a triage aid, not a verdict.
SHARP = 60.0
SOFT = 12.0
#: Below this contrast there is no detail to focus on, so no verdict is given.
FLAT_CONTRAST = 2.0
#: A tile this far below the image's own median tile variance is "locally soft".
RELATIVE_SOFT = 0.15
#: Above this share of soft tiles the blur is global, not localized.
MAX_SOFT_SHARE = 0.4


class BlurAnalyzer(Analyzer):
    name = "blur"
    version = "1.0.0"
    title = "Sharpness and blur"
    description = "Focus measure, blur type, and per-region blur map for redaction detection"
    cost = Cost.MEDIUM

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        gray = ctx.small_gray
        h, w = gray.shape

        global_var = ctx.sharpness
        contrast = float(gray.std())
        # Normalizing by contrast makes the measure comparable across images
        # of very different tonality; the raw value stays for reference.
        normalized = global_var / max(contrast**2, 1e-6)

        anisotropy, angle = gradient_orientation_stats(gray)

        data: dict = {
            "laplacian_variance": round(global_var, 3),
            "normalized_sharpness": round(normalized, 5),
            "contrast_std": round(contrast, 3),
            "gradient_anisotropy": round(anisotropy, 4),
            "dominant_edge_angle_deg": round(angle, 1),
            "analysis_dimensions": [w, h],
        }
        findings: list[Finding] = []

        # -- global verdict -------------------------------------------------
        if contrast < FLAT_CONTRAST:
            data["verdict"] = "no detail"
            data["blur_type"] = None
            data["tiles"] = []
            data["tile_stats"] = {}
            data["localized_blur"] = []
            findings.append(
                Finding(
                    "quality",
                    "No detail to assess focus",
                    round(contrast, 2),
                    Severity.INFO,
                    detail="Image is effectively uniform — sharpness is undefined",
                )
            )
            return self.ok(data, findings)

        if global_var < SOFT:
            verdict = "blurred"
            findings.append(
                Finding("quality", "Image is blurred", round(global_var, 1), Severity.MEDIUM,
                        confidence=0.8,
                        detail="Focus measure well below the sharp threshold; detection and OCR "
                        "will be unreliable")
            )
        elif global_var < SHARP:
            verdict = "soft"
            findings.append(
                Finding("quality", "Image is soft", round(global_var, 1), Severity.LOW,
                        confidence=0.6)
            )
        else:
            verdict = "sharp"
            findings.append(
                Finding("quality", "Image is sharp", round(global_var, 1), Severity.INFO)
            )
        data["verdict"] = verdict

        # -- blur type ------------------------------------------------------
        if verdict != "sharp" and anisotropy > 0.25:
            data["blur_type"] = "motion"
            findings.append(
                Finding(
                    "quality",
                    "Blur appears directional (motion)",
                    f"{angle:.0f}° dominant, anisotropy {anisotropy:.2f}",
                    Severity.LOW,
                    confidence=0.65,
                    detail="Gradient energy concentrated in one orientation — camera shake or "
                    "subject movement rather than misfocus",
                )
            )
        elif verdict != "sharp":
            data["blur_type"] = "defocus"
            findings.append(
                Finding("quality", "Blur appears isotropic (defocus)", round(anisotropy, 2),
                        Severity.LOW, confidence=0.6)
            )
        else:
            data["blur_type"] = None

        # -- tile map -------------------------------------------------------
        tiles, stats = self._tile_map(gray)
        data["tiles"] = tiles
        data["tile_stats"] = stats

        if stats and stats["count"] >= 16:
            self._localized(tiles, stats, global_var, findings, data)

        return self.ok(data, findings)

    # ----------------------------------------------------------------------

    def _tile_map(self, gray: np.ndarray) -> tuple[list[dict], dict]:
        h, w = gray.shape
        th, tw = h // GRID, w // GRID
        if th < MIN_TILE or tw < MIN_TILE:
            return [], {}

        tiles: list[dict] = []
        for row in range(GRID):
            for col in range(GRID):
                y0, x0 = row * th, col * tw
                patch = gray[y0 : y0 + th, x0 : x0 + tw]
                tiles.append(
                    {
                        "bbox": [x0, y0, x0 + tw, y0 + th],
                        "variance": round(laplacian_variance(patch), 3),
                        "contrast": round(float(patch.std()), 3),
                    }
                )

        values = np.array([t["variance"] for t in tiles])
        stats = {
            "count": len(tiles),
            "min": round(float(values.min()), 3),
            "max": round(float(values.max()), 3),
            "median": round(float(np.median(values)), 3),
            "p90": round(float(np.percentile(values, 90)), 3),
        }
        return tiles, stats

    def _localized(
        self,
        tiles: list[dict],
        stats: dict,
        global_var: float,
        findings: list[Finding],
        data: dict,
    ) -> None:
        """Flag tiles that are far softer than the image as a whole.

        Threshold-free by design: compares each tile against this image's own
        median rather than an absolute cutoff, so it works on any content and
        at any resolution.
        """
        median = stats["median"]
        data["localized_blur"] = []
        if median <= 1.0:
            # Uniformly detail-free; "locally soft" would be meaningless.
            return

        # Three conditions, and all three are load-bearing:
        #   1. absolutely soft — actually blurry, not merely less sharp than a
        #      very sharp neighbour. Without this, a QR code or screenshot has
        #      such extreme edge energy that a quarter of the frame looks
        #      "soft" by ratio while being razor sharp in absolute terms.
        #   2. relatively soft — far below *this* image's own median, which is
        #      what makes the check work on any content without calibration.
        #   3. not perfectly uniform — a blank margin was never detailed, so
        #      its softness says nothing. The floor is deliberately low: heavy
        #      blur destroys local contrast too, and requiring much of it would
        #      miss the very regions this check exists to find.
        suspects = [
            t
            for t in tiles
            if t["variance"] < SOFT
            and t["variance"] < median * RELATIVE_SOFT
            and t["contrast"] > 2.0
        ]

        # If most of the frame qualifies, this is global blur, not localized.
        if len(suspects) > MAX_SOFT_SHARE * stats["count"]:
            return
        data["localized_blur"] = suspects

        if len(suspects) >= 2:
            ratio = median / max(max(t["variance"] for t in suspects), 1e-6)
            findings.append(
                Finding(
                    "manipulation",
                    "Localized blur regions",
                    f"{len(suspects)} of {stats['count']} tiles, {ratio:.0f}x softer than median",
                    Severity.MEDIUM,
                    confidence=min(0.75, 0.4 + ratio / 100),
                    bbox=_union([t["bbox"] for t in suspects]),
                    detail="Regions markedly softer than the rest of an otherwise sharp frame — "
                    "consistent with deliberate blurring or pixelation. Shallow depth of "
                    "field and plain backgrounds produce the same pattern; confirm visually",
                )
            )


def _union(boxes: list[list[int]]) -> tuple[int, int, int, int]:
    arr = np.array(boxes)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 2].max()),
        int(arr[:, 3].max()),
    )
