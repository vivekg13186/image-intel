"""Colour statistics, dominant palette and cast detection."""

from __future__ import annotations

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.imgmath import colorfulness, kmeans, srgb_to_lab

MAX_SAMPLES = 20_000
PALETTE_K = 6

# Coarse names for the dominant-colour report. Nearest neighbour in RGB is
# crude but adequate for a human-readable label.
_NAMED: list[tuple[str, tuple[int, int, int]]] = [
    ("black", (0, 0, 0)), ("white", (255, 255, 255)), ("grey", (128, 128, 128)),
    ("silver", (192, 192, 192)), ("charcoal", (54, 54, 54)),
    ("red", (220, 30, 30)), ("maroon", (128, 0, 0)), ("orange", (255, 140, 0)),
    ("brown", (139, 90, 43)), ("beige", (222, 202, 166)), ("yellow", (240, 220, 40)),
    ("olive", (110, 118, 50)), ("green", (40, 160, 60)), ("dark green", (20, 80, 40)),
    ("teal", (0, 128, 128)), ("cyan", (60, 200, 220)), ("sky blue", (120, 180, 230)),
    ("blue", (40, 80, 200)), ("navy", (25, 35, 90)), ("purple", (120, 60, 170)),
    ("magenta", (200, 50, 160)), ("pink", (240, 160, 190)), ("skin", (222, 180, 150)),
]


def _name_of(rgb: tuple[int, int, int]) -> str:
    arr = np.array(rgb, dtype=np.float32)
    return min(_NAMED, key=lambda n: float(((arr - np.array(n[1], dtype=np.float32)) ** 2).sum()))[0]


class ColorAnalyzer(Analyzer):
    name = "color"
    version = "1.0.0"
    title = "Colour analysis"
    description = "Dominant palette, channel statistics, colour cast and saturation"
    cost = Cost.MEDIUM

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        rgb = ctx.small
        h, w = rgb.shape[:2]
        flat = rgb.reshape(-1, 3)

        # Deterministic stride sampling — same pixels every run.
        if len(flat) > MAX_SAMPLES:
            stride = len(flat) // MAX_SAMPLES + 1
            sample = flat[::stride][:MAX_SAMPLES]
        else:
            sample = flat

        channel_means = flat.mean(axis=0)
        channel_stds = flat.std(axis=0)
        max_channel_gap = float(channel_means.max() - channel_means.min())
        is_gray = max_channel_gap < 1.0 and float(np.abs(np.diff(flat.astype(np.int16))).mean()) < 1.0

        cf = colorfulness(rgb)
        palette = self._palette(sample)

        lab = srgb_to_lab(sample.reshape(1, -1, 3)).reshape(-1, 3)
        data = {
            "is_grayscale": bool(is_gray),
            "colorfulness": round(cf, 2),
            "channel_mean": {c: round(float(v), 2) for c, v in zip("rgb", channel_means, strict=True)},
            "channel_std": {c: round(float(v), 2) for c, v in zip("rgb", channel_stds, strict=True)},
            "max_channel_gap": round(max_channel_gap, 2),
            "lab_mean": {
                "L": round(float(lab[:, 0].mean()), 2),
                "a": round(float(lab[:, 1].mean()), 2),
                "b": round(float(lab[:, 2].mean()), 2),
            },
            "mean_saturation": round(float(_saturation(sample).mean()), 4),
            "unique_colors_sampled": int(len(np.unique(sample.reshape(-1, 3), axis=0))),
            "palette": palette,
            "sampled_pixels": int(len(sample)),
            "analysis_dimensions": [w, h],
        }

        findings: list[Finding] = []
        if palette:
            top = palette[0]
            findings.append(
                Finding(
                    "color",
                    "Dominant colour",
                    f"{top['hex']} ({top['name']}, {top['share']:.0%})",
                    Severity.INFO,
                )
            )
        if is_gray:
            findings.append(
                Finding(
                    "color",
                    "Image is greyscale",
                    None,
                    Severity.LOW,
                    detail="Stored in colour but carries no chroma — converted, scanned or duotone",
                )
            )
        elif cf < 12:
            findings.append(
                Finding("color", "Very low colourfulness", round(cf, 1), Severity.INFO,
                        detail="Near-monochrome, heavily desaturated, or a document scan")
            )

        # Colour cast: a strong a*/b* mean shift away from neutral.
        a_mean, b_mean = data["lab_mean"]["a"], data["lab_mean"]["b"]
        cast_strength = float(np.hypot(a_mean, b_mean))
        data["cast_strength"] = round(cast_strength, 2)
        if cast_strength > 12 and not is_gray:
            direction = []
            if b_mean > 6:
                direction.append("warm/yellow")
            elif b_mean < -6:
                direction.append("cool/blue")
            if a_mean > 6:
                direction.append("magenta")
            elif a_mean < -6:
                direction.append("green")
            findings.append(
                Finding(
                    "color",
                    "Strong colour cast",
                    ", ".join(direction) or f"a*{a_mean:+.1f} b*{b_mean:+.1f}",
                    Severity.LOW,
                    detail="Artificial light, an uncorrected white balance, or a applied filter",
                )
            )

        return self.ok(data, findings)

    def _palette(self, sample: np.ndarray) -> list[dict]:
        if len(sample) < PALETTE_K:
            return []
        lab = srgb_to_lab(sample.reshape(1, -1, 3)).reshape(-1, 3)
        _, labels = kmeans(lab, PALETTE_K, seed=0)
        out: list[dict] = []
        for i in range(labels.max() + 1):
            members = sample[labels == i]
            if not len(members):
                continue
            mean = members.mean(axis=0)
            rgb = tuple(int(round(float(v))) for v in mean)
            out.append(
                {
                    "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                    "rgb": list(rgb),
                    "name": _name_of(rgb),  # type: ignore[arg-type]
                    "share": round(float(len(members) / len(sample)), 4),
                }
            )
        return sorted(out, key=lambda c: -c["share"])


def _saturation(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    return np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
