"""Numerical primitives for tamper indicators.

Kept separate from the analyzer so each technique can be tested against inputs
whose correct answer is known — a synthesised splice, a deliberately
double-compressed file — rather than only against whole images where "did it
work" is a matter of opinion.

Every function here returns a *measurement*. Interpretation, thresholds and the
caveats belong in the analyzer, because the same number means different things
on a camera original and on a screenshot.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from imgintel.util.imgmath import convolve3

# -- error level analysis --------------------------------------------------


def error_level(image_rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    """Per-pixel difference after one JPEG round trip.

    Regions that have already been compressed hard sit near their local
    JPEG fixed point and barely change; freshly-pasted or re-rendered
    regions move more. That is the whole idea, and it is also why ELA is so
    widely misread: texture, brightness and edges all shift the level too.
    """
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recompressed:
        resaved = np.asarray(recompressed.convert("RGB"), dtype=np.int16)
    return np.abs(image_rgb.astype(np.int16) - resaved).max(axis=2).astype(np.float32)


def tiled_stats(field: np.ndarray, grid: int = 8) -> tuple[np.ndarray, list[dict]]:
    """Mean of a scalar field per tile, plus the tile geometry."""
    height, width = field.shape
    th, tw = height // grid, width // grid
    if th < 8 or tw < 8:
        return np.zeros((0,), dtype=np.float32), []

    values = np.zeros(grid * grid, dtype=np.float32)
    tiles: list[dict] = []
    for row in range(grid):
        for col in range(grid):
            y0, x0 = row * th, col * tw
            patch = field[y0 : y0 + th, x0 : x0 + tw]
            index = row * grid + col
            values[index] = float(patch.mean())
            tiles.append({"bbox": [x0, y0, x0 + tw, y0 + th], "value": round(values[index], 4)})
    return values, tiles


def outlier_tiles(
    values: np.ndarray, tiles: list[dict], *, z: float = 3.0, min_tiles: int = 16
) -> list[dict]:
    """Tiles whose value is an outlier by median absolute deviation.

    MAD rather than mean/standard deviation because a genuine splice is
    exactly the kind of large outlier that would inflate an SD-based
    threshold enough to hide itself.
    """
    if len(values) < min_tiles:
        return []
    median = float(np.median(values))
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))

    if mad < 1e-6:
        # More than half the tiles are identical, so the MAD is zero and
        # dividing by it would hide the very thing being looked for: one
        # region differing sharply from a uniform background is the clearest
        # possible outlier, not an undefined one. Fall back to the mean
        # deviation, which is non-zero whenever anything differs at all.
        fallback = float(deviations.mean())
        if fallback < 1e-6:
            return []  # genuinely uniform; nothing to report
        scores = deviations / fallback
        # The mean-deviation scale is not comparable with a MAD z-score, so
        # require a larger multiple to keep the threshold's meaning.
        z = z * 2.0
    else:
        # 1.4826 rescales MAD to be comparable with a standard deviation.
        scores = deviations / (1.4826 * mad)
    return [
        {**tiles[i], "z_score": round(float(scores[i]), 2)}
        for i in np.argsort(scores)[::-1]
        if scores[i] >= z
    ]


# -- JPEG structure --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GhostResult:
    """Recompression response across a sweep of qualities."""

    qualities: list[int]
    curve: list[float]
    minimum_quality: int
    #: True when the curve dips sharply at one quality — evidence of a prior
    #: compression at that setting.
    has_dip: bool
    dip_depth: float


#: The sweep must reach well below typical "web quality" or it cannot see the
#: dip at all. A first save at q=45 is entirely ordinary, and a range starting
#: at 50 silently reports "no prior compression" for every one of them —
#: a false negative that looks exactly like a clean image.
GHOST_QUALITIES: tuple[int, ...] = tuple(range(30, 100, 5))


def jpeg_ghost_sweep(
    image_rgb: np.ndarray, qualities: tuple[int, ...] = GHOST_QUALITIES
) -> GhostResult:
    """Mean absolute difference after recompressing at a range of qualities.

    An image previously saved at quality Q has a local minimum near Q,
    because recompressing at its own quality changes it least. A single
    clear dip therefore reveals a prior save even when the metadata is gone.
    """
    curve: list[float] = []
    for quality in qualities:
        buffer = io.BytesIO()
        Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            resaved = np.asarray(recompressed.convert("RGB"), dtype=np.int16)
        curve.append(float(np.abs(image_rgb.astype(np.int16) - resaved).mean()))

    values = np.asarray(curve, dtype=np.float32)
    index = int(values.argmin())
    # A dip only counts if the curve rises again afterwards; a monotonically
    # falling curve just means "higher quality changes less", which is true
    # of every image and says nothing.
    interior = 0 < index < len(values) - 1
    depth = 0.0
    if interior:
        shoulder = min(values[index - 1], values[index + 1])
        depth = float((shoulder - values[index]) / max(shoulder, 1e-6))

    return GhostResult(
        qualities=[int(q) for q in qualities],
        curve=[round(v, 4) for v in curve],
        minimum_quality=int(qualities[index]),
        has_dip=interior and depth > 0.02,
        dip_depth=round(depth, 4),
    )


def jpeg_ghost_map(
    image_rgb: np.ndarray,
    *,
    grid: int = 8,
    qualities: tuple[int, ...] = GHOST_QUALITIES,
) -> dict:
    """Per-region recompression minima — the actual JPEG-ghost technique.

    The *global* sweep above only ever recovers an image's own last-save
    quality: recompressing at the quality it already is changes it least, and
    that dominates the mean. It cannot see a prior compression, because the
    most recent save is what the whole frame now carries.

    Splices are different. A region pasted in from a file compressed at some
    other quality retains that history, so its own recompression minimum sits
    at a different quality from the rest of the frame. Comparing each region's
    minimum against the consensus is what makes this a *localising* test rather
    than a global one — and localisation is the point, since it says where to
    look rather than just that something is off.
    """
    height, width = image_rgb.shape[:2]
    th, tw = height // grid, width // grid
    if th < 16 or tw < 16:
        return {"available": False, "reason": "image too small for a per-region sweep"}

    # (qualities, grid*grid) — mean absolute difference per tile per quality.
    responses = np.zeros((len(qualities), grid * grid), dtype=np.float32)
    for qi, quality in enumerate(qualities):
        buffer = io.BytesIO()
        Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        with Image.open(buffer) as recompressed:
            resaved = np.asarray(recompressed.convert("RGB"), dtype=np.int16)
        difference = np.abs(image_rgb.astype(np.int16) - resaved).mean(axis=2)
        for row in range(grid):
            for col in range(grid):
                y0, x0 = row * th, col * tw
                responses[qi, row * grid + col] = difference[
                    y0 : y0 + th, x0 : x0 + tw
                ].mean()

    tiles = []
    for index in range(grid * grid):
        row, col = divmod(index, grid)
        ghost = _local_minimum(responses[:, index], qualities)
        tiles.append(
            {
                "bbox": [col * tw, row * th, col * tw + tw, row * th + th],
                "ghost_quality": ghost[0],
                "ghost_depth": ghost[1],
            }
        )

    ghosted = [t for t in tiles if t["ghost_quality"] is not None]
    return {
        "available": True,
        "qualities": [int(q) for q in qualities],
        "tiles": tiles,
        "ghost_tiles": sorted(ghosted, key=lambda t: -t["ghost_depth"])[:8],
        "ghost_count": len(ghosted),
        "ghost_fraction": round(len(ghosted) / max(len(tiles), 1), 4),
    }


#: A dip must be this deep relative to its shoulders to count as a ghost.
GHOST_DEPTH = 0.15


def _local_minimum(curve: np.ndarray, qualities: tuple[int, ...]) -> tuple[int | None, float]:
    """The deepest interior local minimum, if any.

    This is the correction that makes ghosts work at all. The *global* minimum
    of a tile's curve is always the image's current quality — recompressing at
    what it already is changes it least — so an argmin finds the last save and
    nothing else. A region carried over from an earlier, harder compression
    shows up as a separate dip *above* that floor, and only a local-minimum
    search can see it.
    """
    best_quality: int | None = None
    best_depth = 0.0
    for i in range(1, len(curve) - 1):
        if not (curve[i] < curve[i - 1] and curve[i] < curve[i + 1]):
            continue
        shoulder = float(min(curve[i - 1], curve[i + 1]))
        depth = (shoulder - float(curve[i])) / max(shoulder, 1e-6)
        if depth > best_depth:
            best_depth, best_quality = depth, int(qualities[i])
    if best_depth < GHOST_DEPTH:
        return None, 0.0
    return best_quality, round(best_depth, 4)


def dct_first_digit_deviation(gray: np.ndarray) -> float:
    """How far the DCT coefficients depart from Benford's law.

    Quantised DCT coefficients of a singly-compressed natural image follow a
    Benford-like first-digit distribution closely. A second quantisation
    disturbs it. Returned as a chi-square-like divergence, not a verdict:
    heavily-edited-but-authentic images move it too.
    """
    from imgintel.util.imgmath import dct2

    height, width = gray.shape
    blocks_y, blocks_x = height // 8, width // 8
    if blocks_y < 4 or blocks_x < 4:
        return 0.0

    counts = np.zeros(9, dtype=np.float64)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = gray[by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8] - 128.0
            coefficients = np.abs(dct2(block)).ravel()[1:]  # drop DC
            coefficients = coefficients[coefficients >= 1.0]
            if not len(coefficients):
                continue
            leading = (coefficients / 10 ** np.floor(np.log10(coefficients))).astype(int)
            counts += np.bincount(leading.clip(1, 9), minlength=10)[1:10]

    total = counts.sum()
    if total < 100:
        return 0.0
    observed = counts / total
    digits = np.arange(1, 10)
    expected = np.log10(1 + 1 / digits)
    return float(np.sum((observed - expected) ** 2 / expected))


#: Standard JPEG luminance table, for recognising encoder provenance.
def quant_table_signature(table: list[int]) -> dict[str, float]:
    """Coarse descriptors of a quantization table.

    Enough to say "these two files came out of the same encoder at the same
    setting" without shipping a database of camera tables.
    """
    values = np.asarray(table, dtype=np.float64)
    return {
        "sum": float(values.sum()),
        "mean": round(float(values.mean()), 3),
        "max": float(values.max()),
        "dc": float(values[0]),
        "zeros": int((values <= 1).sum()),
    }


# -- noise -----------------------------------------------------------------

#: Immerkaer's noise-estimation kernel, reused per tile.
_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_field(gray: np.ndarray, grid: int = 8) -> tuple[np.ndarray, list[dict]]:
    """Per-tile noise sigma.

    A region spliced from another photograph usually carries that camera's
    noise characteristics, which rarely match the host. Denoising or
    upscaling a region leaves the opposite signature — a hole in the noise
    floor.
    """
    height, width = gray.shape
    th, tw = height // grid, width // grid
    if th < 8 or tw < 8:
        return np.zeros((0,), dtype=np.float32), []

    values = np.zeros(grid * grid, dtype=np.float32)
    tiles: list[dict] = []
    for row in range(grid):
        for col in range(grid):
            y0, x0 = row * th, col * tw
            patch = gray[y0 : y0 + th, x0 : x0 + tw]
            conv = convolve3(patch, _NOISE_K)
            sigma = float(
                np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * max(patch.shape[1] - 2, 1)
                                                           * max(patch.shape[0] - 2, 1))
            )
            index = row * grid + col
            values[index] = sigma
            tiles.append({"bbox": [x0, y0, x0 + tw, y0 + th], "value": round(sigma, 4)})
    return values, tiles


# -- copy-move -------------------------------------------------------------


def copy_move_candidates(
    gray: np.ndarray,
    *,
    min_distance: int = 40,
    #: Natural images throw up a handful of coincidentally-consistent pairs;
    #: a real cloned region produces dozens. Measured on 1/f test content,
    #: clean images peak at ~4 and a genuine clone at ~75.
    min_cluster: int = 12,
    tolerance: int = 6,
) -> list[dict]:
    """Regions duplicated elsewhere in the same image.

    Matches keypoints against *themselves*, discards self-matches and near
    neighbours, then keeps only groups sharing a consistent translation
    offset. The offset agreement is what separates a real cloned region from
    the incidental self-similarity every photograph of brickwork contains.
    """
    try:
        import cv2
    except ImportError:
        return []

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(
        np.ascontiguousarray(gray, dtype=np.uint8), None
    )
    if descriptors is None or len(keypoints) < 10:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # k=3 so that after discarding the self-match there are still two
    # candidates left to compare.
    matches = matcher.knnMatch(descriptors, descriptors, k=3)

    offsets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for group in matches:
        for match in group:
            if match.queryIdx == match.trainIdx:
                continue
            src = keypoints[match.queryIdx].pt
            dst = keypoints[match.trainIdx].pt
            dx, dy = dst[0] - src[0], dst[1] - src[1]
            if np.hypot(dx, dy) < min_distance:
                continue  # neighbouring keypoints on the same feature
            key = (int(round(dx / tolerance)), int(round(dy / tolerance)))
            offsets.setdefault(key, []).append((match.queryIdx, match.trainIdx))

    results: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for (kx, ky), pairs in offsets.items():
        if len(pairs) < min_cluster:
            continue
        # A cloned region produces the offset and its negative; report once.
        if (-kx, -ky) in seen:
            continue
        seen.add((kx, ky))

        source = np.array([keypoints[q].pt for q, _ in pairs])
        target = np.array([keypoints[t].pt for _, t in pairs])
        results.append(
            {
                "offset": [int(kx * tolerance), int(ky * tolerance)],
                "pairs": len(pairs),
                "source_bbox": _bounds(source),
                "target_bbox": _bounds(target),
            }
        )
    return sorted(results, key=lambda r: -r["pairs"])


def _bounds(points: np.ndarray) -> list[int]:
    return [
        int(points[:, 0].min()), int(points[:, 1].min()),
        int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max())),
    ]


__all__ = [
    "GhostResult",
    "copy_move_candidates",
    "dct_first_digit_deviation",
    "error_level",
    "jpeg_ghost_sweep",
    "noise_field",
    "outlier_tiles",
    "quant_table_signature",
    "tiled_stats",
]
