"""Small numpy implementations of the image maths Phase 1 needs.

Deliberately dependency-free: implementing a 32x32 DCT and k-means in numpy is
~60 lines and saves pulling in scipy and scikit-learn, which between them add
~120 MB and a pile of platform-specific wheels to a tool whose whole point is
being easy to install everywhere.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# -- transforms ------------------------------------------------------------


@lru_cache(maxsize=8)
def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, so ``M @ x`` is the 1-D DCT of ``x``."""
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    m *= np.sqrt(2.0 / n)
    m[0] *= 1 / np.sqrt(2)
    return m


def dct2(block: np.ndarray) -> np.ndarray:
    """2-D DCT-II of a square array."""
    m = dct_matrix(block.shape[0])
    return m @ block.astype(np.float64) @ m.T


# -- convolutions (valid-region, via slicing — no scipy) --------------------


def convolve3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 convolution over the valid interior. Returns an (H-2, W-2) array."""
    img = img.astype(np.float32)
    out = np.zeros((img.shape[0] - 2, img.shape[1] - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            k = kernel[dy, dx]
            if k:
                out += k * img[dy : dy + out.shape[0], dx : dx + out.shape[1]]
    return out


LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
# Immerkaer's noise-estimation kernel.
NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian — the standard focus measure.

    Higher is sharper. Absolute values are resolution- and content-dependent,
    which is why the blur analyzer reports the tile distribution too.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    return float(convolve3(gray, LAPLACIAN).var())


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise standard-deviation estimate."""
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    conv = convolve3(gray, NOISE_K)
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def gradient_orientation_stats(gray: np.ndarray, bins: int = 18) -> tuple[float, float]:
    """Return (anisotropy, dominant_angle_degrees) of the gradient field.

    Motion blur suppresses gradients along the direction of travel, leaving a
    strongly anisotropic gradient field; defocus blur stays isotropic. This is
    what separates "camera shake" from "out of focus" in the blur analyzer.
    """
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0, 0.0
    gx = convolve3(gray, SOBEL_X)
    gy = convolve3(gray, SOBEL_Y)
    mag = np.hypot(gx, gy)
    total = mag.sum()
    if total <= 0:
        return 0.0, 0.0
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    idx = np.minimum((ang / (180.0 / bins)).astype(np.int32), bins - 1)
    hist = np.bincount(idx.ravel(), weights=mag.ravel(), minlength=bins)
    hist = hist / hist.sum()
    peak = int(np.argmax(hist))
    # 0 = perfectly uniform, 1 = all energy in one orientation.
    anisotropy = float((hist.max() - 1.0 / bins) / (1.0 - 1.0 / bins))
    return anisotropy, float((peak + 0.5) * (180.0 / bins))


# -- colour ----------------------------------------------------------------


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIE L*a*b* (D65). Shape preserved, float32 out."""
    arr = rgb.astype(np.float32) / 255.0
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear.reshape(-1, 3) @ m.T
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    t = xyz / white
    delta = 6.0 / 29.0
    f = np.where(t > delta**3, np.cbrt(np.maximum(t, 1e-12)), t / (3 * delta**2) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])
    return lab.reshape(rgb.shape).astype(np.float32)


def kmeans(points: np.ndarray, k: int, iterations: int = 12, seed: int = 0) -> tuple:
    """Minimal k-means with deterministic k-means++ seeding.

    Determinism matters: the same image must produce the same palette on every
    run, or evidence exports stop being reproducible.
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    k = min(k, n)
    centres = np.empty((k, points.shape[1]), dtype=np.float32)
    centres[0] = points[rng.integers(n)]
    closest = ((points - centres[0]) ** 2).sum(axis=1)
    for i in range(1, k):
        total = closest.sum()
        probs = closest / total if total > 0 else np.full(n, 1 / n)
        centres[i] = points[rng.choice(n, p=probs)]
        closest = np.minimum(closest, ((points - centres[i]) ** 2).sum(axis=1))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iterations):
        d = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            member = points[labels == i]
            if len(member):
                centres[i] = member.mean(axis=0)
    return centres, labels


def colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk colourfulness metric. Roughly: <15 dull, >60 vivid."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std + 0.3 * mean)


# -- misc ------------------------------------------------------------------


def shannon_entropy(gray: np.ndarray) -> float:
    """Entropy of the 8-bit histogram, in bits (0-8)."""
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    hist = hist[hist > 0]
    p = hist / hist.sum()
    return max(0.0, float(-(p * np.log2(p)).sum()))


def bits_to_hex(bits: np.ndarray) -> str:
    """Pack a boolean array (row-major) into a lowercase hex string."""
    flat = bits.ravel().astype(np.uint8)
    pad = (-len(flat)) % 8
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(flat).tobytes().hex()


def hamming_hex(a: str, b: str) -> int:
    """Hamming distance between two equal-length hex hash strings."""
    if len(a) != len(b):
        raise ValueError("hash length mismatch")
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))
