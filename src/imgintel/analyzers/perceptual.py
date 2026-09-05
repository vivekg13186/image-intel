"""Perceptual hashes — fingerprints of the picture rather than the file.

Several are computed because each survives a different transform: pHash tolerates
rescaling and JPEG recompression, dHash is fast and good on crops, aHash is
brittle but nearly free. Keeping all three makes near-duplicate matching robust.

Algorithms follow the conventions of the ``imagehash`` package so values are
directly comparable with hashes produced by other tooling.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.imgmath import bits_to_hex, dct2

HASH_SIZE = 8


def _resized_gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.LANCZOS), dtype=np.float64)


def average_hash(img: Image.Image, n: int = HASH_SIZE) -> str:
    px = _resized_gray(img, (n, n))
    return bits_to_hex(px > px.mean())


def difference_hash(img: Image.Image, n: int = HASH_SIZE) -> str:
    px = _resized_gray(img, (n + 1, n))
    return bits_to_hex(px[:, 1:] > px[:, :-1])


def difference_hash_vertical(img: Image.Image, n: int = HASH_SIZE) -> str:
    px = _resized_gray(img, (n, n + 1))
    return bits_to_hex(px[1:, :] > px[:-1, :])


def perceptual_hash(img: Image.Image, n: int = HASH_SIZE, factor: int = 4) -> str:
    """DCT-based hash. The most robust of the set against recompression."""
    px = _resized_gray(img, (n * factor, n * factor))
    low = dct2(px)[:n, :n]
    return bits_to_hex(low > np.median(low))


def histogram_hash(rgb: np.ndarray, bins: int = 4) -> str:
    """64-bit hash of a 4x4x4 RGB histogram — survives heavy geometric edits.

    Not an ``imagehash`` algorithm; useful as a coarse colour-distribution
    fingerprint when the composition has changed but the palette has not.
    """
    idx = (rgb.astype(np.uint16) * bins // 256).clip(0, bins - 1)
    flat = (idx[..., 0] * bins * bins + idx[..., 1] * bins + idx[..., 2]).ravel()
    hist = np.bincount(flat, minlength=bins**3).astype(np.float64)
    return bits_to_hex(hist > np.median(hist))


class PerceptualHashAnalyzer(Analyzer):
    name = "perceptual"
    version = "1.0.0"
    title = "Perceptual hashes"
    description = "pHash/dHash/aHash fingerprints plus a decoded-pixel digest"
    cost = Cost.MEDIUM

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        img = ctx.pil

        # Digest of decoded pixels: survives metadata stripping and
        # re-containering, so it links files whose byte hashes differ.
        raw_rgb = ctx.pil_raw.convert("RGB")
        pixel_sha = hashlib.sha256(raw_rgb.tobytes()).hexdigest()

        data = {
            "phash": perceptual_hash(img),
            "dhash": difference_hash(img),
            "dhash_vertical": difference_hash_vertical(img),
            "ahash": average_hash(img),
            "histhash": histogram_hash(ctx.small),
            "hash_size": HASH_SIZE,
            "hash_bits": HASH_SIZE * HASH_SIZE,
            "pixel_sha256": pixel_sha,
            "pixel_dimensions": list(raw_rgb.size),
        }

        findings = [
            Finding(
                "identity",
                "Perceptual hash (pHash)",
                data["phash"],
                Severity.INFO,
                detail="Stable across rescaling and recompression; compare with Hamming distance",
            ),
            Finding(
                "identity",
                "Decoded-pixel SHA-256",
                pixel_sha,
                Severity.INFO,
                detail="Unchanged by metadata edits — matches files the file hash would miss",
            ),
        ]
        return self.ok(data, findings)
