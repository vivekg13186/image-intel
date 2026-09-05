"""ORB reference matching, shared by the logo and object-database analyzers.

Descriptor matches on their own produce constant false positives on repetitive
texture — brickwork, foliage, fabric weave. Requiring the matches to agree on a
single homography is what makes a result trustworthy, and it hands back the
location and orientation of the match for free.

Extracted from the logo analyzer when the entity database needed the same
thing for enrolled objects: the two differ only in where the references come
from, not in how matching works.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

N_FEATURES = 1500
#: Lowe's ratio test threshold — the conventional value.
RATIO = 0.75
#: Below this many geometrically consistent matches, call it noise.
MIN_INLIERS = 10
RANSAC_REPROJ = 5.0


@lru_cache(maxsize=2)
def orb(n_features: int = N_FEATURES):
    import cv2  # noqa: PLC0415

    return cv2.ORB_create(nfeatures=n_features)


@dataclass(slots=True)
class Reference:
    """A thing to look for, with its descriptors precomputed."""

    name: str
    keypoints: Any
    descriptors: Any
    shape: tuple[int, int]
    source: str | None = None
    entity_id: int | None = None


def describe(image_gray: np.ndarray, n_features: int = N_FEATURES):
    """Keypoints and descriptors for a greyscale image."""
    return orb(n_features).detectAndCompute(
        np.ascontiguousarray(image_gray, dtype=np.uint8), None
    )


def make_reference(
    name: str,
    image_gray: np.ndarray,
    *,
    source: str | None = None,
    entity_id: int | None = None,
) -> Reference | None:
    """Build a reference, or None if the image is too featureless to match."""
    keypoints, descriptors = describe(image_gray)
    if descriptors is None or len(keypoints) < MIN_INLIERS:
        return None
    return Reference(
        name=name,
        keypoints=keypoints,
        descriptors=descriptors,
        shape=image_gray.shape[:2],
        source=source,
        entity_id=entity_id,
    )


def match_reference(
    reference: Reference,
    scene_keypoints: Any,
    scene_descriptors: Any,
    *,
    min_inliers: int = MIN_INLIERS,
) -> dict[str, Any] | None:
    """Ratio-test matches, then require them to agree on one homography."""
    import cv2  # noqa: PLC0415

    if scene_descriptors is None or len(scene_descriptors) < min_inliers:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(reference.descriptors, scene_descriptors, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < min_inliers:
        return None

    src = np.float32([reference.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([scene_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ)
    if homography is None or mask is None:
        return None
    inliers = int(mask.sum())
    if inliers < min_inliers:
        return None

    # Project the reference's corners to locate the match in the scene.
    height, width = reference.shape
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)

    x0, y0 = projected.min(axis=0)
    x1, y1 = projected.max(axis=0)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    return {
        "name": reference.name,
        "entity_id": reference.entity_id,
        "source": reference.source,
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(inliers / len(good), 3),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "quad": projected.astype(int).tolist(),
    }


def match_all(
    references: list[Reference] | tuple[Reference, ...],
    image_gray: np.ndarray,
    *,
    min_inliers: int = MIN_INLIERS,
) -> tuple[list[dict[str, Any]], int]:
    """Match every reference against one scene. Returns (matches, keypoints)."""
    keypoints, descriptors = describe(image_gray)
    if descriptors is None:
        return [], 0

    matches = [
        result
        for reference in references
        if (result := match_reference(reference, keypoints, descriptors,
                                      min_inliers=min_inliers)) is not None
    ]
    matches.sort(key=lambda m: -m["inliers"])
    return matches, len(keypoints)


def confidence_for(inliers: int) -> float:
    """More geometrically consistent matches means more confidence, capped.

    ORB agreement is corroboration that the same textured thing is present, not
    proof of identity, so the ceiling stays below certainty.
    """
    return round(min(0.9, 0.45 + inliers / 100), 3)


__all__ = [
    "MIN_INLIERS",
    "N_FEATURES",
    "RATIO",
    "Reference",
    "confidence_for",
    "describe",
    "match_all",
    "match_reference",
    "make_reference",
]
