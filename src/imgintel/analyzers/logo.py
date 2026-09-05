"""Logo and brand detection by reference matching.

No general-purpose logo model exists under a permissive licence, and a
fine-tuned one would only know the brands it was trained on. In an actual
investigation the brands that matter are usually specific and known in
advance — a particular company's letterhead, a courier's livery, a marking on
equipment — so the useful design is the opposite of a fixed classifier:
point the tool at a folder of reference images and it finds those.

Matching is ORB keypoints plus a homography check. Descriptor matches alone
produce constant false positives on repetitive texture; requiring the matches
to agree on a single geometric transform is what makes the result trustworthy,
and it yields the location and orientation of the match for free.

Uses ORB rather than SIFT: both ship in OpenCV now, but ORB is faster and its
patent history is unambiguous.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from platformdirs import user_config_dir

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module
from imgintel.util.orbmatch import (
    MIN_INLIERS,
    N_FEATURES,
    RATIO,
    Reference,
    confidence_for,
    make_reference,
    match_all,
)

#: Env var pointing at a folder of reference logo images.
REFERENCE_DIR_ENV = "IMGINTEL_LOGO_DIR"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

__all__ = ["MIN_INLIERS", "N_FEATURES", "RATIO", "LogoAnalyzer", "load_references", "reference_dir"]


def reference_dir() -> Path:
    import os  # noqa: PLC0415

    override = os.environ.get(REFERENCE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir("imgintel")) / "logos"


@lru_cache(maxsize=4)
def load_references(directory: str) -> tuple[Reference, ...]:
    """Compute descriptors for every reference image, once per process."""
    import cv2  # noqa: PLC0415

    root = Path(directory)
    if not root.is_dir():
        return ()

    out: list[Reference] = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        reference = make_reference(path.stem, image, source=str(path))
        if reference is not None:  # None means too featureless to match
            out.append(reference)
    return tuple(out)


class LogoAnalyzer(Analyzer):
    name = "logo"
    version = "1.0.0"
    title = "Logos and brands"
    description = "Matches user-supplied reference logos by ORB features and homography"
    cost = Cost.MEDIUM

    def available(self) -> Availability:
        if not has_module("cv2"):
            return Availability.no("opencv is not installed", "pip install 'imgintel[logo]'")
        directory = reference_dir()
        if not directory.is_dir():
            return Availability.no(
                f"no reference logos in {directory}",
                f"add images there, or set {REFERENCE_DIR_ENV}",
            )
        if not load_references(str(directory)):
            return Availability.no(
                f"{directory} contains no usable reference images",
                "reference logos need enough texture to yield ORB features",
            )
        return Availability.yes()

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        directory = reference_dir()
        references = load_references(str(directory))

        matches, keypoints = match_all(references, ctx.small_gray.astype(np.uint8))
        data = {
            "reference_dir": str(directory),
            "reference_count": len(references),
            "references": [r.name for r in references],
            "scene_keypoints": keypoints,
            "min_inliers": MIN_INLIERS,
            "matches": matches,
        }
        return self.ok(data, _findings(matches))


def _findings(matches: list[dict]) -> list[Finding]:
    return [
        Finding(
            "brand",
            f"Reference logo matched: {match['name']}",
            f"{match['inliers']} consistent keypoints",
            Severity.MEDIUM,
            confidence=confidence_for(match["inliers"]),
            bbox=tuple(match["bbox"]),
            detail="Geometrically verified against a supplied reference image. "
            "Confirm visually — repetitive texture can still produce a match",
        )
        for match in matches
    ]
