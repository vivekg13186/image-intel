"""Shared, lazily-computed per-image state.

Decoding a large JPEG costs more than most analyzers do. Everything derived
from the file is computed at most once here and reused by every analyzer that
asks for it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:  # pragma: no cover
    from imgintel.core.analyzer import AnalyzerResult

# Forensic inputs are sometimes genuinely enormous. Raise Pillow's bomb guard
# but keep a ceiling, and let fileinfo flag anything past the default.
PILLOW_DEFAULT_LIMIT = 89_478_485
Image.MAX_IMAGE_PIXELS = 1_000_000_000

#: Longest edge of ``ctx.small``. Model inputs and expensive pixel passes use
#: this rather than full resolution.
SMALL_EDGE = 1024


def small_size(width: int, height: int, edge: int = SMALL_EDGE) -> tuple[int, int]:
    """The dimensions :attr:`AnalysisContext.small` would have.

    Report rendering needs this to place bounding boxes correctly without
    re-running analysis, so the calculation lives in one place.
    """
    longest = max(width, height)
    if longest <= edge:
        return width, height
    scale = edge / longest
    return max(1, int(width * scale)), max(1, int(height * scale))


class ImageDecodeError(RuntimeError):
    """Raised when the file cannot be decoded as an image."""


@dataclass
class CaseConfig:
    """Investigation-level metadata carried into evidence exports."""

    case_id: str | None = None
    operator: str | None = None
    notes: str | None = None

    # -- biometric authorisation ------------------------------------------
    #
    # Face *identification* is processing of biometric data under GDPR Art. 9,
    # Illinois BIPA and their equivalents; face *detection* is not. Two
    # independent conditions must hold before any identification runs:
    #
    #   1. a lawful basis recorded in the case file (this field), and
    #   2. --allow-biometrics on the individual run.
    #
    # Requiring both is deliberate. The case file answers "were you entitled to
    # do this", which belongs with the case and ends up in the evidence bundle;
    # the flag answers "did you mean to do it on this run", which should never
    # be inherited silently from a config file. Neither alone is sufficient.
    #: Free-text lawful basis, e.g. "Warrant 2026/114, Art. 9(2)(f)".
    biometric_lawful_basis: str | None = None
    #: Who authorised it.
    biometric_authorised_by: str | None = None
    #: ISO date after which the authorisation no longer applies.
    biometric_expires: str | None = None

    @property
    def biometrics_authorized(self) -> bool:
        """True when a lawful basis is recorded and has not expired."""
        if not self.biometric_lawful_basis:
            return False
        return not self.biometric_authorisation_expired

    @property
    def biometric_authorisation_expired(self) -> bool:
        if not self.biometric_expires:
            return False
        from datetime import date  # noqa: PLC0415

        try:
            return date.fromisoformat(self.biometric_expires[:10]) < date.today()
        except ValueError:
            # An unparseable expiry is treated as expired: failing closed is the
            # only safe reading for an authorisation record.
            return True

    def biometric_refusal(self, *, run_flag: bool) -> str | None:
        """Why identification must not run, or None if it may.

        Returns a reason rather than raising so the caller can record it as an
        analyzer status instead of aborting the whole analysis.
        """
        if not self.biometric_lawful_basis:
            return (
                "no lawful basis recorded for biometric identification — set one with "
                "`imgintel case set --biometric-basis '<basis>' --biometric-authorised-by '<name>'`"
            )
        if self.biometric_authorisation_expired:
            return (
                f"biometric authorisation expired on {self.biometric_expires}; "
                "renew it in the case file before matching identities"
            )
        if not run_flag:
            return "biometric identification requires --allow-biometrics on this run"
        return None


class AnalysisContext:
    """Everything an analyzer is allowed to know about one image."""

    def __init__(
        self,
        path: str | Path,
        *,
        allow_network: bool = False,
        allow_biometrics: bool = False,
        entity_db: str | Path | None = None,
        case: CaseConfig | None = None,
    ) -> None:
        self.path = Path(path)
        self.allow_network = allow_network
        #: Per-run consent for biometric identification. Never inherited from
        #: config — see :class:`CaseConfig`.
        self.allow_biometrics = allow_biometrics
        #: Entity database to match against, if any.
        self.entity_db = Path(entity_db) if entity_db else None
        self.case = case or CaseConfig()
        #: Results of analyzers that already ran, keyed by analyzer name.
        self.results: dict[str, AnalyzerResult] = {}
        #: Every analyzer in this run, set by the pipeline before execution.
        #: Lets an analyzer avoid duplicating a finding another one will make.
        self.planned: frozenset[str] = frozenset()
        #: Free-form scratch space for analyzers that want to pass along
        #: non-serializable objects (e.g. a loaded model session).
        self.scratch: dict[str, Any] = {}

    # -- raw bytes ---------------------------------------------------------

    @cached_property
    def raw(self) -> bytes:
        return self.path.read_bytes()

    @cached_property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    # -- decoded forms -----------------------------------------------------

    @cached_property
    def header(self) -> Image.Image:
        """The image opened but **not** decoded.

        Format, dimensions, mode, container info and JPEG quantization tables
        are all available from the header alone. Keeping this separate from
        :attr:`pil_raw` is what makes the ``quick`` profile genuinely quick on
        a 60-megapixel file.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                return Image.open(self.path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
            raise ImageDecodeError(f"{type(exc).__name__}: {exc}") from exc

    @cached_property
    def pil_raw(self) -> Image.Image:
        """The fully decoded image as stored — no EXIF orientation applied.

        Pixel hashing uses this so that two files with identical pixel data but
        different orientation tags do not silently hash the same.
        """
        img = self.header
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                img.load()
        except Exception as exc:  # noqa: BLE001
            raise ImageDecodeError(f"{type(exc).__name__}: {exc}") from exc
        return img

    @cached_property
    def pil(self) -> Image.Image:
        """EXIF-oriented image — what a human sees when they open the file."""
        return ImageOps.exif_transpose(self.pil_raw) or self.pil_raw

    @cached_property
    def rgb(self) -> np.ndarray:
        """H x W x 3 uint8, EXIF-oriented, alpha composited onto white."""
        img = self.pil
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    @cached_property
    def gray(self) -> np.ndarray:
        """H x W float32 luminance in 0..255 (ITU-R BT.601)."""
        r, g, b = (self.rgb[..., i].astype(np.float32) for i in range(3))
        return 0.299 * r + 0.587 * g + 0.114 * b

    @cached_property
    def small(self) -> np.ndarray:
        """RGB downscaled so the longest edge is at most :data:`SMALL_EDGE`."""
        h, w = self.rgb.shape[:2]
        target = small_size(w, h)
        if target == (w, h):
            return self.rgb
        return np.asarray(self.pil.convert("RGB").resize(target, Image.LANCZOS), dtype=np.uint8)

    @cached_property
    def small_gray(self) -> np.ndarray:
        r, g, b = (self.small[..., i].astype(np.float32) for i in range(3))
        return 0.299 * r + 0.587 * g + 0.114 * b

    @cached_property
    def sharpness(self) -> float:
        """Variance-of-Laplacian focus measure over :attr:`small_gray`.

        Lives here rather than in the blur analyzer because quality, blur and
        (later) tamper detection all want it, and it should be computed once.
        """
        from imgintel.util.imgmath import laplacian_variance

        return laplacian_variance(self.small_gray)

    # -- format specifics --------------------------------------------------

    @cached_property
    def dimensions(self) -> tuple[int, int]:
        """(width, height), EXIF-oriented."""
        return self.pil.size

    @cached_property
    def header_dimensions(self) -> tuple[int, int]:
        """(width, height) as stored, read from the header without decoding."""
        return self.header.size

    @cached_property
    def megapixels(self) -> float:
        w, h = self.dimensions
        return (w * h) / 1_000_000

    @cached_property
    def jpeg_quant_tables(self) -> dict[int, list[int]] | None:
        """JPEG quantization tables, or None for non-JPEG input.

        Used by the quality analyzer to estimate the encoder's quality setting
        and (from Phase 6) by tamper detection to fingerprint the encoder.
        """
        tables = getattr(self.header, "quantization", None)
        if not tables:
            return None
        return {int(k): [int(x) for x in v] for k, v in tables.items()}

    @cached_property
    def readable(self) -> bool:
        """True if the header parses as a known image format (no decode)."""
        try:
            _ = self.header
            return True
        except ImageDecodeError:
            return False

    @cached_property
    def decodable(self) -> bool:
        """True if the pixel data actually decodes. Forces a full decode."""
        try:
            _ = self.pil_raw
            return True
        except ImageDecodeError:
            return False

    # -- reading upstream analyzer output ----------------------------------

    def data(self, analyzer: str, default: Any = None) -> Any:
        """Read another analyzer's ``data``. The only cross-plugin coupling."""
        res = self.results.get(analyzer)
        if res is None or not res.succeeded:
            return {} if default is None else default
        return res.data

    def succeeded(self, analyzer: str) -> bool:
        res = self.results.get(analyzer)
        return res is not None and res.succeeded


@dataclass
class BatchContext:
    """Cross-image state for batch runs (dedupe indexes, shared models)."""

    case: CaseConfig = field(default_factory=CaseConfig)
    allow_network: bool = False
    shared: dict[str, Any] = field(default_factory=dict)
