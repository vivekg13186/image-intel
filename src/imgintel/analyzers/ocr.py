"""Text extraction via RapidOCR (PP-OCRv4 weights on onnxruntime).

Runs on ``ctx.small`` rather than full resolution. That is deliberate: every
bounding box in the findings document lives in analysis space, so the HTML
overlay, the sensitive-data highlighter and the OCR boxes all agree without
anyone having to rescale. PP-OCR's detector downscales internally anyway, so
full resolution buys accuracy only on very small text — at which point the
right fix is a tiled pass, not a bigger single input.

Output is per-block with boxes and confidence, never a single text blob:
localizing a match is what makes redaction and PII highlighting possible.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module

#: Blocks below this score are kept in `data` but excluded from `text`.
MIN_CONFIDENCE = 0.45
#: Enough text to look like a document rather than incidental signage.
DOCUMENT_CHARS = 400

#: Accepted backends, best first. RapidOCR was renamed from
#: ``rapidocr-onnxruntime`` to ``rapidocr`` at v3; the old package caps at
#: Python <3.13 and so cannot be installed on current interpreters, while the
#: new one is pure-Python and supports 3.8+. Both are accepted because an
#: existing environment may already have the old one, and the only real
#: differences are the import name, the model generation, and whether the
#: call returns a tuple or an object.
_BACKENDS: tuple[tuple[str, str], ...] = (
    ("rapidocr", "PP-OCRv6"),
    ("rapidocr_onnxruntime", "PP-OCRv4"),
)


def backend_module() -> str | None:
    """The RapidOCR package available here, or None."""
    return next((module for module, _ in _BACKENDS if has_module(module)), None)


def backend_label(module: str | None) -> str:
    for name, generation in _BACKENDS:
        if name == module:
            return f"{name.replace('_', '-')} ({generation})"
    return "unknown"


@lru_cache(maxsize=1)
def _engine() -> tuple[Any, str]:
    """One engine per process — constructing it builds three ONNX sessions."""
    module = backend_module()
    if module is None:
        raise RuntimeError("no RapidOCR backend installed")

    import importlib  # noqa: PLC0415

    package = importlib.import_module(module)

    # RapidOCR v3 logs three INFO lines per engine at construction — across a
    # batch run that is thousands of lines interleaved with the progress bar.
    # It re-applies its own log level from config as each sub-engine is built,
    # so setting the logger level from outside is overwritten; the supported
    # way to quieten it is its own config knob. Warnings still surface.
    try:
        engine = package.RapidOCR(params={"Global.log_level": "warning"})
    except TypeError:
        # v1 takes no params, and does not log at construction anyway.
        engine = package.RapidOCR()
    _quieten_backend_logging()
    return engine, module


def _quieten_backend_logging() -> None:
    """Belt and braces for any logger the config knob does not cover.

    ERROR rather than WARNING: RapidOCR warns "text detection result is empty"
    on every image without text, which is a completely ordinary outcome that
    this analyzer already reports as a finding. Across a batch of landscape
    photographs that is one warning per image and nothing useful. Real
    failures still surface, both here and as an analyzer error.
    """
    for name in ("RapidOCR", "rapidocr", "rapidocr_onnxruntime"):
        logger = logging.getLogger(name)
        if logger.level < logging.ERROR:
            logger.setLevel(logging.ERROR)
        for handler in logger.handlers:
            if handler.level < logging.ERROR:
                handler.setLevel(logging.ERROR)


def _normalize(raw: Any) -> list[tuple[Any, str, float]]:
    """Flatten either RapidOCR result shape into (polygon, text, score).

    v1 returns ``(list_of_triples, elapsed)``; v3 returns a ``RapidOCROutput``
    with ``.boxes``/``.txts``/``.scores``. Both also return an empty result as
    ``None``, which is why every branch tolerates it.
    """
    if raw is None:
        return []

    # v1: the call yields a (result, elapsed) pair.
    if isinstance(raw, tuple) and len(raw) == 2 and not isinstance(raw[0], (str, bytes)):
        raw = raw[0]
    if raw is None:
        return []

    # v3: an output object.
    if hasattr(raw, "boxes") and hasattr(raw, "txts"):
        boxes = raw.boxes if raw.boxes is not None else []
        txts = raw.txts if raw.txts is not None else []
        scores = raw.scores if raw.scores is not None else []
        return [
            (box, str(text), float(score))
            for box, text, score in zip(boxes, txts, scores, strict=False)
        ]

    out: list[tuple[Any, str, float]] = []
    for item in raw:
        if len(item) >= 3:
            out.append((item[0], str(item[1]), float(item[2])))
    return out


def _bbox(polygon: Any) -> tuple[int, int, int, int] | None:
    try:
        pts = np.asarray(polygon, dtype=float).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if not len(pts):
        return None
    return (
        int(pts[:, 0].min()), int(pts[:, 1].min()),
        int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max())),
    )


class OcrAnalyzer(Analyzer):
    name = "ocr"
    version = "1.0.0"
    title = "Text extraction"
    description = "OCR via RapidOCR/PP-OCRv4, with per-block boxes and confidence"
    cost = Cost.HEAVY
    needs_models = ("ocr-det", "ocr-rec")

    def available(self) -> Availability:
        if backend_module() is None:
            return Availability.no("RapidOCR is not installed", "pip install 'imgintel[ocr]'")
        from imgintel.core.modelstore import available as models_available

        ok, reason = models_available(*self.needs_models)
        if ok:
            return Availability.yes()
        # The hint is what makes an unavailable analyzer actionable.
        return Availability.no(reason, f"imgintel models pull {' '.join(self.needs_models)}")

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        image = ctx.small
        height, width = image.shape[:2]

        engine, module = _engine()
        results = _normalize(engine(image))

        blocks: list[dict[str, Any]] = []
        for polygon, text, score in results:
            text = text.strip()
            if not text:
                continue
            blocks.append(
                {
                    "text": text,
                    "confidence": round(score, 4),
                    "bbox": list(_bbox(polygon) or (0, 0, 0, 0)),
                    "polygon": np.asarray(polygon, dtype=float).reshape(-1, 2).astype(int).tolist(),
                }
            )

        # Reading order: top to bottom, then left to right within a line band.
        blocks.sort(key=lambda b: (b["bbox"][1] // 12, b["bbox"][0]))

        confident = [b for b in blocks if b["confidence"] >= MIN_CONFIDENCE]
        text = "\n".join(b["text"] for b in confident)
        mean_conf = (
            round(float(np.mean([b["confidence"] for b in blocks])), 4) if blocks else 0.0
        )

        data = {
            "engine": backend_label(module),
            "analysis_dimensions": [width, height],
            "block_count": len(blocks),
            "confident_block_count": len(confident),
            "char_count": len(text),
            "mean_confidence": mean_conf,
            "min_confidence_threshold": MIN_CONFIDENCE,
            "text": text,
            "blocks": blocks,
        }

        findings = self._findings(blocks, confident, text, mean_conf, width, height)
        return self.ok(data, findings)

    # ----------------------------------------------------------------------

    def _findings(
        self,
        blocks: list[dict],
        confident: list[dict],
        text: str,
        mean_conf: float,
        width: int,
        height: int,
    ) -> list[Finding]:
        if not blocks:
            return [Finding("text", "No text detected", None, Severity.INFO)]

        out = [
            Finding(
                "text",
                "Text detected",
                f"{len(confident)} block(s), {len(text)} characters",
                Severity.INFO,
                confidence=mean_conf,
                detail=_preview(text),
            )
        ]

        if len(text) >= DOCUMENT_CHARS:
            out.append(
                Finding(
                    "text",
                    "Text-heavy image",
                    f"{len(text)} characters",
                    Severity.MEDIUM,
                    confidence=0.7,
                    detail="Consistent with a document, screenshot or sign rather than a scene — "
                    "worth checking for sensitive content",
                )
            )

        weak = [b for b in blocks if b["confidence"] < MIN_CONFIDENCE]
        if weak and len(weak) > len(blocks) * 0.3:
            out.append(
                Finding(
                    "text",
                    "Low-confidence OCR",
                    f"{len(weak)} of {len(blocks)} blocks below {MIN_CONFIDENCE}",
                    Severity.LOW,
                    confidence=0.8,
                    detail="Blur, low resolution, unusual script or heavy compression; "
                    "treat extracted text as unreliable",
                )
            )

        # A large block near the frame edge is often an overlay: a timestamp
        # burn-in, a watermark, or a caption added after capture.
        for block in confident:
            x0, y0, x1, y1 = block["bbox"]
            box_w, box_h = x1 - x0, y1 - y0
            near_edge = y0 < height * 0.08 or y1 > height * 0.92
            wide = box_w > width * 0.35
            if near_edge and wide and box_h > height * 0.03:
                out.append(
                    Finding(
                        "text",
                        "Edge overlay text",
                        block["text"][:80],
                        Severity.LOW,
                        confidence=0.5,
                        bbox=tuple(block["bbox"]),
                        detail="Large text at the frame edge — often a burned-in timestamp, "
                        "watermark or caption added after capture",
                    )
                )
                break

        return out


def _preview(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
