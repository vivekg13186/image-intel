"""Language of the extracted text.

Runs per OCR block, not on the concatenated blob. A sign in one language beside
a caption in another is exactly the situation worth detecting, and joining them
first destroys that signal.

Uses lingua rather than langdetect because it is calibrated for short strings.
Even so, short input is unreliable — "ACME" alone confidently detects as
Somali — so blocks below :data:`MIN_CHARS` are not classified at all, and
low-confidence results are reported as undetermined rather than guessed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module

#: Below this, detection is noise regardless of what the confidence says.
MIN_CHARS = 14
#: Below this confidence the block is recorded as undetermined.
MIN_CONFIDENCE = 0.55


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder  # noqa: PLC0415

    # Low-accuracy mode trades a little precision for a much smaller memory
    # footprint and far faster startup — the right trade for short OCR blocks.
    return (
        LanguageDetectorBuilder.from_all_languages()
        .with_low_accuracy_mode()
        .with_preloaded_language_models()
        .build()
    )


class LanguageAnalyzer(Analyzer):
    name = "language"
    version = "1.0.0"
    title = "Text language"
    description = "Per-block language identification of OCR output"
    requires = ("ocr",)
    cost = Cost.MEDIUM

    def available(self) -> Availability:
        if not has_module("lingua"):
            return Availability.no(
                "lingua-language-detector not installed", "pip install 'imgintel[lang]'"
            )
        return Availability.yes()

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        blocks: list[dict[str, Any]] = ctx.data("ocr").get("blocks", [])
        candidates = [b for b in blocks if len(b.get("text", "").strip()) >= MIN_CHARS]

        if not candidates:
            return self.ok(
                {
                    "languages": [],
                    "per_block": [],
                    "analyzed_blocks": 0,
                    "skipped_short_blocks": len(blocks),
                    "min_chars": MIN_CHARS,
                },
                [],
            )

        detector = _detector()
        per_block: list[dict[str, Any]] = []
        totals: dict[str, float] = {}

        for block in candidates:
            text = block["text"].strip()
            values = detector.compute_language_confidence_values(text)
            best = values[0] if values else None
            if best is None or best.value < MIN_CONFIDENCE:
                per_block.append(
                    {
                        "text": text[:80],
                        "language": None,
                        "confidence": round(best.value, 4) if best else 0.0,
                        "bbox": block.get("bbox"),
                        "note": "below confidence threshold",
                    }
                )
                continue

            name = best.language.name.title()
            per_block.append(
                {
                    "text": text[:80],
                    "language": name,
                    "iso_639_1": best.language.iso_code_639_1.name.lower(),
                    "confidence": round(best.value, 4),
                    "bbox": block.get("bbox"),
                }
            )
            # Weight by text length: a long paragraph should outrank a caption.
            totals[name] = totals.get(name, 0.0) + len(text) * best.value

        ranked = sorted(totals, key=lambda k: -totals[k])
        data = {
            "languages": ranked,
            "primary": ranked[0] if ranked else None,
            "per_block": per_block,
            "analyzed_blocks": len(candidates),
            "skipped_short_blocks": len(blocks) - len(candidates),
            "undetermined_blocks": sum(1 for b in per_block if b["language"] is None),
            "min_chars": MIN_CHARS,
            "min_confidence": MIN_CONFIDENCE,
        }

        findings: list[Finding] = []
        if ranked:
            findings.append(
                Finding(
                    "text",
                    "Primary text language",
                    ranked[0],
                    Severity.INFO,
                    confidence=0.8,
                    detail=None if len(ranked) == 1 else f"Also present: {', '.join(ranked[1:4])}",
                )
            )
        if len(ranked) > 1:
            findings.append(
                Finding(
                    "text",
                    "Multiple languages present",
                    ", ".join(ranked[:4]),
                    Severity.LOW,
                    confidence=0.6,
                    detail="Mixed-language text can indicate a border region, a tourist "
                    "location, or imported signage",
                )
            )
        return self.ok(data, findings)
