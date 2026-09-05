"""A written summary of what every other analyzer found.

Template-driven and deterministic. No language model is involved, and that is
the design rather than a limitation: the same image produces the same words
every time, nothing is asserted that no analyzer measured, and the summary can
go into an evidence bundle without anyone having to ask whether a model
invented part of it.

It reads the normalized ``findings`` list, not analyzers' ``data``. That means
a new analyzer's findings appear in the summary automatically — the split
between rich per-analyzer data and flat findings, which has held since Phase 0,
is what makes this thirty lines rather than three hundred.
"""

from __future__ import annotations

from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext

#: Sections, in the order a reader wants them, and the categories feeding each.
_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Provenance", ("file", "provenance", "metadata", "device", "attribution")),
    ("Location", ("location",)),
    ("Timeline", ("timeline",)),
    ("Content", ("objects", "people", "scene", "text", "vehicles", "brand", "codes")),
    ("Identity", ("identity", "entity")),
    ("Sensitive data", ("sensitive", "credentials")),
    ("Integrity", ("manipulation",)),
    ("Quality", ("quality", "color")),
)


class SummaryAnalyzer(Analyzer):
    name = "summary"
    version = "1.0.0"
    title = "Summary"
    description = "Deterministic written rollup of every other analyzer's findings"
    cost = Cost.CHEAP
    # Runs last: it summarises whatever else was in the run. Soft edges, so a
    # narrow selection still produces a summary of what did run.
    after = (
        "fileinfo", "exif", "gps", "hashes", "perceptual", "color", "quality", "blur",
        "codes", "ocr", "language", "sensitive", "objects", "faces", "scene",
        "vehicles", "logo", "facedb", "objectdb", "tamper", "geoestimate",
    )

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        findings = [f for result in ctx.results.values() for f in result.findings]
        counts = _severity_counts(findings)

        sections = []
        for title, categories in _SECTIONS:
            lines = _section_lines(findings, categories)
            if lines:
                sections.append({"section": title, "points": lines})

        headline = _headline(ctx, findings, counts)
        text = _render(headline, sections, counts)

        data: dict[str, Any] = {
            "method": "deterministic template",
            "headline": headline,
            "sections": sections,
            "severity_counts": counts,
            "findings_summarised": len(findings),
            "text": text,
        }
        return self.ok(
            data,
            [
                Finding(
                    "summary",
                    "Summary",
                    headline,
                    Severity.INFO,
                    detail=text if len(text) < 900 else text[:897] + "…",
                )
            ],
        )


def _severity_counts(findings: list[Finding]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0, "info": 0}
    for finding in findings:
        counts[finding.severity.value] += 1
    return counts


def _section_lines(findings: list[Finding], categories: tuple[str, ...]) -> list[str]:
    """The notable findings in these categories, most significant first."""
    selected = [
        f
        for f in findings
        if f.category in categories and f.severity.rank >= Severity.LOW.rank
    ]
    selected.sort(key=lambda f: (-f.severity.rank, f.label))

    lines = []
    for finding in selected[:6]:
        value = f": {finding.value}" if finding.value is not None else ""
        confidence = (
            f" ({finding.confidence:.0%} confidence)" if finding.confidence < 0.9 else ""
        )
        lines.append(f"{finding.label}{value}{confidence}".strip())
    return lines


def _headline(ctx: AnalysisContext, findings: list[Finding], counts: dict[str, int]) -> str:
    """One sentence naming the image and the most significant thing about it."""
    name = ctx.path.name
    high = [f for f in findings if f.severity is Severity.HIGH]

    if not findings:
        return f"{name}: nothing of note found."
    if not high:
        return (
            f"{name}: {counts['medium']} finding(s) of moderate significance, "
            "nothing high-severity."
        )

    # Prefer the finding categories a reader will care about most.
    for category in ("sensitive", "credentials", "identity", "location", "manipulation"):
        match = next((f for f in high if f.category == category), None)
        if match:
            value = f" ({match.value})" if match.value is not None else ""
            return f"{name}: {match.label.lower()}{value}."
    return f"{name}: {high[0].label.lower()}."


def _render(headline: str, sections: list[dict], counts: dict[str, int]) -> str:
    lines = [headline, ""]
    for section in sections:
        lines.append(f"{section['section']}:")
        lines.extend(f"  - {point}" for point in section["points"])
        lines.append("")

    lines.append(
        f"Findings: {counts['high']} high, {counts['medium']} medium, "
        f"{counts['low']} low, {counts['info']} informational."
    )
    lines.append(
        "This summary is generated mechanically from the findings above and asserts "
        "nothing that was not measured. Findings are indicators; read each one's "
        "explanation before relying on it."
    )
    return "\n".join(lines).strip()
