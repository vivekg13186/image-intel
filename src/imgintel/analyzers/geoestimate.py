"""Where was this taken? — a ranked hypothesis list, never a single point.

Evidence is combined in confidence order:

1. **EXIF GPS** — if present and coherent, the question is answered, and this
   analyzer says so rather than adding noise around a known coordinate.
2. **Text in the image** — phone country codes, postcode formats, country
   domains, company suffixes, currency symbols. Often the strongest evidence
   available, and completely explainable: each hypothesis names the exact
   string that produced it.
3. **Language of the text** — a weak prior, since most languages span borders.
4. **Timezone coherence** — when GPS exists, whether the EXIF offset agrees.

No model is involved. A CLIP-based geo-estimator would add coarse visual
priors, but it would also add an opaque 350 MB dependency whose output cannot
be shown to anyone; the text evidence here is both stronger and defensible.

The output is deliberately a *list* with confidences. A single estimated
location invites being read as a fact.
"""

from __future__ import annotations

from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.util.geocues import COUNTRY_NAMES, cue_from_language, cues_from_text, rank

#: Below this a hypothesis is recorded but not raised as a finding.
REPORT_THRESHOLD = 0.4


class GeoEstimateAnalyzer(Analyzer):
    name = "geoestimate"
    version = "1.0.0"
    title = "Location estimate"
    description = "Ranks where an image was taken from GPS, text and language evidence"
    requires = ("gps",)
    # Text evidence is the main input when present, but the analyzer still
    # reports on GPS alone.
    after = ("ocr", "language", "codes")
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        gps = ctx.data("gps")
        data: dict[str, Any] = {"method": "evidence-ranked", "hypotheses": []}
        findings: list[Finding] = []

        # 1. A GPS fix settles it. Adding weaker guesses alongside a known
        #    coordinate would only dilute the record.
        if gps.get("present") and gps.get("latitude") is not None:
            place = gps.get("place") or {}
            data.update(
                {
                    "resolved": True,
                    "source": "exif_gps",
                    "latitude": gps["latitude"],
                    "longitude": gps["longitude"],
                    "country": place.get("country_code"),
                    "place": place.get("name"),
                    "timezone": gps.get("timezone"),
                }
            )
            findings.append(
                Finding(
                    "location",
                    "Location known from EXIF GPS",
                    f"{gps['latitude']:.5f}, {gps['longitude']:.5f}",
                    Severity.INFO,
                    confidence=0.95,
                    detail="Estimation not required. Note that GPS tags can be edited — the "
                    "timestamp cross-check in the gps analyzer is what tests them",
                )
            )
            return self.ok(data, findings)

        # 2. Text evidence.
        text, sources = _collect_text(ctx)
        cues = cues_from_text(text)

        # 3. Language prior.
        primary_language = ctx.data("language").get("primary")
        language_cue = cue_from_language(primary_language)
        if language_cue:
            cues.append(language_cue)

        data["text_characters"] = len(text)
        data["text_sources"] = sources
        data["language"] = primary_language
        data["cue_count"] = len(cues)

        if not cues:
            data["resolved"] = False
            findings.append(
                Finding(
                    "location",
                    "Location not estimated",
                    None,
                    Severity.INFO,
                    detail="No GPS, and no country cues in any extracted text. Run with "
                    "--profile deep so OCR and barcode payloads are available",
                    confidence=1.0,
                )
            )
            return self.ok(data, findings)

        hypotheses = rank(cues)
        data["hypotheses"] = hypotheses
        data["resolved"] = False
        data["source"] = "text_evidence"

        top = hypotheses[0]
        if top["confidence"] >= REPORT_THRESHOLD:
            evidence = ", ".join(
                f"{e['kind']} '{e['text']}'" for e in top["evidence"][:3]
            )
            findings.append(
                Finding(
                    "location",
                    f"Likely country: {top['country_name']}",
                    f"{top['confidence']:.0%} from {evidence}",
                    Severity.MEDIUM if top["confidence"] >= 0.7 else Severity.LOW,
                    confidence=float(top["confidence"]),
                    detail="Inferred from text visible in the image, not from GPS. Signage, "
                    "packaging and documents can all originate elsewhere than the photograph",
                )
            )

        # A close second is worth naming: it stops a marginal call from
        # reading as a settled one.
        if len(hypotheses) > 1 and hypotheses[1]["confidence"] >= top["confidence"] * 0.7:
            findings.append(
                Finding(
                    "location",
                    "Competing country hypothesis",
                    f"{hypotheses[1]['country_name']} ({hypotheses[1]['confidence']:.0%})",
                    Severity.LOW,
                    confidence=float(hypotheses[1]["confidence"]),
                    detail="Evidence supports more than one country; the leading answer is not "
                    "clearly separated from the next",
                )
            )

        return self.ok(data, findings)


def _collect_text(ctx: AnalysisContext) -> tuple[str, list[str]]:
    """Every source of text that might carry a country cue."""
    parts: list[str] = []
    sources: list[str] = []

    ocr = ctx.data("ocr").get("text")
    if ocr:
        parts.append(ocr)
        sources.append("ocr")

    for code in ctx.data("codes").get("codes", []):
        if code.get("text"):
            parts.append(code["text"])
    if len(sources) < len(parts):
        sources.append("barcode")

    # EXIF free-text fields sometimes carry addresses or descriptions.
    highlights = ctx.data("exif").get("highlights", {})
    for key in ("user_comment", "artist", "copyright"):
        value = highlights.get(key)
        if value:
            parts.append(str(value))
            if "metadata" not in sources:
                sources.append("metadata")

    return "\n".join(parts), sources


__all__ = ["COUNTRY_NAMES", "GeoEstimateAnalyzer", "REPORT_THRESHOLD"]
