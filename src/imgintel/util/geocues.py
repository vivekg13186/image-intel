"""Country cues recoverable from text and metadata.

Nothing here is a model. Phone country codes, postcode formats, top-level
domains, currency symbols and vehicle-plate shapes are simply *facts about
places*, and an image containing a UK postcode next to a +44 phone number and
a .co.uk domain has told you where it is without any inference at all.

That is why this sits alongside the model-based approaches rather than under
them: it is often the strongest evidence in the frame, and it is fully
explainable — every hypothesis can name the exact string that produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: ISO 3166-1 alpha-2 -> display name, for the countries the cues cover.
COUNTRY_NAMES: dict[str, str] = {
    "GB": "United Kingdom", "US": "United States", "CA": "Canada", "AU": "Australia",
    "NZ": "New Zealand", "IE": "Ireland", "FR": "France", "DE": "Germany",
    "ES": "Spain", "IT": "Italy", "NL": "Netherlands", "BE": "Belgium",
    "CH": "Switzerland", "AT": "Austria", "PT": "Portugal", "SE": "Sweden",
    "NO": "Norway", "DK": "Denmark", "FI": "Finland", "PL": "Poland",
    "RU": "Russia", "UA": "Ukraine", "IN": "India", "CN": "China",
    "JP": "Japan", "KR": "South Korea", "SG": "Singapore", "MY": "Malaysia",
    "TH": "Thailand", "ID": "Indonesia", "PH": "Philippines", "VN": "Vietnam",
    "AE": "United Arab Emirates", "SA": "Saudi Arabia", "IL": "Israel",
    "TR": "Turkey", "EG": "Egypt", "ZA": "South Africa", "NG": "Nigeria",
    "KE": "Kenya", "BR": "Brazil", "AR": "Argentina", "MX": "Mexico",
    "CL": "Chile", "CO": "Colombia", "PE": "Peru",
}

#: International dialling prefixes. Longest-first matching matters: +1 must not
#: shadow +1-something, and +4 must not shadow +44.
DIALLING_CODES: dict[str, str] = {
    "44": "GB", "353": "IE", "33": "FR", "49": "DE", "34": "ES", "39": "IT",
    "31": "NL", "32": "BE", "41": "CH", "43": "AT", "351": "PT", "46": "SE",
    "47": "NO", "45": "DK", "358": "FI", "48": "PL", "380": "UA", "7": "RU",
    "91": "IN", "86": "CN", "81": "JP", "82": "KR", "65": "SG", "60": "MY",
    "66": "TH", "62": "ID", "63": "PH", "84": "VN", "971": "AE", "966": "SA",
    "972": "IL", "90": "TR", "20": "EG", "27": "ZA", "234": "NG", "254": "KE",
    "55": "BR", "54": "AR", "52": "MX", "56": "CL", "57": "CO", "51": "PE",
    "61": "AU", "64": "NZ", "1": "US",  # shared with CA; reported as ambiguous
}
#: Codes that do not identify a single country.
AMBIGUOUS_CODES = {"1": ("US", "CA"), "7": ("RU", "KZ")}

#: Country-code TLDs seen in URLs and business names.
CCTLD: dict[str, str] = {
    "uk": "GB", "ie": "IE", "fr": "FR", "de": "DE", "es": "ES", "it": "IT",
    "nl": "NL", "be": "BE", "ch": "CH", "at": "AT", "pt": "PT", "se": "SE",
    "no": "NO", "dk": "DK", "fi": "FI", "pl": "PL", "ua": "UA", "ru": "RU",
    "in": "IN", "cn": "CN", "jp": "JP", "kr": "KR", "sg": "SG", "my": "MY",
    "th": "TH", "id": "ID", "ph": "PH", "vn": "VN", "ae": "AE", "sa": "SA",
    "il": "IL", "tr": "TR", "eg": "EG", "za": "ZA", "ng": "NG", "ke": "KE",
    "br": "BR", "ar": "AR", "mx": "MX", "cl": "CL", "co": "CO", "pe": "PE",
    "au": "AU", "nz": "NZ", "ca": "CA", "us": "US",
}


@dataclass(frozen=True, slots=True)
class Cue:
    """One observation and the country it points at."""

    country: str
    kind: str
    evidence: str
    weight: float
    detail: str = ""


#: (kind, regex, country, weight, detail). Weights reflect how strongly a
#: single occurrence pins a country, not how common the pattern is.
_PATTERNS: tuple[tuple[str, re.Pattern, str, float, str], ...] = (
    (
        "postcode",
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
        "GB", 0.9, "UK postcode format",
    ),
    (
        "postcode",
        re.compile(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b"),
        "CA", 0.9, "Canadian postal code format",
    ),
    (
        "postcode",
        re.compile(r"\b\d{5}(?:-\d{4})?\b(?=\s*(?:USA|US|United States)?)"),
        "US", 0.25, "5-digit ZIP-like number (weak on its own)",
    ),
    (
        "road",
        re.compile(r"\b(?:A|B|M)\d{1,4}\b(?=\s|,|$)"),
        "GB", 0.3, "UK road numbering",
    ),
    (
        # German street names are compounds — Hauptstrasse, Bahnhofstraße —
        # so this has to match as a suffix, not as a standalone word.
        "address",
        re.compile(r"\b\w{3,}(?:strasse|straße|weg|platz)\b", re.IGNORECASE),
        "DE", 0.6, "German street-type suffix",
    ),
    (
        "address",
        re.compile(r"\b(?:Rue|Avenue|Boulevard|Place)\s+(?:de|du|des|la|le)\b", re.IGNORECASE),
        "FR", 0.6, "French address form",
    ),
    (
        "address",
        re.compile(r"\b(?:Calle|Avenida|Plaza)\b", re.IGNORECASE),
        "ES", 0.5, "Spanish street-type word",
    ),
    (
        "vehicle",
        re.compile(r"\b[A-Z]{2}\d{2}\s?[A-Z]{3}\b"),
        "GB", 0.55, "UK vehicle registration format",
    ),
    (
        "business",
        re.compile(r"\b(?:Ltd|Limited|PLC)\b"),
        "GB", 0.35, "UK company suffix",
    ),
    (
        "business",
        re.compile(r"\bGmbH\b"),
        "DE", 0.7, "German company suffix",
    ),
    (
        "business",
        re.compile(r"\b(?:S\.?A\.?R\.?L|SAS)\b"),
        "FR", 0.6, "French company suffix",
    ),
    (
        "business",
        re.compile(r"\bPty\.?\s?Ltd\b", re.IGNORECASE),
        "AU", 0.7, "Australian company suffix",
    ),
    (
        # No trailing \b: a word boundary after "Inc." needs a word character
        # following it, so "ACME Inc." at the end of a line would never match.
        "business",
        re.compile(r"\b(?:Inc|LLC|Corp)\.?(?=\s|,|$)"),
        "US", 0.35, "US company suffix",
    ),
    (
        "tax",
        re.compile(r"\bVAT\s*(?:No|Number|Reg)", re.IGNORECASE),
        "GB", 0.3, "VAT registration wording",
    ),
)

_CURRENCY: tuple[tuple[str, str, float], ...] = (
    ("£", "GB", 0.6), ("€", "EU", 0.3), ("¥", "JP", 0.4),
    ("₹", "IN", 0.7), ("₽", "RU", 0.6), ("₩", "KR", 0.7),
)

_PHONE = re.compile(r"\+\s?(\d{1,3})[\s\-().]?\d")
_URL_HOST = re.compile(r"(?:https?://)?(?:[\w-]+\.)+([a-z]{2,})(?:\.([a-z]{2}))?", re.IGNORECASE)


def cues_from_text(text: str) -> list[Cue]:
    """Every country cue found in a block of text."""
    if not text:
        return []
    out: list[Cue] = []

    for kind, pattern, country, weight, detail in _PATTERNS:
        for match in pattern.finditer(text):
            out.append(Cue(country, kind, match.group(0).strip(), weight, detail))

    out.extend(_phone_cues(text))
    out.extend(_domain_cues(text))

    for symbol, country, weight in _CURRENCY:
        if symbol in text:
            out.append(
                Cue(country, "currency", symbol, weight, f"{symbol} currency symbol")
            )
    return out


def _phone_cues(text: str) -> list[Cue]:
    out: list[Cue] = []
    for match in _PHONE.finditer(text):
        digits = match.group(1)
        # Longest prefix first: +44 must win over +4.
        for length in (3, 2, 1):
            prefix = digits[:length]
            if prefix not in DIALLING_CODES:
                continue
            if prefix in AMBIGUOUS_CODES:
                for country in AMBIGUOUS_CODES[prefix]:
                    out.append(
                        Cue(country, "phone", f"+{prefix}", 0.35,
                            f"+{prefix} covers {', '.join(AMBIGUOUS_CODES[prefix])}")
                    )
            else:
                out.append(
                    Cue(DIALLING_CODES[prefix], "phone", f"+{prefix}", 0.8,
                        f"international dialling code +{prefix}")
                )
            break
    return out


def _domain_cues(text: str) -> list[Cue]:
    out: list[Cue] = []
    for match in _URL_HOST.finditer(text):
        first, second = match.group(1).lower(), (match.group(2) or "").lower()
        # co.uk style: the country code is the final label.
        tld = second or first
        if tld in CCTLD:
            out.append(
                Cue(CCTLD[tld], "domain", match.group(0), 0.7, f".{tld} country domain")
            )
    return out


def cue_from_language(language: str | None) -> Cue | None:
    """A weak prior from the written language.

    Weak deliberately: most languages span many countries, and signage in a
    language is at best a hint about which.
    """
    if not language:
        return None
    mapping = {
        "German": "DE", "French": "FR", "Spanish": "ES", "Italian": "IT",
        "Dutch": "NL", "Portuguese": "PT", "Swedish": "SE", "Norwegian": "NO",
        "Danish": "DK", "Finnish": "FI", "Polish": "PL", "Russian": "RU",
        "Ukrainian": "UA", "Japanese": "JP", "Korean": "KR", "Thai": "TH",
        "Turkish": "TR", "Hebrew": "IL", "Greek": "GR", "Vietnamese": "VN",
    }
    country = mapping.get(language)
    if not country:
        return None
    return Cue(country, "language", language, 0.25, f"text is in {language}")


def rank(cues: list[Cue]) -> list[dict]:
    """Combine cues into ranked country hypotheses with their evidence.

    Weights combine as independent evidence — ``1 - prod(1 - w)`` — rather than
    summing, so ten weak hints never outweigh one decisive one, and repeating
    the same weak hint does not manufacture certainty.
    """
    by_country: dict[str, list[Cue]] = {}
    for cue in cues:
        by_country.setdefault(cue.country, []).append(cue)

    hypotheses = []
    for country, group in by_country.items():
        # Take the strongest cue of each kind: five UK postcodes in one
        # document are one observation about the country, not five.
        strongest: dict[str, Cue] = {}
        for cue in group:
            if cue.kind not in strongest or cue.weight > strongest[cue.kind].weight:
                strongest[cue.kind] = cue

        combined = 1.0
        for cue in strongest.values():
            combined *= 1 - cue.weight
        confidence = 1 - combined

        hypotheses.append(
            {
                "country": country,
                "country_name": COUNTRY_NAMES.get(country, country),
                "confidence": round(confidence, 4),
                "cue_kinds": sorted(strongest),
                "evidence": [
                    {
                        "kind": cue.kind,
                        "text": cue.evidence[:80],
                        "weight": cue.weight,
                        "detail": cue.detail,
                    }
                    for cue in sorted(strongest.values(), key=lambda c: -c.weight)
                ],
            }
        )
    return sorted(hypotheses, key=lambda h: -h["confidence"])


__all__ = [
    "AMBIGUOUS_CODES",
    "CCTLD",
    "COUNTRY_NAMES",
    "DIALLING_CODES",
    "Cue",
    "cue_from_language",
    "cues_from_text",
    "rank",
]
