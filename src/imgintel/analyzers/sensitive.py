"""Sensitive information visible in the image.

Matching runs **per OCR block**, so every hit carries the box it came from.
That is what makes an automated redaction pass possible later, and it is why
this analyzer depends on `ocr` rather than re-running its own text extraction.

Patterns are paired with validators wherever one exists — Luhn for card
numbers, mod-97 for IBANs, libphonenumber for phone numbers. A regex alone
produces far too many false positives on OCR output, which is full of serial
numbers, prices and reference codes that look like everything.

**Values are masked in findings and full in data.** Findings feed the HTML
report and CSV, which get shared; the JSON keeps what the investigator needs.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.deps import has_module

# -- validators ------------------------------------------------------------


def luhn_valid(text: str) -> bool:
    digits = [c for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha():
        return False
    rearranged = compact[4:] + compact[:4]
    try:
        numeric = "".join(str(int(c, 36)) for c in rearranged)
    except ValueError:
        return False
    return int(numeric) % 97 == 1


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    length = len(text)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def high_entropy(text: str, threshold: float = 3.4) -> bool:
    """Distinguishes a real secret from a repeated or structured string."""
    return shannon_entropy(text) >= threshold


def card_brand(text: str) -> str:
    digits = "".join(c for c in text if c.isdigit())
    if digits.startswith("4"):
        return "Visa"
    if digits[:2] in {"34", "37"}:
        return "American Express"
    if 51 <= int(digits[:2] or 0) <= 55 or digits[:4].isdigit() and 2221 <= int(digits[:4]) <= 2720:
        return "Mastercard"
    if digits[:4] == "6011" or digits[:2] == "65":
        return "Discover"
    return "unknown issuer"


# -- pattern table ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pattern:
    key: str
    label: str
    regex: re.Pattern
    severity: Severity
    detail: str
    validator: Callable[[str], bool] | None = None
    confidence: float = 0.8


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        "email", "Email address",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        Severity.MEDIUM, "Directly identifying and contactable", confidence=0.9,
    ),
    Pattern(
        "credit_card", "Payment card number",
        # OCR frequently loses the separators, so accept both forms.
        re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
        Severity.HIGH, "Passes the Luhn checksum — treat as a live card number",
        validator=luhn_valid, confidence=0.85,
    ),
    Pattern(
        "iban", "Bank account (IBAN)",
        re.compile(r"\b[A-Z]{2}\d{2}[ ]?(?:[A-Z0-9]{4}[ ]?){2,7}[A-Z0-9]{1,4}\b"),
        Severity.HIGH, "Passes the mod-97 checksum", validator=iban_valid, confidence=0.9,
    ),
    Pattern(
        "jwt", "JSON Web Token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{0,}"),
        Severity.HIGH, "A bearer token — may still grant access", confidence=0.95,
    ),
    Pattern(
        "aws_key", "AWS access key ID",
        re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA)[0-9A-Z]{16}\b"),
        Severity.HIGH, "Cloud credential", confidence=0.95,
    ),
    Pattern(
        "github_token", "GitHub token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
        Severity.HIGH, "Source-control credential", confidence=0.95,
    ),
    Pattern(
        "generic_secret", "API key or secret",
        re.compile(r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b", re.IGNORECASE),
        Severity.HIGH, "Key-shaped, high-entropy string",
        validator=high_entropy, confidence=0.6,
    ),
    Pattern(
        "us_ssn", "US Social Security number",
        re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b"),
        Severity.HIGH, "US national identifier", confidence=0.75,
    ),
    Pattern(
        "uk_nino", "UK National Insurance number",
        re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}[ ]?\d{2}[ ]?\d{2}[ ]?\d{2}[ ]?[A-D]\b"),
        Severity.HIGH, "UK national identifier", confidence=0.7,
    ),
    Pattern(
        "ipv4", "IP address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        ),
        Severity.LOW, "Network address — may reveal infrastructure", confidence=0.7,
    ),
    Pattern(
        "btc_address", "Bitcoin address",
        re.compile(r"\b(?:bc1[a-z0-9]{25,62}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b"),
        Severity.MEDIUM, "Traceable on-chain", confidence=0.7,
    ),
    Pattern(
        "eth_address", "Ethereum address",
        re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
        Severity.MEDIUM, "Traceable on-chain", confidence=0.85,
    ),
    Pattern(
        "uk_postcode", "UK postcode",
        re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?[ ]?\d[A-Z]{2}\b"),
        Severity.MEDIUM, "Narrows an address to a handful of properties", confidence=0.75,
    ),
)

#: Phone numbers get their own pass — libphonenumber is far better than a regex.
_PHONE_KEYWORD = re.compile(
    r"(?:tel|phone|mobile|cell|fax|whatsapp)[\s.:]*([+\d][\d\s().\-]{7,20})", re.IGNORECASE
)


# -- masking ---------------------------------------------------------------


def mask(value: str, key: str) -> str:
    """Reveal enough to recognise the hit, not enough to reuse it."""
    stripped = value.strip()
    if key == "email":
        local, _, domain = stripped.partition("@")
        head = local[:2] if len(local) > 3 else local[:1]
        return f"{head}{'*' * max(3, len(local) - len(head))}@{domain}"
    if key in ("credit_card", "iban"):
        digits = re.sub(r"[\s-]", "", stripped)
        return f"{digits[:4]}{'*' * max(4, len(digits) - 8)}{digits[-4:]}"
    if len(stripped) <= 8:
        return stripped[0] + "*" * (len(stripped) - 1)
    keep = max(3, len(stripped) // 5)
    return f"{stripped[:keep]}{'*' * (len(stripped) - keep * 2)}{stripped[-keep:]}"


# -- analyzer --------------------------------------------------------------


class SensitiveAnalyzer(Analyzer):
    name = "sensitive"
    version = "1.0.0"
    title = "Sensitive information"
    description = "Locates PII, credentials and financial identifiers in extracted text"
    requires = ("ocr",)
    # Barcode payloads are scanned too when `codes` is in the run, but its
    # absence must not disable PII detection on OCR text.
    after = ("codes",)
    cost = Cost.MEDIUM

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        sources = list(self._sources(ctx))
        matches: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        for text, bbox, origin in sources:
            for match in self._scan(text):
                dedupe_key = (match["type"], match["value"])
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                match["bbox"] = list(bbox) if bbox else None
                match["source"] = origin
                matches.append(match)

        by_type: dict[str, int] = {}
        for match in matches:
            by_type[match["type"]] = by_type.get(match["type"], 0) + 1

        data = {
            "match_count": len(matches),
            "types": sorted(by_type),
            "counts_by_type": by_type,
            "sources_scanned": len(sources),
            "phone_validation": has_module("phonenumbers"),
            # Full values live here; findings carry masked forms.
            "matches": matches,
            "note": "Values in `findings` are masked; `matches` holds the full values.",
        }
        return self.ok(data, self._findings(matches, by_type))

    # ------------------------------------------------------------------

    def _sources(self, ctx: AnalysisContext) -> Iterator[tuple[str, tuple | None, str]]:
        """Text to scan, each with the box it came from."""
        for block in ctx.data("ocr").get("blocks", []):
            text = block.get("text", "")
            if text:
                bbox = block.get("bbox")
                yield text, tuple(bbox) if bbox else None, "ocr"

        # Barcode payloads are text too, and often the richest source.
        for code in ctx.data("codes").get("codes", []):
            text = code.get("text", "")
            if text:
                bbox = code.get("bbox")
                yield text, tuple(bbox) if bbox else None, "barcode"

    def _scan(self, text: str) -> Iterator[dict[str, Any]]:
        for pattern in PATTERNS:
            for found in pattern.regex.finditer(text):
                value = found.group(0).strip()
                if pattern.validator and not pattern.validator(value):
                    continue
                entry: dict[str, Any] = {
                    "type": pattern.key,
                    "label": pattern.label,
                    "value": value,
                    "masked": mask(value, pattern.key),
                    "severity": pattern.severity.value,
                    "confidence": pattern.confidence,
                    "detail": pattern.detail,
                    "context": _context(text, found.start(), found.end()),
                }
                if pattern.key == "credit_card":
                    entry["brand"] = card_brand(value)
                yield entry

        yield from self._phones(text)

    def _phones(self, text: str) -> Iterator[dict[str, Any]]:
        """libphonenumber where available, keyword-anchored regex otherwise."""
        if has_module("phonenumbers"):
            import phonenumbers  # noqa: PLC0415

            for match in phonenumbers.PhoneNumberMatcher(text, None):
                if not phonenumbers.is_valid_number(match.number):
                    continue
                formatted = phonenumbers.format_number(
                    match.number, phonenumbers.PhoneNumberFormat.E164
                )
                region = phonenumbers.region_code_for_number(match.number) or "?"
                yield {
                    "type": "phone",
                    "label": "Phone number",
                    "value": formatted,
                    "masked": mask(formatted, "phone"),
                    "severity": Severity.MEDIUM.value,
                    "confidence": 0.9,
                    "detail": f"Valid number, region {region}",
                    "region": region,
                    "context": _context(text, match.start, match.start + len(match.raw_string)),
                }
            return

        for match in _PHONE_KEYWORD.finditer(text):
            value = match.group(1).strip()
            if sum(c.isdigit() for c in value) < 7:
                continue
            yield {
                "type": "phone",
                "label": "Phone number",
                "value": value,
                "masked": mask(value, "phone"),
                "severity": Severity.MEDIUM.value,
                "confidence": 0.6,
                "detail": "Matched by keyword; install imgintel[pii] to validate properly",
                "context": _context(text, match.start(), match.end()),
            }

    def _findings(self, matches: list[dict], by_type: dict[str, int]) -> list[Finding]:
        if not matches:
            return []

        out = [
            Finding(
                "sensitive",
                "Sensitive information visible",
                ", ".join(f"{k} x{v}" for k, v in sorted(by_type.items())),
                Severity.HIGH if _has_high(matches) else Severity.MEDIUM,
                confidence=0.85,
                detail="Values are masked here; the JSON output holds them in full",
            )
        ]
        for match in matches:
            out.append(
                Finding(
                    "sensitive",
                    match["label"],
                    match["masked"],
                    Severity(match["severity"]),
                    confidence=match["confidence"],
                    bbox=tuple(match["bbox"]) if match.get("bbox") else None,
                    detail=match["detail"],
                )
            )
        return out


def _has_high(matches: list[dict]) -> bool:
    return any(m["severity"] == "high" for m in matches)


def _context(text: str, start: int, end: int, window: int = 24) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{text[left:start]}[…]{text[end:right]}{suffix}"
