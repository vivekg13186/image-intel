"""Analyzers written as YAML rules instead of Python.

Most custom checks people actually want are one of three shapes: match a
pattern against extracted text, assert something about a metadata field, or
compare a number against a threshold. Writing a Python class for each is more
ceremony than the check deserves, and it puts the capability out of reach of
anyone who does not write Python — which in an investigative team is most of
the people who know what should be checked.

A rule file is a list of rules; each becomes one finding when it matches.

    name: house-rules
    description: Checks specific to our casework
    rules:
      - id: internal-doc
        when:
          text_matches: '(?i)\\bCOMPANY CONFIDENTIAL\\b'
        finding:
          category: sensitive
          label: Internal document marking
          severity: high
          detail: Carries our internal classification marking

      - id: drone-capture
        when:
          exif_contains: {field: model, value: FC3411}
        finding:
          category: device
          label: DJI drone capture
          severity: medium

      - id: tiny-image
        when:
          value_below: {path: "fileinfo.megapixels", threshold: 0.2}
        finding:
          category: quality
          label: Image too small for reliable analysis
          severity: low

Rules are data, not code: a malformed file produces an error naming the rule,
never a traceback from inside a run, and nothing in a rule file can execute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext

#: Conditions a rule may use. Deliberately small: each is unambiguous, and
#: anything needing more than these is better written as Python.
CONDITIONS = (
    "text_matches",      # regex against all extracted text
    "text_contains",     # plain substring, case-insensitive
    "exif_contains",     # {field, value} — substring of an EXIF highlight
    "exif_missing",      # field name absent or empty
    "value_above",       # {path, threshold}
    "value_below",       # {path, threshold}
    "value_equals",      # {path, value}
    "finding_present",   # another analyzer raised this label
)


class RuleError(ValueError):
    """A rule file is malformed. Names the rule so it can be fixed."""


@dataclass(slots=True)
class Rule:
    id: str
    condition: str
    argument: Any
    category: str
    label: str
    severity: Severity = Severity.INFO
    confidence: float = 0.8
    detail: str | None = None
    value_template: str | None = None

    def evaluate(self, ctx: AnalysisContext, text: str) -> Finding | None:
        matched, evidence = _EVALUATORS[self.condition](ctx, text, self.argument)
        if not matched:
            return None
        return Finding(
            category=self.category,
            label=self.label,
            value=evidence,
            severity=self.severity,
            confidence=self.confidence,
            detail=self.detail,
        )


@dataclass(slots=True)
class RuleSet:
    name: str
    description: str
    rules: list[Rule] = field(default_factory=list)
    source: str | None = None


# -- condition evaluators --------------------------------------------------
#
# Each returns (matched, evidence). Evidence is what the finding shows, so it
# has to be the thing that actually matched rather than the rule that matched
# it — otherwise a report says "a rule fired" instead of what it found.


def _text_matches(ctx: AnalysisContext, text: str, argument: Any):
    try:
        pattern = re.compile(str(argument))
    except re.error as exc:
        raise RuleError(f"invalid regular expression {argument!r}: {exc}") from exc
    match = pattern.search(text)
    return (True, match.group(0)[:120]) if match else (False, None)


def _text_contains(ctx: AnalysisContext, text: str, argument: Any):
    needle = str(argument)
    index = text.lower().find(needle.lower())
    if index < 0:
        return False, None
    return True, text[index : index + len(needle)]


def _exif_contains(ctx: AnalysisContext, text: str, argument: Any):
    field_name, wanted = _pair(argument, "field", "value")
    actual = _exif_value(ctx, field_name)
    if actual is None:
        return False, None
    return (str(wanted).lower() in str(actual).lower()), str(actual)[:120]


def _exif_missing(ctx: AnalysisContext, text: str, argument: Any):
    value = _exif_value(ctx, str(argument))
    return (value in (None, "")), str(argument)


def _value_above(ctx: AnalysisContext, text: str, argument: Any):
    path, threshold = _pair(argument, "path", "threshold")
    value = _lookup(ctx, str(path))
    if not isinstance(value, (int, float)):
        return False, None
    return (value > float(threshold)), value


def _value_below(ctx: AnalysisContext, text: str, argument: Any):
    path, threshold = _pair(argument, "path", "threshold")
    value = _lookup(ctx, str(path))
    if not isinstance(value, (int, float)):
        return False, None
    return (value < float(threshold)), value


def _value_equals(ctx: AnalysisContext, text: str, argument: Any):
    path, wanted = _pair(argument, "path", "value")
    value = _lookup(ctx, str(path))
    if value is None:
        return False, None
    return (str(value).lower() == str(wanted).lower()), value


def _finding_present(ctx: AnalysisContext, text: str, argument: Any):
    wanted = str(argument).lower()
    for result in ctx.results.values():
        for finding in result.findings:
            if wanted in finding.label.lower():
                return True, finding.label
    return False, None


_EVALUATORS = {
    "text_matches": _text_matches,
    "text_contains": _text_contains,
    "exif_contains": _exif_contains,
    "exif_missing": _exif_missing,
    "value_above": _value_above,
    "value_below": _value_below,
    "value_equals": _value_equals,
    "finding_present": _finding_present,
}


def _pair(argument: Any, first: str, second: str) -> tuple[Any, Any]:
    if not isinstance(argument, dict) or first not in argument or second not in argument:
        raise RuleError(f"expected a mapping with '{first}' and '{second}', got {argument!r}")
    return argument[first], argument[second]


def _exif_value(ctx: AnalysisContext, field_name: str) -> Any:
    """Look in the friendly highlights first, then the raw tags."""
    exif = ctx.data("exif")
    highlights = exif.get("highlights", {})
    if field_name in highlights:
        return highlights[field_name]
    lowered = field_name.lower()
    for key, value in exif.get("tags", {}).items():
        if key.split(":", 1)[-1].lower() == lowered:
            return value
    return None


def _lookup(ctx: AnalysisContext, path: str) -> Any:
    """Read ``analyzer.key.subkey`` out of another analyzer's data."""
    parts = path.split(".")
    if len(parts) < 2:
        raise RuleError(f"value path must be 'analyzer.key', got {path!r}")
    node: Any = ctx.data(parts[0])
    for key in parts[1:]:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


# -- loading ---------------------------------------------------------------


def load_ruleset(path: Path) -> RuleSet:
    """Parse a YAML (or JSON) rule file into a validated RuleSet."""
    text = Path(path).read_text(encoding="utf-8")
    payload = _parse(text, path)

    if not isinstance(payload, dict):
        raise RuleError(f"{path}: expected a mapping at the top level")
    name = str(payload.get("name") or Path(path).stem)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise RuleError(
            f"{path}: name {name!r} must be lowercase letters, digits, '-' or '_'"
        )

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise RuleError(f"{path}: 'rules' must be a non-empty list")

    rules = [_parse_rule(entry, index, path) for index, entry in enumerate(raw_rules)]
    duplicates = {r.id for r in rules if [x.id for x in rules].count(r.id) > 1}
    if duplicates:
        raise RuleError(f"{path}: duplicate rule id(s): {', '.join(sorted(duplicates))}")

    return RuleSet(
        name=name,
        description=str(payload.get("description") or f"Rules from {Path(path).name}"),
        rules=rules,
        source=str(path),
    )


def _parse(text: str, path: Path) -> Any:
    """YAML if PyYAML is installed, otherwise JSON.

    Rule files are usually YAML, but JSON is a valid subset and needs no
    dependency — so a rule file still works on a minimal install.
    """
    try:
        import yaml  # noqa: PLC0415

        return yaml.safe_load(text)
    except ImportError:
        import json  # noqa: PLC0415

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuleError(
                f"{path}: PyYAML is not installed, so rule files must be JSON. "
                f"Install it with: pip install 'imgintel[rules]' ({exc})"
            ) from exc
    except Exception as exc:  # noqa: BLE001 - yaml raises its own error types
        raise RuleError(f"{path}: could not parse: {exc}") from exc


def _parse_rule(entry: Any, index: int, path: Path) -> Rule:
    where = f"{path}: rule {index + 1}"
    if not isinstance(entry, dict):
        raise RuleError(f"{where}: expected a mapping")

    rule_id = str(entry.get("id") or f"rule-{index + 1}")
    when = entry.get("when")
    if not isinstance(when, dict) or len(when) != 1:
        raise RuleError(
            f"{where} ({rule_id}): 'when' must contain exactly one condition, "
            f"one of: {', '.join(CONDITIONS)}"
        )
    condition, argument = next(iter(when.items()))
    if condition not in _EVALUATORS:
        raise RuleError(
            f"{where} ({rule_id}): unknown condition {condition!r}. "
            f"Available: {', '.join(CONDITIONS)}"
        )

    finding = entry.get("finding")
    if not isinstance(finding, dict) or not finding.get("label"):
        raise RuleError(f"{where} ({rule_id}): 'finding' needs at least a 'label'")

    severity_name = str(finding.get("severity", "info")).lower()
    try:
        severity = Severity(severity_name)
    except ValueError as exc:
        raise RuleError(
            f"{where} ({rule_id}): severity must be one of "
            f"{', '.join(s.value for s in Severity)}, got {severity_name!r}"
        ) from exc

    confidence = finding.get("confidence", 0.8)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise RuleError(f"{where} ({rule_id}): confidence must be a number") from exc
    if not 0.0 <= confidence <= 1.0:
        raise RuleError(f"{where} ({rule_id}): confidence must be between 0 and 1")

    return Rule(
        id=rule_id,
        condition=condition,
        argument=argument,
        category=str(finding.get("category", "custom")),
        label=str(finding["label"]),
        severity=severity,
        confidence=confidence,
        detail=finding.get("detail"),
    )


# -- the analyzer ----------------------------------------------------------


class RuleSetAnalyzer(Analyzer):
    """One analyzer per rule file, built at load time."""

    cost = Cost.CHEAP
    # Soft edges: rules read whatever ran, and work on less when less ran.
    after = ("exif", "fileinfo", "ocr", "codes", "quality", "gps")

    def __init__(self, ruleset: RuleSet) -> None:
        self.ruleset = ruleset
        self.name = f"rules:{ruleset.name}"
        self.version = "1.0.0"
        self.title = ruleset.name
        self.description = ruleset.description

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        text = _all_text(ctx)
        findings: list[Finding] = []
        matched: list[str] = []
        errors: list[dict[str, str]] = []

        for rule in self.ruleset.rules:
            try:
                finding = rule.evaluate(ctx, text)
            except RuleError as exc:
                # One broken rule must not lose the others' results.
                errors.append({"rule": rule.id, "error": str(exc)})
                continue
            if finding is not None:
                findings.append(finding)
                matched.append(rule.id)

        return self.ok(
            {
                "ruleset": self.ruleset.name,
                "source": self.ruleset.source,
                "rules_evaluated": len(self.ruleset.rules),
                "rules_matched": matched,
                "rule_errors": errors,
                "text_characters": len(text),
            },
            findings,
        )


def _all_text(ctx: AnalysisContext) -> str:
    """Everything a text rule should be able to see."""
    parts: list[str] = []
    ocr = ctx.data("ocr").get("text")
    if ocr:
        parts.append(ocr)
    for code in ctx.data("codes").get("codes", []):
        if code.get("text"):
            parts.append(code["text"])
    highlights = ctx.data("exif").get("highlights", {})
    parts.extend(str(v) for v in highlights.values() if v)
    return "\n".join(parts)


def load_rule_analyzers(directory: Path) -> tuple[list[RuleSetAnalyzer], list[str]]:
    """Every rule file in a directory. Returns (analyzers, errors)."""
    directory = Path(directory)
    if not directory.is_dir():
        return [], []

    analyzers: list[RuleSetAnalyzer] = []
    errors: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".yaml", ".yml", ".json"):
            continue
        try:
            analyzers.append(RuleSetAnalyzer(load_ruleset(path)))
        except (RuleError, OSError) as exc:
            errors.append(str(exc))
    return analyzers, errors


__all__ = [
    "CONDITIONS",
    "Rule",
    "RuleError",
    "RuleSet",
    "RuleSetAnalyzer",
    "load_rule_analyzers",
    "load_ruleset",
]
