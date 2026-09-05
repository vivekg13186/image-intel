"""Conformance checks for analyzers.

Catches the mistakes that are easy to make and hard to notice: a `CHEAP`
analyzer that decodes pixels (which quietly breaks the `quick` profile's whole
promise), an `available()` that raises instead of returning, findings with no
explanation, non-determinism.

The pixel-decode check is the interesting one. Rather than reading the source
and guessing, it runs the analyzer against a real image with the decode path
instrumented, so it reports what the analyzer *did* rather than what it looks
like it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgintel.core.analyzer import Analyzer, Cost

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(slots=True)
class Issue:
    level: str  # "error" | "warning"
    message: str


@dataclass(slots=True)
class Report:
    analyzer: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "warning"]


def validate(analyzer: Analyzer, sample: Path | None = None) -> Report:
    """Check one analyzer against the contract."""
    report = Report(analyzer=analyzer.name or type(analyzer).__name__)
    _check_identity(analyzer, report)
    _check_availability(analyzer, report)
    if sample is not None:
        _check_behaviour(analyzer, sample, report)
    return report


def _check_identity(analyzer: Analyzer, report: Report) -> None:
    if not analyzer.name:
        report.issues.append(Issue("error", "name is empty"))
    elif not analyzer.name.replace("_", "").replace(":", "").replace("-", "").isalnum():
        report.issues.append(
            Issue("warning", f"name {analyzer.name!r} should be lowercase and alphanumeric")
        )
    if not analyzer.version:
        report.issues.append(Issue("error", "version is empty"))
    if not analyzer.description:
        report.issues.append(
            Issue("warning", "description is empty; `imgintel plugins` will show nothing")
        )
    if not isinstance(analyzer.cost, Cost):
        report.issues.append(Issue("error", f"cost must be a Cost, got {analyzer.cost!r}"))
    for name, value in (("requires", analyzer.requires), ("after", analyzer.after)):
        if not isinstance(value, tuple):
            report.issues.append(
                Issue("error", f"{name} must be a tuple, got {type(value).__name__}")
            )
    overlap = set(analyzer.requires) & set(analyzer.after)
    if overlap:
        report.issues.append(
            Issue(
                "warning",
                f"{', '.join(sorted(overlap))} appears in both requires and after; "
                "requires already orders it",
            )
        )


def _check_availability(analyzer: Analyzer, report: Report) -> None:
    try:
        availability = analyzer.available()
    except Exception as exc:  # noqa: BLE001 - that is the failure being tested
        report.issues.append(
            Issue("error", f"available() raised {type(exc).__name__}: {exc}; it must return")
        )
        return
    if not hasattr(availability, "ok"):
        report.issues.append(Issue("error", "available() must return an Availability"))
        return
    if not availability.ok and not availability.reason:
        report.issues.append(Issue("warning", "unavailable but gives no reason"))
    if not availability.ok and not availability.hint:
        report.issues.append(
            Issue("warning", "unavailable but gives no hint; the user cannot act on it")
        )


def _check_behaviour(analyzer: Analyzer, sample: Path, report: Report) -> None:
    """Run it twice against a real image and watch what it touches."""
    from imgintel.core import context as ctx_module
    from imgintel.core.context import AnalysisContext

    if not analyzer.available().ok:
        report.issues.append(
            Issue("warning", "not available here, so behaviour was not checked")
        )
        return

    decoded: list[str] = []
    # Keep the *original descriptor object*, not just its function.
    # Rebuilding a cached_property and assigning it to the class skips
    # __set_name__, which leaves it without an attribute name and makes every
    # later access raise "Cannot use cached_property instance without calling
    # __set_name__". Restoring the exact object avoids the whole problem.
    original_descriptor = AnalysisContext.__dict__["pil_raw"]
    original_func = original_descriptor.func

    def spy(self):
        decoded.append("pil_raw")
        return original_func(self)

    AnalysisContext.pil_raw = property(spy)  # type: ignore[assignment]
    try:
        first = _run_once(analyzer, sample, ctx_module)
        touched_pixels = bool(decoded)
        decoded.clear()
        second = _run_once(analyzer, sample, ctx_module)
    finally:
        AnalysisContext.pil_raw = original_descriptor  # type: ignore[assignment]

    if first is None or second is None:
        report.issues.append(Issue("error", "analyze() raised; see the run output"))
        return
    first_block, first_findings = first
    second_block, _ = second

    # An analyzer that short-circuited never reached its own work, so its
    # behaviour here says nothing about what it costs.
    if first_block.status != "ok":
        report.issues.append(
            Issue(
                "warning",
                f"{first_block.status} on the sample ({first_block.error}); "
                "cost and determinism were not checked",
            )
        )
        return

    # The CHEAP contract is the one that breaks a whole profile if violated.
    if analyzer.cost is Cost.CHEAP and touched_pixels:
        report.issues.append(
            Issue(
                "error",
                "declared CHEAP but decoded pixels. CHEAP puts the analyzer in the "
                "`quick` profile, which promises header-only work; use Cost.MEDIUM",
            )
        )
    if analyzer.cost is not Cost.CHEAP and not touched_pixels:
        report.issues.append(
            Issue("warning", f"declared {analyzer.cost.value} but never decoded pixels; "
                             "Cost.CHEAP would put it in the fast profile")
        )

    if first_block.status == "ok" and first_block.data != second_block.data:
        report.issues.append(
            Issue("error", "not deterministic: two runs on the same image differed")
        )

    for finding in first_findings:
        if not finding.label:
            report.issues.append(Issue("error", "a finding has no label"))
        if not 0.0 <= finding.confidence <= 1.0:
            report.issues.append(
                Issue("error", f"finding {finding.label!r} has confidence "
                               f"{finding.confidence}, outside 0..1")
            )
        # These come back as schema models, where severity is a plain string.
        if _SEVERITY_RANK.get(str(finding.severity), 0) >= 2 and not finding.detail:
            report.issues.append(
                Issue(
                    "warning",
                    f"{finding.label!r} is {finding.severity} severity but has no "
                    "detail; a reader cannot judge it or see what would explain it innocently",
                )
            )


def _run_once(analyzer: Analyzer, sample: Path, ctx_module: Any):
    """Run one analyzer and return (its block, its findings).

    Findings live on the document rather than the block — the block is the
    serialized schema model — so both have to come back together.
    """
    from imgintel.core.engine import analyze_image
    from imgintel.core.registry import Registry

    registry = Registry()
    registry.load_builtin()
    if analyzer.name not in registry:
        registry.register(analyzer, "validate")

    try:
        doc = analyze_image(sample, registry=registry, only=[analyzer.name])
    except Exception:  # noqa: BLE001
        return None
    block = doc.analyzers.get(analyzer.name)
    if block is None:
        return None
    return block, [f for f in doc.findings if f.analyzer == analyzer.name]


def validate_ruleset(path: Path) -> Report:
    """Check a declarative rule file parses and is well-formed."""
    from imgintel.plugins.declarative import RuleError, load_ruleset

    report = Report(analyzer=str(path))
    try:
        ruleset = load_ruleset(path)
    except (RuleError, OSError) as exc:
        report.issues.append(Issue("error", str(exc)))
        return report

    report.analyzer = f"rules:{ruleset.name}"
    for rule in ruleset.rules:
        if rule.severity.rank >= 2 and not rule.detail:
            report.issues.append(
                Issue(
                    "warning",
                    f"rule {rule.id!r} is {rule.severity.value} severity with no detail",
                )
            )
        if rule.condition == "text_matches":
            import re

            try:
                re.compile(str(rule.argument))
            except re.error as exc:
                report.issues.append(
                    Issue("error", f"rule {rule.id!r}: invalid regex: {exc}")
                )
    return report


__all__ = ["Issue", "Report", "validate", "validate_ruleset"]
