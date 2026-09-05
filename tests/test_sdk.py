"""The plugin SDK: public surface, declarative rules, scaffolding, validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from imgintel.core.engine import analyze_image
from imgintel.core.registry import Registry


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer}: {block.error}"
    return block.data


# -- the public SDK --------------------------------------------------------
#
# The point of `imgintel.plugins` is that these names keep working while
# `imgintel.core` is free to move. Tests that assert the surface exists are
# the only thing that makes that promise real.


def test_sdk_exports_everything_a_plugin_needs():
    import imgintel.plugins as sdk

    for name in (
        "Analyzer", "AnalyzerResult", "Availability", "Cost", "Finding", "Severity",
        "AnalysisContext", "CaseConfig", "API_VERSION", "require_api",
        "has_module", "binary_path",
    ):
        assert hasattr(sdk, name), f"{name} missing from the public SDK"
        assert name in sdk.__all__, f"{name} exists but is not in __all__"


def test_a_plugin_can_be_written_using_only_the_sdk():
    """No import from imgintel.core anywhere in a working analyzer."""
    from imgintel.plugins import Analyzer, Cost, Finding, Severity

    class Minimal(Analyzer):
        name = "sdk-only"
        version = "1.0.0"
        description = "Uses nothing but the public SDK"
        cost = Cost.CHEAP

        def analyze(self, ctx):
            return self.ok({"ok": True}, [Finding("test", "Ran", True, Severity.INFO)])

    analyzer = Minimal()
    assert analyzer.available().ok
    assert analyzer.ok().status == "ok"


def test_sdk_version_is_declared_and_parseable():
    from imgintel.plugins import API_VERSION

    major, minor = (int(p) for p in API_VERSION.split(".")[:2])
    assert major >= 1
    assert minor >= 0


def test_require_api_accepts_a_satisfied_requirement():
    from imgintel.plugins import API_VERSION, require_api

    require_api(API_VERSION)  # must not raise
    require_api("1.0")


def test_require_api_rejects_a_newer_requirement():
    """A plugin needing a later API must fail at import, not mid-run."""
    from imgintel.plugins import ApiVersionError, require_api

    with pytest.raises(ApiVersionError, match="99"):
        require_api("1.99")


def test_require_api_rejects_a_different_major_version():
    from imgintel.plugins import ApiVersionError, require_api

    with pytest.raises(ApiVersionError):
        require_api("0.9")


# -- declarative rules -----------------------------------------------------


def write_rules(path: Path, rules: list[dict], name: str = "test-rules") -> Path:
    """Rule files are written as JSON — a valid YAML subset needing no PyYAML."""
    path.write_text(json.dumps({"name": name, "rules": rules}), encoding="utf-8")
    return path


def test_text_rule_matches_ocr_output(images, tmp_path):
    from imgintel.plugins.declarative import RuleSetAnalyzer, load_ruleset

    path = write_rules(
        tmp_path / "r.json",
        [
            {
                "id": "acme",
                "when": {"text_matches": r"(?i)\bACME\b"},
                "finding": {"category": "custom", "label": "ACME mentioned", "severity": "high",
                            "detail": "Named client"},
            }
        ],
    )
    registry = Registry()
    registry.load_builtin()
    registry.register(RuleSetAnalyzer(load_ruleset(path)), "test")

    doc = analyze_image(images["document"], registry=registry, profile="deep")
    data = data_of(doc, "rules:test-rules")
    assert data["rules_matched"] == ["acme"]

    finding = next(f for f in doc.findings if f.label == "ACME mentioned")
    assert finding.severity == "high"
    assert finding.value  # the matched text, not the rule name


def test_threshold_rule_reads_any_analyzer_path(images, tmp_path):
    from imgintel.plugins.declarative import RuleSetAnalyzer, load_ruleset

    path = write_rules(
        tmp_path / "r.json",
        [
            {
                "id": "small",
                "when": {"value_below": {"path": "fileinfo.megapixels", "threshold": 99}},
                "finding": {"label": "Small image", "severity": "low"},
            }
        ],
    )
    registry = Registry()
    registry.load_builtin()
    registry.register(RuleSetAnalyzer(load_ruleset(path)), "test")

    doc = analyze_image(images["sharp"], registry=registry, profile="quick")
    assert data_of(doc, "rules:test-rules")["rules_matched"] == ["small"]


def test_metadata_rule_detects_a_missing_field(images, tmp_path):
    from imgintel.plugins.declarative import RuleSetAnalyzer, load_ruleset

    path = write_rules(
        tmp_path / "r.json",
        [
            {
                "id": "no-make",
                "when": {"exif_missing": "make"},
                "finding": {"label": "No camera make", "severity": "low"},
            }
        ],
    )
    registry = Registry()
    registry.load_builtin()
    registry.register(RuleSetAnalyzer(load_ruleset(path)), "test")

    doc = analyze_image(images["sharp"], registry=registry, profile="quick")
    assert data_of(doc, "rules:test-rules")["rules_matched"] == ["no-make"]


def test_rule_that_does_not_match_produces_no_finding(images, tmp_path):
    from imgintel.plugins.declarative import RuleSetAnalyzer, load_ruleset

    path = write_rules(
        tmp_path / "r.json",
        [
            {
                "id": "absent",
                "when": {"text_contains": "definitely-not-in-this-image-xyzzy"},
                "finding": {"label": "Should not fire"},
            }
        ],
    )
    registry = Registry()
    registry.load_builtin()
    registry.register(RuleSetAnalyzer(load_ruleset(path)), "test")

    doc = analyze_image(images["document"], registry=registry, profile="deep")
    assert data_of(doc, "rules:test-rules")["rules_matched"] == []
    assert "Should not fire" not in {f.label for f in doc.findings}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"rules": []}, "non-empty list"),
        ({"rules": [{"when": {}, "finding": {"label": "x"}}]}, "exactly one condition"),
        ({"rules": [{"when": {"nope": 1}, "finding": {"label": "x"}}]}, "unknown condition"),
        ({"rules": [{"when": {"text_contains": "a"}}]}, "needs at least a 'label'"),
        (
            {"rules": [{"when": {"text_contains": "a"},
                        "finding": {"label": "x", "severity": "urgent"}}]},
            "severity must be one of",
        ),
        (
            {"rules": [{"when": {"text_contains": "a"},
                        "finding": {"label": "x", "confidence": 5}}]},
            "between 0 and 1",
        ),
    ],
)
def test_malformed_rules_are_rejected_with_a_useful_message(tmp_path, payload, expected):
    """A rule file is data; a mistake in it must name the problem, not traceback."""
    from imgintel.plugins.declarative import RuleError, load_ruleset

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "bad", **payload}), encoding="utf-8")

    with pytest.raises(RuleError, match=expected):
        load_ruleset(path)


def test_duplicate_rule_ids_are_rejected(tmp_path):
    from imgintel.plugins.declarative import RuleError, load_ruleset

    path = write_rules(
        tmp_path / "r.json",
        [
            {"id": "same", "when": {"text_contains": "a"}, "finding": {"label": "A"}},
            {"id": "same", "when": {"text_contains": "b"}, "finding": {"label": "B"}},
        ],
    )
    with pytest.raises(RuleError, match="duplicate rule id"):
        load_ruleset(path)


def test_one_broken_rule_does_not_lose_the_others(images, tmp_path):
    """An invalid regex is caught at load; a runtime error must be isolated."""
    from imgintel.plugins.declarative import Rule, RuleSet, RuleSetAnalyzer
    from imgintel.plugins.declarative import Severity as RuleSeverity

    good = Rule(
        id="good", condition="exif_missing", argument="make",
        category="custom", label="Good rule fired", severity=RuleSeverity.LOW,
    )
    broken = Rule(
        id="broken", condition="value_above", argument="not-a-mapping",
        category="custom", label="Broken", severity=RuleSeverity.LOW,
    )
    registry = Registry()
    registry.load_builtin()
    registry.register(RuleSetAnalyzer(RuleSet("mixed", "test", [broken, good])), "test")

    doc = analyze_image(images["sharp"], registry=registry, profile="quick")
    data = data_of(doc, "rules:mixed")

    assert "good" in data["rules_matched"], "a broken rule must not stop the others"
    assert data["rule_errors"] and data["rule_errors"][0]["rule"] == "broken"


def test_rules_load_without_the_python_plugin_opt_in(images, tmp_path, monkeypatch):
    """Rule files are parsed, never executed, so they need no consent flag."""
    from imgintel.core import registry as registry_module

    write_rules(
        tmp_path / "auto.json",
        [{"id": "x", "when": {"exif_missing": "make"}, "finding": {"label": "No make"}}],
        name="auto",
    )
    monkeypatch.setattr(registry_module, "user_rules_dir", lambda: tmp_path)

    reg = Registry.discover(allow_local=False)
    assert "rules:auto" in reg, "rule files should load without --allow-local-plugins"


def test_rule_conditions_are_all_implemented():
    from imgintel.plugins.declarative import _EVALUATORS, CONDITIONS

    assert set(CONDITIONS) == set(_EVALUATORS), "documented conditions must all exist"


# -- scaffolding -----------------------------------------------------------


def test_scaffolded_python_plugin_is_complete(tmp_path):
    from imgintel.plugins.scaffold import scaffold_python

    written = scaffold_python(tmp_path, "demo")
    names = {p.name for p in written}
    assert {"analyzer.py", "pyproject.toml", "README.md"} <= names

    package = tmp_path / "imgintel-demo"
    pyproject = (package / "pyproject.toml").read_text()
    assert 'imgintel.analyzers' in pyproject, "must declare the entry point"
    assert "demo = " in pyproject


def test_scaffolded_analyzer_actually_runs(tmp_path, images):
    """Scaffolding that produces something broken is worse than none."""
    from imgintel.plugins.scaffold import scaffold_python

    scaffold_python(tmp_path, "demo")
    source = tmp_path / "imgintel-demo" / "src" / "demo"
    sys.path.insert(0, str(source.parent))
    try:
        import importlib

        module = importlib.import_module("demo.analyzer")
        analyzer = module.DemoAnalyzer()

        registry = Registry()
        registry.load_builtin()
        registry.register(analyzer, "test")

        doc = analyze_image(images["sharp"], registry=registry, only=["demo"])
        assert doc.analyzers["demo"].status == "ok"
    finally:
        sys.path.remove(str(source.parent))
        sys.modules.pop("demo.analyzer", None)
        sys.modules.pop("demo", None)


def test_scaffolded_rules_file_parses(tmp_path):
    from imgintel.plugins.declarative import load_ruleset
    from imgintel.plugins.scaffold import scaffold_rules

    written = scaffold_rules(tmp_path, "house")
    pytest.importorskip("yaml")
    ruleset = load_ruleset(written[0])
    assert ruleset.name == "house"
    assert len(ruleset.rules) >= 3


def test_scaffold_rejects_unsafe_names(tmp_path):
    from imgintel.plugins.scaffold import ScaffoldError, scaffold_python

    for name in ("../escape", "Has Spaces", "9leading", ""):
        with pytest.raises(ScaffoldError):
            scaffold_python(tmp_path, name)


def test_scaffold_refuses_to_overwrite(tmp_path):
    from imgintel.plugins.scaffold import ScaffoldError, scaffold_python

    scaffold_python(tmp_path, "demo")
    with pytest.raises(ScaffoldError, match="already exists"):
        scaffold_python(tmp_path, "demo")


# -- validation ------------------------------------------------------------


def test_every_builtin_analyzer_conforms(registry):
    """The contract is only real if the built-ins obey it."""
    from imgintel.plugins.validate import validate

    failures = {}
    for analyzer in registry.all():
        report = validate(analyzer)
        if not report.ok:
            failures[analyzer.name] = [i.message for i in report.errors]
    assert not failures, failures


def test_validator_leaves_the_context_class_intact(images):
    """Regression: the behaviour check patches AnalysisContext and must undo it.

    Restoring by rebuilding a cached_property skips ``__set_name__``, which
    leaves the descriptor nameless and makes every later decode raise. That
    corrupted the class for the rest of the process, so unrelated tests failed
    only when run after this one.
    """
    from imgintel.core.context import AnalysisContext
    from imgintel.plugins import Analyzer, Cost
    from imgintel.plugins.validate import validate

    before = AnalysisContext.__dict__["pil_raw"]

    class Probe(Analyzer):
        name = "probe-restore"
        version = "1.0.0"
        description = "touches pixels"
        cost = Cost.MEDIUM

        def analyze(self, ctx):
            return self.ok({"mean": round(float(ctx.gray.mean()), 3)})

    validate(Probe(), images["sharp"])

    after = AnalysisContext.__dict__["pil_raw"]
    assert after is before, "the original descriptor must be restored, not rebuilt"
    assert getattr(after, "attrname", None) == "pil_raw"

    # And decoding still works afterwards.
    assert AnalysisContext(images["sharp"]).pil_raw is not None


def test_validator_catches_a_cheap_analyzer_that_decodes_pixels(images):
    """The mistake that silently breaks the quick profile's whole promise."""
    from imgintel.plugins import Analyzer, Cost
    from imgintel.plugins.validate import validate

    class Liar(Analyzer):
        name = "liar"
        version = "1.0.0"
        description = "Claims to be cheap but decodes pixels"
        cost = Cost.CHEAP

        def analyze(self, ctx):
            return self.ok({"mean": float(ctx.gray.mean())})

    report = validate(Liar(), images["sharp"])
    assert not report.ok
    assert any("CHEAP" in i.message for i in report.errors)


def test_validator_catches_an_available_that_raises():
    from imgintel.plugins import Analyzer
    from imgintel.plugins.validate import validate

    class Exploder(Analyzer):
        name = "exploder"
        version = "1.0.0"
        description = "Raises from available()"

        def available(self):
            raise RuntimeError("boom")

        def analyze(self, ctx):
            return self.ok()

    report = validate(Exploder())
    assert not report.ok
    assert any("available() raised" in i.message for i in report.errors)


def test_validator_catches_non_determinism(images):
    from imgintel.plugins import Analyzer, Cost
    from imgintel.plugins.validate import validate

    class Random(Analyzer):
        name = "randomish"
        version = "1.0.0"
        description = "Different every run"
        cost = Cost.MEDIUM

        def analyze(self, ctx):
            import time

            _ = ctx.gray  # touch pixels so the cost check is satisfied
            return self.ok({"now": time.perf_counter_ns()})

    report = validate(Random(), images["sharp"])
    assert not report.ok
    assert any("deterministic" in i.message for i in report.errors)


def test_validator_warns_about_missing_identity():
    from imgintel.plugins import Analyzer
    from imgintel.plugins.validate import validate

    class Nameless(Analyzer):
        name = "nameless"
        version = ""
        description = ""

        def analyze(self, ctx):
            return self.ok()

    report = validate(Nameless())
    assert not report.ok
    assert any("version" in i.message for i in report.errors)


def test_validator_checks_rule_files(tmp_path):
    from imgintel.plugins.validate import validate_ruleset

    good = write_rules(
        tmp_path / "good.json",
        [{"id": "a", "when": {"text_contains": "x"}, "finding": {"label": "X"}}],
    )
    assert validate_ruleset(good).ok

    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "bad", "rules": "not a list"}', encoding="utf-8")
    assert not validate_ruleset(bad).ok


# -- entry points ----------------------------------------------------------


def test_entry_point_group_is_stable():
    """Renaming this breaks every installed plugin."""
    from imgintel.core.registry import ENTRY_POINT_GROUP

    assert ENTRY_POINT_GROUP == "imgintel.analyzers"


def test_registry_reports_where_each_analyzer_came_from(registry):
    for analyzer in registry.all():
        origin = registry.origin(analyzer.name)
        assert origin and origin != "unknown"


@pytest.mark.slow
def test_a_real_package_is_discovered_after_install(tmp_path, images):
    """End to end: build a plugin package, pip install it, see it appear.

    Skipped unless pip can install into this environment; the mechanism is
    otherwise covered by the local-plugin and manual-registration tests.
    """
    package = tmp_path / "imgintel-probe"
    (package / "src" / "probe").mkdir(parents=True)
    (package / "src" / "probe" / "__init__.py").write_text("")
    (package / "src" / "probe" / "analyzer.py").write_text(
        "from imgintel.plugins import Analyzer, Cost\n\n\n"
        "class ProbeAnalyzer(Analyzer):\n"
        "    name = 'probe'\n"
        "    version = '1.0.0'\n"
        "    description = 'probe'\n"
        "    cost = Cost.CHEAP\n\n"
        "    def analyze(self, ctx):\n"
        "        return self.ok({'probed': True})\n"
    )
    (package / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        '[project]\nname = "imgintel-probe"\nversion = "0.1.0"\n\n'
        '[project.entry-points."imgintel.analyzers"]\n'
        'probe = "probe.analyzer:ProbeAnalyzer"\n\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/probe"]\n'
    )

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q", str(package)],
        capture_output=True, text=True, timeout=180, check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"pip install unavailable here: {result.stderr[:200]}")

    try:
        reg = Registry()
        reg.load_entry_points()
        assert "probe" in reg
        assert reg.origin("probe").startswith("package:")
    finally:
        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "imgintel-probe"],
            capture_output=True, timeout=120, check=False,
        )
