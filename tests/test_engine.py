"""Phase 0: the engine itself — contract, ordering, isolation, schema."""

from __future__ import annotations

import json

import pytest

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext
from imgintel.core.engine import SelectionError, analyze_image, plan
from imgintel.core.pipeline import CycleError, Pipeline, topo_sort
from imgintel.core.profiles import PROFILES
from imgintel.core.registry import Registry
from imgintel.core.schema import SCHEMA_VERSION, FindingsDocument

# -- fakes -----------------------------------------------------------------


class _Base(Analyzer):
    def analyze(self, ctx):  # noqa: D102
        return self.ok({"ran": True})


def _make(name, *, requires=(), cost=Cost.CHEAP, **kw):
    return type(f"A_{name}", (_Base,), {"name": name, "requires": requires, "cost": cost, **kw})()


class Exploder(_Base):
    name = "exploder"

    def analyze(self, ctx):
        raise RuntimeError("boom")


class Unavailable(_Base):
    name = "unavail"

    def available(self):
        return Availability.no("missing widget", "pip install widget")


class Networked(_Base):
    name = "networked"
    needs_network = True


# -- registry --------------------------------------------------------------


def test_builtin_analyzers_are_discovered(registry):
    for expected in ("fileinfo", "exif", "gps", "hashes", "perceptual", "color", "quality", "blur"):
        assert expected in registry, f"{expected} not registered"
    assert not registry.errors


def test_duplicate_registration_is_rejected():
    reg = Registry()
    reg.register(_make("dup"))
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_make("dup"))


def test_resolve_pulls_in_dependencies():
    reg = Registry()
    reg.register(_make("base"))
    reg.register(_make("leaf", requires=("base",)))
    chosen, unknown = reg.resolve(["leaf"])
    assert {a.name for a in chosen} == {"base", "leaf"}
    assert unknown == []


def test_resolve_reports_unknown_names(registry):
    _, unknown = registry.resolve(["nope"])
    assert unknown == ["nope"]


# -- ordering --------------------------------------------------------------


def test_dependencies_run_before_dependents():
    order = [a.name for a in topo_sort([_make("z", requires=("a",)), _make("a")])]
    assert order.index("a") < order.index("z")


def test_cheap_analyzers_run_before_expensive_ones():
    order = [a.name for a in topo_sort([_make("m", cost=Cost.MEDIUM), _make("c", cost=Cost.CHEAP)])]
    assert order == ["c", "m"]


def test_cycles_are_detected():
    with pytest.raises(CycleError):
        topo_sort([_make("a", requires=("b",)), _make("b", requires=("a",))])


def test_ordering_is_deterministic():
    analyzers = [_make(n) for n in "fedcba"]
    assert [a.name for a in topo_sort(analyzers)] == [a.name for a in topo_sort(analyzers)]


# -- isolation -------------------------------------------------------------


def test_one_failure_does_not_stop_the_others(images):
    ctx = AnalysisContext(images["sharp"])
    results = {r.analyzer: r for r in Pipeline([Exploder(), _make("fine")]).run(ctx)}
    assert results["exploder"].status == "error"
    assert "boom" in results["exploder"].error
    assert results["fine"].status == "ok"


def test_unavailable_analyzer_reports_its_hint(images):
    ctx = AnalysisContext(images["sharp"])
    (result,) = Pipeline([Unavailable()]).run(ctx)
    assert result.status == "unavailable"
    assert "missing widget" in result.error
    assert "pip install widget" in result.error


def test_dependent_is_skipped_when_dependency_fails(images):
    ctx = AnalysisContext(images["sharp"])
    results = {
        r.analyzer: r
        for r in Pipeline([Exploder(), _make("child", requires=("exploder",))]).run(ctx)
    }
    assert results["child"].status == "skipped"
    assert "did not succeed" in results["child"].error


def test_network_analyzers_are_blocked_by_default(images):
    ctx = AnalysisContext(images["sharp"], allow_network=False)
    (result,) = Pipeline([Networked()]).run(ctx)
    assert result.status == "skipped"
    assert "network" in result.error


def test_results_are_timed(images):
    ctx = AnalysisContext(images["sharp"])
    (result,) = Pipeline([_make("timed")]).run(ctx)
    assert result.duration_ms >= 0


# -- profiles --------------------------------------------------------------


def test_quick_profile_is_only_cheap_analyzers(registry):
    """The property, not a fixed list.

    An exact set breaks every time a genuinely cheap analyzer is added, which
    says nothing about whether the profile still means what it claims. What
    matters is that everything in it is CHEAP — the companion test proves that
    also means no pixel decode.
    """
    chosen = plan(registry, profile="quick")
    assert chosen, "quick must not be empty"
    for analyzer in chosen:
        assert analyzer.cost is Cost.CHEAP, f"{analyzer.name} is {analyzer.cost} in quick"

    names = {a.name for a in chosen}
    assert {"fileinfo", "exif", "gps", "hashes"} <= names, "the metadata core must be present"


def test_deep_profile_is_a_superset_of_standard(registry):
    standard = {a.name for a in plan(registry, profile="standard")}
    deep = {a.name for a in plan(registry, profile="deep")}
    assert standard <= deep


def test_profiles_never_include_a_stranded_analyzer(registry):
    """A cost ceiling can cut a dependency out from under an analyzer.

    `sensitive` is MEDIUM but requires `ocr`, which is HEAVY. Listing it in
    `standard` would promise an analyzer that can only report "dependency not
    selected" — noise in every report and a profile listing that lies.
    """
    for profile in PROFILES:
        chosen = {a.name for a in plan(registry, profile=profile)}
        for analyzer in plan(registry, profile=profile):
            missing = set(analyzer.requires) - chosen
            assert not missing, f"{profile}: {analyzer.name} needs {missing}"


def test_profile_listing_accounts_for_every_planned_analyzer(registry, images):
    """Everything planned appears in the document, with a reason if it did not run.

    Not every planned analyzer *can* run: whether a model is installed is a
    runtime fact the planner cannot know, so an analyzer may be planned and
    then skipped because its dependency turned out to be unavailable. What
    must never happen is one vanishing without explanation.
    """
    for profile in PROFILES:
        planned = {a.name for a in plan(registry, profile=profile)}
        doc = analyze_image(images["sharp"], registry=registry, profile=profile)
        assert set(doc.analyzers) == planned, f"{profile}: {planned ^ set(doc.analyzers)}"
        for name, block in doc.analyzers.items():
            if block.status != "ok":
                assert block.error, f"{profile}: {name} is {block.status} with no reason"


def test_unknown_profile_raises(registry):
    with pytest.raises(KeyError):
        plan(registry, profile="turbo")


def test_explicit_selection_overrides_profile(registry):
    names = {a.name for a in plan(registry, profile="quick", only=["blur"])}
    assert names == {"blur"}


def test_selection_of_unknown_analyzer_raises(registry):
    with pytest.raises(SelectionError):
        plan(registry, only=["nonexistent"])


def test_exclude_keeps_required_dependencies(registry):
    names = {a.name for a in plan(registry, only=["gps"], exclude=["exif"])}
    assert "exif" in names, "gps depends on exif, so exif must survive exclusion"


def test_quick_profile_never_decodes_pixels(registry, images, monkeypatch):
    """Regression: `quick` must stay header-only, or it is not quick.

    Decoding a 60 MP file dominates everything else in the profile, so this is
    a performance contract, not an implementation detail.
    """
    from imgintel.core import context as ctx_mod

    decoded = []
    original = ctx_mod.AnalysisContext.pil_raw.func

    def spy(self):
        decoded.append(self.path.name)
        return original(self)

    monkeypatch.setattr(ctx_mod.AnalysisContext, "pil_raw", property(spy))
    analyze_image(images["gps"] if "gps" in images else images["sharp"],
                  registry=registry, profile="quick")
    assert decoded == [], f"quick profile decoded pixels: {decoded}"


def test_every_profile_is_runnable(registry, images):
    for name in PROFILES:
        doc = analyze_image(images["sharp"], registry=registry, profile=name)
        assert doc.summary.analyzers_failed == 0


# -- schema ----------------------------------------------------------------


def test_document_is_json_serializable_with_no_numpy_leakage(images):
    doc = analyze_image(images["sharp"])
    payload = json.loads(doc.model_dump_json())
    assert payload["tool"]["schema_version"] == SCHEMA_VERSION

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        else:
            assert type(node).__module__ == "builtins", f"non-builtin leaked: {type(node)}"

    walk(payload)


def test_summary_counts_match_the_findings(images):
    doc = analyze_image(images["sharp"])
    assert doc.summary.findings_total == len(doc.findings)
    assert sum(doc.summary.by_severity.values()) == len(doc.findings)


def test_findings_are_sorted_by_severity(images):
    doc = analyze_image(images["gps"] if "gps" in images else images["mislabeled"])
    rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    ranks = [rank[f.severity] for f in doc.findings]
    assert ranks == sorted(ranks)


def test_schema_is_valid_json_schema():
    schema = FindingsDocument.model_json_schema()
    assert "properties" in schema and "findings" in schema["properties"]


def test_findings_carry_their_analyzer(images):
    doc = analyze_image(images["sharp"])
    assert all(f.analyzer in doc.analyzers for f in doc.findings)


# -- context ---------------------------------------------------------------


def test_derived_images_are_cached(images):
    ctx = AnalysisContext(images["sharp"])
    assert ctx.rgb is ctx.rgb
    assert ctx.gray is ctx.gray


def test_small_is_bounded(images):
    from imgintel.core.context import SMALL_EDGE

    assert max(AnalysisContext(images["sharp"]).small.shape[:2]) <= SMALL_EDGE


def test_undecodable_file_is_reported_not_raised(images):
    doc = analyze_image(images["notanimage"])
    assert doc.analyzers["fileinfo"].status == "ok"
    assert any("decodable" in f.label.lower() for f in doc.findings)


def test_empty_file_is_handled(images):
    doc = analyze_image(images["empty"])
    assert any(f.label == "Empty file" for f in doc.findings)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        analyze_image("/nonexistent/nope.jpg")


# -- finding contract ------------------------------------------------------


def test_finding_serializes_cleanly():
    f = Finding("cat", "label", 42, Severity.HIGH, confidence=0.5, bbox=(1, 2, 3, 4))
    assert f.as_dict() == {
        "category": "cat",
        "label": "label",
        "value": 42,
        "severity": "high",
        "confidence": 0.5,
        "bbox": [1, 2, 3, 4],
        "detail": None,
    }


def test_analyzer_result_helpers():
    a = _make("helper")
    assert a.ok().status == "ok"
    assert a.skipped("why").status == "skipped"
    assert isinstance(a.ok(), AnalyzerResult)
