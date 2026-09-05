"""The plugin contract — the thing Phases 3-7 depend on staying stable."""

from __future__ import annotations

import textwrap

from imgintel.core.engine import analyze_image
from imgintel.core.registry import Registry

PLUGIN_SOURCE = textwrap.dedent(
    '''
    """A third-party analyzer that consumes a built-in one's output."""

    from imgintel.core.analyzer import Analyzer, Cost, Finding, Severity


    class AspectRatioAnalyzer(Analyzer):
        name = "aspect"
        version = "0.1.0"
        title = "Aspect ratio"
        description = "Flags unusual aspect ratios"
        requires = ("fileinfo",)
        cost = Cost.CHEAP

        def analyze(self, ctx):
            info = ctx.data("fileinfo")
            ratio = info.get("aspect_ratio")
            findings = []
            if ratio and (ratio > 3 or ratio < 0.33):
                findings.append(
                    Finding("composition", "Unusual aspect ratio", ratio, Severity.LOW)
                )
            return self.ok({"aspect_ratio": ratio}, findings)
    '''
)


def test_local_plugin_is_discovered_and_runs(tmp_path, images):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "aspect.py").write_text(PLUGIN_SOURCE)

    registry = Registry()
    registry.load_builtin()
    registry.load_local(plugin_dir)
    assert not registry.errors, registry.errors
    assert "aspect" in registry
    assert registry.origin("aspect") == "local:aspect.py"

    doc = analyze_image(images["sharp"], registry=registry, profile="quick")
    block = doc.analyzers["aspect"]
    assert block.status == "ok"
    assert block.data["aspect_ratio"] > 1  # 1200x900


def test_local_plugins_are_not_loaded_without_opt_in(tmp_path, monkeypatch):
    """Loose .py files are arbitrary code — they must never load implicitly.

    Uses a deliberately unlikely name: an installed third-party plugin could
    otherwise provide the same one and make this assert the wrong thing.
    """
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "localonly.py").write_text(
        PLUGIN_SOURCE.replace('"aspect"', '"localonly-probe"').replace(
            "AspectRatioAnalyzer", "LocalOnlyProbe"
        )
    )
    monkeypatch.setattr("imgintel.core.registry.user_plugin_dir", lambda: plugin_dir)

    assert "localonly-probe" not in Registry.discover()
    assert "localonly-probe" in Registry.discover(allow_local=True)


def test_broken_plugin_is_reported_not_fatal(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "broken.py").write_text("this is not valid python !!!")

    registry = Registry()
    registry.load_builtin()
    registry.load_local(plugin_dir)
    assert len(registry) >= 8, "built-ins must survive a broken plugin"
    assert any("broken.py" in e.source for e in registry.errors)


def test_plugin_dependency_ordering(tmp_path, images):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "aspect.py").write_text(PLUGIN_SOURCE)

    registry = Registry()
    registry.load_builtin()
    registry.load_local(plugin_dir)

    doc = analyze_image(images["sharp"], registry=registry, only=["aspect"])
    # `requires` pulled fileinfo in even though it was not asked for.
    assert set(doc.analyzers) == {"aspect", "fileinfo"}
    assert doc.analyzers["aspect"].status == "ok"


def test_every_builtin_declares_its_contract(registry):
    for analyzer in registry.all():
        assert analyzer.name and analyzer.version and analyzer.description
        assert analyzer.available().ok in (True, False)
        for dep in analyzer.requires:
            assert dep in registry, f"{analyzer.name} requires unknown {dep}"
