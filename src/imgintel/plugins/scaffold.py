"""Templates for `imgintel plugins new`.

Scaffolding exists because the gap between "I know what should be checked" and
"I have a working plugin" is mostly boilerplate — an entry-point table, a
pyproject, remembering that `cost` decides the profile. Generating a plugin
that already runs turns that into editing one function.

Both templates produce something that works before it is edited, so the first
thing a user does is see it appear in `imgintel plugins`, not debug packaging.
"""

from __future__ import annotations

import re
from pathlib import Path

from imgintel.plugins import API_VERSION

SAFE_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class ScaffoldError(ValueError):
    pass


def check_name(name: str) -> str:
    if not SAFE_NAME.match(name):
        raise ScaffoldError(
            f"{name!r} must start with a lowercase letter and contain only "
            "lowercase letters, digits, '-' or '_'"
        )
    return name


def _module_name(name: str) -> str:
    return name.replace("-", "_")


ANALYZER_TEMPLATE = '''\
"""The {name} analyzer."""

from imgintel.plugins import Analyzer, Availability, Cost, Finding, Severity, require_api

# Fails loudly now rather than with an AttributeError mid-run.
require_api("{api}")


class {klass}(Analyzer):
    #: Unique. Used in --analyzers and in other analyzers' `requires`.
    name = "{name}"
    version = "0.1.0"
    title = "{title}"
    description = "TODO: one line, shown by `imgintel plugins`"

    #: Analyzers that must run first. Their output is read via ctx.data(...).
    requires = ("fileinfo",)
    #: Run after these *if present*, but do not require them.
    after = ()

    #: CHEAP means metadata only and NO pixel decode — it puts this analyzer
    #: in the `quick` profile, which promises exactly that. Use MEDIUM once
    #: you touch ctx.small / ctx.rgb / ctx.gray, HEAVY for model inference.
    cost = Cost.CHEAP

    def available(self) -> Availability:
        """Report missing binaries or models. Must never raise."""
        return Availability.yes()

    def analyze(self, ctx):
        # `data` is yours to shape; `findings` is the flat, normalized list
        # that every report and export consumes. Keep the split — it is why
        # adding an analyzer needs no changes to any output code.
        info = ctx.data("fileinfo")
        findings = []

        if info.get("megapixels", 0) > 50:
            findings.append(
                Finding(
                    category="quality",
                    label="Very large image",
                    value=info["megapixels"],
                    severity=Severity.LOW,
                    confidence=0.9,
                    detail="Explain what this means, and what could explain it innocently.",
                )
            )

        return self.ok({{"megapixels": info.get("megapixels")}}, findings)
'''

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "imgintel-{name}"
version = "0.1.0"
description = "{title} analyzer for imgintel"
requires-python = ">=3.10"
dependencies = ["imgintel"]

# This is what makes the analyzer appear after `pip install`. No registration
# call, no configuration.
[project.entry-points."imgintel.analyzers"]
{name} = "{module}.analyzer:{klass}"

[tool.hatch.build.targets.wheel]
packages = ["src/{module}"]
"""

TEST_TEMPLATE = '''\
"""Tests for the {name} analyzer."""

from imgintel.core.engine import analyze_image
from imgintel.core.registry import Registry

from {module}.analyzer import {klass}


def registry():
    reg = Registry()
    reg.load_builtin()
    reg.register({klass}(), "test")
    return reg


def test_analyzer_runs(tmp_path):
    from PIL import Image

    image = tmp_path / "test.png"
    Image.new("RGB", (64, 48), (128, 128, 128)).save(image)

    doc = analyze_image(image, registry=registry(), only=["{name}"])
    block = doc.analyzers["{name}"]
    assert block.status == "ok", block.error


def test_contract_is_complete():
    analyzer = {klass}()
    assert analyzer.name and analyzer.version and analyzer.description
    assert analyzer.available().ok in (True, False)
'''

README_TEMPLATE = """\
# imgintel-{name}

A plugin for [imgintel]({url}).

## Install

```bash
pip install -e .
imgintel plugins            # {name} should now be listed
imgintel analyze photo.jpg -a {name}
```

## Develop

The analyzer is in `src/{module}/analyzer.py`. See the imgintel plugin guide
for the full contract; the short version:

- `cost` decides which profile includes it. `CHEAP` promises no pixel decode.
- `data` is your rich output; `findings` is the flat list reports consume.
- `available()` reports missing dependencies; it must never raise.
- `precheck()` refuses on grounds of permission, before any capability check.

```bash
pytest
```
"""

RULES_TEMPLATE = """\
# A declarative imgintel ruleset. No Python required.
#
# Drop this in your rules directory (`imgintel plugins dirs` shows where) and
# it is picked up automatically. Rule files are data — they are parsed, never
# executed — so they need no --allow-local-plugins opt-in.
name: {name}
description: TODO describe what these rules check

rules:
  # Match a regular expression against all extracted text: OCR output,
  # barcode payloads and EXIF free-text fields.
  - id: example-text
    when:
      text_matches: '(?i)\\bCONFIDENTIAL\\b'
    finding:
      category: sensitive
      label: Confidentiality marking
      severity: high
      confidence: 0.9
      detail: Explain what this means and what could explain it innocently.

  # Compare a number from any analyzer's output against a threshold.
  - id: example-threshold
    when:
      value_below: {{path: "fileinfo.megapixels", threshold: 0.2}}
    finding:
      category: quality
      label: Image too small for reliable analysis
      severity: low

  # Assert something about metadata.
  - id: example-metadata
    when:
      exif_missing: make
    finding:
      category: provenance
      label: No camera make recorded
      severity: low
"""


def scaffold_python(directory: Path, name: str, *, url: str = "") -> list[Path]:
    """Write a complete, installable plugin package."""
    check_name(name)
    module = _module_name(name)
    klass = "".join(part.title() for part in module.split("_")) + "Analyzer"
    title = name.replace("-", " ").replace("_", " ").title()

    package = directory / f"imgintel-{name}"
    if package.exists() and any(package.iterdir()):
        raise ScaffoldError(f"{package} already exists and is not empty")

    source = package / "src" / module
    source.mkdir(parents=True, exist_ok=True)
    (package / "tests").mkdir(exist_ok=True)

    written = [
        _write(source / "__init__.py", f'"""imgintel plugin: {name}."""\n'),
        _write(
            source / "analyzer.py",
            ANALYZER_TEMPLATE.format(name=name, klass=klass, title=title, api=API_VERSION),
        ),
        _write(
            package / "pyproject.toml",
            PYPROJECT_TEMPLATE.format(name=name, module=module, klass=klass, title=title),
        ),
        _write(
            package / "tests" / f"test_{module}.py",
            TEST_TEMPLATE.format(name=name, module=module, klass=klass),
        ),
        _write(
            package / "README.md",
            README_TEMPLATE.format(
                name=name, module=module, url=url or "https://github.com/vivek/image-intel"
            ),
        ),
    ]
    return written


def scaffold_rules(directory: Path, name: str) -> list[Path]:
    """Write a starter rule file."""
    check_name(name)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    if path.exists():
        raise ScaffoldError(f"{path} already exists")
    return [_write(path, RULES_TEMPLATE.format(name=name))]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["ScaffoldError", "check_name", "scaffold_python", "scaffold_rules"]
