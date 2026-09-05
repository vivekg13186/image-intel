"""The case file — investigation metadata that persists between runs.

A JSON file rather than CLI flags because the things it holds are properties of
the *investigation*, not of one command: the case reference, the operator, and
the lawful basis under which biometric identification may be performed. Typing
a warrant number into every invocation is how it ends up wrong.

The one thing deliberately *not* stored here is per-run consent. See
:class:`~imgintel.core.context.CaseConfig` for why authorisation and intent are
kept separate.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from imgintel.core.context import CaseConfig

CASE_FILENAME = "imgintel-case.json"


def default_path(directory: Path | None = None) -> Path:
    return (directory or Path.cwd()) / CASE_FILENAME


def load(path: Path | None = None) -> CaseConfig:
    """Read a case file, returning an empty config when there is none."""
    target = path or default_path()
    if not target.exists():
        return CaseConfig()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return CaseConfig()
    if not isinstance(payload, dict):
        return CaseConfig()

    fields = {f for f in CaseConfig.__dataclass_fields__}
    return CaseConfig(**{k: v for k, v in payload.items() if k in fields})


def save(case: CaseConfig, path: Path | None = None) -> Path:
    target = path or default_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(case), indent=2, sort_keys=True), encoding="utf-8")
    return target


def merge(base: CaseConfig, **overrides) -> CaseConfig:
    """Apply non-None overrides — CLI flags win over the case file."""
    data = asdict(base)
    data.update({k: v for k, v in overrides.items() if v is not None})
    return CaseConfig(**data)


__all__ = ["CASE_FILENAME", "default_path", "load", "merge", "save"]
