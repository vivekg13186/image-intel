"""The one function that ties everything together.

The CLI, the (Phase 2) batch runner and any library consumer all go through
here, so behaviour cannot drift between them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from imgintel.core.analyzer import Analyzer
from imgintel.core.context import AnalysisContext, CaseConfig
from imgintel.core.pipeline import Pipeline, ProgressHook
from imgintel.core.profiles import DEFAULT_PROFILE, select
from imgintel.core.registry import Registry
from imgintel.core.schema import CaseInfo, FindingsDocument, RunInfo


class SelectionError(ValueError):
    """Raised when the requested analyzers cannot be resolved."""


def plan(
    registry: Registry,
    *,
    profile: str | None = DEFAULT_PROFILE,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    allow_network: bool = False,
) -> list[Analyzer]:
    """Work out which analyzers to run.

    ``only`` overrides the profile entirely (and pulls in dependencies), which
    is what you want when debugging a single analyzer.
    """
    if only:
        chosen, unknown = registry.resolve(only)
        if unknown:
            raise SelectionError(
                f"unknown analyzer(s): {', '.join(unknown)}. "
                f"Available: {', '.join(registry.names())}"
            )
    else:
        chosen = select(registry.all(), profile or DEFAULT_PROFILE)

    if exclude:
        drop = set(exclude)
        # Never drop something another selected analyzer depends on.
        needed = {d for a in chosen if a.name not in drop for d in a.requires}
        chosen = [a for a in chosen if a.name not in drop or a.name in needed]

    if not allow_network:
        chosen = [a for a in chosen if not a.needs_network]
    return chosen


def analyze_image(
    path: str | Path,
    *,
    registry: Registry | None = None,
    profile: str | None = DEFAULT_PROFILE,
    only: list[str] | None = None,
    exclude: list[str] | None = None,
    allow_network: bool = False,
    allow_biometrics: bool = False,
    entity_db: str | Path | None = None,
    case: CaseConfig | None = None,
    command: str | None = None,
    on_progress: ProgressHook | None = None,
) -> FindingsDocument:
    """Analyze one image and return the canonical findings document."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory; use `imgintel batch`")

    registry = registry or Registry.discover()
    analyzers = plan(
        registry,
        profile=profile,
        only=only,
        exclude=exclude,
        allow_network=allow_network,
    )

    ctx = AnalysisContext(
        path,
        allow_network=allow_network,
        allow_biometrics=allow_biometrics,
        entity_db=entity_db,
        case=case or CaseConfig(),
    )
    pipeline = Pipeline(analyzers, on_progress=on_progress)

    started = time.perf_counter()
    results = pipeline.run(ctx)
    elapsed = (time.perf_counter() - started) * 1000

    try:
        size = path.stat().st_size
    except OSError:
        size = None

    cfg = case or CaseConfig()
    return FindingsDocument.build(
        path=str(path.resolve()),
        filename=path.name,
        size_bytes=size,
        results=results,
        run=RunInfo(
            profile=None if only else (profile or DEFAULT_PROFILE),
            analyzers_requested=only or [],
            analyzers_run=pipeline.names,
            allow_network=allow_network,
            allow_biometrics=allow_biometrics,
            entity_db=str(entity_db) if entity_db else None,
            duration_ms=round(elapsed, 3),
            command=command,
        ),
        case=CaseInfo(
            case_id=cfg.case_id,
            operator=cfg.operator,
            notes=cfg.notes,
            biometric_lawful_basis=cfg.biometric_lawful_basis,
            biometric_authorised_by=cfg.biometric_authorised_by,
            biometric_expires=cfg.biometric_expires,
        ),
    )


AnalyzeFn = Callable[..., FindingsDocument]
