"""Dependency-ordered execution with per-analyzer isolation.

One analyzer blowing up must never lose the results of the other twenty-nine —
that is the whole reason this is a pipeline rather than a script.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable

from imgintel.core.analyzer import Analyzer, AnalyzerResult
from imgintel.core.context import AnalysisContext, ImageDecodeError

ProgressHook = Callable[[str, str], None]  # (analyzer_name, phase)


class CycleError(RuntimeError):
    """Raised when analyzer ``requires`` declarations form a loop."""


def topo_sort(analyzers: Iterable[Analyzer]) -> list[Analyzer]:
    """Order analyzers so every dependency precedes its dependents.

    Both hard (``requires``) and soft (``after``) edges are honoured; soft
    edges only apply when the other analyzer is actually in the selection.

    Ties are broken by (cost, name) so cheap analyzers run first and ordering
    is deterministic across runs — which matters for reproducible evidence.
    """
    by_name = {a.name: a for a in analyzers}
    ordered: list[Analyzer] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if state.get(name) == 1:
            return
        if state.get(name) == 0:
            raise CycleError(" -> ".join([*trail, name]))
        analyzer = by_name.get(name)
        if analyzer is None:
            return  # dependency outside the selection; handled at run time
        state[name] = 0
        edges = set(analyzer.requires) | {a for a in analyzer.after if a in by_name}
        for dep in sorted(edges):
            visit(dep, (*trail, name))
        state[name] = 1
        ordered.append(analyzer)

    for name in sorted(by_name, key=lambda n: (by_name[n].cost.rank, n)):
        visit(name, ())
    return ordered


class Pipeline:
    """Runs a set of analyzers over one image."""

    def __init__(
        self,
        analyzers: Iterable[Analyzer],
        *,
        on_progress: ProgressHook | None = None,
    ) -> None:
        self.analyzers = topo_sort(analyzers)
        self.on_progress = on_progress

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.analyzers]

    def run(self, ctx: AnalysisContext) -> list[AnalyzerResult]:
        results: list[AnalyzerResult] = []
        selected = {a.name for a in self.analyzers}
        ctx.planned = frozenset(selected)

        for analyzer in self.analyzers:
            result = self._run_one(analyzer, ctx, selected)
            ctx.results[analyzer.name] = result
            results.append(result)
        return results

    def _run_one(
        self,
        analyzer: Analyzer,
        ctx: AnalysisContext,
        selected: set[str],
    ) -> AnalyzerResult:
        name = analyzer.name

        # 0. Permission, before capability. An analyzer that may not run must
        #    report that, not a missing dependency it was never going to reach.
        try:
            refusal = analyzer.precheck(ctx)
        except Exception as exc:  # noqa: BLE001
            return AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="error",
                error=f"precheck raised: {type(exc).__name__}: {exc}",
            )
        if refusal:
            return AnalyzerResult(
                analyzer=name, version=analyzer.version, status="skipped", error=refusal
            )

        # 1. Unmet dependencies — skip rather than fail.
        missing = [d for d in analyzer.requires if d not in selected]
        if missing:
            return AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="skipped",
                error=f"dependency not selected: {', '.join(sorted(missing))}",
            )
        unmet = [d for d in analyzer.requires if not ctx.succeeded(d)]
        if unmet:
            return AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="skipped",
                error=f"dependency did not succeed: {', '.join(sorted(unmet))}",
            )

        # 2. Network gate — an offline-by-default tool must never surprise you.
        if analyzer.needs_network and not ctx.allow_network:
            return AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="skipped",
                error="requires network access (pass --allow-network)",
            )

        # 3. Missing binaries, models or optional imports.
        try:
            avail = analyzer.available()
        except Exception as exc:  # noqa: BLE001
            return AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="error",
                error=f"availability check raised: {type(exc).__name__}: {exc}",
            )
        if not avail.ok:
            msg = avail.reason
            if avail.hint:
                msg = f"{msg} ({avail.hint})"
            return AnalyzerResult(
                analyzer=name, version=analyzer.version, status="unavailable", error=msg
            )

        # 4. Run it.
        if self.on_progress:
            self.on_progress(name, "start")
        started = time.perf_counter()
        try:
            result = analyzer.analyze(ctx)
        except ImageDecodeError as exc:
            result = AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="error",
                error=f"image could not be decoded: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            result = AnalyzerResult(
                analyzer=name,
                version=analyzer.version,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                data={"traceback": traceback.format_exc(limit=6)},
            )
        result.duration_ms = (time.perf_counter() - started) * 1000
        # Guard against an analyzer returning a result labelled as another.
        result.analyzer = name
        if self.on_progress:
            self.on_progress(name, "done")
        return result
