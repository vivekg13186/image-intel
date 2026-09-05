"""The canonical output document.

JSON is the source of truth; CSV, HTML and the evidence bundle are all
projections of it. ``SCHEMA_VERSION`` is bumped whenever the shape changes so
downstream consumers can detect it.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from imgintel import __version__
from imgintel.core.analyzer import AnalyzerResult

SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FindingModel(BaseModel):
    analyzer: str
    category: str
    label: str
    value: Any = None
    severity: Literal["info", "low", "medium", "high"] = "info"
    confidence: float = 1.0
    bbox: list[int] | None = None
    detail: str | None = None


class AnalyzerBlock(BaseModel):
    analyzer: str
    version: str
    status: Literal["ok", "skipped", "unavailable", "error"]
    duration_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ToolInfo(BaseModel):
    name: str = "imgintel"
    version: str = __version__
    schema_version: str = SCHEMA_VERSION
    python: str = Field(default_factory=platform.python_version)
    platform: str = Field(default_factory=lambda: f"{platform.system()} {platform.release()}")


class TargetInfo(BaseModel):
    path: str
    filename: str
    size_bytes: int | None = None
    analyzed_at: str = Field(default_factory=_now)


class CaseInfo(BaseModel):
    case_id: str | None = None
    operator: str | None = None
    notes: str | None = None
    # Recorded so an evidence bundle shows the authority under which any
    # biometric identification was carried out — or that none existed.
    biometric_lawful_basis: str | None = None
    biometric_authorised_by: str | None = None
    biometric_expires: str | None = None


class RunInfo(BaseModel):
    profile: str | None = None
    analyzers_requested: list[str] = Field(default_factory=list)
    analyzers_run: list[str] = Field(default_factory=list)
    allow_network: bool = False
    #: Whether this run permitted biometric identification, and against what.
    allow_biometrics: bool = False
    entity_db: str | None = None
    duration_ms: float = 0.0
    command: str | None = None


class Summary(BaseModel):
    findings_total: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    analyzers_ok: int = 0
    analyzers_failed: int = 0
    analyzers_unavailable: int = 0


class FindingsDocument(BaseModel):
    tool: ToolInfo = Field(default_factory=ToolInfo)
    target: TargetInfo
    case: CaseInfo = Field(default_factory=CaseInfo)
    run: RunInfo = Field(default_factory=RunInfo)
    summary: Summary = Field(default_factory=Summary)
    findings: list[FindingModel] = Field(default_factory=list)
    analyzers: dict[str, AnalyzerBlock] = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        path: str,
        filename: str,
        size_bytes: int | None,
        results: list[AnalyzerResult],
        run: RunInfo | None = None,
        case: CaseInfo | None = None,
    ) -> FindingsDocument:
        blocks: dict[str, AnalyzerBlock] = {}
        findings: list[FindingModel] = []
        by_sev: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        ok = failed = unavailable = 0

        for res in results:
            blocks[res.analyzer] = AnalyzerBlock(
                analyzer=res.analyzer,
                version=res.version,
                status=res.status,
                duration_ms=round(res.duration_ms, 3),
                error=res.error,
                data=_jsonable(res.data),
            )
            if res.status == "ok":
                ok += 1
            elif res.status == "error":
                failed += 1
            elif res.status == "unavailable":
                unavailable += 1

            for f in res.findings:
                d = f.as_dict()
                findings.append(FindingModel(analyzer=res.analyzer, **d))
                by_sev[d["severity"]] = by_sev.get(d["severity"], 0) + 1
                by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1

        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        findings.sort(key=lambda f: (order[f.severity], f.category, f.label))

        return cls(
            target=TargetInfo(path=path, filename=filename, size_bytes=size_bytes),
            case=case or CaseInfo(),
            run=run or RunInfo(),
            summary=Summary(
                findings_total=len(findings),
                by_severity=by_sev,
                by_category=by_cat,
                analyzers_ok=ok,
                analyzers_failed=failed,
                analyzers_unavailable=unavailable,
            ),
            findings=findings,
            analyzers=blocks,
        )


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays and other stragglers into JSON-safe types.

    Analyzers work in numpy; the schema must not leak numpy types into output.
    """
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (f != f or f in (float("inf"), float("-inf"))) else round(f, 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return value
