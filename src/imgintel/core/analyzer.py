"""The plugin contract.

Every feature in imgintel — EXIF parsing, blur detection, face matching — is an
``Analyzer``. The engine knows nothing about any of them beyond this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    from imgintel.core.context import AnalysisContext

Status = Literal["ok", "skipped", "unavailable", "error"]


class Cost(str, Enum):
    """Rough execution cost. Drives profile selection and scheduling.

    CHEAP   — metadata only, no pixel decode. Sub-millisecond.
    MEDIUM  — needs the decoded image. Tens of milliseconds.
    HEAVY   — model inference or multi-pass pixel work. Hundreds of ms or more.
    """

    CHEAP = "cheap"
    MEDIUM = "medium"
    HEAVY = "heavy"

    @property
    def rank(self) -> int:
        return {"cheap": 0, "medium": 1, "heavy": 2}[self.value]


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return {"info": 0, "low": 1, "medium": 2, "high": 3}[self.value]


@dataclass(slots=True)
class Finding:
    """A single normalized, human-facing observation.

    This is deliberately flat and analyzer-agnostic. Renderers, the summarizer
    and CSV export consume *only* findings — never an analyzer's ``data`` blob.
    That is what lets a new analyzer be added without touching any output code.
    """

    category: str
    label: str
    value: Any = None
    severity: Severity = Severity.INFO
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "value": self.value,
            "severity": self.severity.value,
            "confidence": round(float(self.confidence), 4),
            "bbox": list(self.bbox) if self.bbox else None,
            "detail": self.detail,
        }


@dataclass(slots=True)
class Availability:
    """Whether an analyzer can run here, and if not, what to do about it."""

    ok: bool
    reason: str = ""
    hint: str = ""

    @classmethod
    def yes(cls) -> Availability:
        return cls(True)

    @classmethod
    def no(cls, reason: str, hint: str = "") -> Availability:
        return cls(False, reason, hint)


@dataclass(slots=True)
class AnalyzerResult:
    analyzer: str
    version: str
    status: Status
    data: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"


class Analyzer(ABC):
    """Base class for all analyzers.

    Subclasses set the class attributes, implement :meth:`analyze`, and
    optionally override :meth:`available` to report missing dependencies.
    """

    #: Stable identifier used in CLI selection, JSON keys and ``requires``.
    name: str = ""
    #: Bump whenever the shape of ``data`` changes.
    version: str = "0.1.0"
    #: Human-readable heading for reports.
    title: str = ""
    #: One-line explanation shown by ``imgintel plugins``.
    description: str = ""
    #: Names of analyzers that must run (successfully) before this one.
    #: A missing or failed dependency causes this analyzer to be skipped.
    requires: tuple[str, ...] = ()
    #: Soft ordering: run after these *if they are in the selection*, but do
    #: not require them. Use it when an analyzer can enrich its output with
    #: another's results but works fine without them — `sensitive` scans
    #: barcode payloads when `codes` ran, and OCR text alone when it did not.
    after: tuple[str, ...] = ()
    cost: Cost = Cost.CHEAP
    #: True if the analyzer contacts a remote service. Blocked unless --allow-network.
    needs_network: bool = False
    #: Model-store keys required. Checked by ``doctor`` and the model puller.
    needs_models: tuple[str, ...] = ()

    def precheck(self, ctx: AnalysisContext) -> str | None:
        """Refuse on grounds of *permission*, before any capability check.

        Return a reason to skip, or None to proceed. Called before
        dependencies, the network gate and :meth:`available`, because
        permission is logically prior to capability: an analyzer that may not
        run should say so, not report that a model it was never going to use
        is missing.

        The difference matters in an evidence bundle. "Face matching was not
        performed because no lawful basis was recorded" and "face matching was
        not performed because the model was absent" are very different answers
        to the same question, and only the first is true when both hold.
        """
        return None

    def available(self) -> Availability:
        """Check imports, binaries and models. Must never raise."""
        return Availability.yes()

    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        """Do the work. May raise — the pipeline isolates and records failures."""

    # -- convenience constructors so subclasses stay terse ------------------

    def ok(
        self,
        data: dict[str, Any] | None = None,
        findings: list[Finding] | None = None,
    ) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer=self.name,
            version=self.version,
            status="ok",
            data=data or {},
            findings=findings or [],
        )

    def skipped(self, reason: str) -> AnalyzerResult:
        return AnalyzerResult(
            analyzer=self.name,
            version=self.version,
            status="skipped",
            error=reason,
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Analyzer {self.name} v{self.version}>"
