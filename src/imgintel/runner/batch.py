"""Batch analysis across a directory tree.

Two things dominate correctness here and neither is obvious:

1. **Model and registry loading must happen once per worker, not per image.**
   Discovery plus (from Phase 3) ONNX session construction costs far more than
   analyzing one image. The pool initializer builds them once and every task in
   that process reuses them. Getting this wrong makes batch tens of times
   slower and looks like "the tool is just slow".

2. **Every input must appear in the output.** A file that crashes the decoder
   gets a row with an ``error`` column, never silent omission — an investigator
   who sees 998 rows for 1000 files should be able to find the missing two.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgintel.core.context import CaseConfig
from imgintel.core.engine import analyze_image
from imgintel.core.registry import Registry
from imgintel.core.schema import FindingsDocument
from imgintel.report.flatten import failed_row, flatten

IMAGE_EXTENSIONS = frozenset(
    {
        ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".gif", ".bmp", ".dib",
        ".tif", ".tiff", ".webp", ".heic", ".heif", ".avif", ".jxl",
        ".ico", ".psd", ".qoi",
    }
)

# Per-worker state, populated once by the pool initializer.
_REGISTRY: Registry | None = None
_OPTIONS: dict[str, Any] = {}


@dataclass(slots=True)
class BatchConfig:
    profile: str | None = "standard"
    only: list[str] | None = None
    exclude: list[str] | None = None
    allow_network: bool = False
    allow_biometrics: bool = False
    entity_db: str | None = None
    allow_local_plugins: bool = False
    case: CaseConfig = field(default_factory=CaseConfig)
    workers: int = 0  # 0 = auto
    write_json: bool = False

    def resolved_workers(self, n_items: int) -> int:
        if self.workers > 0:
            return max(1, min(self.workers, n_items))
        return max(1, min((os.cpu_count() or 2), n_items, 8))


@dataclass(slots=True)
class BatchResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    documents: list[FindingsDocument] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    skipped: int = 0
    workers: int = 1

    @property
    def analyzed(self) -> int:
        return len(self.documents)

    @property
    def total(self) -> int:
        return len(self.rows)


def find_images(root: Path, *, recursive: bool = True, follow_symlinks: bool = False) -> list[Path]:
    """Every image file under ``root``, sorted for deterministic ordering."""
    root = Path(root)
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    found = []
    for path in root.glob(pattern):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not follow_symlinks and path.is_symlink():
            continue
        if path.is_file():
            found.append(path)
    return sorted(found)


# -- worker side -----------------------------------------------------------


def _init_worker(options: dict[str, Any]) -> None:
    """Build the registry once in this process. Runs on worker start."""
    global _REGISTRY, _OPTIONS  # noqa: PLW0603 - process-local warm state
    _OPTIONS = options
    _REGISTRY = Registry.discover(allow_local=options.get("allow_local_plugins", False))


def _analyze_one(path_str: str) -> tuple[str, str | None, str | None]:
    """Return (path, document_json, error). Never raises."""
    try:
        doc = analyze_image(
            path_str,
            registry=_REGISTRY,
            profile=_OPTIONS.get("profile"),
            only=_OPTIONS.get("only"),
            exclude=_OPTIONS.get("exclude"),
            allow_network=_OPTIONS.get("allow_network", False),
            allow_biometrics=_OPTIONS.get("allow_biometrics", False),
            entity_db=_OPTIONS.get("entity_db"),
            case=CaseConfig(**_OPTIONS.get("case", {})),
            command=_OPTIONS.get("command"),
        )
        return path_str, doc.model_dump_json(), None
    except Exception as exc:  # noqa: BLE001 - a bad file must not kill the pool
        return path_str, None, f"{type(exc).__name__}: {exc}"


# -- driver ----------------------------------------------------------------


def run_batch(
    paths: Iterable[Path],
    config: BatchConfig,
    *,
    already_done: set[str] | None = None,
    on_result: Callable[[str, FindingsDocument | None, str | None], None] | None = None,
    command: str | None = None,
) -> BatchResult:
    """Analyze many images, in parallel when it is worth it."""
    done = already_done or set()
    all_paths = list(paths)  # may be a generator; needed twice
    todo = [p for p in all_paths if str(p.resolve()) not in done and str(p) not in done]
    result = BatchResult(skipped=len(all_paths) - len(todo))

    if not todo:
        return result

    options: dict[str, Any] = {
        "profile": config.profile,
        "only": config.only,
        "exclude": config.exclude,
        "allow_network": config.allow_network,
        "allow_biometrics": config.allow_biometrics,
        "entity_db": config.entity_db,
        "allow_local_plugins": config.allow_local_plugins,
        "case": {
            "case_id": config.case.case_id,
            "operator": config.case.operator,
            "notes": config.case.notes,
            "biometric_lawful_basis": config.case.biometric_lawful_basis,
            "biometric_authorised_by": config.case.biometric_authorised_by,
            "biometric_expires": config.case.biometric_expires,
        },
        "command": command,
    }
    workers = config.resolved_workers(len(todo))
    result.workers = workers

    # One image, or one worker: skip the pool entirely. Process startup would
    # cost more than the work, and in-process runs are far easier to debug.
    if workers == 1:
        _init_worker(options)
        stream: Iterator[tuple[str, str | None, str | None]] = (
            _analyze_one(str(p)) for p in todo
        )
        _collect(stream, result, on_result)
        return result

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(options,)
    ) as pool:
        futures = {pool.submit(_analyze_one, str(p)): p for p in todo}
        _collect((f.result() for f in as_completed(futures)), result, on_result)

    # as_completed returns out of order; restore input order for reproducibility.
    order = {str(p): i for i, p in enumerate(todo)}
    result.rows.sort(key=lambda r: order.get(str(r.get("path_input", r.get("path"))), 0))
    return result


def _collect(
    stream: Iterable[tuple[str, str | None, str | None]],
    result: BatchResult,
    on_result: Callable[[str, FindingsDocument | None, str | None], None] | None,
) -> None:
    for path_str, doc_json, error in stream:
        if error or doc_json is None:
            message = error or "unknown error"
            result.failures.append((path_str, message))
            row = failed_row(path_str, message)
            row["path_input"] = path_str
            result.rows.append(row)
            if on_result:
                on_result(path_str, None, message)
            continue

        doc = FindingsDocument.model_validate_json(doc_json)
        result.documents.append(doc)
        row = flatten(doc)
        row["path_input"] = path_str
        result.rows.append(row)
        if on_result:
            on_result(path_str, doc, None)
