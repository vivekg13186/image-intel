"""CSV projections.

Two shapes, because they answer different questions:

* **summary** — one row per image. For triage: sort by findings_high, filter by
  camera serial, pivot on country.
* **findings** — one row per finding. For working through what was found.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from imgintel.core.schema import FindingsDocument
from imgintel.report.flatten import FLAT_COLUMNS, flatten

FINDING_COLUMNS = (
    "path", "filename", "sha256", "analyzer", "category", "severity",
    "confidence", "label", "value", "detail", "bbox",
)


def _open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" is required by the csv module; utf-8-sig so Excel on Windows
    # does not mangle non-ASCII place names and EXIF strings.
    return path.open("w", newline="", encoding="utf-8-sig")


def write_summary_csv(rows: Iterable[dict[str, Any]], path: Path) -> int:
    """One row per image, using the canonical :data:`FLAT_COLUMNS` order."""
    count = 0
    with _open(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FLAT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in FLAT_COLUMNS})
            count += 1
    return count


def write_findings_csv(docs: Iterable[FindingsDocument], path: Path) -> int:
    """One row per finding, across any number of documents."""
    count = 0
    with _open(path) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FINDING_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for doc in docs:
            sha = (doc.analyzers.get("hashes").data.get("sha256")
                   if doc.analyzers.get("hashes") else None)
            for finding in doc.findings:
                writer.writerow(
                    {
                        "path": doc.target.path,
                        "filename": doc.target.filename,
                        "sha256": sha,
                        "analyzer": finding.analyzer,
                        "category": finding.category,
                        "severity": finding.severity,
                        "confidence": finding.confidence,
                        "label": finding.label,
                        "value": _cell(finding.value),
                        "detail": finding.detail,
                        "bbox": _cell(finding.bbox),
                    }
                )
                count += 1
    return count


def summary_row(doc: FindingsDocument) -> dict[str, Any]:
    return flatten(doc)


def _cell(value: Any) -> Any:
    """Flatten containers; leave scalars alone so numeric columns stay numeric."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)
