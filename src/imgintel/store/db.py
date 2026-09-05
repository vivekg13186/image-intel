"""The case index: one sqlite file per output directory.

Serves two jobs that would otherwise need separate machinery:

* **Resume** — a batch run that dies at image 40,000 must not restart at zero.
* **Cross-image queries** — dedupe, "which of these came off the same camera
  body", "everything within 200 m of this point" — all need the whole set in
  one place, not a directory of JSON files.

Sqlite is in the standard library, needs no server, and the resulting file is
itself an artifact you can hand to someone.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imgintel.report.flatten import FLAT_COLUMNS

SCHEMA_VERSION = 1

# Columns get their SQL affinity from the flat record's natural types.
_NUMERIC = {
    "size_bytes", "width", "height", "megapixels", "latitude", "longitude",
    "altitude_m", "bearing_deg", "jpeg_quality", "sharpness", "noise_sigma",
    "entropy", "localized_blur_regions", "ocr_text_chars", "ocr_blocks",
    "pii_count", "findings_high", "findings_medium", "findings_low",
    "findings_info", "findings_total", "analyzers_ok", "analyzers_failed",
    "duration_ms", "object_count", "people_count", "face_count", "vehicle_count",
    "tamper_indicator_count", "estimated_country_confidence",
}

# Columns worth an index for cross-image queries.
_INDEXED = (
    "sha256", "phash", "pixel_sha256", "camera", "serial", "country", "identities",
    "estimated_country",
)


def _column_sql() -> str:
    parts = []
    for name in FLAT_COLUMNS:
        affinity = "REAL" if name in _NUMERIC else "TEXT"
        if name == "path":
            parts.append("path TEXT PRIMARY KEY")
        else:
            parts.append(f"{name} {affinity}")
    return ", ".join(parts)


class ImageIndex:
    """Sqlite-backed store of flat records plus a resumable job table."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        # WAL lets the writer proceed while a reader (e.g. a progress view)
        # holds a snapshot; both are common during a long batch run.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create()

    # -- lifecycle ---------------------------------------------------------

    def _create(self) -> None:
        with self._conn:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS images ({_column_sql()})")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                       path TEXT PRIMARY KEY, status TEXT NOT NULL,
                       error TEXT, updated_utc TEXT NOT NULL)"""
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            for column in _INDEXED:
                self._conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_images_{column} ON images({column})"
                )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ImageIndex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def upsert(self, row: dict[str, Any]) -> None:
        self.upsert_many([row])

    def upsert_many(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        placeholders = ", ".join("?" * len(FLAT_COLUMNS))
        columns = ", ".join(FLAT_COLUMNS)
        sql = f"INSERT OR REPLACE INTO images ({columns}) VALUES ({placeholders})"
        payload = [tuple(_sql_value(row.get(c)) for c in FLAT_COLUMNS) for row in rows]
        with self._conn:
            self._conn.executemany(sql, payload)

    def mark(self, path: str, status: str, error: str | None = None) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?)",
                (path, status, error, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def mark_many(self, entries: list[tuple[str, str, str | None]]) -> None:
        if not entries:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO jobs VALUES (?, ?, ?, ?)",
                [(p, s, e, now) for p, s, e in entries],
            )

    # -- reading -----------------------------------------------------------

    def completed(self) -> set[str]:
        """Paths already finished — the basis for ``--resume``."""
        cur = self._conn.execute("SELECT path FROM jobs WHERE status = 'done'")
        return {r["path"] for r in cur}

    def rows(self, where: str = "", params: tuple = ()) -> Iterator[dict[str, Any]]:
        sql = "SELECT * FROM images"
        if where:
            sql += f" WHERE {where}"
        for row in self._conn.execute(sql, params):
            yield dict(row)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def hashes(self, column: str = "phash") -> list[tuple[str, str]]:
        """(hash, path) pairs for building a BK-tree."""
        if column not in FLAT_COLUMNS:
            raise ValueError(f"unknown column {column!r}")
        cur = self._conn.execute(
            f"SELECT {column} AS h, path FROM images WHERE {column} IS NOT NULL AND {column} != ''"
        )
        return [(r["h"], r["path"]) for r in cur]

    def exact_duplicates(self, column: str = "sha256") -> list[tuple[str, list[str]]]:
        """Groups of paths sharing an identical digest."""
        if column not in FLAT_COLUMNS:
            raise ValueError(f"unknown column {column!r}")
        cur = self._conn.execute(
            f"""SELECT {column} AS digest, GROUP_CONCAT(path, char(10)) AS paths
                FROM images WHERE {column} IS NOT NULL AND {column} != ''
                GROUP BY {column} HAVING COUNT(*) > 1"""
        )
        return [(r["digest"], r["paths"].split("\n")) for r in cur]

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            """SELECT COUNT(*) AS images,
                      SUM(findings_high) AS high,
                      SUM(findings_medium) AS medium,
                      COUNT(latitude) AS with_gps,
                      COUNT(DISTINCT camera) AS cameras,
                      COUNT(DISTINCT serial) AS serials
               FROM images"""
        ).fetchone()
        failed = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status = 'failed'"
        ).fetchone()[0]
        return {**{k: row[k] for k in row.keys()}, "failed": failed}


def _sql_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    return str(value)


def open_index(directory: Path, name: str = "index.db") -> ImageIndex:
    return ImageIndex(Path(directory) / name)


__all__ = ["ImageIndex", "open_index"]
