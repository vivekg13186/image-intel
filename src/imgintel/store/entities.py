"""The entity database: enrolled identities and objects, and matching against them.

Two kinds of entity share one store because the workflow is identical — enrol
some references, then ask "is this in the set?":

* **person** — face embeddings. Biometric data, gated (see :class:`CaseConfig`).
* **object** — ORB descriptors of a specific thing: a vehicle, a bag, a mark.

Matching uses a **threshold *and* a margin**. A threshold alone answers "does
this resemble Alice?", which in a database of a thousand people is nearly
always yes for someone. The margin asks the question that matters — "does it
resemble Alice *distinctly more* than anyone else?" — and returns
``inconclusive`` when it does not. An identification tool that cannot say "I
don't know" is not usable as evidence.

No vector-database dependency: brute-force cosine over a few thousand
normalised vectors is well under a millisecond in numpy, exact rather than
approximate, and trivial to audit. Graduating to sqlite-vec or faiss makes
sense past ~10^6 vectors, which is not this tool's scale.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = 1
EntityKind = Literal["person", "object"]

#: Cosine similarity below this is never a match, whatever the margin.
DEFAULT_THRESHOLD = 0.363  # SFace's published same-identity threshold
#: The best candidate must beat the runner-up by this much to be conclusive.
DEFAULT_MARGIN = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise so a dot product is the cosine similarity."""
    array = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(array))
    return array / norm if norm > 0 else array


@dataclass(slots=True)
class Entity:
    id: int
    name: str
    kind: EntityKind
    notes: str | None = None
    reference_count: int = 0
    created_utc: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Match:
    """One candidate, with everything needed to judge it."""

    entity: str
    entity_id: int
    kind: EntityKind
    similarity: float
    margin: float
    #: "match" | "inconclusive" | "no match"
    verdict: str
    threshold: float
    runner_up: str | None = None
    runner_up_similarity: float | None = None
    reference_id: int | None = None

    @property
    def conclusive(self) -> bool:
        return self.verdict == "match"

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_id": self.entity_id,
            "kind": self.kind,
            "similarity": round(self.similarity, 4),
            "margin": round(self.margin, 4),
            "verdict": self.verdict,
            "threshold": self.threshold,
            "runner_up": self.runner_up,
            "runner_up_similarity": (
                None if self.runner_up_similarity is None else round(self.runner_up_similarity, 4)
            ),
            "reference_id": self.reference_id,
        }


class EntityStore:
    """Sqlite-backed entity database with an in-memory match index."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create()
        self._cache: tuple[np.ndarray, list[sqlite3.Row]] | None = None

    # -- lifecycle ---------------------------------------------------------

    def _create(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    notes TEXT,
                    metadata TEXT,
                    created_utc TEXT NOT NULL,
                    UNIQUE(name, kind)
                );
                CREATE TABLE IF NOT EXISTS refs (
                    id INTEGER PRIMARY KEY,
                    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    vector BLOB NOT NULL,
                    dims INTEGER NOT NULL,
                    embedder TEXT NOT NULL,
                    source_path TEXT,
                    source_sha256 TEXT,
                    bbox TEXT,
                    enrolled_utc TEXT NOT NULL,
                    enrolled_by TEXT
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY,
                    action TEXT NOT NULL,
                    detail TEXT,
                    operator TEXT,
                    case_id TEXT,
                    at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
                CREATE INDEX IF NOT EXISTS idx_refs_entity ON refs(entity_id);
                CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);
                """
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EntityStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- enrolment ---------------------------------------------------------

    def add_entity(
        self,
        name: str,
        kind: EntityKind,
        *,
        notes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        row = self._conn.execute(
            "SELECT id FROM entities WHERE name = ? AND kind = ?", (name, kind)
        ).fetchone()
        if row is not None:
            return int(row["id"])
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO entities (name, kind, notes, metadata, created_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, kind, notes, json.dumps(metadata or {}), _now()),
            )
        return int(cursor.lastrowid)

    def add_reference(
        self,
        entity_id: int,
        vector: np.ndarray,
        *,
        embedder: str,
        source_path: str | None = None,
        source_sha256: str | None = None,
        bbox: list[int] | None = None,
        enrolled_by: str | None = None,
    ) -> int:
        """Store one reference vector, with the provenance of where it came from.

        Provenance is not optional bookkeeping: "which image was this identity
        built from" is the first question anyone will ask about a match.
        """
        unit = normalise(vector)
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO refs (entity_id, vector, dims, embedder, source_path, "
                "source_sha256, bbox, enrolled_utc, enrolled_by) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    entity_id,
                    unit.astype(np.float32).tobytes(),
                    len(unit),
                    embedder,
                    source_path,
                    source_sha256,
                    json.dumps(bbox) if bbox else None,
                    _now(),
                    enrolled_by,
                ),
            )
        self._cache = None
        return int(cursor.lastrowid)

    def forget(self, name: str, kind: EntityKind | None = None) -> int:
        """Delete an entity and its references. Returns rows removed.

        A subject-access or erasure request has to be actionable, so removal is
        real deletion rather than a soft-delete flag.
        """
        query = "DELETE FROM entities WHERE name = ?"
        params: tuple = (name,)
        if kind:
            query += " AND kind = ?"
            params = (name, kind)
        with self._conn:
            cursor = self._conn.execute(query, params)
        self._cache = None
        return cursor.rowcount

    def log(
        self,
        action: str,
        detail: str | None = None,
        *,
        operator: str | None = None,
        case_id: str | None = None,
    ) -> None:
        """Append to the audit trail. Every enrolment and search is recorded."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit (action, detail, operator, case_id, at_utc) VALUES (?,?,?,?,?)",
                (action, detail, operator, case_id, _now()),
            )

    # -- reading -----------------------------------------------------------

    def entities(self, kind: EntityKind | None = None) -> list[Entity]:
        query = (
            "SELECT e.*, COUNT(r.id) AS n FROM entities e "
            "LEFT JOIN refs r ON r.entity_id = e.id"
        )
        params: tuple = ()
        if kind:
            query += " WHERE e.kind = ?"
            params = (kind,)
        query += " GROUP BY e.id ORDER BY e.name"
        return [
            Entity(
                id=int(row["id"]),
                name=row["name"],
                kind=row["kind"],
                notes=row["notes"],
                reference_count=int(row["n"]),
                created_utc=row["created_utc"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in self._conn.execute(query, params)
        ]

    def audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._conn.execute(
                "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,)
            )
        ]

    def count(self, kind: EntityKind | None = None) -> int:
        if kind:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM entities WHERE kind = ?", (kind,)
                ).fetchone()[0]
            )
        return int(self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0])

    def embedders(self) -> set[str]:
        return {
            row[0] for row in self._conn.execute("SELECT DISTINCT embedder FROM refs")
        }

    def raw_references(self, kind: EntityKind, embedder: str) -> list[dict[str, Any]]:
        """Reference rows with their raw payloads.

        For backends whose reference is not a comparable vector — ORB stores
        descriptors plus keypoint geometry — so they can rebuild their own
        structures instead of going through the cosine index.
        """
        return [
            {
                "id": int(row["id"]),
                "entity_id": int(row["entity_id"]),
                "name": row["name"],
                "payload": row["vector"],
                "bbox": json.loads(row["bbox"]) if row["bbox"] else None,
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
            }
            for row in self._conn.execute(
                "SELECT r.id, r.entity_id, r.vector, r.bbox, r.source_path, "
                "r.source_sha256, e.name FROM refs r JOIN entities e ON e.id = r.entity_id "
                "WHERE e.kind = ? AND r.embedder = ? ORDER BY r.id",
                (kind, embedder),
            )
        ]

    def add_raw_reference(
        self,
        entity_id: int,
        payload: bytes,
        *,
        embedder: str,
        source_path: str | None = None,
        source_sha256: str | None = None,
        bbox: list[int] | None = None,
        enrolled_by: str | None = None,
    ) -> int:
        """Store an opaque reference payload (see :meth:`raw_references`)."""
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO refs (entity_id, vector, dims, embedder, source_path, "
                "source_sha256, bbox, enrolled_utc, enrolled_by) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    entity_id,
                    payload,
                    0,  # not a fixed-dimension vector
                    embedder,
                    source_path,
                    source_sha256,
                    json.dumps(bbox) if bbox else None,
                    _now(),
                    enrolled_by,
                ),
            )
        self._cache = None
        return int(cursor.lastrowid)

    # -- matching ----------------------------------------------------------

    def _matrix(self, kind: EntityKind, embedder: str) -> tuple[np.ndarray, list[sqlite3.Row]]:
        """All reference vectors for a kind/embedder as one array, cached."""
        rows = list(
            self._conn.execute(
                "SELECT r.id, r.entity_id, r.vector, r.dims, e.name, e.kind "
                "FROM refs r JOIN entities e ON e.id = r.entity_id "
                "WHERE e.kind = ? AND r.embedder = ? ORDER BY r.id",
                (kind, embedder),
            )
        )
        if not rows:
            return np.zeros((0, 0), dtype=np.float32), []
        dims = int(rows[0]["dims"])
        matrix = np.zeros((len(rows), dims), dtype=np.float32)
        for i, row in enumerate(rows):
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            if len(vector) != dims:
                continue  # dimension drift from a different model; skip it
            matrix[i] = vector
        return matrix, rows

    def search(
        self,
        vector: np.ndarray,
        *,
        kind: EntityKind,
        embedder: str,
        threshold: float = DEFAULT_THRESHOLD,
        margin: float = DEFAULT_MARGIN,
        top_k: int = 5,
    ) -> tuple[Match | None, list[dict[str, Any]]]:
        """Best candidate (or None) plus the ranked shortlist.

        The shortlist is returned even when the verdict is inconclusive: an
        analyst needs to see the near-misses to judge the call, and hiding them
        would make a marginal result look decisive.
        """
        matrix, rows = self._matrix(kind, embedder)
        if not len(rows):
            return None, []

        query = normalise(vector)
        if query.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"embedding is {query.shape[0]}-dimensional but the database holds "
                f"{matrix.shape[1]}-dimensional vectors for {embedder}"
            )

        similarities = matrix @ query
        order = np.argsort(similarities)[::-1]

        # Rank by entity, not by reference: five photos of Alice are one
        # candidate, and letting them occupy the whole shortlist would hide
        # the runner-up that the margin test depends on.
        best_per_entity: dict[int, tuple[float, sqlite3.Row]] = {}
        for index in order:
            row = rows[int(index)]
            entity_id = int(row["entity_id"])
            score = float(similarities[int(index)])
            if entity_id not in best_per_entity or score > best_per_entity[entity_id][0]:
                best_per_entity[entity_id] = (score, row)

        ranked = sorted(best_per_entity.values(), key=lambda pair: -pair[0])
        shortlist = [
            {
                "entity": row["name"],
                "entity_id": int(row["entity_id"]),
                "similarity": round(score, 4),
                "reference_id": int(row["id"]),
            }
            for score, row in ranked[:top_k]
        ]

        top_score, top_row = ranked[0]
        runner_up_score = ranked[1][0] if len(ranked) > 1 else None
        gap = top_score - runner_up_score if runner_up_score is not None else float("inf")

        if top_score < threshold:
            verdict = "no match"
        elif gap < margin:
            verdict = "inconclusive"
        else:
            verdict = "match"

        return (
            Match(
                entity=top_row["name"],
                entity_id=int(top_row["entity_id"]),
                kind=kind,
                similarity=top_score,
                margin=0.0 if gap == float("inf") else gap,
                verdict=verdict,
                threshold=threshold,
                runner_up=ranked[1][1]["name"] if len(ranked) > 1 else None,
                runner_up_similarity=runner_up_score,
                reference_id=int(top_row["id"]),
            ),
            shortlist,
        )

    def stats(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM entities) AS entities, "
            "(SELECT COUNT(*) FROM refs) AS refs, "
            "(SELECT COUNT(*) FROM entities WHERE kind='person') AS people, "
            "(SELECT COUNT(*) FROM entities WHERE kind='object') AS objects"
        ).fetchone()
        return {**{k: row[k] for k in row.keys()}, "embedders": sorted(self.embedders())}


def open_entities(path: str | Path) -> EntityStore:
    return EntityStore(path)


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_THRESHOLD",
    "Entity",
    "EntityStore",
    "Match",
    "normalise",
    "open_entities",
]
