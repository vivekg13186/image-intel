"""BK-tree over fixed-length hex hashes.

Near-duplicate search is a metric-space radius query, and Hamming distance
obeys the triangle inequality, so a BK-tree prunes it to roughly O(log n)
instead of the O(n^2) all-pairs comparison that makes dedupe unusable past a
few thousand images.

The tree is built in memory from the index each time; at the scale where that
stops being fine (millions of images) the right answer is a different index
entirely, not a persisted BK-tree.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from imgintel.util.imgmath import hamming_hex


@dataclass(slots=True)
class _Node:
    key: str
    items: list[Any] = field(default_factory=list)
    children: dict[int, _Node] = field(default_factory=dict)


class BKTree:
    """Metric tree supporting ``within(hash, radius)`` queries."""

    def __init__(self, entries: Iterable[tuple[str, Any]] = ()) -> None:
        self._root: _Node | None = None
        self._size = 0
        self._bits: int | None = None
        for key, payload in entries:
            self.add(key, payload)

    def __len__(self) -> int:
        return self._size

    @property
    def bits(self) -> int | None:
        """Hash width in bits — the maximum meaningful radius."""
        return self._bits

    def add(self, key: str, payload: Any) -> None:
        key = key.lower()
        if self._bits is None:
            self._bits = len(key) * 4
        elif len(key) * 4 != self._bits:
            raise ValueError(
                f"hash width mismatch: tree holds {self._bits}-bit hashes, got {len(key) * 4}-bit"
            )

        self._size += 1
        if self._root is None:
            self._root = _Node(key, [payload])
            return

        node = self._root
        while True:
            distance = hamming_hex(node.key, key)
            if distance == 0:
                node.items.append(payload)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _Node(key, [payload])
                return
            node = child

    def within(self, key: str, radius: int) -> list[tuple[int, Any]]:
        """Every payload whose hash is within ``radius`` bits, nearest first."""
        if self._root is None:
            return []
        key = key.lower()
        if self._bits is not None and len(key) * 4 != self._bits:
            raise ValueError("hash width mismatch")

        found: list[tuple[int, Any]] = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            distance = hamming_hex(node.key, key)
            if distance <= radius:
                found.extend((distance, item) for item in node.items)
            # Triangle inequality: only children in this distance band can match.
            low, high = distance - radius, distance + radius
            stack.extend(
                child for d, child in node.children.items() if low <= d <= high
            )
        found.sort(key=lambda pair: pair[0])
        return found

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        if self._root is None:
            return
        stack = [self._root]
        while stack:
            node = stack.pop()
            for item in node.items:
                yield node.key, item
            stack.extend(node.children.values())


def cluster(
    entries: list[tuple[str, Any]], radius: int
) -> list[list[tuple[int, Any]]]:
    """Group entries into near-duplicate clusters on a single hash."""
    return cluster_multi([[key] for key, _ in entries], [payload for _, payload in entries], radius)


def cluster_multi(
    hash_sets: list[list[str | None]],
    payloads: list[Any],
    radius: int,
) -> list[list[tuple[int, Any]]]:
    """Cluster on several hashes at once, linking on *any* of them.

    Each perceptual hash fails on a different transform: pHash tolerates
    recompression but aliases on high-frequency patterns under downscaling,
    while dHash handles that case and is weaker elsewhere. Consulting only one
    throws away the reason for computing several — a plain resize can land 16
    bits away in pHash and 4 bits away in dHash, and calling that "a different
    image" is simply wrong.

    Two images are linked when *any* hash puts them within ``radius``, and
    clusters are the connected components of that graph (single linkage), so a
    chain of resize → recompress → crop stays together.
    """
    count = len(payloads)
    if not count:
        return []
    width = max(len(h) for h in hash_sets) if hash_sets else 0

    # One tree per hash position; payloads are indices so queries return them.
    neighbours: dict[int, set[int]] = {i: set() for i in range(count)}
    best: dict[tuple[int, int], int] = {}

    for column in range(width):
        tree = BKTree()
        present: list[tuple[int, str]] = []
        for i, hashes in enumerate(hash_sets):
            key = hashes[column] if column < len(hashes) else None
            if key:
                present.append((i, key))
        for i, key in present:
            tree.add(key, i)
        for i, key in present:
            for distance, j in tree.within(key, radius):
                if i == j:
                    continue
                neighbours[i].add(j)
                pair = (min(i, j), max(i, j))
                best[pair] = min(best.get(pair, distance), distance)

    seen: set[int] = set()
    clusters: list[list[tuple[int, Any]]] = []
    for i in range(count):
        if i in seen:
            continue
        group, queue = [], [i]
        seen.add(i)
        while queue:  # breadth-first walk over the neighbour graph
            current = queue.pop()
            group.append(current)
            for neighbour in neighbours.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        if len(group) > 1:
            anchor = min(group)
            clusters.append(
                sorted(
                    (
                        (0 if g == anchor else best.get((min(anchor, g), max(anchor, g)), radius),
                         payloads[g])
                        for g in group
                    ),
                    key=lambda pair: pair[0],
                )
            )
    return clusters


def build_index(entries: Iterable[tuple[str, Any]]) -> BKTree:
    """Convenience constructor, skipping entries with no hash."""
    return BKTree((key, payload) for key, payload in entries if key)


__all__ = ["BKTree", "build_index", "cluster", "cluster_multi", "hamming_hex"]
