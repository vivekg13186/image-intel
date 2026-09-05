"""Execution profiles.

Without these, a user asking "what camera took this?" pays the full deep-scan
cost. Profiles are defined by cost ceiling rather than by an explicit analyzer
list, so a newly installed plugin lands in the right tier automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from imgintel.core.analyzer import Analyzer, Cost


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    max_cost: Cost
    allow_network: bool
    description: str

    def includes(self, analyzer: Analyzer) -> bool:
        if analyzer.cost.rank > self.max_cost.rank:
            return False
        if analyzer.needs_network and not self.allow_network:
            return False
        return True


PROFILES: dict[str, Profile] = {
    "quick": Profile(
        name="quick",
        max_cost=Cost.CHEAP,
        allow_network=False,
        description="Metadata only — no pixel decode. Milliseconds per image.",
    ),
    "standard": Profile(
        name="standard",
        max_cost=Cost.MEDIUM,
        allow_network=False,
        description="Metadata plus pixel-level analysis. The sensible default.",
    ),
    "deep": Profile(
        name="deep",
        max_cost=Cost.HEAVY,
        allow_network=False,
        description="Everything local, including model inference and forensics.",
    ),
}

DEFAULT_PROFILE = "standard"


def select(analyzers: list[Analyzer], profile: str) -> list[Analyzer]:
    """Analyzers in this profile, with unsatisfiable ones removed.

    A cost ceiling can strand an analyzer: `sensitive` is MEDIUM but requires
    `ocr`, which is HEAVY. Including it in `standard` would list an analyzer
    that can only ever report "dependency not selected" — noise in every
    report, and a profile listing that lies about what will run.

    Dropping is iterative because a dependency chain can be several deep.
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; choose from {', '.join(PROFILES)}")
    prof = PROFILES[profile]
    chosen = {a.name: a for a in analyzers if prof.includes(a)}

    while True:
        stranded = {
            name
            for name, analyzer in chosen.items()
            if any(dep not in chosen for dep in analyzer.requires)
        }
        if not stranded:
            return [chosen[n] for n in sorted(chosen)]
        for name in stranded:
            del chosen[name]
