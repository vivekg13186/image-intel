"""Cryptographic digests over the file bytes.

These identify the *file*. The perceptual analyzer identifies the *picture* —
they answer different questions and both matter.
"""

from __future__ import annotations

import hashlib

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext

_ALGOS = ("md5", "sha1", "sha256", "blake2b")
_CHUNK = 1 << 20  # 1 MiB


class HashAnalyzer(Analyzer):
    name = "hashes"
    version = "1.0.0"
    title = "Cryptographic hashes"
    description = "MD5/SHA-1/SHA-256/BLAKE2b over the raw file bytes"
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        digests = {name: hashlib.new(name) for name in _ALGOS}
        with ctx.path.open("rb") as fh:
            while chunk := fh.read(_CHUNK):
                for d in digests.values():
                    d.update(chunk)

        data = {name: d.hexdigest() for name, d in digests.items()}
        data["blake2b"] = data["blake2b"][:64]  # truncate 512-bit to 256-bit
        data["bytes_hashed"] = ctx.path.stat().st_size

        findings = [
            Finding(
                "identity",
                "SHA-256",
                data["sha256"],
                Severity.INFO,
                detail="Identifies this exact file; changes if any metadata is edited",
            )
        ]
        return self.ok(data, findings)
