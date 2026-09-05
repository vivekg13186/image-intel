"""The public plugin SDK — the only imgintel API third parties should use.

Everything a plugin needs is re-exported here. Importing from
``imgintel.core.*`` instead couples your plugin to imgintel's internals, and
those move: `precheck` arrived in Phase 5 and `after` in Phase 3, both by
changing files under ``core``. This module is the promise that *these* names
keep working.

    from imgintel.plugins import Analyzer, Cost, Finding, Severity

    class AspectRatio(Analyzer):
        name = "aspect"
        version = "1.0.0"
        description = "Flags unusual aspect ratios"
        requires = ("fileinfo",)
        cost = Cost.CHEAP

        def analyze(self, ctx):
            ratio = ctx.data("fileinfo").get("aspect_ratio")
            findings = []
            if ratio and (ratio > 3 or ratio < 0.33):
                findings.append(
                    Finding("composition", "Unusual aspect ratio", ratio, Severity.LOW)
                )
            return self.ok({"aspect_ratio": ratio}, findings)

Register it by shipping a package with an ``imgintel.analyzers`` entry point:

    [project.entry-points."imgintel.analyzers"]
    aspect = "my_package.analyzer:AspectRatio"

``imgintel plugins new`` scaffolds all of this.

Stability
---------
:data:`API_VERSION` follows semantic versioning independently of imgintel's own
version. Within a major version, everything exported here keeps working:
names are not removed, required arguments are not added, and return shapes do
not change. New optional hooks may appear — a plugin that does not implement
them keeps working, which is how ``precheck`` was added without breaking
anything.

Check it at import time if you depend on a recent addition::

    from imgintel.plugins import API_VERSION, require_api

    require_api("1.0")  # raises if this imgintel is too old
"""

from __future__ import annotations

#: Semantic version of *this interface*, not of imgintel.
API_VERSION = "1.0"

from imgintel.core.analyzer import (  # noqa: E402
    Analyzer,
    AnalyzerResult,
    Availability,
    Cost,
    Finding,
    Severity,
    Status,
)
from imgintel.core.context import AnalysisContext, CaseConfig  # noqa: E402
from imgintel.core.deps import binary_path, has_module, module_version  # noqa: E402

# Helpers a non-trivial analyzer is likely to want. Re-exported so a plugin
# never has to reach into a private module for the same maths the built-ins
# use — and so these can be reimplemented without breaking anyone.
from imgintel.util.imgmath import (  # noqa: E402
    bits_to_hex,
    hamming_hex,
    laplacian_variance,
    shannon_entropy,
)


class ApiVersionError(RuntimeError):
    """Raised when the installed imgintel is older than a plugin requires."""


def require_api(minimum: str) -> None:
    """Fail loudly at import time if this imgintel predates what you need.

    Better than a mystifying ``AttributeError`` three layers into a run, and
    it names both versions so the fix is obvious.
    """
    def parts(value: str) -> tuple[int, ...]:
        return tuple(int(p) for p in value.split(".")[:2])

    if parts(API_VERSION) < parts(minimum):
        raise ApiVersionError(
            f"this plugin needs imgintel plugin API >= {minimum}, "
            f"but the installed imgintel provides {API_VERSION}"
        )
    if parts(API_VERSION)[0] != parts(minimum)[0]:
        raise ApiVersionError(
            f"this plugin targets plugin API {minimum}, which is not compatible "
            f"with the installed API {API_VERSION}"
        )


__all__ = [
    "API_VERSION",
    "AnalysisContext",
    "Analyzer",
    "AnalyzerResult",
    "ApiVersionError",
    "Availability",
    "CaseConfig",
    "Cost",
    "Finding",
    "Severity",
    "Status",
    "binary_path",
    "bits_to_hex",
    "hamming_hex",
    "has_module",
    "laplacian_variance",
    "module_version",
    "require_api",
    "shannon_entropy",
]
