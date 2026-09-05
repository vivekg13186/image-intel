"""Reverse image search — prepared, not performed.

Every reverse-search engine requires either uploading the image or having it
already on a public URL. imgintel does neither by default, for a reason that
matters in this domain: uploading evidence to a third party is a disclosure
decision, and it belongs to the investigator rather than to a tool run.

So this analyzer prepares the search instead of running it. It emits
ready-to-use URLs for the major engines, the file digests those engines index
by, and enough context to paste into a manual search. Everything is offline;
nothing leaves the machine.

The provider interface exists for anyone who wants API-backed lookups. A
provider must declare ``needs_network``, and the pipeline will refuse to run it
without ``--allow-network`` — the same gate every other network-touching
analyzer passes through.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Protocol

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext


class SearchProvider(Protocol):
    """An engine that can look up an image.

    Implement and register via the ``imgintel.search_providers`` entry point to
    add API-backed search. ``search`` is only ever called when the run
    permitted network access.
    """

    name: str
    needs_network: bool

    def url_for(self, image_url: str) -> str:
        """A URL a human can open to run this search themselves."""
        ...

    def search(self, image_bytes: bytes) -> list[dict[str, Any]]:
        """Perform the search. Only called with --allow-network."""
        ...


#: Engines whose search-by-URL form is stable enough to construct.
_ENGINES: tuple[tuple[str, str, str], ...] = (
    (
        "Google Lens",
        "https://lens.google.com/uploadbyurl?url={url}",
        "Broadest index; good for products, landmarks and stock imagery",
    ),
    (
        "Yandex",
        "https://yandex.com/images/search?rpt=imageview&url={url}",
        "Often strongest on faces and on imagery from Eastern Europe",
    ),
    (
        "Bing Visual Search",
        "https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{url}",
        "Good product and page-context coverage",
    ),
    (
        "TinEye",
        "https://tineye.com/search?url={url}",
        "Indexes exact and edited copies; sorts by first-seen date, which is what "
        "establishes whether an image predates the event it is claimed to show",
    ),
)

#: Engines that only accept an upload, with the page to drop the file onto.
_UPLOAD_ONLY: tuple[tuple[str, str], ...] = (
    ("Google Images", "https://images.google.com/"),
    ("Baidu", "https://graph.baidu.com/"),
)


class ReverseSearchAnalyzer(Analyzer):
    name = "reversesearch"
    version = "1.0.0"
    title = "Reverse image search"
    description = "Prepares reverse-search links and digests; performs no network requests"
    # Not marked needs_network: preparing links is entirely offline. A provider
    # that actually searches declares it for itself.
    cost = Cost.CHEAP
    after = ("hashes", "perceptual")

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        hashes = ctx.data("hashes")
        perceptual = ctx.data("perceptual")

        # Search-by-URL needs the image to already be reachable. When it is
        # only a local file, say so plainly rather than emitting URLs that
        # cannot work.
        engines = [
            {
                "engine": name,
                "url_template": template,
                "note": note,
                "requires_public_url": True,
            }
            for name, template, note in _ENGINES
        ]

        data: dict[str, Any] = {
            "performed": False,
            "reason": "reverse search requires uploading the image or hosting it publicly; "
            "imgintel prepares the search and leaves that decision to the operator",
            "engines": engines,
            "upload_pages": [
                {"engine": name, "url": url} for name, url in _UPLOAD_ONLY
            ],
            "digests": {
                "sha256": hashes.get("sha256"),
                "pixel_sha256": perceptual.get("pixel_sha256"),
                "phash": perceptual.get("phash"),
            },
            "providers_available": [],
        }

        findings = [
            Finding(
                "provenance",
                "Reverse search prepared, not performed",
                f"{len(engines)} engine(s)",
                Severity.INFO,
                detail="No request was made and the image was not uploaded anywhere. Use the "
                "links in the JSON output, or drop the file onto one of the upload pages. "
                "TinEye's first-seen date is the useful one for establishing whether an "
                "image predates what it is claimed to show",
            )
        ]
        return self.ok(data, findings)


def urls_for(image_url: str) -> list[dict[str, str]]:
    """Concrete search URLs for an image that is already publicly reachable."""
    quoted = urllib.parse.quote(image_url, safe="")
    return [
        {"engine": name, "url": template.format(url=quoted), "note": note}
        for name, template, note in _ENGINES
    ]


__all__ = ["ReverseSearchAnalyzer", "SearchProvider", "urls_for"]
