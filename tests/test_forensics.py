"""Manipulation indicators, location estimation, reverse search, summaries."""

from __future__ import annotations

import numpy as np
import pytest

from imgintel.core.deps import has_module
from imgintel.core.engine import analyze_image
from imgintel.util.forensics import (
    dct_first_digit_deviation,
    error_level,
    jpeg_ghost_map,
    jpeg_ghost_sweep,
    noise_field,
    outlier_tiles,
    tiled_stats,
)
from imgintel.util.geocues import cue_from_language, cues_from_text, rank


def _needs(module: str):
    return pytest.mark.skipif(not has_module(module), reason=f"{module} not installed")


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer}: {block.error}"
    return block.data


def labels(doc) -> set[str]:
    return {f.label for f in doc.findings}


def raised(doc) -> set[str]:
    return set(data_of(doc, "tamper").get("indicators_raised", []))


# -- the false-positive control --------------------------------------------
#
# The single most important test here. A tamper detector that fires on clean
# images is worse than no detector: it trains its user to ignore it.


def test_clean_image_raises_no_indicators(images):
    doc = analyze_image(images["clean"], only=["tamper"])
    assert raised(doc) == set(), f"clean image raised {raised(doc)}"


def test_clean_image_produces_no_manipulation_findings(images):
    doc = analyze_image(images["clean"], only=["tamper"])
    manipulation = [f for f in doc.findings if f.category == "manipulation"]
    assert not manipulation, [f.label for f in manipulation]


@pytest.mark.parametrize("fixture", ["photo", "sharp", "clean"])
def test_ordinary_images_stay_quiet(images, fixture):
    doc = analyze_image(images[fixture], only=["tamper"])
    assert "Multiple independent indicators raised" not in labels(doc)


# -- true positives --------------------------------------------------------


def test_splice_raises_several_independent_indicators(images):
    """A region from a differently-compressed file should show up three ways."""
    doc = analyze_image(images["spliced"], only=["tamper"])
    indicators = raised(doc)

    assert "jpeg_ghost" in indicators, "the localised recompression ghost is the key signal"
    assert "noise" in indicators, "a foreign region carries a foreign noise floor"
    assert len(indicators) >= 3
    assert "Multiple independent indicators raised" in labels(doc)


def test_splice_ghost_recovers_the_donor_quality(images):
    """The donor was compressed at q=35; the ghost should say so."""
    doc = analyze_image(images["spliced"], only=["tamper"])
    ghost = data_of(doc, "tamper")["indicators"]["jpeg_ghost"]

    assert ghost["raised"]
    assert not ghost["whole_image"], "a splice is localised, not frame-wide"
    qualities = [t["ghost_quality"] for t in ghost["ghost_tiles"]]
    assert any(30 <= q <= 45 for q in qualities), qualities


def test_splice_ghost_localises_the_region(images):
    """Saying *where* is what makes this more than a global alarm."""
    doc = analyze_image(images["spliced"], only=["tamper"])
    finding = next(f for f in doc.findings if f.label == "Localised recompression ghost")

    assert finding.bbox is not None
    x0, y0, x1, y1 = finding.bbox
    # Source splice was x 250-520, y 180-360 at full size; analysis space is
    # scaled but the region must still land in the upper-middle of the frame.
    data = data_of(doc, "tamper")
    width, height = data["analysis_dimensions"]
    assert 0.2 * width <= x0 <= 0.75 * width, (x0, width)
    assert 0.15 * height <= y0 <= 0.7 * height, (y0, height)


@_needs("cv2")
def test_copy_move_is_detected_with_its_offset(images):
    doc = analyze_image(images["copymove"], only=["tamper"])
    assert "copy_move" in raised(doc)

    candidate = data_of(doc, "tamper")["indicators"]["copy_move"]["candidates"][0]
    # Clone was from (120,100) to (430,330): a displacement of (310, 230),
    # scaled into analysis space and possibly reported in either direction.
    dx, dy = (abs(v) for v in candidate["offset"])
    assert 200 <= dx <= 340, candidate["offset"]
    assert 140 <= dy <= 260, candidate["offset"]
    assert candidate["pairs"] >= 12


@_needs("cv2")
def test_copy_move_does_not_fire_on_clean_content(images):
    assert "copy_move" not in raised(analyze_image(images["clean"], only=["tamper"]))


def test_double_compression_shows_in_dct_statistics(images):
    doc = analyze_image(images["doublejpeg"], only=["tamper"])
    assert "dct_first_digit" in raised(doc)


# -- forensics primitives --------------------------------------------------


def test_ghost_sweep_recovers_a_known_save_quality(images):
    """Recompressing at an image's own quality changes it least."""
    from PIL import Image

    with Image.open(images["clean"]) as raw:
        rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)

    import io

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        at70 = np.asarray(reopened.convert("RGB"), dtype=np.uint8)

    result = jpeg_ghost_sweep(at70)
    assert abs(result.minimum_quality - 70) <= 5, result.minimum_quality


def test_ghost_map_needs_local_minima_not_the_global_one():
    """The correction that made ghosts work: argmin only finds the last save.

    A monotonically falling curve has its minimum at the top of the sweep and
    contains no ghost. A curve with an interior dip does.
    """
    from imgintel.util.forensics import _local_minimum

    qualities = tuple(range(30, 100, 5))
    monotonic = np.linspace(5.0, 1.0, len(qualities)).astype(np.float32)
    assert _local_minimum(monotonic, qualities) == (None, 0.0)

    dipped = monotonic.copy()
    dipped[2] = 0.5  # a sharp dip at q=40
    quality, depth = _local_minimum(dipped, qualities)
    assert quality == 40
    assert depth > 0.15


def test_ghost_map_is_empty_on_a_single_save(images):
    from PIL import Image

    with Image.open(images["clean"]) as raw:
        rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    result = jpeg_ghost_map(rgb)
    assert result["available"]
    assert result["ghost_count"] == 0


def test_outlier_tiles_uses_a_robust_threshold():
    """MAD, not standard deviation — a real splice would inflate an SD."""
    values = np.array([1.0] * 30 + [50.0], dtype=np.float32)
    tiles = [{"bbox": [0, 0, 1, 1], "value": float(v)} for v in values]
    outliers = outlier_tiles(values, tiles, z=3.0)
    assert len(outliers) == 1
    assert outliers[0]["value"] == 50.0


def test_outlier_tiles_finds_nothing_in_uniform_data():
    values = np.full(64, 5.0, dtype=np.float32)
    tiles = [{"bbox": [0, 0, 1, 1], "value": 5.0} for _ in values]
    assert outlier_tiles(values, tiles) == []


def test_error_level_and_noise_fields_have_matching_geometry(images):
    from PIL import Image

    with Image.open(images["clean"]) as raw:
        rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    gray = rgb.mean(axis=2).astype(np.float32)

    _, ela_tiles = tiled_stats(error_level(rgb))
    _, noise_tiles = noise_field(gray)
    assert len(ela_tiles) == len(noise_tiles) == 64
    assert ela_tiles[0]["bbox"] == noise_tiles[0]["bbox"]


def test_benford_divergence_separates_single_from_double(images):
    """Calibration check for the threshold the analyzer uses."""
    from PIL import Image

    from imgintel.analyzers.tamper import BENFORD_SUSPICIOUS

    def gray_of(path):
        with Image.open(path) as raw:
            return np.asarray(raw.convert("RGB"), dtype=np.float32).mean(axis=2)

    single = dct_first_digit_deviation(gray_of(images["clean"]))
    double = dct_first_digit_deviation(gray_of(images["doublejpeg"]))

    assert single < BENFORD_SUSPICIOUS < double, (single, double)


# -- content gating --------------------------------------------------------


def test_pixel_forensics_are_skipped_on_synthetic_images(images):
    """None of these measures mean anything on a screenshot or a QR code."""
    doc = analyze_image(images["document"], only=["tamper"])
    data = data_of(doc, "tamper")
    assert data["content_type"] == "graphic"
    assert "skipped_reason" in data
    assert "Pixel forensics not applicable" in labels(doc)


def test_tamper_never_declares_an_image_fake(images):
    """The wording is part of the contract, not decoration."""
    doc = analyze_image(images["spliced"], only=["tamper"])
    forbidden = ("fake", "forged", "manipulated image", "authentic", "genuine")
    for finding in doc.findings:
        text = f"{finding.label} {finding.value} {finding.detail or ''}".lower()
        for word in forbidden:
            assert word not in text, f"{finding.label!r} contains {word!r}"
    assert "disclaimer" in data_of(doc, "tamper")


# -- geolocation cues ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "country", "kind"),
    [
        ("Call us on +44 20 7946 0958", "GB", "phone"),
        ("Tel +33 1 42 68 53 00", "FR", "phone"),
        ("12 Harbour Road, Bristol BS1 5TY", "GB", "postcode"),
        ("Toronto ON M5V 3L9", "CA", "postcode"),
        ("visit acme-logistics.co.uk today", "GB", "domain"),
        ("Musterfirma GmbH", "DE", "business"),
        ("Wollongong Pty Ltd", "AU", "business"),
        ("Hauptstrasse 14", "DE", "address"),
        ("Total: £42.50", "GB", "currency"),
    ],
)
def test_country_cues_are_recognised(text, country, kind):
    cues = cues_from_text(text)
    assert any(c.country == country and c.kind == kind for c in cues), [
        (c.country, c.kind, c.evidence) for c in cues
    ]


def test_shared_dialling_codes_are_reported_as_ambiguous():
    """+1 covers the US and Canada; claiming one would be a fabrication."""
    countries = {c.country for c in cues_from_text("+1 415 555 0100") if c.kind == "phone"}
    assert countries == {"US", "CA"}
    weights = [c.weight for c in cues_from_text("+1 415 555 0100") if c.kind == "phone"]
    assert all(w < 0.5 for w in weights), "an ambiguous code must carry low weight"


def test_longest_dialling_prefix_wins():
    """+353 is Ireland, not '+3' something."""
    countries = {c.country for c in cues_from_text("+353 1 234 5678") if c.kind == "phone"}
    assert countries == {"IE"}


def test_independent_cues_combine_without_summing_past_one():
    cues = cues_from_text(
        "ACME Ltd, 12 Harbour Road, Bristol BS1 5TY. Tel +44 20 7946 0958. acme.co.uk"
    )
    hypotheses = rank(cues)
    assert hypotheses[0]["country"] == "GB"
    assert 0.0 < hypotheses[0]["confidence"] <= 1.0
    assert len(hypotheses[0]["evidence"]) >= 3


def test_repeating_one_weak_cue_does_not_manufacture_certainty():
    """Ten copies of the same hint are one observation, not ten."""
    once = rank(cues_from_text("Inc."))
    many = rank(cues_from_text(" ".join(["Inc."] * 10)))
    assert once[0]["confidence"] == many[0]["confidence"]


def test_language_is_only_a_weak_prior():
    cue = cue_from_language("German")
    assert cue is not None and cue.country == "DE"
    assert cue.weight <= 0.3, "language spans borders; it must not dominate"
    assert cue_from_language("English") is None, "English is far too widespread to place"


def test_hypotheses_carry_their_evidence():
    hypotheses = rank(cues_from_text("Bristol BS1 5TY +44 20 7946 0958"))
    top = hypotheses[0]
    assert top["evidence"], "a hypothesis with no visible evidence is unusable"
    for item in top["evidence"]:
        assert item["kind"] and item["text"] and item["detail"]


# -- geoestimate analyzer --------------------------------------------------


def test_gps_settles_the_question(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    data = data_of(analyze_image(images["gps"], profile="standard"), "geoestimate")
    assert data["resolved"] is True
    assert data["source"] == "exif_gps"
    assert not data["hypotheses"], "no need to guess around a known coordinate"


def test_no_evidence_reports_nothing_rather_than_guessing(images):
    doc = analyze_image(images["clean"], only=["geoestimate"])
    data = data_of(doc, "geoestimate")
    assert data["resolved"] is False
    assert data["hypotheses"] == []
    assert "Location not estimated" in labels(doc)


def test_geoestimate_is_cheap_and_needs_no_model(registry):
    from imgintel.core.analyzer import Cost

    assert registry.get("geoestimate").cost is Cost.CHEAP
    assert not registry.get("geoestimate").needs_models


# -- reverse search --------------------------------------------------------


def test_reverse_search_performs_no_network_request(images):
    """Uploading evidence is the operator's decision, not a tool's."""
    doc = analyze_image(images["clean"], only=["reversesearch"])
    data = data_of(doc, "reversesearch")

    assert data["performed"] is False
    assert data["engines"]
    assert not registry_needs_network()
    assert "prepared, not performed" in " ".join(labels(doc)).lower()


def registry_needs_network() -> bool:
    from imgintel.core.registry import Registry

    return Registry.discover().get("reversesearch").needs_network


def test_reverse_search_urls_are_well_formed():
    from urllib.parse import urlparse

    from imgintel.analyzers.reversesearch import urls_for

    entries = urls_for("https://example.com/a b.jpg")
    assert len(entries) >= 4
    for entry in entries:
        parsed = urlparse(entry["url"])
        assert parsed.scheme == "https" and parsed.netloc
        assert " " not in entry["url"], "the image URL must be percent-encoded"


def test_reverse_search_exposes_digests_for_manual_lookup(images):
    doc = analyze_image(images["clean"], only=["reversesearch", "hashes", "perceptual"])
    digests = data_of(doc, "reversesearch")["digests"]
    assert len(digests["sha256"]) == 64
    assert digests["phash"]


# -- summary ---------------------------------------------------------------


def test_summary_is_deterministic(images):
    """No language model, so the same image must produce the same words."""
    first = data_of(analyze_image(images["document"], profile="standard"), "summary")
    second = data_of(analyze_image(images["document"], profile="standard"), "summary")
    assert first["text"] == second["text"]


def test_summary_headline_names_the_most_significant_finding(images):
    doc = analyze_image(images["document"], profile="deep")
    headline = data_of(doc, "summary")["headline"]
    assert images["document"].name in headline
    assert "sensitive" in headline.lower()


def test_summary_only_reports_what_was_measured(images):
    """Every claim must trace to a finding, since nothing here is generated."""
    doc = analyze_image(images["document"], profile="standard")
    data = data_of(doc, "summary")

    measured = {f.label for result in doc.analyzers.values() for f in []}  # placeholder
    _ = measured
    all_labels = {f.label for f in doc.findings}
    for section in data["sections"]:
        for point in section["points"]:
            label = point.split(":")[0].strip()
            assert any(label in existing for existing in all_labels), point


def test_summary_carries_the_indicator_caveat(images):
    text = data_of(analyze_image(images["clean"], profile="standard"), "summary")["text"]
    assert "indicators" in text.lower()
    assert "generated mechanically" in text.lower()


def test_summary_handles_an_image_with_nothing_notable(images):
    doc = analyze_image(images["clean"], only=["summary", "fileinfo"])
    assert data_of(doc, "summary")["headline"]


def test_summary_runs_last(registry):
    """It summarises the run, so everything else must already have reported."""
    from imgintel.core.engine import plan
    from imgintel.core.pipeline import topo_sort

    order = [a.name for a in topo_sort(plan(registry, profile="standard"))]
    assert order[-1] == "summary", order[-3:]
