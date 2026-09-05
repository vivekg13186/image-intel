"""Phase 1: each analyzer produces correct, meaningful output."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from imgintel.core.context import AnalysisContext
from imgintel.core.engine import analyze_image
from imgintel.util.imgmath import (
    colorfulness,
    dct2,
    estimate_noise_sigma,
    hamming_hex,
    laplacian_variance,
    shannon_entropy,
    srgb_to_lab,
)


def labels(doc) -> set[str]:
    return {f.label for f in doc.findings}


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer} did not run: {block.error}"
    return block.data


# -- fileinfo --------------------------------------------------------------


def test_reports_true_format_and_size(images):
    d = data_of(analyze_image(images["sharp"]), "fileinfo")
    assert d["format_actual"] == "JPEG"
    assert d["size_bytes"] == images["sharp"].stat().st_size
    assert d["width"] == 1200 and d["height"] == 900


def test_extension_mismatch_is_flagged_high(images):
    doc = analyze_image(images["mislabeled"])
    mismatch = next(f for f in doc.findings if f.label == "Extension does not match content")
    assert mismatch.severity == "high"
    assert "PNG" in str(mismatch.value)


def test_filename_provenance_is_recognized(images):
    doc = analyze_image(images["mislabeled"])
    assert "WhatsApp" in data_of(doc, "fileinfo")["filename_sources"]


def test_matching_extension_produces_no_mismatch(images):
    assert "Extension does not match content" not in labels(analyze_image(images["sharp"]))


# -- hashes ----------------------------------------------------------------


def test_file_hashes_match_hashlib(images):
    d = data_of(analyze_image(images["sharp"]), "hashes")
    expected = hashlib.sha256(images["sharp"].read_bytes()).hexdigest()
    assert d["sha256"] == expected
    assert len(d["md5"]) == 32 and len(d["sha1"]) == 40 and len(d["sha256"]) == 64


# -- perceptual ------------------------------------------------------------


def test_hashes_are_stable_and_correctly_sized(images):
    a = data_of(analyze_image(images["sharp"]), "perceptual")
    b = data_of(analyze_image(images["sharp"]), "perceptual")
    assert a["phash"] == b["phash"]
    assert len(a["phash"]) == 16  # 64 bits
    assert len(a["dhash"]) == 16


@pytest.mark.parametrize("size", [(600, 450), (300, 225)])
def test_phash_survives_rescaling_and_recompression(images, tmp_path, size):
    original = data_of(analyze_image(images["photo"]), "perceptual")["phash"]
    smaller = tmp_path / f"small_{size[0]}.jpg"
    Image.open(images["photo"]).resize(size, Image.LANCZOS).save(smaller, quality=70)
    rescaled = data_of(analyze_image(smaller), "perceptual")["phash"]
    assert hamming_hex(original, rescaled) <= 8, f"pHash should tolerate a downscale to {size}"


def test_phash_distinguishes_different_images(images):
    a = data_of(analyze_image(images["sharp"]), "perceptual")["phash"]
    b = data_of(analyze_image(images["aigen"]), "perceptual")["phash"]
    assert hamming_hex(a, b) > 8


def test_pixel_hash_ignores_metadata(images, tmp_path):
    """The point of the pixel digest: stripping EXIF must not change it.

    ``piexif.remove`` rewrites the metadata segments without touching the
    entropy-coded scan, so the decoded pixels are bit-identical while every
    byte-level hash changes.
    """
    piexif = pytest.importorskip("piexif")
    src = images.get("gps")
    if src is None:
        pytest.skip("no exif fixture")

    stripped = tmp_path / "stripped.jpg"
    stripped.write_bytes(src.read_bytes())
    piexif.remove(str(stripped))

    a = data_of(analyze_image(src), "perceptual")
    b = data_of(analyze_image(stripped), "perceptual")
    assert a["pixel_sha256"] == b["pixel_sha256"], "pixels must survive metadata removal"
    assert a["phash"] == b["phash"]

    assert (
        data_of(analyze_image(src), "hashes")["sha256"]
        != data_of(analyze_image(stripped), "hashes")["sha256"]
    ), "file hashes must differ even though the pixels match"


# -- exif ------------------------------------------------------------------


def test_camera_lens_and_serial_are_extracted(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    doc = analyze_image(images["gps"])
    hl = data_of(doc, "exif")["highlights"]
    assert hl["make"] == "Canon"
    assert "R5" in hl["model"]
    assert hl["serial"] == "042031000537"
    assert "RF24-70" in hl["lens"]

    # Make="Canon", Model="Canon EOS R5" must not become "Canon Canon EOS R5".
    camera = next(f for f in doc.findings if f.label == "Camera")
    assert camera.value == "Canon EOS R5"


@pytest.mark.parametrize(
    ("make", "model", "expected"),
    [
        ("Canon", "Canon EOS R5", "Canon EOS R5"),
        ("NIKON CORPORATION", "NIKON Z 8", "NIKON CORPORATION NIKON Z 8"),
        ("Apple", "iPhone 15 Pro", "Apple iPhone 15 Pro"),
        ("", "Pixel 8", "Pixel 8"),
        ("Sony", "", "Sony"),
    ],
)
def test_camera_label_joining(make, model, expected):
    from imgintel.analyzers.exif import _camera_label

    assert _camera_label({"make": make, "model": model}) == expected


def test_editing_software_is_flagged(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    doc = analyze_image(images["gps"])
    edit = next(f for f in doc.findings if f.label == "Edited in image-editing software")
    assert edit.severity == "medium"
    assert "Photoshop" in str(edit.value)


def test_serial_number_is_flagged_as_identifying(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    doc = analyze_image(images["gps"])
    assert any("BodySerialNumber" in f.label for f in doc.findings)


def test_stripped_image_is_reported_as_such(images):
    """A JPEG carrying only JFIF density fields has had its metadata removed."""
    assert "No capture metadata" in labels(analyze_image(images["sharp"]))


def test_generative_ai_metadata_is_flagged_high(images):
    doc = analyze_image(images["aigen"])
    ai = next(f for f in doc.findings if f.label == "Generative-AI metadata present")
    assert ai.severity == "high"
    assert "stable diffusion" in str(ai.value)


# -- gps -------------------------------------------------------------------


def test_coordinates_are_parsed_correctly(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    d = data_of(analyze_image(images["gps"]), "gps")
    assert d["present"] is True
    assert d["latitude"] == pytest.approx(48.8584, abs=1e-3)
    assert d["longitude"] == pytest.approx(2.2945, abs=1e-3)
    assert d["altitude_m"] == pytest.approx(33.0, abs=0.1)
    assert d["img_direction_deg"] == pytest.approx(135.0, abs=0.1)
    assert d["google_maps"].startswith("https://")


def test_hemisphere_reference_is_applied():
    from imgintel.analyzers.gps import _signed

    assert _signed(((51, 1), (30, 1), (0, 1)), "S", "S") == pytest.approx(-51.5)
    assert _signed(((51, 1), (30, 1), (0, 1)), "N", "S") == pytest.approx(51.5)
    assert _signed(2.5, "W", "W") == pytest.approx(-2.5)


def test_absent_gps_is_not_an_error(images):
    d = data_of(analyze_image(images["sharp"]), "gps")
    assert d["present"] is False


def test_consistent_timestamps_are_confirmed(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    pytest.importorskip("timezonefinder")
    doc = analyze_image(images["gps"])
    assert "Timestamp consistent with GPS timezone" in labels(doc)


def test_inconsistent_timestamps_are_flagged(images):
    """The cross-check that catches an edited clock or a faked location."""
    if "gps_badtime" not in images:
        pytest.skip("piexif not installed")
    pytest.importorskip("timezonefinder")
    doc = analyze_image(images["gps_badtime"])
    bad = next(f for f in doc.findings if f.label == "Timestamp inconsistent with GPS timezone")
    assert bad.severity == "medium"
    assert data_of(doc, "gps")["implied_utc_offset_hours"] == pytest.approx(8.0)
    assert data_of(doc, "gps")["timezone_offset_hours"] == pytest.approx(2.0)


def test_reverse_geocoding_when_available(images):
    if "gps" not in images:
        pytest.skip("piexif not installed")
    pytest.importorskip("reverse_geocoder")
    place = data_of(analyze_image(images["gps"]), "gps")["place"]
    assert place["country_code"] == "FR"


# -- blur ------------------------------------------------------------------


def test_sharp_and_blurred_are_distinguished(images):
    sharp = data_of(analyze_image(images["sharp"]), "blur")["laplacian_variance"]
    blurred = data_of(analyze_image(images["blurry"]), "blur")["laplacian_variance"]
    assert sharp > blurred * 5


def test_global_blur_is_flagged(images):
    doc = analyze_image(images["blurry"])
    assert data_of(doc, "blur")["verdict"] == "blurred"
    assert "Image is blurred" in labels(doc)


def test_localized_blur_is_detected_with_a_bbox(images):
    """A sharp frame with one blurred rectangle — the redaction signal."""
    doc = analyze_image(images["redacted"])
    finding = next(f for f in doc.findings if f.label == "Localized blur regions")
    assert finding.severity == "medium"
    assert finding.bbox is not None
    x0, y0, x1, y1 = finding.bbox
    # Source rectangle was (300,300)-(600,500) at 1200x900; small is 1024 wide.
    scale = 1024 / 1200
    assert x0 == pytest.approx(300 * scale, abs=140)
    assert y0 == pytest.approx(300 * scale, abs=140)


def test_localized_blur_not_reported_on_uniformly_blurred_images(images):
    assert not data_of(analyze_image(images["blurry"]), "blur")["localized_blur"]


def test_localized_blur_not_reported_on_sharp_images(images):
    assert "Localized blur regions" not in labels(analyze_image(images["sharp"]))


def test_uniform_image_gets_no_focus_verdict(images):
    assert data_of(analyze_image(images["flat"]), "blur")["verdict"] == "no detail"


# -- quality ---------------------------------------------------------------


@pytest.mark.parametrize("quality", [50, 75, 90, 95])
def test_jpeg_quality_is_recovered(tmp_path, quality):
    """Encode at a known quality and check the estimate comes back close."""
    rng = np.random.default_rng(1)
    arr = (rng.normal(128, 50, (400, 400, 3))).clip(0, 255).astype(np.uint8)
    path = tmp_path / f"q{quality}.jpg"
    Image.fromarray(arr).save(path, quality=quality)
    estimate = data_of(analyze_image(path), "quality")["jpeg"]["quality_estimate"]
    assert abs(estimate - quality) <= 3, f"estimated {estimate}, encoded at {quality}"


def test_non_jpeg_has_no_quantization_tables(images):
    assert data_of(analyze_image(images["flat"]), "quality")["jpeg"] is None


def test_resolution_class_and_exposure(images):
    d = data_of(analyze_image(images["sharp"]), "quality")
    assert d["megapixels"] == pytest.approx(1.08, abs=0.01)
    assert d["resolution_class"] == "web"
    assert 0 <= d["exposure"]["clipped_highlights"] <= 1


def test_flat_image_flagged_as_low_information(images):
    doc = analyze_image(images["flat"])
    assert "Very low information content" in labels(doc)
    assert data_of(doc, "quality")["entropy_bits"] == pytest.approx(0.0, abs=1e-6)


def test_blurred_image_is_not_flagged_for_a_clean_noise_floor(images):
    """Regression: a blurry photo legitimately has no noise left."""
    assert "Implausibly clean noise floor" not in labels(analyze_image(images["blurry"]))


# -- color -----------------------------------------------------------------


def test_palette_shares_sum_to_one(images):
    palette = data_of(analyze_image(images["sharp"]), "color")["palette"]
    assert palette
    assert sum(c["share"] for c in palette) == pytest.approx(1.0, abs=0.01)
    assert all(c["hex"].startswith("#") and len(c["hex"]) == 7 for c in palette)


def test_palette_is_deterministic(images):
    a = data_of(analyze_image(images["sharp"]), "color")["palette"]
    b = data_of(analyze_image(images["sharp"]), "color")["palette"]
    assert [c["hex"] for c in a] == [c["hex"] for c in b]


def test_grayscale_is_detected(images):
    assert data_of(analyze_image(images["flat"]), "color")["is_grayscale"] is True


def test_color_image_is_not_grayscale(images):
    assert data_of(analyze_image(images["sharp"]), "color")["is_grayscale"] is False


# -- imgmath ---------------------------------------------------------------


def test_dct_matches_a_known_transform():
    """DC coefficient of a constant block equals mean * N for an orthonormal DCT."""
    block = np.full((8, 8), 10.0)
    out = dct2(block)
    assert out[0, 0] == pytest.approx(10.0 * 8)
    assert np.abs(out[1:, 1:]).max() < 1e-9


def test_lab_conversion_of_known_colors():
    lab = srgb_to_lab(np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8))
    assert lab[0, 0, 0] == pytest.approx(100.0, abs=0.5)  # white -> L*=100
    assert lab[0, 1, 0] == pytest.approx(0.0, abs=0.5)  # black -> L*=0


def test_laplacian_variance_ranks_sharpness():
    rng = np.random.default_rng(0)
    noisy = rng.normal(128, 40, (200, 200)).astype(np.float32)
    smooth = np.full((200, 200), 128.0, dtype=np.float32)
    assert laplacian_variance(noisy) > laplacian_variance(smooth)


def test_noise_estimate_tracks_injected_noise():
    rng = np.random.default_rng(0)
    flat = np.full((300, 300), 128.0, dtype=np.float32)
    assert estimate_noise_sigma(flat) == pytest.approx(0.0, abs=0.01)
    for sigma in (5.0, 15.0):
        noisy = flat + rng.normal(0, sigma, flat.shape)
        assert estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.15)


def test_entropy_bounds():
    assert shannon_entropy(np.zeros((50, 50))) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    assert shannon_entropy(rng.integers(0, 256, (400, 400))) == pytest.approx(8.0, abs=0.05)


def test_colorfulness_ordering():
    grey = np.full((100, 100, 3), 128, np.uint8)
    vivid = np.zeros((100, 100, 3), np.uint8)
    vivid[:, :50, 0] = 255
    vivid[:, 50:, 2] = 255
    assert colorfulness(vivid) > colorfulness(grey)


def test_hamming_of_identical_hashes_is_zero():
    assert hamming_hex("ff00ff00", "ff00ff00") == 0
    assert hamming_hex("00000000", "00000001") == 1
    with pytest.raises(ValueError):
        hamming_hex("ff", "ffff")


# -- context sharing -------------------------------------------------------


def test_sharpness_is_computed_once(images, monkeypatch):
    ctx = AnalysisContext(images["sharp"])
    calls = {"n": 0}
    real = laplacian_variance

    import imgintel.util.imgmath as m

    def counted(arr):
        calls["n"] += 1
        return real(arr)

    monkeypatch.setattr(m, "laplacian_variance", counted)
    assert ctx.sharpness == ctx.sharpness
    assert calls["n"] == 1, "sharpness must be cached on the context"
