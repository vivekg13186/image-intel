"""Text extraction and what it reveals: OCR, barcodes, language, sensitive data."""

from __future__ import annotations

import pytest

from imgintel.analyzers.codes import interpret
from imgintel.analyzers.sensitive import (
    card_brand,
    high_entropy,
    iban_valid,
    luhn_valid,
    mask,
)
from imgintel.core.deps import has_module
from imgintel.core.engine import analyze_image


def _needs(module: str):
    """Skip when an optional backend is absent — the suite must pass either way."""
    return pytest.mark.skipif(not has_module(module), reason=f"{module} not installed")


def _needs_ocr():
    """Either RapidOCR package satisfies the OCR analyzer."""
    from imgintel.analyzers.ocr import backend_module

    return pytest.mark.skipif(backend_module() is None, reason="no RapidOCR backend installed")


needs_ocr = _needs_ocr()


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer}: {block.error}"
    return block.data


def labels(doc) -> set[str]:
    return {f.label for f in doc.findings}


# -- model store -----------------------------------------------------------


def test_model_catalogue_is_coherent():
    from imgintel.core.modelstore import MODELS

    for key, spec in MODELS.items():
        assert spec.key == key
        assert spec.filename and spec.description and spec.licence
        assert spec.licence != "unknown", f"{key} must record a licence"
        if spec.bundled:
            assert spec.provided_by, f"{key} is bundled but names no provider"
        else:
            assert spec.urls, f"{key} must be fetchable"


def test_unpinned_models_say_so_rather_than_claiming_verification():
    """A model imgintel has not verified must never look verified.

    Recording a guessed hash would reject every legitimate download; silently
    accepting any file would make findings unreproducible. The third option —
    install it, say it is unpinned, and offer `models pin` — is the honest one.
    """
    from imgintel.core.modelstore import MODELS, expected_sha256, status

    for key, spec in MODELS.items():
        if spec.bundled or spec.sha256 or expected_sha256(key):
            continue
        st = status(key)
        if st.present:
            assert st.verified is None, f"{key} is unpinned but reports verified={st.verified}"
            assert "unpinned" in st.note


def test_missing_model_reports_a_fix_not_a_crash():
    from imgintel.core.modelstore import available

    ok, reason = available("does-not-exist")
    assert not ok
    assert "unknown model" in reason


@needs_ocr
def test_bundled_models_are_located_inside_the_wheel():
    from imgintel.core.modelstore import status

    st = status("ocr-det")
    assert st.present and st.kind == "bundled"
    assert st.path and st.path.endswith(".onnx")


@needs_ocr
def test_installed_models_records_licences_for_evidence():
    from imgintel.core.modelstore import installed_models

    models = installed_models()
    assert "ocr-det" in models
    assert models["ocr-det"]["licence"] == "Apache-2.0"
    assert len(models["ocr-det"]["sha256"]) == 64


# -- OCR -------------------------------------------------------------------


@needs_ocr
def test_ocr_reads_the_document(images):
    doc = analyze_image(images["document"], only=["ocr"])
    data = data_of(doc, "ocr")
    text = data["text"].lower()

    assert data["block_count"] >= 4
    assert "acme" in text
    assert "harbour" in text or "bristol" in text
    assert data["mean_confidence"] > 0.8


@needs_ocr
def test_ocr_blocks_carry_boxes_in_analysis_space(images):
    doc = analyze_image(images["document"], only=["ocr"])
    data = data_of(doc, "ocr")
    width, height = data["analysis_dimensions"]
    for block in data["blocks"]:
        x0, y0, x1, y1 = block["bbox"]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height


@needs_ocr
def test_ocr_reports_no_text_without_crashing(images):
    doc = analyze_image(images["flat"], only=["ocr"])
    assert data_of(doc, "ocr")["block_count"] == 0
    assert "No text detected" in labels(doc)


def test_ocr_is_heavy_and_excluded_from_standard(registry):
    from imgintel.core.engine import plan

    assert "ocr" not in {a.name for a in plan(registry, profile="standard")}
    assert "ocr" in {a.name for a in plan(registry, profile="deep")}


# -- backend compatibility -------------------------------------------------
#
# RapidOCR was renamed from `rapidocr-onnxruntime` to `rapidocr` at v3, and the
# two return different shapes. The old package caps at Python <3.13 so it
# cannot be installed on current interpreters — which is exactly why both
# shapes are covered here rather than only whichever one happens to be present.


def test_normalize_handles_the_v3_object_result():
    """v3: a RapidOCROutput with .boxes / .txts / .scores."""
    import numpy as np

    from imgintel.analyzers.ocr import _normalize

    class Output:
        boxes = np.array([[[0, 0], [10, 0], [10, 5], [0, 5]]], dtype=np.float32)
        txts = ("hello",)
        scores = (0.97,)

    (polygon, text, score), = _normalize(Output())
    assert text == "hello"
    assert score == pytest.approx(0.97)
    assert np.asarray(polygon).shape == (4, 2)


def test_normalize_handles_the_v1_tuple_result():
    """v1: the call returns (list_of_triples, elapsed)."""
    from imgintel.analyzers.ocr import _normalize

    raw = ([[[[0, 0], [8, 0], [8, 4], [0, 4]], "world", 0.88]], 0.031)
    (_, text, score), = _normalize(raw)
    assert text == "world"
    assert score == pytest.approx(0.88)


@pytest.mark.parametrize("empty", [None, ([], 0.0), []])
def test_normalize_handles_empty_results(empty):
    from imgintel.analyzers.ocr import _normalize

    assert _normalize(empty) == []


def test_normalize_handles_an_empty_v3_object():
    from imgintel.analyzers.ocr import _normalize

    class Output:
        boxes = None
        txts = None
        scores = None

    assert _normalize(Output()) == []


@needs_ocr
def test_backend_is_reported_in_the_output(images):
    """Which engine produced the text belongs in the record."""
    from imgintel.analyzers.ocr import backend_module

    doc = analyze_image(images["document"], only=["ocr"])
    engine = data_of(doc, "ocr")["engine"]
    assert backend_module().replace("_", "-") in engine
    assert "PP-OCR" in engine


@needs_ocr
def test_ocr_models_resolve_across_backend_generations():
    """v1 ships PP-OCRv4, v3 ships PP-OCRv6 — both must resolve by role."""
    from imgintel.core.modelstore import status

    for key, marker in (("ocr-det", "det"), ("ocr-rec", "rec"), ("ocr-cls", "cls")):
        st = status(key)
        assert st.present, f"{key} not found in the installed backend"
        assert marker in st.path.lower()


# -- barcodes --------------------------------------------------------------


@_needs("zxingcpp")
def test_decodes_a_qr_code(images):
    doc = analyze_image(images["qr_url"], only=["codes"])
    data = data_of(doc, "codes")
    assert data["count"] == 1
    assert "acme-logistics.co.uk" in data["codes"][0]["text"]


@_needs("zxingcpp")
def test_wifi_password_is_a_high_severity_finding(images):
    """The payload is the intel — "QR code detected" would miss this entirely."""
    doc = analyze_image(images["qr_wifi"], only=["codes"])
    finding = next(f for f in doc.findings if f.label == "WiFi credentials in QR code")
    assert finding.severity == "high"
    assert "AcmeGuest" in str(finding.value)
    payload = data_of(doc, "codes")["codes"][0]["payload"]
    assert payload["password"] == "Summer2024!"


@_needs("zxingcpp")
def test_shortened_url_is_flagged(images):
    doc = analyze_image(images["qr_shortener"], only=["codes"])
    payload = data_of(doc, "codes")["codes"][0]["payload"]
    assert "url_shortener" in payload["flags"]
    assert "insecure_scheme" in payload["flags"]
    finding = next(f for f in doc.findings if f.label == "URL in code")
    assert finding.severity == "medium"


@_needs("zxingcpp")
def test_codes_bboxes_are_in_analysis_space(images):
    doc = analyze_image(images["qr_url"], only=["codes", "fileinfo"])
    from imgintel.core.context import small_size

    info = doc.analyzers["fileinfo"].data
    width, height = small_size(info["width"], info["height"])
    for code in data_of(doc, "codes")["codes"]:
        x0, y0, x1, y1 = code["bbox"]
        assert 0 <= x0 < x1 <= width + 1
        assert 0 <= y0 < y1 <= height + 1


@pytest.mark.parametrize(
    ("payload", "kind", "check"),
    [
        ("WIFI:T:WPA;S:Net;P:pw;;", "wifi", lambda p: p["ssid"] == "Net" and p["password"] == "pw"),
        ("otpauth://totp/Acme:bob?secret=JBSW&issuer=Acme", "otpauth",
         lambda p: p["secret_present"] and p["issuer"] == "Acme"),
        ("https://example.com/a?b=1", "url", lambda p: p["host"] == "example.com"),
        ("http://192.168.1.1/", "url", lambda p: "ip_literal_host" in p["flags"]),
        ("https://xn--80ak6aa92e.com", "url", lambda p: "punycode_host" in p["flags"]),
        ("geo:51.5,-0.12", "geo", lambda p: p["latitude"] == 51.5),
        ("mailto:a@b.com", "email", lambda p: p["address"] == "a@b.com"),
        ("bitcoin:1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "crypto",
         lambda p: p["chain"] == "bitcoin"),
        ("BEGIN:VCARD\nFN:Jane Doe\nTEL:+441234567890\nEND:VCARD", "contact",
         lambda p: p["name"] == "Jane Doe"),
        ("just some text", "text", lambda p: p["value"] == "just some text"),
    ],
)
def test_payload_interpretation(payload, kind, check):
    result = interpret(payload)
    assert result["kind"] == kind
    assert check(result)


def test_wifi_payload_handles_escaped_separators():
    payload = interpret(r"WIFI:T:WPA;S:My\;Net;P:pa\;ss;;")
    assert payload["ssid"] == "My;Net"
    assert payload["password"] == "pa;ss"


# -- language --------------------------------------------------------------


@_needs("lingua")
@needs_ocr
def test_language_detected_from_ocr(images):
    doc = analyze_image(images["document"], only=["language"])
    data = data_of(doc, "language")
    assert data["primary"] == "English"


@_needs("lingua")
def test_short_strings_are_not_classified(images):
    """"ACME" alone detects as Somali with 0.69 confidence — so do not ask."""
    from imgintel.analyzers.language import MIN_CHARS

    assert MIN_CHARS > 4


# -- sensitive data --------------------------------------------------------


@pytest.mark.parametrize(
    ("number", "valid"),
    [
        ("4111111111111111", True),      # Visa test number
        ("4111 1111 1111 1111", True),   # spaced, as OCR often returns it
        ("5500005555555559", True),
        ("4111111111111112", False),     # fails Luhn
        ("1234567890123", False),
        ("123", False),
    ],
)
def test_luhn(number, valid):
    assert luhn_valid(number) is valid


@pytest.mark.parametrize(
    ("iban", "valid"),
    [
        ("GB82WEST12345698765432", True),
        ("GB82 WEST 1234 5698 7654 32", True),
        ("GB82WEST12345698765433", False),
        ("NOTANIBAN", False),
    ],
)
def test_iban_checksum(iban, valid):
    assert iban_valid(iban) is valid


def test_card_brands():
    assert card_brand("4111111111111111") == "Visa"
    assert card_brand("378282246310005") == "American Express"
    assert card_brand("5500005555555559") == "Mastercard"


def test_entropy_separates_secrets_from_placeholders():
    assert high_entropy("k3J8vN2pQrX9zLmW4tYb")
    assert not high_entropy("aaaaaaaaaaaaaaaaaaaa")


@pytest.mark.parametrize(
    ("value", "key"),
    [("j.smith@acme.co.uk", "email"), ("4111111111111111", "credit_card"),
     ("+442079460958", "phone")],
)
def test_masking_hides_the_value_but_keeps_it_recognisable(value, key):
    masked = mask(value, key)
    assert masked != value
    assert "*" in masked
    assert len(masked) >= 4


@needs_ocr
def test_finds_pii_in_a_real_document(images):
    doc = analyze_image(images["document"], only=["sensitive"])
    data = data_of(doc, "sensitive")
    assert "email" in data["types"]
    assert "credit_card" in data["types"]

    card = next(m for m in data["matches"] if m["type"] == "credit_card")
    assert card["value"].replace(" ", "").startswith("4111")
    assert card["brand"] == "Visa"
    assert card["bbox"], "matches must carry the box they came from"


@needs_ocr
def test_findings_mask_values_while_data_keeps_them(images):
    """Reports get shared; the JSON is for the investigator."""
    doc = analyze_image(images["document"], only=["sensitive"])
    data = data_of(doc, "sensitive")

    full = {m["value"].replace(" ", "") for m in data["matches"]}
    assert any(v.startswith("4111111111") for v in full), "data must hold full values"

    for finding in doc.findings:
        assert "4111111111111111" not in str(finding.value)


@needs_ocr
@_needs("phonenumbers")
def test_phone_numbers_are_validated_not_just_matched(images):
    doc = analyze_image(images["document"], only=["sensitive"])
    phones = [m for m in data_of(doc, "sensitive")["matches"] if m["type"] == "phone"]
    assert phones
    assert phones[0]["region"] == "GB"
    assert phones[0]["value"].startswith("+44")


@_needs("zxingcpp")
@needs_ocr
def test_sensitive_also_scans_barcode_payloads(images):
    doc = analyze_image(images["qr_url"], only=["sensitive", "codes"])
    assert doc.analyzers["sensitive"].status == "ok"
    assert data_of(doc, "sensitive")["sources_scanned"] >= 1


def test_sensitive_runs_after_codes_when_both_present(registry):
    """`after` is a soft edge: ordering without a hard dependency."""
    from imgintel.core.pipeline import topo_sort

    chosen, _ = registry.resolve(["sensitive", "codes"])
    order = [a.name for a in topo_sort(chosen)]
    assert order.index("codes") < order.index("sensitive")


def test_sensitive_still_runs_without_codes(registry, images):
    """Soft dependency: zxing being absent must not disable PII detection."""
    doc = analyze_image(images["document"], registry=registry, only=["sensitive"])
    assert doc.analyzers["sensitive"].status in ("ok", "skipped", "unavailable")
    if doc.analyzers["sensitive"].status == "ok":
        assert "codes" not in doc.analyzers


# -- content classification ------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [("document", "graphic"), ("qr_wifi", "graphic"),
     ("flat", "blank"), ("sharp", "photograph"), ("photo", "photograph")],
)
def test_content_type_classification(images, fixture, expected):
    """Gates every photographic assumption downstream."""
    if fixture not in images:
        pytest.skip(f"{fixture} fixture unavailable")
    doc = analyze_image(images[fixture], only=["quality"])
    assert data_of(doc, "quality")["content_type"] == expected


def test_graphics_do_not_get_exposure_complaints(images):
    """A white document is not a blown-out photograph."""
    doc = analyze_image(images["document"], profile="standard")
    assert "Blown highlights" not in labels(doc)
    assert "Very low information content" not in labels(doc)
    assert "Synthetic image, not a camera photograph" in labels(doc)


def test_qr_codes_are_not_reported_as_selectively_blurred(images):
    """Regression: hard-edged graphics made a quarter of the frame look soft."""
    doc = analyze_image(images["qr_wifi"], profile="standard")
    assert "Localized blur regions" not in labels(doc)
    assert not data_of(doc, "blur")["localized_blur"]


def test_photographic_redaction_is_still_detected(images):
    doc = analyze_image(images["redacted"], profile="standard")
    assert "Localized blur regions" in labels(doc)
