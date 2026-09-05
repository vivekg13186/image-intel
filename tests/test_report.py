"""Phase 2: flat records, CSV, HTML and evidence bundles."""

from __future__ import annotations

import csv
import json

import pytest

from imgintel.core.context import CaseConfig
from imgintel.core.engine import analyze_image
from imgintel.core.schema import FindingsDocument
from imgintel.report import (
    FLAT_COLUMNS,
    export_bundle,
    failed_row,
    flatten,
    render_html,
    verify_bundle,
    write_findings_csv,
    write_summary_csv,
)
from imgintel.report.evidence import MANIFEST_NAME


@pytest.fixture(scope="module")
def doc(images) -> FindingsDocument:
    return analyze_image(
        images.get("gps", images["sharp"]),
        case=CaseConfig(case_id="CASE-001", operator="tester"),
        command="imgintel analyze test",
    )


# -- flatten ---------------------------------------------------------------


def test_flat_row_has_exactly_the_declared_columns(doc):
    row = flatten(doc)
    assert tuple(row) == FLAT_COLUMNS


def test_flat_row_pulls_from_every_analyzer(doc):
    row = flatten(doc)
    assert row["sha256"] and len(row["sha256"]) == 64
    assert row["phash"]
    assert row["format"] == "JPEG"
    assert row["findings_total"] == doc.summary.findings_total
    assert row["width"] and row["height"]


def test_flat_row_tolerates_missing_analyzers(images):
    """A quick-profile run and a deep run must produce the same columns."""
    quick = flatten(analyze_image(images["sharp"], profile="quick"))
    assert tuple(quick) == FLAT_COLUMNS
    assert quick["sha256"]
    assert quick["phash"] is None  # perceptual did not run
    assert quick["blur_verdict"] is None


def test_failed_row_is_still_complete():
    row = failed_row("/x/y.jpg", "boom")
    assert tuple(row) == FLAT_COLUMNS
    assert row["error"] == "boom"
    assert row["filename"] == "y.jpg"


# -- CSV -------------------------------------------------------------------


def test_summary_csv_roundtrips(doc, tmp_path):
    out = tmp_path / "summary.csv"
    assert write_summary_csv([flatten(doc)], out) == 1
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 1
    assert rows[0]["filename"] == doc.target.filename
    assert list(rows[0]) == list(FLAT_COLUMNS)


def test_findings_csv_has_one_row_per_finding(doc, tmp_path):
    out = tmp_path / "findings.csv"
    count = write_findings_csv([doc], out)
    assert count == len(doc.findings)
    rows = list(csv.DictReader(out.read_text(encoding="utf-8-sig").splitlines()))
    assert {r["severity"] for r in rows} <= {"high", "medium", "low", "info"}
    assert all(r["analyzer"] for r in rows)


def test_csv_is_excel_safe(doc, tmp_path):
    """utf-8-sig, or Excel on Windows mangles non-ASCII place names."""
    out = tmp_path / "s.csv"
    write_summary_csv([flatten(doc)], out)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf")


# -- HTML ------------------------------------------------------------------


def test_html_escapes_section_titles(images):
    """A bare ampersand in a heading is invalid markup.

    The document fixture is used because a key-facts card only renders when it
    has content, and this is the one with a location estimate in it.
    """
    doc = analyze_image(images["document"], profile="deep")
    html = render_html(doc, images["document"])
    assert "Integrity &amp; estimation" in html, "section should render for this fixture"
    assert "Integrity & estimation" not in html


def test_html_is_self_contained(doc, images):
    html = render_html(doc, images.get("gps", images["sharp"]))
    assert html.startswith("<!doctype html>")
    assert "data:image/jpeg;base64," in html, "thumbnail must be embedded"
    # No external fetches: the report must render offline, years later.
    for probe in ("src=\"http", "href=\"http://", "fonts.googleapis"):
        assert probe not in html
    assert "<style>" in html


def test_html_contains_the_findings(doc, images):
    html = render_html(doc, images.get("gps", images["sharp"]))
    for finding in doc.findings[:5]:
        assert finding.label in html


def test_html_escapes_untrusted_values(images, tmp_path):
    """EXIF and OCR text are attacker-controlled; they must not become markup."""
    from PIL import Image

    evil = tmp_path / "evil.jpg"
    with Image.open(images["sharp"]) as im:
        im.save(evil)
    doc = analyze_image(evil)
    doc.findings[0].label = "<script>alert(1)</script>"
    doc.findings[0].value = '<img src=x onerror=alert(2)>"'
    doc.findings[0].detail = "</div><script>alert(3)</script>"
    html = render_html(doc, evil)

    # The test is that nothing becomes *markup* — the literal characters may
    # well survive as escaped text, which is exactly the desired outcome.
    assert "<script>" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;img src=x onerror=alert(2)&gt;&quot;" in html
    assert "&lt;/div&gt;" in html


def test_html_draws_boxes_for_findings_that_have_them(images):
    doc = analyze_image(images["redacted"])
    assert any(f.bbox for f in doc.findings), "fixture should produce a bbox finding"
    html = render_html(doc, images["redacted"])
    assert "<svg" in html and "<rect" in html


def test_html_survives_a_missing_image(doc, tmp_path):
    html = render_html(doc, tmp_path / "gone.jpg")
    assert "<!doctype html>" in html  # report still renders, just without a thumbnail


# -- evidence --------------------------------------------------------------


def test_bundle_contains_everything_needed(doc, images, tmp_path):
    source = images.get("gps", images["sharp"])
    bundle = export_bundle(doc, tmp_path, image_path=source)
    directory = bundle.directory

    assert (directory / "findings.json").exists()
    assert (directory / "report.html").exists()
    assert (directory / "findings.csv").exists()
    assert (directory / "provenance.json").exists()
    assert (directory / MANIFEST_NAME).exists()
    assert "CASE-001" in directory.name


def test_bundle_original_is_bit_identical(doc, images, tmp_path):
    """The subject file must never be rewritten — that is the whole point."""
    source = images.get("gps", images["sharp"])
    bundle = export_bundle(doc, tmp_path, image_path=source)
    copied = bundle.directory / "original" / source.name
    assert copied.read_bytes() == source.read_bytes()


def test_provenance_records_versions_and_models(doc, images, tmp_path):
    bundle = export_bundle(doc, tmp_path, image_path=images.get("gps", images["sharp"]))
    prov = json.loads((bundle.directory / "provenance.json").read_text())

    assert prov["tool"]["name"] == "imgintel"
    assert prov["case"]["case_id"] == "CASE-001"
    assert prov["subject"]["sha256"]
    # Analyzer versions must be recorded: a finding from blur 1.0.0 may not
    # reproduce under 2.0.0, and the bundle has to say which ran.
    assert prov["analyzers"]["hashes"]["version"]
    assert "environment" in prov and "optional_dependencies" in prov


def test_verify_passes_on_a_fresh_bundle(doc, images, tmp_path):
    bundle = export_bundle(doc, tmp_path, image_path=images.get("gps", images["sharp"]))
    result = verify_bundle(bundle.directory)
    assert result.ok, result.summary
    assert result.checked >= 5


def test_verify_detects_a_modified_file(doc, images, tmp_path):
    bundle = export_bundle(doc, tmp_path, image_path=images.get("gps", images["sharp"]))
    target = bundle.directory / "findings.json"
    target.write_text(target.read_text() + " ")

    result = verify_bundle(bundle.directory)
    assert not result.ok
    assert "findings.json" in result.modified


def test_verify_detects_a_deleted_file(doc, images, tmp_path):
    bundle = export_bundle(doc, tmp_path, image_path=images.get("gps", images["sharp"]))
    (bundle.directory / "findings.csv").unlink()

    result = verify_bundle(bundle.directory)
    assert not result.ok
    assert "findings.csv" in result.missing


def test_verify_detects_a_planted_file(doc, images, tmp_path):
    """An added file is as much a problem as a changed one."""
    bundle = export_bundle(doc, tmp_path, image_path=images.get("gps", images["sharp"]))
    (bundle.directory / "extra.txt").write_text("planted")

    result = verify_bundle(bundle.directory)
    assert not result.ok
    assert "extra.txt" in result.unlisted


def test_verify_reports_a_missing_manifest(tmp_path):
    empty = tmp_path / "notabundle"
    empty.mkdir()
    assert not verify_bundle(empty).ok
