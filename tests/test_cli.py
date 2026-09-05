"""The CLI surface: exit codes, output files, and machine-readable output."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from imgintel.cli import app

runner = CliRunner()


def test_analyze_renders_a_report(images):
    result = runner.invoke(app, ["analyze", str(images["sharp"])])
    assert result.exit_code == 0, result.output
    assert "imgintel" in result.output


def test_analyze_writes_json(images, tmp_path):
    out = tmp_path / "nested" / "findings.json"
    result = runner.invoke(app, ["analyze", str(images["sharp"]), "--json", str(out)])
    assert result.exit_code == 0, result.output

    doc = json.loads(out.read_text())
    assert doc["target"]["filename"] == "sharp.jpg"
    assert doc["analyzers"]["hashes"]["status"] == "ok"
    assert doc["run"]["profile"] == "standard"
    assert doc["run"]["command"]


def test_stdout_json_is_pure_json(images):
    result = runner.invoke(app, ["analyze", str(images["sharp"]), "--stdout-json"])
    assert result.exit_code == 0
    json.loads(result.stdout)  # must parse with no surrounding chatter


def test_case_metadata_is_recorded(images, tmp_path):
    out = tmp_path / "c.json"
    runner.invoke(
        app,
        ["analyze", str(images["sharp"]), "--json", str(out),
         "--case", "CASE-001", "--operator", "vivek", "--quiet"],
    )
    doc = json.loads(out.read_text())
    assert doc["case"]["case_id"] == "CASE-001"
    assert doc["case"]["operator"] == "vivek"


def test_analyzer_selection(images, tmp_path):
    out = tmp_path / "s.json"
    runner.invoke(
        app,
        ["analyze", str(images["sharp"]), "-a", "hashes,fileinfo", "--json", str(out), "--quiet"],
    )
    doc = json.loads(out.read_text())
    assert set(doc["analyzers"]) == {"hashes", "fileinfo"}


def test_quick_profile_skips_pixel_analyzers(images, tmp_path):
    out = tmp_path / "q.json"
    runner.invoke(
        app, ["analyze", str(images["sharp"]), "-p", "quick", "--json", str(out), "--quiet"]
    )
    doc = json.loads(out.read_text())
    assert "blur" not in doc["analyzers"]
    assert "exif" in doc["analyzers"]


def test_missing_file_exits_two(tmp_path):
    result = runner.invoke(app, ["analyze", str(tmp_path / "nope.jpg")])
    assert result.exit_code == 2


def test_unknown_analyzer_exits_two(images):
    result = runner.invoke(app, ["analyze", str(images["sharp"]), "-a", "telepathy"])
    assert result.exit_code == 2
    assert "unknown analyzer" in result.output.lower()


def test_unknown_profile_exits_two(images):
    assert runner.invoke(app, ["analyze", str(images["sharp"]), "-p", "turbo"]).exit_code == 2


@pytest.mark.parametrize("command", [["doctor"], ["plugins"], ["profiles"], ["schema"]])
def test_informational_commands_succeed(command):
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output


def test_schema_command_emits_valid_json():
    result = runner.invoke(app, ["schema"])
    assert "findings" in json.loads(result.stdout)["properties"]


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "imgintel" in result.output
