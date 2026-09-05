"""Evidence bundles.

A bundle is a directory that can be handed to someone else and still mean
something: the original file bit-for-bit, the findings, a readable report, and
a provenance record naming every version and hash involved. ``verify_bundle``
re-hashes everything, which is what makes the manifest more than decoration.

The original is copied, never rewritten. Nothing in here modifies the subject
file — that property is the whole point.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from imgintel import __version__
from imgintel.core.schema import SCHEMA_VERSION, FindingsDocument
from imgintel.report.csv_out import write_findings_csv
from imgintel.report.html_out import render_html

MANIFEST_NAME = "MANIFEST.sha256"
PROVENANCE_NAME = "provenance.json"
_CHUNK = 1 << 20
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(text: str, limit: int = 48) -> str:
    return _SAFE.sub("_", text).strip("_")[:limit] or "item"


@dataclass(slots=True)
class BundleResult:
    directory: Path
    files: list[str] = field(default_factory=list)
    manifest: Path | None = None

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass(slots=True)
class VerifyResult:
    directory: Path
    ok: bool
    checked: int = 0
    modified: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return f"{self.checked} file(s) verified, all hashes match"
        problems = []
        if self.modified:
            problems.append(f"{len(self.modified)} modified")
        if self.missing:
            problems.append(f"{len(self.missing)} missing")
        if self.unlisted:
            problems.append(f"{len(self.unlisted)} not in manifest")
        return ", ".join(problems)


def export_bundle(
    doc: FindingsDocument,
    out_dir: Path,
    *,
    image_path: Path | str | None = None,
    include_html: bool = True,
    include_csv: bool = True,
    copy_original: bool = True,
) -> BundleResult:
    """Write a self-describing evidence directory under ``out_dir``."""
    source = Path(image_path) if image_path else Path(doc.target.path)
    parts = [p for p in (doc.case.case_id, _slug(source.name), _utc_stamp()) if p]
    bundle = out_dir / _slug("_".join(str(p) for p in parts), limit=160)
    bundle.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    if copy_original and source.exists():
        originals = bundle / "original"
        originals.mkdir(exist_ok=True)
        target = originals / source.name
        # copy2 preserves mtime/atime, which are themselves evidence.
        shutil.copy2(source, target)
        written.append(f"original/{source.name}")

    (bundle / "findings.json").write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    written.append("findings.json")

    if include_html:
        (bundle / "report.html").write_text(
            render_html(doc, source if source.exists() else None), encoding="utf-8"
        )
        written.append("report.html")

    if include_csv:
        write_findings_csv([doc], bundle / "findings.csv")
        written.append("findings.csv")

    (bundle / PROVENANCE_NAME).write_text(
        json.dumps(build_provenance(doc, source), indent=2), encoding="utf-8"
    )
    written.append(PROVENANCE_NAME)

    manifest = write_manifest(bundle)
    return BundleResult(directory=bundle, files=written, manifest=manifest)


def build_provenance(doc: FindingsDocument, source: Path) -> dict[str, Any]:
    """Everything needed to reproduce, or challenge, this analysis."""
    from imgintel.core.deps import check_all

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": {
            "name": "imgintel",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "command": doc.run.command,
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "case": doc.case.model_dump(),
        # Whether biometric identification was permitted on this run, and under
        # what authority. Recorded even when it was not performed: "we did not
        # run face matching" is itself part of the record.
        "biometrics": {
            "permitted_this_run": doc.run.allow_biometrics,
            "lawful_basis": doc.case.biometric_lawful_basis,
            "authorised_by": doc.case.biometric_authorised_by,
            "expires": doc.case.biometric_expires,
            "entity_db": doc.run.entity_db,
            "performed": (
                doc.analyzers["facedb"].status == "ok" if "facedb" in doc.analyzers else False
            ),
            "not_performed_because": (
                doc.analyzers["facedb"].error
                if "facedb" in doc.analyzers and doc.analyzers["facedb"].status != "ok"
                else None
            ),
        },
        "subject": {
            "path": str(source),
            "filename": source.name,
            "size_bytes": doc.target.size_bytes,
            "sha256": _subject_hash(doc, source),
            "analyzed_at": doc.target.analyzed_at,
        },
        "run": doc.run.model_dump(),
        # Analyzer versions matter: a finding produced by blur 1.0.0 may not be
        # reproducible under 2.0.0, and the bundle should say which ran.
        "analyzers": {
            name: {"version": block.version, "status": block.status}
            for name, block in sorted(doc.analyzers.items())
        },
        "models": _model_provenance(),
        "optional_dependencies": {
            dep.key: {"present": dep.present, "version": dep.version} for dep in check_all()
        },
    }


def _subject_hash(doc: FindingsDocument, source: Path) -> str | None:
    block = doc.analyzers.get("hashes")
    if block is not None and block.status == "ok":
        return block.data.get("sha256")
    return _sha256(source) if source.exists() else None


def _model_provenance() -> dict[str, Any]:
    """Hashes and licences of every model file present, for auditability."""
    try:
        from imgintel.core.modelstore import installed_models

        return installed_models()
    except Exception:  # noqa: BLE001 - model store is optional
        return {}


def write_manifest(bundle: Path) -> Path:
    """Write ``sha256  relative/path`` lines for every file except the manifest."""
    lines = []
    for path in sorted(bundle.rglob("*")):
        if path.is_dir() or path.name == MANIFEST_NAME:
            continue
        rel = path.relative_to(bundle).as_posix()
        lines.append(f"{_sha256(path)}  {rel}")
    manifest = bundle / MANIFEST_NAME
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(bundle: Path) -> VerifyResult:
    """Re-hash every listed file and report anything that changed."""
    manifest = bundle / MANIFEST_NAME
    if not manifest.exists():
        return VerifyResult(bundle, ok=False, missing=[MANIFEST_NAME])

    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        if rel:
            expected[rel.strip()] = digest.strip()

    modified, missing = [], []
    for rel, digest in expected.items():
        path = bundle / rel
        if not path.exists():
            missing.append(rel)
        elif _sha256(path) != digest:
            modified.append(rel)

    present = {
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    }
    unlisted = sorted(present - set(expected))

    return VerifyResult(
        directory=bundle,
        ok=not (modified or missing or unlisted),
        checked=len(expected),
        modified=sorted(modified),
        missing=sorted(missing),
        unlisted=unlisted,
    )
