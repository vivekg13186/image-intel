"""Model weights: fetched on demand, pinned by hash, recorded for evidence.

Three properties matter more than convenience here:

* **The wheel stays small.** Weights live in the platform cache directory, not
  in the package, so an update is a manifest change rather than a release.
* **Every file is pinned.** A model is verified against a recorded sha256 on
  download and on demand. "Which weights produced this finding" must have an
  answer months later, in writing.
* **Absence is not failure.** A missing model makes its analyzer report
  ``unavailable`` with a ``imgintel models pull`` hint — never a crash, and
  never a silent skip.

Some backends (RapidOCR) ship their own weights inside their wheel. Those are
registered here as ``bundled`` so ``imgintel models`` still accounts for them,
but nothing is downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

_CHUNK = 1 << 20
USER_AGENT = "imgintel-modelstore/1.0"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A model imgintel knows how to obtain and verify."""

    key: str
    filename: str
    description: str
    #: SHA-256 of the file. None only for `bundled` models we do not control.
    sha256: str | None = None
    size_bytes: int | None = None
    #: Mirrors, tried in order. Empty for bundled models.
    urls: tuple[str, ...] = ()
    licence: str = "unknown"
    source: str = ""
    #: "download" or "bundled" (shipped inside a dependency's wheel).
    kind: str = "download"
    #: pip extra that provides a bundled model.
    provided_by: str | None = None
    #: Packages that may carry this bundled model, best first.
    bundled_in: tuple[str, ...] = ()
    #: Glob patterns matching the file inside those packages. Patterns rather
    #: than an exact name because the same role is served by different model
    #: generations across releases — RapidOCR v1 ships PP-OCRv4 and v3 ships
    #: PP-OCRv6, and pinning the filename would break on upgrade.
    bundled_patterns: tuple[str, ...] = ()

    @property
    def bundled(self) -> bool:
        return self.kind == "bundled"


#: The catalogue.
#:
#: Detection models carry ``sha256=None`` — "unpinned". Upstream projects
#: republish release assets and imgintel has not verified these specific files,
#: so recording a hash here that might be wrong would reject every legitimate
#: download. Instead the user verifies the file against the upstream checksum
#: once and runs ``imgintel models pin <key>``, which records the digest
#: locally so every later verification, and every evidence bundle, is exact.
MODELS: dict[str, ModelSpec] = {
    "ocr-det": ModelSpec(
        key="ocr-det",
        filename="PP-OCR_det.onnx",
        description="PP-OCR text detection (ONNX)",
        licence="Apache-2.0",
        source="rapidocr",
        kind="bundled",
        provided_by="imgintel[ocr]",
        bundled_in=("rapidocr", "rapidocr_onnxruntime"),
        bundled_patterns=("*det*.onnx",),
    ),
    "ocr-rec": ModelSpec(
        key="ocr-rec",
        filename="PP-OCR_rec.onnx",
        description="PP-OCR text recognition (ONNX)",
        licence="Apache-2.0",
        source="rapidocr",
        kind="bundled",
        provided_by="imgintel[ocr]",
        bundled_in=("rapidocr", "rapidocr_onnxruntime"),
        bundled_patterns=("*rec*.onnx",),
    ),
    "ocr-cls": ModelSpec(
        key="ocr-cls",
        filename="PP-OCR_cls.onnx",
        description="PP-OCR text-angle classification (ONNX)",
        licence="Apache-2.0",
        source="rapidocr",
        kind="bundled",
        provided_by="imgintel[ocr]",
        bundled_in=("rapidocr", "rapidocr_onnxruntime"),
        bundled_patterns=("*cls*.onnx",),
    ),
    "detect": ModelSpec(
        key="detect",
        filename="yolox_s.onnx",
        description="YOLOX-S object detection, 80 COCO classes (ONNX)",
        size_bytes=36_000_000,
        urls=(
            "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
            "0.1.1rc0/yolox_s.onnx",
        ),
        # Apache-2.0 for both code and weights: no AGPL obligation, unlike
        # Ultralytics YOLO, and no non-commercial rider, unlike InsightFace.
        licence="Apache-2.0",
        source="Megvii-BaseDetection/YOLOX",
    ),
    "face": ModelSpec(
        key="face",
        filename="face_detection_yunet_2023mar.onnx",
        description="YuNet face detection with 5-point landmarks (ONNX)",
        size_bytes=232_000,
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        licence="MIT",
        source="opencv/opencv_zoo",
    ),
    "face-embed": ModelSpec(
        key="face-embed",
        filename="face_recognition_sface_2021dec.onnx",
        description="SFace face recognition embeddings, 128-d (ONNX)",
        size_bytes=38_800_000,
        urls=(
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
        # Apache-2.0, unlike InsightFace's models, which are restricted to
        # non-commercial research. Pairs with the YuNet landmarks for alignment.
        licence="Apache-2.0",
        source="opencv/opencv_zoo",
    ),
}


#: Extra directories to search, colon/semicolon-separated (os.pathsep).
MODEL_DIR_ENV = "IMGINTEL_MODEL_DIR"


def cache_dir() -> Path:
    """Where downloads land. Always writable, always searched first."""
    return Path(user_cache_dir("imgintel")) / "models"


def search_dirs() -> list[Path]:
    """Every directory checked for a model, in priority order.

    Downloading is not always possible — corporate proxies, air-gapped
    forensic workstations, or simply a machine that cannot reach the mirror.
    Pointing ``IMGINTEL_MODEL_DIR`` at a folder of files fetched elsewhere is a
    first-class way to install models, not a workaround.
    """
    dirs = [cache_dir()]
    raw = os.environ.get(MODEL_DIR_ENV, "")
    dirs.extend(Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip())
    return dirs


def model_path(key: str) -> Path:
    """Where a download for ``key`` is written."""
    return cache_dir() / MODELS[key].filename


def find_model(key: str) -> Path | None:
    """Locate an already-installed model across every search directory."""
    spec = MODELS[key]
    for directory in search_dirs():
        candidate = directory / spec.filename
        if candidate.exists():
            return candidate
        # Also accept the key as a filename, so a user who renamed the file
        # while downloading it does not silently get "missing model".
        alias = directory / f"{key}{Path(spec.filename).suffix}"
        if alias.exists():
            return alias
    return None


# -- state -----------------------------------------------------------------


@dataclass(slots=True)
class ModelStatus:
    key: str
    description: str
    kind: str
    licence: str
    present: bool
    path: str | None = None
    sha256: str | None = None
    verified: bool | None = None
    size_bytes: int | None = None
    note: str = ""


def _pins_file() -> Path:
    return cache_dir() / "pins.json"


def load_receipts() -> dict[str, dict[str, Any]]:
    """Most recent acquisition record per model key."""
    log = cache_dir() / "receipts.json"
    if not log.exists():
        return {}
    try:
        entries = json.loads(log.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("key"):
            out[str(entry["key"])] = entry
    return out


def load_pins() -> dict[str, str]:
    """Locally recorded digests for models the catalogue does not pin."""
    path = _pins_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def expected_sha256(key: str) -> str | None:
    """The digest this model must match: catalogue first, then local pin."""
    return MODELS[key].sha256 or load_pins().get(key)


def pin(key: str) -> ModelStatus:
    """Record the digest of the installed model so future checks are exact.

    For an unpinned model this is the step that turns "some file called
    yolox_s.onnx" into "this exact file, and imgintel will tell you if it ever
    changes". Verify the download against the upstream checksum first — pinning
    a tampered file just makes the tampering reproducible.
    """
    if key not in MODELS:
        raise DownloadError(f"unknown model key {key!r}")
    st = status(key)
    if not st.present or st.path is None:
        raise DownloadError(f"{key} is not installed; nothing to pin")

    digest = _sha256(Path(st.path))
    pins = load_pins()
    pins[key] = digest
    path = _pins_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8")
    return status(key, verify=True)


def unpin(key: str) -> None:
    pins = load_pins()
    if pins.pop(key, None) is not None:
        _pins_file().write_text(json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8")


def _bundled_path(spec: ModelSpec) -> Path | None:
    """Locate a model shipped inside a dependency's wheel.

    Searches each candidate package in order and matches by pattern, so a
    backend rename or a model-generation bump does not silently turn into
    "missing model".
    """
    import importlib  # noqa: PLC0415

    for package in spec.bundled_in:
        try:
            module = importlib.import_module(package)
        except ImportError:
            continue
        if module.__file__ is None:
            continue
        root = Path(module.__file__).parent
        for pattern in spec.bundled_patterns or (spec.filename,):
            match = next((p for p in sorted(root.rglob(pattern)) if p.is_file()), None)
            if match is not None:
                return match
    return None


def status(key: str, *, verify: bool = False) -> ModelStatus:
    spec = MODELS[key]
    path = _bundled_path(spec) if spec.bundled else find_model(key)

    if path is None or not path.exists():
        note = (
            f"install {spec.provided_by}"
            if spec.bundled
            else f"run: imgintel models pull {key} — or download {spec.filename} "
            f"and run: imgintel models import <file>"
        )
        return ModelStatus(
            key=key, description=spec.description, kind=spec.kind,
            licence=spec.licence, present=False, note=note,
        )

    expected = expected_sha256(key)
    digest = _sha256(path) if (verify or expected) else None

    verified: bool | None = None
    note = ""
    if expected and digest:
        verified = digest == expected
        if not verified:
            note = "HASH MISMATCH — this file is not the model that was pinned"
    elif not spec.bundled:
        note = f"unpinned — verify the download, then: imgintel models pin {key}"

    return ModelStatus(
        key=key,
        description=spec.description,
        kind=spec.kind,
        licence=spec.licence,
        present=True,
        path=str(path),
        sha256=digest,
        verified=verified,
        size_bytes=path.stat().st_size,
        note=note,
    )


def all_status(*, verify: bool = False) -> list[ModelStatus]:
    return [status(key, verify=verify) for key in MODELS]


def available(*keys: str) -> tuple[bool, str]:
    """Check a set of models. Returns (ok, human-readable reason)."""
    missing = [k for k in keys if k in MODELS and not status(k).present]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        return False, f"unknown model(s): {', '.join(unknown)}"
    if missing:
        notes = {status(k).note for k in missing}
        return False, f"missing model(s): {', '.join(missing)} — {'; '.join(sorted(notes))}"
    return True, ""


def installed_models() -> dict[str, Any]:
    """Provenance record: what is on disk, hashed, with licences.

    Copied verbatim into evidence bundles — this is the audit trail for which
    weights produced a given finding.
    """
    receipts = load_receipts()
    out: dict[str, Any] = {}
    for key in MODELS:
        st = status(key, verify=True)
        if not st.present:
            continue
        record = {
            "description": st.description,
            "licence": st.licence,
            "sha256": st.sha256,
            "size_bytes": st.size_bytes,
            "kind": st.kind,
            # An unpinned model is still recorded by hash, but the bundle must
            # say the digest was never checked against a known-good reference —
            # that difference matters if the findings are ever challenged.
            "pinned": expected_sha256(key) is not None,
            "verified": st.verified,
        }
        receipt = receipts.get(key)
        if receipt:
            record["source_url"] = receipt.get("url")
            record["fetched_utc"] = receipt.get("fetched_utc")
            # Absent from older receipts, so default to the safe reading only
            # when the field genuinely is not there.
            record["tls_verified"] = receipt.get("tls_verified", True)
        out[key] = record
    return out


# -- fetching --------------------------------------------------------------


class DownloadError(RuntimeError):
    pass


def pull(
    key: str,
    *,
    force: bool = False,
    progress: Any = None,
    timeout: int = 120,
    insecure: bool = False,
) -> ModelStatus:
    """Fetch one model, verify it, and install it atomically.

    ``insecure`` disables TLS verification. It is recorded in the receipt and
    surfaced in evidence provenance, because "these weights arrived over an
    unauthenticated connection" is exactly the kind of thing that should not
    quietly vanish from the record.
    """
    spec = MODELS[key]
    if spec.bundled:
        st = status(key)
        if not st.present:
            raise DownloadError(
                f"{key} ships with {spec.provided_by}; install that rather than downloading"
            )
        return st

    existing = find_model(key)
    if existing is not None and not force:
        st = status(key, verify=True)
        if st.verified is not False:
            return st

    target = model_path(key)

    if not spec.urls:
        raise DownloadError(f"no download URL recorded for {key}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    errors: list[str] = []

    certificate_failure: CertificateError | None = None

    for url in spec.urls:
        try:
            _download(url, tmp, timeout=timeout, progress=progress, insecure=insecure)
        except CertificateError as exc:
            certificate_failure = exc
            errors.append(f"{url}: certificate verification failed")
            tmp.unlink(missing_ok=True)
            continue
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            tmp.unlink(missing_ok=True)
            continue

        digest = _sha256(tmp)
        expected = expected_sha256(key)
        if expected and digest != expected:
            tmp.unlink(missing_ok=True)
            errors.append(f"{url}: hash mismatch (got {digest[:16]}…)")
            continue

        # Atomic install: an interrupted download can never be mistaken for a
        # complete model.
        tmp.replace(target)
        _record_receipt(key, digest, url, tls_verified=not insecure)
        return status(key, verify=True)

    # A trust-store problem has a specific fix; do not bury it under a generic
    # "could not fetch" that sends the user looking at the network.
    if certificate_failure is not None:
        raise certificate_failure

    raise DownloadError(f"could not fetch {key}:\n  " + "\n  ".join(errors))


def pull_all(
    *, force: bool = False, progress: Any = None, insecure: bool = False
) -> list[ModelStatus]:
    out = []
    for key, spec in MODELS.items():
        if spec.bundled:
            out.append(status(key))
            continue
        try:
            out.append(pull(key, force=force, progress=progress, insecure=insecure))
        except DownloadError:
            out.append(status(key))
    return out


def import_file(
    source: Path,
    key: str | None = None,
    *,
    move: bool = False,
    force: bool = False,
) -> ModelStatus:
    """Install a model from a local file, verifying it against the catalogue.

    The counterpart to :func:`pull` for anyone who fetched the weights another
    way. Identification is by filename first, then by hash — so a file renamed
    during download is still recognised, and a file whose hash matches a
    different model than its name claims is caught rather than trusted.
    """
    source = Path(source).expanduser()
    if not source.is_file():
        raise DownloadError(f"no such file: {source}")

    digest = _sha256(source)
    resolved = key or _identify(source, digest)
    if resolved is None:
        known = ", ".join(k for k, s in MODELS.items() if not s.bundled)
        raise DownloadError(
            f"{source.name} does not match any known model by name or hash. "
            f"Pass the key explicitly, e.g. --key {known.split(',')[0].strip()}"
        )
    if resolved not in MODELS:
        raise DownloadError(f"unknown model key {resolved!r}")

    spec = MODELS[resolved]
    if spec.bundled:
        raise DownloadError(f"{resolved} ships with {spec.provided_by}; nothing to import")
    if spec.sha256 and digest != spec.sha256:
        raise DownloadError(
            f"{source.name} does not match the pinned hash for {resolved}.\n"
            f"  expected {spec.sha256}\n  got      {digest}\n"
            "Refusing to install: an unverified model would make every finding it "
            "produces unreproducible."
        )

    target = model_path(resolved)
    if target.exists() and not force:
        return status(resolved, verify=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    if move:
        shutil.move(str(source), tmp)
    else:
        shutil.copy2(source, tmp)
    tmp.replace(target)
    _record_receipt(resolved, digest, f"file://{source}")
    return status(resolved, verify=True)


def _identify(source: Path, digest: str) -> str | None:
    for key, spec in MODELS.items():
        if spec.bundled:
            continue
        if spec.sha256 and spec.sha256 == digest:
            return key
    for key, spec in MODELS.items():
        if not spec.bundled and spec.filename == source.name:
            return key
    return None


def import_directory(directory: Path, *, force: bool = False) -> list[ModelStatus]:
    """Import every recognisable model file from a folder."""
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        raise DownloadError(f"not a directory: {directory}")

    out: list[ModelStatus] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (".onnx", ".bin", ".tflite"):
            continue
        try:
            out.append(import_file(path, force=force))
        except DownloadError:
            continue  # unrecognised files in the folder are not an error
    return out


def ssl_context(*, insecure: bool = False):
    """The TLS context for model downloads.

    A certificate-verification failure on macOS is almost never a bad server —
    it is a Python install with no CA bundle wired up, because the framework
    installer leaves that to a script most people never run. So try certifi's
    bundle (pip pulls it in, so it is usually already there) before concluding
    anything is wrong.
    """
    import ssl  # noqa: PLC0415

    if insecure:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    try:
        import certifi  # noqa: PLC0415

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class CertificateError(DownloadError):
    """TLS verification failed — usually a local trust-store problem."""


_SSL_GUIDANCE = (
    "TLS certificate verification failed. This is almost always a local trust-store\n"
    "problem rather than a bad server. In order of preference:\n"
    "  1. pip install certifi         (imgintel uses its CA bundle automatically)\n"
    "  2. macOS python.org builds:    run '/Applications/Python 3.x/Install Certificates.command'\n"
    "  3. Download the file in a browser, then: imgintel models import <file>\n"
    "  4. Last resort:                imgintel models pull <key> --insecure\n"
    "\n"
    "Option 3 is the safest here: these models are unpinned, so an insecure download\n"
    "has no integrity check at all, and ONNX graphs are parsed by onnxruntime."
)


def _download(
    url: str,
    target: Path,
    *,
    timeout: int,
    progress: Any = None,
    insecure: bool = False,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    context = ssl_context(insecure=insecure)
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=timeout, context=context
        ) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with target.open("wb") as fh:
                while chunk := response.read(_CHUNK):
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
    except urllib.error.URLError as exc:
        if _is_certificate_error(exc):
            raise CertificateError(f"{exc}\n\n{_SSL_GUIDANCE}") from exc
        raise DownloadError(str(exc)) from exc


def _is_certificate_error(exc: BaseException) -> bool:
    import ssl  # noqa: PLC0415

    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLCertVerificationError):
            return True
        reason = getattr(exc, "reason", None)
        if isinstance(reason, BaseException):
            exc = reason
            continue
        exc = exc.__cause__ or exc.__context__
    return False


def _record_receipt(key: str, digest: str, url: str, *, tls_verified: bool = True) -> None:
    """Append to a local log of what was fetched, from where, when, and how."""
    log = cache_dir() / "receipts.json"
    entries = []
    if log.exists():
        try:
            entries = json.loads(log.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.append(
        {
            "key": key,
            "sha256": digest,
            "url": url,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tls_verified": tls_verified,
            "spec": asdict(MODELS[key]),
        }
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def purge() -> int:
    """Delete the downloaded model cache. Returns bytes freed."""
    directory = cache_dir()
    if not directory.exists():
        return 0
    freed = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    shutil.rmtree(directory)
    return freed


__all__ = [
    "MODELS",
    "MODEL_DIR_ENV",
    "CertificateError",
    "DownloadError",
    "load_receipts",
    "ssl_context",
    "ModelSpec",
    "ModelStatus",
    "all_status",
    "available",
    "cache_dir",
    "expected_sha256",
    "find_model",
    "import_directory",
    "import_file",
    "installed_models",
    "load_pins",
    "model_path",
    "pin",
    "pull",
    "pull_all",
    "purge",
    "search_dirs",
    "status",
    "unpin",
]
