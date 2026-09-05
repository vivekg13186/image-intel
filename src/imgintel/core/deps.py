"""Optional-dependency detection.

Most of the support burden on a tool like this is "why isn't OCR working".
``imgintel doctor`` answers that precisely, with a per-OS fix.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache


@dataclass(slots=True)
class DepStatus:
    key: str
    kind: str  # "python" | "binary"
    present: bool
    version: str | None
    enables: str
    hint: str


def _macos() -> bool:
    return platform.system() == "Darwin"


def _windows() -> bool:
    return platform.system() == "Windows"


def _install_hint(brew: str, apt: str, win: str, pip: str | None = None) -> str:
    if pip:
        return f"pip install {pip}"
    if _macos():
        return f"brew install {brew}"
    if _windows():
        return win
    return f"apt install {apt}"


@cache
def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@cache
def module_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


@cache
def binary_path(name: str) -> str | None:
    return shutil.which(name)


@cache
def binary_version(name: str, flag: str = "-ver") -> str | None:
    path = binary_path(name)
    if not path:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [path, flag], capture_output=True, text=True, timeout=10, check=False
        )
        text = (out.stdout or out.stderr).strip().splitlines()
        return text[0].strip() if text else None
    except Exception:  # noqa: BLE001
        return None


# Registered optional dependencies. Phase 3+ will append OCR/ONNX entries here.
OPTIONAL_DEPS: list[dict] = [
    {
        "key": "exiftool",
        "kind": "binary",
        "probe": lambda: binary_path("exiftool"),
        "version": lambda: binary_version("exiftool", "-ver"),
        "enables": "Full EXIF/XMP/IPTC/MakerNotes extraction (camera serials, lens, owner)",
        "hint": _install_hint(
            "exiftool", "libimage-exiftool-perl", "winget install OliverBetz.ExifTool"
        ),
    },
    {
        "key": "pyexiftool",
        "kind": "python",
        "probe": lambda: has_module("exiftool"),
        "version": lambda: module_version("pyexiftool"),
        "enables": "Python bindings used to drive the exiftool binary",
        "hint": "pip install 'imgintel[exif]'",
    },
    {
        "key": "reverse_geocoder",
        "kind": "python",
        "probe": lambda: has_module("reverse_geocoder"),
        "version": lambda: module_version("reverse_geocoder"),
        "enables": "Offline place names for GPS coordinates",
        "hint": "pip install 'imgintel[places]'",
    },
    {
        "key": "timezonefinder",
        "kind": "python",
        "probe": lambda: has_module("timezonefinder"),
        "version": lambda: module_version("timezonefinder"),
        "enables": "Timezone from GPS, enabling EXIF timestamp cross-checks",
        "hint": "pip install 'imgintel[timezone]'",
    },
    {
        "key": "rapidocr",
        "kind": "python",
        # Either package satisfies the OCR analyzer; `rapidocr` is the
        # renamed successor and the only one installable on Python 3.13+.
        "probe": lambda: has_module("rapidocr") or has_module("rapidocr_onnxruntime"),
        "version": lambda: module_version("rapidocr")
        or module_version("rapidocr-onnxruntime"),
        "enables": "OCR text extraction (PP-OCR weights ship inside the wheel)",
        "hint": "pip install 'imgintel[ocr]'",
    },
    {
        "key": "onnxruntime",
        "kind": "python",
        "probe": lambda: has_module("onnxruntime"),
        "version": lambda: module_version("onnxruntime"),
        "enables": "Object detection and OCR inference",
        "hint": "pip install 'imgintel[detect]'",
    },
    {
        "key": "opencv",
        "kind": "python",
        "probe": lambda: has_module("cv2"),
        "version": lambda: module_version("opencv-python-headless")
        or module_version("opencv-python"),
        "enables": "Face detection (YuNet) and reference logo matching",
        "hint": "pip install 'imgintel[faces,logo]'",
    },
    {
        "key": "zxing-cpp",
        "kind": "python",
        "probe": lambda: has_module("zxingcpp"),
        "version": lambda: module_version("zxing-cpp"),
        "enables": "QR code and barcode decoding",
        "hint": "pip install 'imgintel[codes]'",
    },
    {
        "key": "lingua",
        "kind": "python",
        "probe": lambda: has_module("lingua"),
        "version": lambda: module_version("lingua-language-detector"),
        "enables": "Language identification of extracted text",
        "hint": "pip install 'imgintel[lang]'",
    },
    {
        "key": "phonenumbers",
        "kind": "python",
        "probe": lambda: has_module("phonenumbers"),
        "version": lambda: module_version("phonenumbers"),
        "enables": "Validated phone-number detection (falls back to a weaker regex)",
        "hint": "pip install 'imgintel[pii]'",
    },
]


def check_all() -> list[DepStatus]:
    out: list[DepStatus] = []
    for spec in OPTIONAL_DEPS:
        try:
            present = bool(spec["probe"]())
        except Exception:  # noqa: BLE001
            present = False
        version = None
        if present:
            try:
                version = spec["version"]()
            except Exception:  # noqa: BLE001
                version = None
        out.append(
            DepStatus(
                key=spec["key"],
                kind=spec["kind"],
                present=present,
                version=version,
                enables=spec["enables"],
                hint=spec["hint"],
            )
        )
    return out
