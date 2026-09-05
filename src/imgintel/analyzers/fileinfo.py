"""Filesystem facts, container format, and what the filename gives away."""

from __future__ import annotations

import re
import stat as stat_mod
from datetime import datetime, timezone

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Cost, Finding, Severity
from imgintel.core.context import PILLOW_DEFAULT_LIMIT, AnalysisContext

# Magic-byte signatures. A tiny table beats a libmagic dependency here: we only
# care about image containers, and this keeps the install pure-wheel.
_SIGNATURES: list[tuple[int, bytes, str]] = [
    (0, b"\xff\xd8\xff", "JPEG"),
    (0, b"\x89PNG\r\n\x1a\n", "PNG"),
    (0, b"GIF87a", "GIF"),
    (0, b"GIF89a", "GIF"),
    (0, b"BM", "BMP"),
    (0, b"II*\x00", "TIFF"),
    (0, b"MM\x00*", "TIFF"),
    (0, b"\x00\x00\x01\x00", "ICO"),
    (0, b"8BPS", "PSD"),
    (0, b"\xff\x0a", "JPEG XL"),
    (0, b"\x00\x00\x00\x0cJXL \r\n\x87\n", "JPEG XL"),
    (0, b"qoif", "QOI"),
]
_FTYP_BRANDS = {
    b"heic": "HEIC",
    b"heix": "HEIC",
    b"heim": "HEIC",
    b"hevc": "HEIC",
    b"mif1": "HEIF",
    b"msf1": "HEIF",
    b"avif": "AVIF",
    b"avis": "AVIF",
}
_EXT_ALIASES = {
    "jpg": "JPEG", "jpeg": "JPEG", "jpe": "JPEG", "jfif": "JPEG",
    "png": "PNG", "gif": "GIF", "bmp": "BMP", "dib": "BMP",
    "tif": "TIFF", "tiff": "TIFF", "webp": "WEBP", "ico": "ICO",
    "psd": "PSD", "heic": "HEIC", "heif": "HEIF", "avif": "AVIF",
    "jxl": "JPEG XL", "qoi": "QOI",
}

# Filenames leak provenance. This is cheap intel that people routinely miss.
_NAME_PATTERNS: list[tuple[str, str, str]] = [
    (r"^IMG[-_]\d{8}[-_]WA\d{4}", "WhatsApp", "WhatsApp re-encodes and strips EXIF"),
    (r"^signal-\d{4}-\d{2}-\d{2}", "Signal", "Signal export naming"),
    (r"^(Screenshot|Screen Shot|screenshot)[ _-]", "Screenshot", "Screen capture, not a camera"),
    (r"^IMG_\d{4}\.(jpe?g|heic)$", "Apple/generic camera", "Sequential camera counter"),
    (r"^DSC[_N]?\d{4}", "Sony/Nikon camera", "DCF sequential naming"),
    (r"^(DJI|dji)_\d+", "DJI drone", "Drone imagery"),
    (r"^PXL_\d{8}_\d+", "Google Pixel", "Pixel camera naming"),
    (r"^(GOPR|GP\d\d)\d+", "GoPro", "GoPro naming"),
    (r"^(FB_IMG|received_)\d+", "Facebook/Messenger", "Downloaded from Facebook"),
    (r"^(image|unnamed|download)\s*\(?\d*\)?$", "Browser download", "Generic saved-from-web name"),
    (r"[0-9a-f]{32,}", "Content-addressed", "Hash-like filename, likely CDN or cache"),
]


def _sniff(head: bytes) -> str | None:
    for offset, sig, name in _SIGNATURES:
        if head[offset : offset + len(sig)] == sig:
            return name
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in _FTYP_BRANDS:
            return _FTYP_BRANDS[brand]
        return "ISOBMFF"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class FileInfoAnalyzer(Analyzer):
    name = "fileinfo"
    version = "1.0.0"
    title = "File information"
    description = "Filesystem metadata, true container format, filename provenance"
    cost = Cost.CHEAP

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        path = ctx.path
        st = path.stat()
        head = ctx.raw[:64] if st.st_size else b""

        sniffed = _sniff(head)
        ext = path.suffix.lower().lstrip(".")
        declared = _EXT_ALIASES.get(ext)

        data: dict = {
            "path": str(path.resolve()),
            "filename": path.name,
            "extension": ext or None,
            "size_bytes": st.st_size,
            "size_human": _human(st.st_size),
            "modified_utc": _iso(st.st_mtime),
            "created_utc": _iso(getattr(st, "st_birthtime", st.st_ctime)),
            "accessed_utc": _iso(st.st_atime),
            "mode": stat_mod.filemode(st.st_mode),
            "format_declared": declared,
            "format_actual": sniffed,
            "magic_hex": head[:12].hex(),
        }
        findings: list[Finding] = []

        if st.st_size == 0:
            findings.append(
                Finding("file", "Empty file", 0, Severity.HIGH, detail="Zero bytes on disk")
            )
            return self.ok(data, findings)

        # Container-level detail from the header only — no pixel decode, so
        # this stays fast on very large files and in the `quick` profile.
        if ctx.readable:
            img = ctx.header
            w, h = img.size
            data.update(
                {
                    "pillow_format": img.format,
                    "mode": img.mode,
                    "width": w,
                    "height": h,
                    "megapixels": round((w * h) / 1_000_000, 3),
                    "aspect_ratio": round(w / h, 4) if h else None,
                    "orientation": "landscape" if w > h else "portrait" if h > w else "square",
                    "has_alpha": img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info,
                    "n_frames": getattr(img, "n_frames", 1),
                    "is_animated": bool(getattr(img, "is_animated", False)),
                    "icc_profile_present": "icc_profile" in img.info,
                }
            )
            if data["is_animated"]:
                findings.append(
                    Finding("file", "Animated image", f"{data['n_frames']} frames", Severity.INFO)
                )
            if w * h > PILLOW_DEFAULT_LIMIT:
                findings.append(
                    Finding(
                        "file",
                        "Very large pixel count",
                        f"{data['megapixels']} MP",
                        Severity.MEDIUM,
                        detail="Exceeds the usual decompression-bomb threshold",
                    )
                )
            bpp = (st.st_size * 8) / (w * h) if w * h else 0
            data["bits_per_pixel"] = round(bpp, 4)
        else:
            data["pillow_format"] = None
            findings.append(
                Finding(
                    "file",
                    "Not decodable as an image",
                    None,
                    Severity.HIGH,
                    detail="Header is missing, truncated, corrupt, or deliberately malformed",
                )
            )

        # Extension vs magic bytes. A mismatch is itself an intel signal.
        if sniffed and declared and sniffed != declared:
            findings.append(
                Finding(
                    "file",
                    "Extension does not match content",
                    f"{ext} claims {declared}, bytes say {sniffed}",
                    Severity.HIGH,
                    detail="Renamed file, wrong export, or deliberate disguise",
                )
            )
        elif sniffed is None:
            findings.append(
                Finding(
                    "file",
                    "Unrecognized container signature",
                    head[:8].hex(),
                    Severity.MEDIUM,
                )
            )
        elif not declared:
            findings.append(
                Finding("file", "Missing or unknown extension", ext or "(none)", Severity.LOW)
            )

        # Filename provenance.
        sources = []
        for pattern, source, why in _NAME_PATTERNS:
            if re.search(pattern, path.name, re.IGNORECASE):
                sources.append(source)
                findings.append(
                    Finding("provenance", f"Filename suggests: {source}", path.name,
                            Severity.INFO, confidence=0.6, detail=why)
                )
        data["filename_sources"] = sources

        return self.ok(data, findings)


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"
