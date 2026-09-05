"""Synthetic fixtures.

Everything is generated so the suite has no binary assets and runs identically
on every platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter, PngImagePlugin

try:
    import piexif as _piexif
except ImportError:  # pragma: no cover
    _piexif = None


def _colorize(base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.stack([base, (base * 0.8 + 30).clip(0, 255), (base * 0.6 + 60).clip(0, 255)], -1)
    return (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)


def _texture(h: int = 900, w: int = 1200, seed: int = 7) -> np.ndarray:
    """High-frequency detail — gives every tile strong local variance.

    Used for the blur and quality fixtures, where what matters is that sharp
    regions are measurably sharper than blurred ones.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(xx / 7.0) * 60 + np.cos(yy / 5.0) * 60 + 128).clip(0, 255)
    return _colorize(base, seed)


def _photo(h: int = 900, w: int = 1200, seed: int = 11) -> np.ndarray:
    """Low-frequency structure — the spatial statistics of an actual photograph.

    High-frequency test patterns alias badly under downscaling, which would
    unfairly punish perceptual hashing; real scenes do not behave that way.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    base = (
        np.sin(xx / 180.0) * 70 + np.cos(yy / 140.0) * 70 + np.sin((xx + yy) / 300.0) * 40 + 128
    ).clip(0, 255)
    return _colorize(base, seed)


@pytest.fixture(scope="session")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("images")
    tex = _texture()
    out: dict[str, Path] = {}

    out["sharp"] = d / "sharp.jpg"
    Image.fromarray(tex).save(out["sharp"], quality=92)

    out["photo"] = d / "photo.jpg"
    Image.fromarray(_photo()).save(out["photo"], quality=92)

    out["blurry"] = d / "blurry.jpg"
    Image.fromarray(tex).filter(ImageFilter.GaussianBlur(6)).save(out["blurry"], quality=92)

    # Sharp frame with one heavily blurred rectangle — the redaction case.
    im = Image.fromarray(tex)
    im.paste(im.crop((300, 300, 600, 500)).filter(ImageFilter.GaussianBlur(12)), (300, 300))
    out["redacted"] = d / "redacted.jpg"
    im.save(out["redacted"], quality=95)

    out["flat"] = d / "flat.png"
    Image.fromarray(np.full((300, 300, 3), 200, np.uint8)).save(out["flat"])

    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a cat, Steps: 20, Sampler: Euler a, Model: stable diffusion 1.5")
    out["aigen"] = d / "aigen.png"
    Image.fromarray(tex[:400, :400]).save(out["aigen"], pnginfo=meta)

    # PNG bytes carrying a .jpg extension, named like a WhatsApp export.
    out["mislabeled"] = d / "IMG-20230415-WA0007.jpg"
    Image.fromarray(tex[:200, :200]).save(out["mislabeled"], format="PNG")

    out["empty"] = d / "empty.jpg"
    out["empty"].write_bytes(b"")

    out["notanimage"] = d / "notanimage.png"
    out["notanimage"].write_bytes(b"this is plain text, not a picture at all" * 10)

    if _piexif is not None:
        out["gps"] = d / "paris.jpg"
        out["gps_badtime"] = d / "paris_badtime.jpg"
        _write_gps_images(tex, out["gps"], out["gps_badtime"])

    out.update(_text_images(d))
    out.update(_code_images(d))
    out.update(_tamper_images(d))
    return out


def _natural(h: int = 600, w: int = 800, seed: int = 31) -> np.ndarray:
    """An image with the statistics of a real photograph.

    Natural scenes have an approximately 1/f amplitude spectrum, and the
    forensic measures depend on that: DCT coefficients follow a Benford-like
    law only for such content, and the smooth periodic gradients used elsewhere
    in this suite are so self-similar that copy-move detection trips on them.
    Synthesising 1/f noise gives fixtures the techniques behave normally on.
    """
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq = np.sqrt(fy**2 + fx**2)
    freq[0, 0] = 1e-6

    planes = []
    for _ in range(3):
        spectrum = np.fft.fft2(rng.normal(size=(h, w))) / freq**1.1
        plane = np.real(np.fft.ifft2(spectrum))
        planes.append((plane - plane.min()) / (np.ptp(plane) + 1e-9) * 255)
    return np.stack(planes, -1).astype(np.uint8)


def _roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """Compress and decompress in memory, leaving that quality's fingerprint."""
    import io

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as reopened:
        return np.asarray(reopened.convert("RGB"), dtype=np.uint8)


def _tamper_images(d: Path) -> dict[str, Path]:
    """Images whose manipulation history is known exactly.

    The right answer here is not a matter of opinion: one has a region spliced
    in from a file compressed at a known different quality, one has a region
    cloned within itself at a known offset, and one is a clean single-save
    original that must stay quiet. That last one is the important fixture —
    a tamper detector that fires on everything is worse than none.
    """
    photo = _natural(600, 800, seed=31)
    out: dict[str, Path] = {}

    # Clean baseline: saved once, never touched. The false-positive control.
    out["clean"] = d / "clean.jpg"
    Image.fromarray(photo).save(out["clean"], quality=95)

    # Spliced: a region from a file compressed hard at q=35, pasted into a
    # q=95 host. That gives it both a foreign noise floor and a recompression
    # ghost at 35 that the rest of the frame does not have.
    donor = _roundtrip(_natural(600, 800, seed=99), 35)
    spliced = photo.copy()
    spliced[180:360, 250:520] = donor[180:360, 250:520]
    out["spliced"] = d / "spliced.jpg"
    Image.fromarray(spliced).save(out["spliced"], quality=95)

    # Copy-move: a textured region duplicated at a known displacement.
    cloned = photo.copy()
    cloned[330:490, 430:630] = cloned[100:260, 120:320]
    out["copymove"] = d / "copymove.jpg"
    Image.fromarray(cloned).save(out["copymove"], quality=95)

    # Double-compressed whole image: saved low, reopened, saved high.
    out["doublejpeg"] = d / "doublejpeg.jpg"
    Image.fromarray(_roundtrip(photo, 45)).save(out["doublejpeg"], quality=94)

    return out


def _text_images(d: Path) -> dict[str, Path]:
    """A rendered document carrying realistic PII for the OCR chain."""
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (900, 400), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    try:
        bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        plain = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except OSError:  # pragma: no cover - font layout varies by platform
        bold = plain = ImageFont.load_default()

    lines = [
        (40, "ACME LOGISTICS LTD", bold),
        (110, "Contact: j.smith@acme-logistics.co.uk", plain),
        (160, "Tel: +44 20 7946 0958", plain),
        # 4111... is the standard Visa test number: valid Luhn, never issued.
        (210, "Card 4111 1111 1111 1111  exp 09/27", plain),
        (260, "12 Harbour Road, Bristol BS1 5TY", plain),
    ]
    for y, text, font in lines:
        draw.text((40, y), text, font=font, fill=(30, 30, 40))

    path = d / "document.png"
    img.save(path)
    return {"document": path}


def _code_images(d: Path) -> dict[str, Path]:
    try:
        import segno
    except ImportError:  # pragma: no cover - segno is a dev dependency
        return {}

    out = {}
    out["qr_wifi"] = d / "qr_wifi.png"
    segno.make("WIFI:T:WPA;S:AcmeGuest;P:Summer2024!;;").save(
        out["qr_wifi"], scale=8, border=4
    )
    out["qr_url"] = d / "qr_url.png"
    segno.make("https://acme-logistics.co.uk/track?id=99213").save(
        out["qr_url"], scale=8, border=4
    )
    out["qr_shortener"] = d / "qr_short.png"
    segno.make("http://bit.ly/3xYzAbC").save(out["qr_shortener"], scale=8, border=4)
    return out


@pytest.fixture(scope="session")
def corpus(images, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small directory tree with real exact- and near-duplicates."""
    import shutil

    root = tmp_path_factory.mktemp("corpus")
    (root / "sub").mkdir()

    for key in ("sharp", "photo", "blurry", "document"):
        if key in images:
            shutil.copy(images[key], root / images[key].name)

    # Exact duplicate: same bytes, different path.
    shutil.copy(images["sharp"], root / "sub" / "sharp_copy.jpg")
    # Near duplicates: resized, and recompressed.
    with Image.open(images["photo"]) as im:
        im.resize((600, 450), Image.LANCZOS).save(root / "sub" / "photo_small.jpg", quality=70)
        im.save(root / "sub" / "photo_q40.jpg", quality=40)
    # A file that will not decode — batch must still account for it.
    (root / "broken.jpg").write_bytes(b"\xff\xd8\xff not actually a jpeg")
    return root


def _dms(value: float) -> tuple:
    d = int(abs(value))
    m = int((abs(value) - d) * 60)
    s = round((abs(value) - d - m / 60) * 3600 * 100)
    return ((d, 1), (m, 1), (s, 100))


def _write_gps_images(tex: np.ndarray, good: Path, bad: Path) -> None:
    """Eiffel Tower, captured 14:30 local. CEST is UTC+2, so GPS UTC is 12:30."""
    img = Image.fromarray(tex[:600, :800])
    exif = {
        "0th": {
            _piexif.ImageIFD.Make: b"Canon",
            _piexif.ImageIFD.Model: b"Canon EOS R5",
            _piexif.ImageIFD.Software: b"Adobe Photoshop 24.0 (Macintosh)",
            _piexif.ImageIFD.Artist: b"J. Investigator",
        },
        "Exif": {
            _piexif.ExifIFD.DateTimeOriginal: b"2023:06:15 14:30:00",
            _piexif.ExifIFD.BodySerialNumber: b"042031000537",
            _piexif.ExifIFD.LensModel: b"RF24-70mm F2.8 L IS USM",
            _piexif.ExifIFD.ISOSpeedRatings: 400,
        },
        "GPS": {
            _piexif.GPSIFD.GPSLatitudeRef: b"N",
            _piexif.GPSIFD.GPSLatitude: _dms(48.8584),
            _piexif.GPSIFD.GPSLongitudeRef: b"E",
            _piexif.GPSIFD.GPSLongitude: _dms(2.2945),
            _piexif.GPSIFD.GPSAltitude: (3300, 100),
            _piexif.GPSIFD.GPSAltitudeRef: 0,
            _piexif.GPSIFD.GPSImgDirection: (1350, 10),
            _piexif.GPSIFD.GPSImgDirectionRef: b"T",
            _piexif.GPSIFD.GPSDateStamp: b"2023:06:15",
            _piexif.GPSIFD.GPSTimeStamp: ((12, 1), (30, 1), (0, 1)),
        },
        "1st": {},
        "thumbnail": None,
    }
    img.save(good, exif=_piexif.dump(exif), quality=90)

    # Same location, GPS clock six hours off — an impossible offset for Paris.
    exif["GPS"][_piexif.GPSIFD.GPSTimeStamp] = ((6, 1), (30, 1), (0, 1))
    img.save(bad, exif=_piexif.dump(exif), quality=90)


@pytest.fixture(scope="session")
def registry():
    from imgintel.core.registry import Registry

    return Registry.discover()
