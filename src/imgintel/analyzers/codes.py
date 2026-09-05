"""QR codes and barcodes — and, more importantly, what their payloads mean.

Decoding is plumbing. The intelligence is in the payload: a WiFi QR carries a
network password, an ``otpauth:`` QR carries a TOTP seed, a URL carries a host
that may be a homograph of a real one. Each of those is a finding in its own
right, and none of them is visible if you stop at "QR code detected".

zxing-cpp is used rather than pyzbar: it ships self-contained wheels on every
platform (pyzbar needs a system libzbar, which is the single most common
install failure on Windows) and reads more symbologies.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

import numpy as np

from imgintel.core.analyzer import Analyzer, AnalyzerResult, Availability, Cost, Finding, Severity
from imgintel.core.context import AnalysisContext, small_size
from imgintel.core.deps import has_module

#: Link shorteners hide the real destination.
_SHORTENERS = {
    "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc", "s.id",
}
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_CRYPTO_SCHEMES = {"bitcoin", "ethereum", "litecoin", "monero", "bitcoincash", "ton"}


class CodeAnalyzer(Analyzer):
    name = "codes"
    version = "1.0.0"
    title = "QR codes and barcodes"
    description = "Decodes 1D/2D symbologies and interprets their payloads"
    cost = Cost.MEDIUM

    def available(self) -> Availability:
        if not has_module("zxingcpp"):
            return Availability.no("zxing-cpp not installed", "pip install 'imgintel[codes]'")
        return Availability.yes()

    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult:
        import zxingcpp  # noqa: PLC0415

        # Try analysis resolution first so boxes match every other analyzer.
        # Small or dense codes sometimes need full resolution, so fall back and
        # scale the coordinates back into analysis space.
        results = zxingcpp.read_barcodes(ctx.small)
        scale = 1.0
        source = "small"

        if not results and ctx.small.shape[:2] != ctx.rgb.shape[:2]:
            results = zxingcpp.read_barcodes(ctx.rgb)
            source = "full"
            full_h, full_w = ctx.rgb.shape[:2]
            small_w, _ = small_size(full_w, full_h)
            scale = small_w / full_w if full_w else 1.0

        codes: list[dict[str, Any]] = []
        findings: list[Finding] = []

        for result in results:
            text = result.text or ""
            bbox = _position_bbox(result, scale)
            payload = interpret(text)
            entry = {
                "format": str(result.format).replace("BarcodeFormat.", ""),
                "text": text,
                "bytes": len(text),
                "bbox": list(bbox) if bbox else None,
                "payload": payload,
                "decoded_at": source,
            }
            codes.append(entry)
            findings.extend(_payload_findings(entry, bbox))

        data = {
            "count": len(codes),
            "decoded_at": source,
            "codes": codes,
        }

        if codes:
            kinds = sorted({c["payload"]["kind"] for c in codes})
            findings.insert(
                0,
                Finding(
                    "codes",
                    f"{len(codes)} code(s) decoded",
                    ", ".join(kinds),
                    Severity.INFO,
                    bbox=codes[0]["bbox"] and tuple(codes[0]["bbox"]),
                ),
            )
        return self.ok(data, findings)


# -- payload interpretation ------------------------------------------------


def interpret(text: str) -> dict[str, Any]:
    """Classify a payload and pull out the fields that carry meaning."""
    raw = text.strip()
    lowered = raw.lower()

    if lowered.startswith("wifi:"):
        return _wifi(raw)
    if lowered.startswith("otpauth://"):
        return _otpauth(raw)
    if lowered.startswith(("begin:vcard", "mecard:")):
        return _contact(raw)
    if lowered.startswith("geo:"):
        return _geo(raw)
    if lowered.startswith("mailto:"):
        return {"kind": "email", "address": raw[7:].split("?")[0]}
    if lowered.startswith(("tel:", "sms:", "smsto:")):
        return {"kind": "phone", "number": raw.split(":", 1)[1].split("?")[0]}
    if lowered.split(":", 1)[0] in _CRYPTO_SCHEMES:
        scheme, _, rest = raw.partition(":")
        return {"kind": "crypto", "chain": scheme.lower(), "address": rest.split("?")[0]}
    if lowered.startswith(("http://", "https://")):
        return _url(raw)
    if re.fullmatch(r"\d{8,14}", raw):
        return {"kind": "product_code", "value": raw}
    return {"kind": "text", "value": raw[:500]}


def _url(raw: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    info: dict[str, Any] = {
        "kind": "url",
        "url": raw[:1000],
        "scheme": parsed.scheme,
        "host": host,
        "port": parsed.port,
        "path": parsed.path,
        "query": dict(urllib.parse.parse_qsl(parsed.query)[:20]) if parsed.query else {},
        "tld": host.rsplit(".", 1)[-1] if "." in host else None,
        "flags": [],
    }
    if parsed.scheme == "http":
        info["flags"].append("insecure_scheme")
    if host in _SHORTENERS:
        info["flags"].append("url_shortener")
    if _IPV4.match(host):
        info["flags"].append("ip_literal_host")
    if host.startswith("xn--") or ".xn--" in host:
        info["flags"].append("punycode_host")
    if parsed.username or parsed.password:
        info["flags"].append("credentials_in_url")
    if parsed.port and parsed.port not in (80, 443):
        info["flags"].append("nonstandard_port")
    return info


def _wifi(raw: str) -> dict[str, Any]:
    """Parse ``WIFI:T:WPA;S:ssid;P:password;H:true;;``."""
    body = raw[5:]
    fields: dict[str, str] = {}
    # Values may contain escaped separators, so split on unescaped ';' only.
    for part in re.split(r"(?<!\\);", body):
        key, _, value = part.partition(":")
        if key:
            fields[key.upper()] = value.replace("\\;", ";").replace("\\:", ":")
    return {
        "kind": "wifi",
        "ssid": fields.get("S"),
        "password": fields.get("P"),
        "auth": fields.get("T") or "nopass",
        "hidden": fields.get("H", "").lower() == "true",
    }


def _otpauth(raw: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(raw)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    return {
        "kind": "otpauth",
        "type": parsed.netloc,
        "label": urllib.parse.unquote(parsed.path.lstrip("/")),
        "issuer": query.get("issuer"),
        "secret_present": bool(query.get("secret")),
        "algorithm": query.get("algorithm", "SHA1"),
    }


def _contact(raw: str) -> dict[str, Any]:
    info: dict[str, Any] = {"kind": "contact", "raw": raw[:500]}
    patterns = {
        "name": r"(?:^|\n)(?:FN|N):([^\n;]+)",
        "email": r"(?:^|\n|;)EMAIL[^:]*:([^\n;]+)",
        "phone": r"(?:^|\n|;)TEL[^:]*:([^\n;]+)",
        "org": r"(?:^|\n)ORG:([^\n;]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            info[key] = match.group(1).strip()
    return info


def _geo(raw: str) -> dict[str, Any]:
    body = raw[4:].split("?")[0]
    parts = body.split(",")
    try:
        return {
            "kind": "geo",
            "latitude": float(parts[0]),
            "longitude": float(parts[1]),
        }
    except (ValueError, IndexError):
        return {"kind": "geo", "raw": raw[:200]}


def _payload_findings(entry: dict, bbox: tuple | None) -> list[Finding]:
    payload = entry["payload"]
    kind = payload["kind"]
    fmt = entry["format"]
    out: list[Finding] = []

    if kind == "wifi":
        out.append(
            Finding(
                "credentials",
                "WiFi credentials in QR code",
                f"SSID {payload.get('ssid')} ({payload.get('auth')})",
                Severity.HIGH,
                confidence=0.95,
                bbox=bbox,
                detail="The network password is embedded in the code and recoverable "
                "from this image",
            )
        )
    elif kind == "otpauth" and payload.get("secret_present"):
        out.append(
            Finding(
                "credentials",
                "Two-factor authentication seed in QR code",
                payload.get("label") or payload.get("issuer"),
                Severity.HIGH,
                confidence=0.95,
                bbox=bbox,
                detail="Anyone with this image can generate valid TOTP codes for that account",
            )
        )
    elif kind == "url":
        flags = payload.get("flags") or []
        severity = Severity.MEDIUM if flags else Severity.INFO
        out.append(
            Finding(
                "codes",
                "URL in code",
                payload["url"][:120],
                severity,
                bbox=bbox,
                detail=_url_detail(flags) if flags else f"Host: {payload.get('host')}",
            )
        )
    elif kind == "crypto":
        out.append(
            Finding(
                "codes",
                "Cryptocurrency address in code",
                f"{payload.get('chain')}: {payload.get('address')}",
                Severity.MEDIUM,
                bbox=bbox,
                detail="Traceable on-chain; commonly seen in payment demands and scams",
            )
        )
    elif kind in ("contact", "email", "phone"):
        value = payload.get("name") or payload.get("address") or payload.get("number")
        out.append(
            Finding(
                "attribution",
                f"Contact details in code ({kind})",
                value,
                Severity.MEDIUM,
                bbox=bbox,
                detail="Personally identifying information encoded in the image",
            )
        )
    elif kind == "geo":
        out.append(
            Finding(
                "location",
                "Coordinates in code",
                f"{payload.get('latitude')}, {payload.get('longitude')}",
                Severity.HIGH,
                bbox=bbox,
            )
        )
    else:
        out.append(
            Finding(
                "codes",
                f"{fmt} payload",
                str(payload.get("value") or entry["text"])[:120],
                Severity.INFO,
                bbox=bbox,
            )
        )
    return out


def _url_detail(flags: list[str]) -> str:
    explanations = {
        "insecure_scheme": "plain HTTP",
        "url_shortener": "shortened link hides the real destination",
        "ip_literal_host": "bare IP address instead of a domain",
        "punycode_host": "punycode host — check for lookalike characters",
        "credentials_in_url": "credentials embedded in the URL",
        "nonstandard_port": "non-standard port",
    }
    return "; ".join(explanations.get(f, f) for f in flags)


def _position_bbox(result: Any, scale: float) -> tuple[int, int, int, int] | None:
    position = getattr(result, "position", None)
    if position is None:
        return None
    points = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, name, None)
        if point is not None:
            points.append((getattr(point, "x", 0), getattr(point, "y", 0)))
    if not points:
        return None
    arr = np.asarray(points, dtype=float) * scale
    return (
        int(arr[:, 0].min()), int(arr[:, 1].min()),
        int(np.ceil(arr[:, 0].max())), int(np.ceil(arr[:, 1].max())),
    )
