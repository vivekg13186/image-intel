"""Self-contained HTML report.

Everything is inlined — the thumbnail as a data URI, the CSS in a style block —
so the file can be attached to an email or dropped into an evidence bundle and
still render years later with no network and no sibling files.

Bounding boxes are drawn as an SVG overlay rather than burned into the image,
which keeps them crisp at any zoom and lets each carry a hover label.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
from typing import Any

from imgintel.core.context import SMALL_EDGE, small_size
from imgintel.core.schema import FindingsDocument

SEVERITY_ORDER = ("high", "medium", "low", "info")
SEVERITY_COLOR = {
    "high": "#c0392b",
    "medium": "#c77700",
    "low": "#1f6feb",
    "info": "#57606a",
}

_CSS = """
:root{--bg:#f6f7f9;--fg:#1c2128;--muted:#57606a;--card:#fff;--line:#d8dee4;
--high:#c0392b;--medium:#c77700;--low:#1f6feb;--info:#57606a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
margin:32px 0 12px;font-weight:600}
.sub{color:var(--muted);font-size:13px;word-break:break-all;margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px;
margin-bottom:14px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 20px}
.chip{border-radius:999px;padding:4px 12px;font-size:12px;font-weight:600;color:#fff}
.chip.zero{background:#8b949e;opacity:.55}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.kv{display:flex;justify-content:space-between;gap:14px;padding:5px 0;
border-bottom:1px solid #eef1f4;font-size:13px}
.kv:last-child{border-bottom:0}
.kv .k{color:var(--muted);white-space:nowrap}
.kv .v{text-align:right;word-break:break-word;font-variant-numeric:tabular-nums}
.finding{border-left:4px solid var(--line);padding:10px 14px;margin-bottom:8px;
background:var(--card);border-radius:0 6px 6px 0;border-top:1px solid var(--line);
border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.finding.high{border-left-color:var(--high)}
.finding.medium{border-left-color:var(--medium)}
.finding.low{border-left-color:var(--low)}
.finding.info{border-left-color:var(--info)}
.finding .top{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.finding .label{font-weight:600}
.finding .meta{color:var(--muted);font-size:11px;white-space:nowrap}
.finding .val{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
word-break:break-all;margin-top:3px}
.finding .detail{color:var(--muted);font-size:12.5px;margin-top:5px}
.imgwrap{position:relative;display:inline-block;max-width:100%;line-height:0}
.imgwrap img{max-width:100%;height:auto;border-radius:6px;display:block}
.imgwrap svg{position:absolute;inset:0;width:100%;height:100%}
a{color:#0969da}
details{margin-bottom:8px}
summary{cursor:pointer;font-weight:600;padding:9px 12px;background:var(--card);
border:1px solid var(--line);border-radius:6px;font-size:13px}
summary::marker{color:var(--muted)}
pre{background:#0d1117;color:#c9d1d9;padding:14px;border-radius:0 0 6px 6px;overflow:auto;
font-size:12px;margin:0;max-height:460px;line-height:1.5}
.status{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:600}
.status.ok{background:#dafbe1;color:#116329}
.status.error{background:#ffebe9;color:#82071e}
.status.unavailable,.status.skipped{background:#fff1c2;color:#7a5b00}
footer{margin-top:40px;padding-top:16px;border-top:1px solid var(--line);
color:var(--muted);font-size:12px}
.note{background:#fff8e1;border:1px solid #f0d68a;border-radius:6px;padding:11px 14px;
font-size:12.5px;color:#5c4700;margin-bottom:18px}
@media print{body{background:#fff}.card,.finding,summary{break-inside:avoid}details{display:none}}
"""

_DISCLAIMER = (
    "Findings are indicators, not verdicts. Severity reflects investigative "
    "significance; confidence reflects certainty. Read each finding's explanation "
    "before acting on it — several have benign explanations."
)


def render_html(
    doc: FindingsDocument,
    image_path: Path | str | None = None,
    *,
    embed_image: bool = True,
    include_raw_data: bool = True,
) -> str:
    """Render a complete standalone report document."""
    src = Path(image_path) if image_path else Path(doc.target.path)
    thumb, dims = (_thumbnail(src) if embed_image else (None, None))

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>imgintel — {html.escape(doc.target.filename)}</title>",
        f"<style>{_CSS}</style></head><body><div class='wrap'>",
        _header(doc),
        _chips(doc),
        f"<div class='note'>{_DISCLAIMER}</div>",
    ]

    if thumb:
        parts.append(_image_section(doc, thumb, dims))

    parts.append(_summary_section(doc))
    parts.append(_key_facts(doc))
    parts.append(_findings(doc))
    parts.append(_analyzer_status(doc))
    if include_raw_data:
        parts.append(_raw_data(doc))
    parts.append(_footer(doc))
    parts.append("</div></body></html>")
    return "\n".join(p for p in parts if p)


# -- sections --------------------------------------------------------------


def _header(doc: FindingsDocument) -> str:
    bits = [f"<h1>{html.escape(doc.target.filename)}</h1>",
            f"<p class='sub'>{html.escape(doc.target.path)}</p>"]
    if doc.case.case_id or doc.case.operator:
        meta = " · ".join(
            filter(None, [
                f"Case {html.escape(doc.case.case_id)}" if doc.case.case_id else None,
                f"Operator {html.escape(doc.case.operator)}" if doc.case.operator else None,
            ])
        )
        bits.append(f"<p class='sub'>{meta}</p>")
    return "".join(bits)


def _chips(doc: FindingsDocument) -> str:
    chips = []
    for sev in SEVERITY_ORDER:
        n = doc.summary.by_severity.get(sev, 0)
        cls = "chip" if n else "chip zero"
        style = f"background:{SEVERITY_COLOR[sev]}" if n else ""
        chips.append(f"<span class='{cls}' style='{style}'>{n} {sev}</span>")
    chips.append(
        f"<span class='chip' style='background:#57606a'>"
        f"{doc.summary.analyzers_ok} analyzers · {doc.run.duration_ms:.0f} ms</span>"
    )
    return f"<div class='chips'>{''.join(chips)}</div>"


def _image_section(doc: FindingsDocument, thumb: str, dims: tuple[int, int]) -> str:
    """Thumbnail with an SVG overlay of every finding that carries a bbox."""
    width, height = dims
    boxes = [f for f in doc.findings if f.bbox]
    shapes = []
    for f in boxes:
        x0, y0, x1, y1 = f.bbox
        color = SEVERITY_COLOR.get(f.severity, "#57606a")
        title = html.escape(f"{f.label} ({f.analyzer})")
        shapes.append(
            f"<g><title>{title}</title>"
            f"<rect x='{x0}' y='{y0}' width='{max(1, x1 - x0)}' height='{max(1, y1 - y0)}' "
            f"fill='{color}' fill-opacity='0.12' stroke='{color}' stroke-width='2'/></g>"
        )
    overlay = (
        f"<svg viewBox='0 0 {width} {height}' preserveAspectRatio='xMidYMid meet'>"
        f"{''.join(shapes)}</svg>"
        if shapes
        else ""
    )
    caption = (
        f"<p class='sub' style='margin-top:8px'>{len(boxes)} region(s) highlighted. "
        f"Coordinates are in analysis space (longest edge {SMALL_EDGE}px).</p>"
        if shapes
        else ""
    )
    return (
        f"<div class='card'><div class='imgwrap'>"
        f"<img src='{thumb}' alt='Analyzed image' width='{width}' height='{height}'>"
        f"{overlay}</div>{caption}</div>"
    )


def _summary_section(doc: FindingsDocument) -> str:
    """The written rollup, when the summary analyzer ran."""
    block = doc.analyzers.get("summary")
    if block is None or block.status != "ok":
        return ""
    text = block.data.get("text")
    if not text:
        return ""
    return (
        "<h2>Summary</h2><div class='card'>"
        f"<pre style='background:transparent;color:inherit;padding:0;max-height:none;"
        f"white-space:pre-wrap;font:inherit'>{html.escape(text)}</pre></div>"
    )


def _key_facts(doc: FindingsDocument) -> str:
    """The handful of fields someone checks first, in three grouped cards."""
    from imgintel.report.flatten import flatten

    row = flatten(doc)
    # Entries are (label, value, value_is_trusted_html). Only the two helpers
    # that build markup themselves are marked raw; everything else is escaped.
    groups: list[tuple[str, list[tuple[str, Any, bool]]]] = [
        ("File", [
            ("Format", row["format"], False),
            ("Size", _bytes(row["size_bytes"]), False),
            ("Dimensions", _dims(row), False),
            ("SHA-256", row["sha256"], False),
            ("pHash", row["phash"], False),
        ]),
        ("Device & provenance", [
            ("Camera", row["camera"], False),
            ("Lens", row["lens"], False),
            ("Serial", row["serial"], False),
            ("Software", row["software"], False),
            ("Artist", row["artist"], False),
            ("Captured", row["captured"], False),
            ("Filename hints", row["filename_source"], False),
        ]),
        ("Location", [
            ("Coordinates", _coords(doc, row), True),
            ("Place", row["place"], False),
            ("Altitude", f"{row['altitude_m']} m" if row["altitude_m"] is not None else None, False),
            ("Bearing", f"{row['bearing_deg']}°" if row["bearing_deg"] is not None else None, False),
            ("Timezone", row["timezone"], False),
        ]),
        ("Pixels", [
            ("JPEG quality", row["jpeg_quality"], False),
            ("Sharpness", row["sharpness"], False),
            ("Focus", row["blur_verdict"], False),
            ("Noise σ", row["noise_sigma"], False),
            ("Entropy", row["entropy"], False),
            ("Dominant colour", _swatch(row["dominant_color"]), True),
        ]),
        ("Content", [
            ("Text characters", row["ocr_text_chars"], False),
            ("Text blocks", row["ocr_blocks"], False),
            ("Languages", row["languages"], False),
            ("Barcodes", row["barcodes"], False),
            ("Sensitive data", row["pii_types"], False),
        ]),
        ("Detections", [
            ("Scene", row["scene"], False),
            ("Objects", row["objects"], False),
            ("People", row["people_count"], False),
            ("Faces", row["face_count"], False),
            ("Vehicles", row["vehicle_count"], False),
            ("Plate text", row["plate_text"], False),
            ("Logos", row["logos"], False),
        ]),
        ("Entity matches", [
            ("Identities", row["identities"], False),
            ("Identity verdicts", row["identity_verdicts"], False),
            ("Matched objects", row["matched_objects"], False),
        ]),
        ("Integrity & estimation", [
            ("Tamper indicators", row["tamper_indicators"], False),
            ("Estimated country", row["estimated_country"], False),
        ]),
    ]

    cards = []
    for title, entries in groups:
        rows = [
            f"<div class='kv'><span class='k'>{html.escape(k)}</span>"
            f"<span class='v'>{v if raw else html.escape(str(v))}</span></div>"
            for k, v, raw in entries
            if v is not None and v != ""
        ]
        if rows:
            # Escaped like everything else: "Integrity & estimation" would
            # otherwise emit a bare ampersand.
            cards.append(
                f"<div class='card'><h2 style='margin-top:0'>{html.escape(title)}</h2>"
                f"{''.join(rows)}</div>"
            )
    return f"<h2>Key facts</h2><div class='grid'>{''.join(cards)}</div>" if cards else ""


def _findings(doc: FindingsDocument) -> str:
    if not doc.findings:
        return "<h2>Findings</h2><div class='card'>No findings.</div>"
    out = ["<h2>Findings</h2>"]
    for f in doc.findings:
        value = (
            f"<div class='val'>{html.escape(str(f.value))}</div>" if f.value is not None else ""
        )
        detail = f"<div class='detail'>{html.escape(f.detail)}</div>" if f.detail else ""
        conf = f" · {f.confidence:.0%} confidence" if f.confidence < 1 else ""
        out.append(
            f"<div class='finding {f.severity}'>"
            f"<div class='top'><span class='label'>{html.escape(f.label)}</span>"
            f"<span class='meta'>{f.category} · {f.analyzer}{conf}</span></div>"
            f"{value}{detail}</div>"
        )
    return "".join(out)


def _analyzer_status(doc: FindingsDocument) -> str:
    problems = [b for b in doc.analyzers.values() if b.status != "ok"]
    if not problems:
        return ""
    rows = "".join(
        f"<div class='kv'><span class='k'>{html.escape(b.analyzer)} "
        f"<span class='status {b.status}'>{b.status}</span></span>"
        f"<span class='v'>{html.escape(b.error or '')}</span></div>"
        for b in sorted(problems, key=lambda x: x.analyzer)
    )
    return f"<h2>Analyzers not run</h2><div class='card'>{rows}</div>"


def _raw_data(doc: FindingsDocument) -> str:
    out = ["<h2>Raw analyzer output</h2>"]
    for name, block in sorted(doc.analyzers.items()):
        body = json.dumps(block.data, indent=2, default=str, ensure_ascii=False)
        out.append(
            f"<details><summary>{html.escape(name)} "
            f"<span class='status {block.status}'>{block.status}</span> "
            f"· {block.duration_ms:.1f} ms</summary>"
            f"<pre>{html.escape(body)}</pre></details>"
        )
    return "".join(out)


def _footer(doc: FindingsDocument) -> str:
    t = doc.tool
    bits = [
        f"{t.name} {t.version} · schema {t.schema_version} · Python {t.python} · {t.platform}",
        f"Analyzed {doc.target.analyzed_at} · profile {doc.run.profile or 'custom'}",
        f"Analyzers: {', '.join(doc.run.analyzers_run)}",
    ]
    if doc.run.command:
        bits.append(f"Command: {html.escape(doc.run.command)}")
    return "<footer>" + "<br>".join(bits) + "</footer>"


# -- helpers ---------------------------------------------------------------


def _thumbnail(path: Path, edge: int = SMALL_EDGE) -> tuple[str | None, tuple[int, int] | None]:
    """Base64 JPEG at exactly the analysis dimensions, so bboxes line up."""
    try:
        from PIL import Image, ImageOps

        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw) or raw
            img = img.convert("RGB")
            target = small_size(*img.size, edge=edge)
            if target != img.size:
                img = img.resize(target, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}", target
    except Exception:  # noqa: BLE001 - a missing thumbnail must not lose the report
        return None, None


def _bytes(n: int | None) -> str | None:
    if n is None:
        return None
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    return None


def _dims(row: dict) -> str | None:
    if not row["width"]:
        return None
    return f"{row['width']} x {row['height']} ({row['megapixels']} MP)"


def _coords(doc: FindingsDocument, row: dict) -> str | None:
    if row["latitude"] is None:
        return None
    block = doc.analyzers.get("gps")
    url = block.data.get("google_maps") if block else None
    text = f"{row['latitude']}, {row['longitude']}"
    return f"<a href='{html.escape(url)}'>{text}</a>" if url else text


def _swatch(hex_color: str | None) -> str | None:
    if not hex_color:
        return None
    safe = html.escape(hex_color)
    return (
        f"<span style='display:inline-block;width:11px;height:11px;border-radius:2px;"
        f"background:{safe};border:1px solid #ccc;vertical-align:-1px;margin-right:6px'></span>"
        f"{safe}"
    )
