# Writing analyzers

Everything imgintel does is an analyzer. There is no privileged built-in path — the twenty-five shipped analyzers use exactly the interface described here, so anything they can do, a plugin can do.

There are two ways in, and the first one is often enough.

## 1. Rules, without Python

Most custom checks are one of three shapes: match a pattern against extracted text, assert something about metadata, or compare a number to a threshold. Those need a YAML file, not a class.

```bash
imgintel plugins new house-rules --kind rules   # writes a starter file
imgintel plugins dirs                           # shows where rules live
```

```yaml
name: house-rules
description: Checks specific to our casework

rules:
  - id: internal-marking
    when:
      text_matches: '(?i)\bCOMPANY CONFIDENTIAL\b'
    finding:
      category: sensitive
      label: Internal document marking
      severity: high
      detail: Carries our internal classification marking

  - id: drone-capture
    when:
      exif_contains: {field: model, value: FC3411}
    finding:
      category: device
      label: DJI drone capture
      severity: medium

  - id: too-small
    when:
      value_below: {path: "fileinfo.megapixels", threshold: 0.2}
    finding:
      category: quality
      label: Image too small for reliable analysis
      severity: low
```

Rule files load **automatically** — no `--allow-local-plugins`. They are data: parsed, never executed, and unable to do anything the condition evaluators cannot. The opt-in exists for arbitrary Python, and demanding it here would only push people toward the riskier option.

Conditions:

| Condition | Argument | Matches when |
|---|---|---|
| `text_matches` | regex | the pattern is found in any extracted text |
| `text_contains` | string | case-insensitive substring is present |
| `exif_contains` | `{field, value}` | that EXIF field contains that value |
| `exif_missing` | field name | the field is absent or empty |
| `value_above` | `{path, threshold}` | `analyzer.key` exceeds the threshold |
| `value_below` | `{path, threshold}` | `analyzer.key` is below the threshold |
| `value_equals` | `{path, value}` | `analyzer.key` equals that value |
| `finding_present` | label fragment | another analyzer raised a matching finding |

"Extracted text" means OCR output, barcode payloads and EXIF free-text fields. A `path` is `analyzer.key.subkey` into any analyzer's `data` — `imgintel analyze IMG --show-data` shows what is available.

YAML needs `pip install 'imgintel[rules]'`; JSON rule files work with no extra dependency.

Check a file before relying on it:

```bash
imgintel plugins validate ~/.config/imgintel/rules/house-rules.yaml
```

## 2. Python analyzers

```bash
imgintel plugins new anpr        # scaffolds an installable package
cd imgintel-anpr && pip install -e .
imgintel plugins                 # anpr is now listed
```

Import from **`imgintel.plugins`** — the public SDK. Importing from `imgintel.core.*` couples you to internals that move: `precheck` arrived in Phase 5 and `after` in Phase 3, both by changing files under `core`.

```python
from imgintel.plugins import Analyzer, Availability, Cost, Finding, Severity, require_api

require_api("1.0")   # fails at import, not mysteriously mid-run


class PlateAnalyzer(Analyzer):
    name = "anpr"                  # unique; used in --analyzers and in `requires`
    version = "1.0.0"              # bump when the shape of `data` changes
    title = "Number plates"
    description = "One line, shown by `imgintel plugins`"

    requires = ("objects",)        # must run first; skipped if it fails
    after = ("ocr",)               # run after if present, but not required
    cost = Cost.MEDIUM
    needs_network = False
    needs_models = ()

    def precheck(self, ctx):
        """Refuse on grounds of permission. Return a reason, or None."""
        return None

    def available(self) -> Availability:
        """Report missing binaries or models. Must never raise."""
        return Availability.yes()

    def analyze(self, ctx):
        detections = ctx.data("objects").get("detections", [])
        findings = [...]
        return self.ok({"count": len(detections)}, findings)
```

### API stability

`imgintel.plugins.API_VERSION` is semantic and independent of imgintel's own version. Within a major version the exported names keep working: nothing is removed, no required argument is added, no return shape changes. New *optional* hooks may appear — a plugin that ignores them keeps working, which is how `precheck` was added without breaking anything.

`require_api("1.0")` fails loudly at import if the installed imgintel is too old.

## The context

`ctx` is the only input. Derived forms are computed lazily and cached, so reading them is free after the first analyzer does.

| Attribute | Type | Decodes? | Notes |
|---|---|---|---|
| `ctx.path` | `Path` | no | |
| `ctx.raw` | `bytes` | no | whole file |
| `ctx.header` | `PIL.Image` | **no** | opened, not decoded — size, mode, format, `info` |
| `ctx.jpeg_quant_tables` | `dict \| None` | no | from the header |
| `ctx.readable` | `bool` | no | header parses |
| `ctx.pil_raw` | `PIL.Image` | yes | decoded, **no** EXIF orientation applied |
| `ctx.pil` | `PIL.Image` | yes | EXIF-oriented — what a human sees |
| `ctx.rgb` | `ndarray` HxWx3 uint8 | yes | oriented, alpha composited on white |
| `ctx.gray` | `ndarray` HxW float32 | yes | BT.601 luminance, 0–255 |
| `ctx.small` / `ctx.small_gray` | `ndarray` | yes | longest edge ≤ 1024 — **prefer these** |
| `ctx.sharpness` | `float` | yes | variance-of-Laplacian over `small_gray` |
| `ctx.dimensions`, `ctx.megapixels` | | yes | oriented |
| `ctx.decodable` | `bool` | yes | pixel data actually decodes |
| `ctx.allow_network` | `bool` | no | |
| `ctx.allow_biometrics` | `bool` | no | per-run consent; never inherited from config |
| `ctx.entity_db` | `Path \| None` | no | entity database to match against |
| `ctx.case` | `CaseConfig` | no | case id, operator, biometric lawful basis |
| `ctx.planned` | `frozenset[str]` | no | every analyzer in this run |
| `ctx.scratch` | `dict` | no | non-serializable handoff (loaded models, etc.) |

Work on `ctx.small` unless you specifically need full resolution. It bounds your analyzer's cost regardless of input size, and it is what every shipped pixel analyzer uses.

**Bounding boxes are always in `ctx.small` coordinates.** OCR blocks, barcode corners, blur tiles, PII matches and the HTML overlay all agree because of this. If you must run at full resolution (as `codes` does when a small QR fails to decode), scale back before emitting; `imgintel.core.context.small_size(w, h)` gives the target dimensions.

If two analyzers would both compute something, add it to `AnalysisContext` as a `cached_property` rather than duplicating it. That is how `sharpness` came to live there.

## Choosing a cost

`cost` is the only thing deciding which profile includes an analyzer, so it is a contract with the user, not a hint.

| Cost | Meaning | Rule of thumb |
|---|---|---|
| `CHEAP` | Header and metadata only — **must not touch `ctx.rgb`, `ctx.gray` or `ctx.small`** | sub-millisecond |
| `MEDIUM` | Needs the decoded image, single pass | tens of ms |
| `HEAVY` | Model inference or multi-pass pixel work | 100 ms+ |

A `CHEAP` analyzer that decodes pixels silently breaks the `quick` profile's whole promise. `imgintel plugins validate --sample IMAGE` catches it by running your analyzer with the decode path instrumented — it reports what your code *did*, not what it looks like it does.

Profile selection is dependency-aware: an analyzer whose `requires` are out of reach is dropped from a profile rather than listed and then always skipped.

## Refusing: permission vs capability

| Hook | Question | Runs |
|---|---|---|
| `precheck(ctx)` | *May* this run? | first — before dependencies, network gate and `available()` |
| `available()` | *Can* this run? | after dependencies are satisfied |

The ordering matters. A face-identification run with no lawful basis must record "no lawful basis", not "the model is missing" — both may be true, and only the first is the governing reason. Whatever `precheck` returns is what ends up in the report and the evidence bundle.

`available()` must never raise; return an `Availability` with a reason *and an actionable hint*:

```python
def available(self):
    from imgintel.plugins import binary_path
    if not binary_path("tesseract"):
        return Availability.no("tesseract not found", "brew install tesseract")
    return Availability.yes()
```

Exceptions from `analyze()` are caught and recorded as `status: error` with a traceback, so a bug in your analyzer never loses another's results. Rely on that for genuine bugs, not for expected conditions.

## Hard vs soft dependencies

| Field | Meaning | If the other analyzer is absent or failed |
|---|---|---|
| `requires` | I cannot work without this | this analyzer is **skipped** |
| `after` | Order me after this **if present** | this analyzer **runs anyway** |

`sensitive` uses both: it `requires = ("ocr",)` because it has no text to scan otherwise, and declares `after = ("codes",)` so it can also scan barcode payloads when they exist — without zxing-cpp's absence disabling PII detection.

Never rely on alphabetical or cost ordering. It happens to work and will silently stop the moment someone renames an analyzer.

Use `ctx.planned` to avoid duplicating a finding another analyzer will make.

## Findings vs data

This split is the core of the design.

- **`data`** is your own structure. Rich, nested, whatever you need. It lands in `analyzers.<name>.data` verbatim (numpy types are coerced for you).
- **`findings`** is a flat, normalized list. Renderers, CSV export, the summary and future outputs consume *only* findings.

Because renderers never look at `data`, adding an analyzer needs zero changes to any output code. If you want a renderer to special-case your analyzer, the information belongs in a `Finding` instead.

```python
Finding(
    category="location",       # groups related findings across analyzers
    label="GPS coordinates",   # short, human-readable
    value="48.8584, 2.2945",   # the actual value
    severity=Severity.HIGH,    # INFO | LOW | MEDIUM | HIGH
    confidence=0.9,            # 0..1
    bbox=(x0, y0, x1, y1),     # optional, in ctx.small coordinates
    detail="What this means, and what could explain it innocently",
)
```

**Severity means investigative significance, not certainty.** Use `confidence` for certainty. Camera model is `INFO` even though we are sure of it; a GPS fix is `HIGH` because it changes the picture.

Write `detail` for the person who has to act on the finding, and include the benign explanation when there is one. "Regions markedly softer than the rest of an otherwise sharp frame — shallow depth of field produces the same pattern; confirm visually" is useful. "SUSPICIOUS" is not.

**Mask sensitive values in findings, keep them whole in `data`.** Findings feed the HTML report and CSV, which get shared; the JSON is for the investigator.

## Models

```python
needs_models = ("detect",)

def available(self):
    from imgintel.core.modelstore import available as models_available
    ok, reason = models_available(*self.needs_models)
    if ok:
        return Availability.yes()
    return Availability.no(reason, f"imgintel models pull {' '.join(self.needs_models)}")
```

Register the model in `imgintel.core.modelstore.MODELS` with mirror URLs, or as `kind="bundled"` when a dependency's wheel carries it. `imgintel models list` then accounts for it and evidence bundles record its hash and licence.

Set `sha256` only for a file you have actually verified. Leaving it `None` marks the model *unpinned*: it installs and runs, says so, and the user can `imgintel models pin` after checking it themselves. **A guessed hash is worse than none** — it rejects every legitimate download.

**Record a real licence.** `licence="unknown"` fails a test. Weights carry licences independent of the library loading them; the two that catch people are Ultralytics YOLO (AGPL-3.0) and InsightFace's pretrained models (non-commercial research only).

**Build heavy state once per process**, via `@lru_cache` on a module-level factory — `imgintel.util.onnx.load_session` already does this for ONNX graphs. Not a micro-optimization: constructing `TimezoneFinder` per image once cost 340 ms and made `quick` 250× slower than it should have been.

## Testing

`tests/fakemodel.py` builds a real ONNX graph returning detections you choose, so session, letterbox, decode, NMS and coordinate mapping are exercised against boxes known to the pixel — with no download in CI. Prefer that over mocking the session: a mock verifies your code calls what you expected, which is not the same as verifying it works.

For fixtures, note that image statistics matter. Smooth periodic gradients are so self-similar that copy-move detection trips on them, and their DCT coefficients do not follow the Benford-like law double-compression detection depends on. `tests/conftest.py` synthesises 1/f noise for anything forensic.

```bash
imgintel plugins validate anpr --sample photo.jpg
```

Validation checks the contract, and with `--sample` it also runs your analyzer twice to catch cost dishonesty and non-determinism.

## Checklist

- [ ] Imports come from `imgintel.plugins`, not `imgintel.core`
- [ ] `name`, `version`, `description` set; `name` unique
- [ ] `cost` honest — `CHEAP` means it never touches pixels
- [ ] `precheck()` for what it may not do; `available()` for what it cannot
- [ ] `available()` returns rather than raises, with an actionable hint
- [ ] Heavy state built once per process, not per image
- [ ] `data` is JSON-safe in spirit (numpy is coerced; custom objects are not)
- [ ] Bounding boxes in `ctx.small` coordinates
- [ ] Findings carry severity, confidence and a `detail` including benign explanations
- [ ] Sensitive values masked in findings, whole in `data`
- [ ] Cross-analyzer reads go through `requires`/`after` + `ctx.data()`, never imports
- [ ] Deterministic: same input, same output, seeds included
- [ ] `imgintel plugins validate <name> --sample IMAGE` is clean
