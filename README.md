# imgintel

Extract intelligence from images, from the command line. Runs entirely offline.

```
$ imgintel analyze photo.jpg

╭─────────────────────────── imgintel ────────────────────────────╮
│ photo.jpg                                                       │
│ 22 findings · 12 ok · 0 failed · 84 ms                          │
╰─────────────────────────────────────────────────────────────────╯
 Sev      Category      Finding                        Value
 high     location      GPS coordinates                48.858400, 2.294500
 medium   attribution   Identifying tag: SerialNumber  042031000537
 medium   location      Nearest known place            Vanves, Ile-de-France, FR
 medium   manipulation  Edited in image-editing software  Adobe Photoshop 24.0
 low      attribution   Identifying tag: Artist        J. Investigator
 info     device        Camera                         Canon EOS R5
 info     location      Camera bearing                 135° (SE)
 info     timeline      Timestamp consistent with GPS timezone  Europe/Paris
```

---

## Install

Python 3.10–3.14.

```bash
pip install -e ".[all]"     # everything
pip install -e .            # core only — metadata, hashes, quality, blur
```

Then check what you have:

```bash
imgintel doctor             # what's installed, what's missing, how to fix it
```

Extras are independent — if one fails, install the rest. A missing extra degrades its analyzer rather than breaking the tool.

| Extra | Unlocks |
|---|---|
| `[exif]` | XMP, IPTC, MakerNotes — camera serials, lens IDs, owner names |
| `[timezone]` | The GPS/EXIF timestamp cross-check |
| `[places]` | Offline place names for coordinates |
| `[ocr]` | Text extraction (weights ship in the wheel — no download) |
| `[codes]` | QR and barcode decoding |
| `[lang]` | Language identification |
| `[pii]` | Validated phone-number detection |
| `[detect]` | Object detection |
| `[faces]` | Face detection and identification |
| `[logo]` | Reference logo and object matching |
| `[rules]` | YAML rule files (JSON works without it) |

> **Python 3.13+:** use `rapidocr`, not `rapidocr-onnxruntime` — the old package caps at `<3.13` and pip fails with a confusing "Ignored the following versions" message. `[ocr]` installs the right one.

## Quick start

```bash
imgintel analyze photo.jpg                    # standard checks, ~80 ms
imgintel analyze photo.jpg -p deep            # everything, ~700 ms
imgintel analyze photo.jpg --html report.html # shareable report
imgintel batch ./photos --out results/        # a whole directory
imgintel dedupe ./photos                      # find duplicates
imgintel compare a.jpg b.jpg                  # same image?
```

Exit codes: `0` clean, `1` an analyzer errored (partial results still written), `2` bad invocation.

## What it finds

**Provenance** — where the file came from

- Extension/content mismatch (a `.jpg` whose bytes are PNG)
- Filename origin: WhatsApp, Signal, Pixel, DJI, GoPro, screenshots
- Stripped metadata — distinguishing "no EXIF" from "only JFIF padding", which is the signal a platform re-encoded it
- Device identity: body serial numbers, owner names, lens IDs
- Generative-AI markers: Stable Diffusion / ComfyUI parameters, C2PA manifests
- Screenshot vs photograph, from entropy against edge energy
- Platform dimension presets (1080×1080 and friends)

**Location and time**

- GPS coordinates, altitude, camera bearing, offline place names
- **GPS/timestamp cross-check** — EXIF local time versus GPS UTC implies a UTC offset; comparing it against the timezone the coordinates fall in catches edited clocks and faked locations
- Country estimate from text alone: phone codes, postcodes, country domains, company suffixes, currency symbols

**Content**

- Text (OCR), with per-block boxes and confidence
- **QR payloads that matter** — a `WIFI:` code carries a live network password; an `otpauth:` code carries a 2FA seed
- Suspicious URLs: shorteners, bare IPs, punycode lookalikes
- Sensitive data: card numbers (Luhn-checked), IBANs (mod-97), phone numbers, API keys, JWTs — each with the box it came from
- Objects, people, faces, vehicles and plate regions, scene, logos

**Integrity**

- Seven independent manipulation indicators — see [the note below](#on-manipulation-detection)
- **Localised blur** — regions far softer than an otherwise sharp frame. The redaction signal, and it needs no calibration: each tile is compared against the image's own median

Every finding carries a severity, a confidence, and usually an explanation of what could cause it innocently.

## Profiles

| Profile | Per image | What runs |
|---|---|---|
| `quick` | ~1 ms | Metadata, hashes, GPS — **no pixel decode at all** |
| `standard` | ~80 ms | The above plus perceptual hashes, colour, quality, blur, barcodes |
| `deep` | ~700 ms | Everything: OCR, sensitive data, detection, forensics, entity matching |

Times are steady-state. A cold start adds ~0.2 s, and the first GPS image loads the offline gazetteer (~0.7 s, once per process).

## Working at scale

```bash
imgintel batch ./photos --out results/ --workers 8 --resume --html
```

```
results/
  summary.csv      one row per image — 60 columns, sortable, pivotable
  findings.csv     one row per finding
  findings.jsonl   full documents
  index.db         sqlite: resume, dedupe, cross-image queries
  html/            per-image reports
```

Every input gets a row, including files that fail to decode. Silent omission is how someone concludes a file was clean when it was never read.

```bash
imgintel dedupe results/index.db --threshold 8
```

Finds exact duplicates (SHA-256), pixel-identical files with different metadata, and near-duplicates. Linking uses **whichever perceptual hash is closest** — each fails on a different transform, and a plain resize can land 16 bits away in pHash but 4 in dHash.

## Evidence bundles

```bash
imgintel analyze photo.jpg --json findings.json --case C-001 --operator vivek
imgintel evidence export findings.json --out ./evidence
imgintel evidence verify ./evidence/C-001_photo.jpg_20260905T101500Z
```

A bundle still means something when handed to someone else: the original file bit-for-bit, `findings.json`, a self-contained `report.html`, `findings.csv`, a `provenance.json` naming every tool, analyzer and model version with hashes, and a `MANIFEST.sha256` over all of it.

`verify` re-hashes everything and reports modified, missing **and planted** files, exiting non-zero on any of them. The subject file is only ever copied, never rewritten.

## Models

OCR needs no download — the weights ship in the wheel. Object and face detection need two files:

```bash
imgintel models list                    # what's needed, its licence, where it goes
imgintel models pull                    # fetch them
imgintel models import ~/Downloads      # ...or install files you downloaded yourself
imgintel models pin                     # record digests once you've verified them
```

If the machine can't reach the mirrors, download the files anywhere and either `models import` them or point `IMGINTEL_MODEL_DIR` at the folder. That's a first-class path — air-gapped forensic workstations are normal.

<details>
<summary><b>If <code>models pull</code> fails with CERTIFICATE_VERIFY_FAILED</b></summary>

That's a local trust-store problem, not a bad server — macOS python.org builds ship without a CA bundle wired up.

```bash
pip install certifi                                    # imgintel picks it up automatically
open "/Applications/Python 3.14/Install Certificates.command"
imgintel models import ~/Downloads/yolox_s.onnx        # or download in a browser
imgintel models pull face --insecure                   # last resort
```

Downloading in a browser and importing is the better fallback. These models are unpinned, so `--insecure` means *no* integrity check at all — neither certificate nor hash — and ONNX graphs are parsed by onnxruntime. It works if you ask for it, warns loudly, and records `tls_verified: false` in the model receipt so it appears in `models list` and in evidence provenance.
</details>

<details>
<summary><b>Why models ship "unpinned"</b></summary>

The detection models carry no recorded hash. imgintel has not verified those specific published files, and a guessed hash would reject every legitimate download. So it installs them, says plainly they're unpinned, and offers `models pin` to record the digest once *you* have checked it against the upstream checksum. From then on verification is exact, and evidence bundles record `pinned: true`.

A model that was never checked against a known-good reference is recorded as such. That distinction matters if findings are challenged.
</details>

## Matching people and objects

Find the same person or object across a set of images.

```bash
# objects need no special authority — a photograph of a bag is not biometric data
imgintel db enrol blue-holdall ./refs/bag*.jpg --kind object --db case.db
imgintel analyze scene.jpg -a objectdb --entity-db case.db

# faces require a recorded lawful basis AND a per-run flag
imgintel case set --case-id C-001 --operator vivek \
  --biometric-basis "Warrant 2026/114, Art. 9(2)(f)" \
  --biometric-authorised-by "DI Shaw" --biometric-expires 2027-01-01
imgintel db enrol "Jane Doe" ./refs/jane*.jpg --kind person --db case.db --allow-biometrics
imgintel analyze scene.jpg -p deep --entity-db case.db --allow-biometrics

imgintel db list --db case.db
imgintel db audit --db case.db      # every enrolment and search, timestamped
imgintel db forget "Jane Doe" --db case.db
```

**Objects** are matched by ORB features and a homography check — no model, no download, and a match reports how many keypoints agreed and where. Strong on rigid textured subjects, unreliable on deformable ones, and the finding says so.

**Faces** use a **threshold *and* a margin**. A threshold alone answers "does this resemble Jane?" — across a thousand enrolled people that's nearly always yes for someone. The margin asks whether it resembles Jane *distinctly more* than the next candidate, and returns `inconclusive` when it doesn't. Near-misses stay visible rather than being resolved by taking the higher score.

## Extending it

Two ways in, and the first is often enough.

**Rules, no Python.** Most custom checks are "match this pattern", "assert this metadata", or "compare this number":

```bash
imgintel plugins new house-rules --kind rules
```

```yaml
name: house-rules
rules:
  - id: internal-marking
    when:
      text_matches: '(?i)\bCOMPANY CONFIDENTIAL\b'
    finding:
      category: sensitive
      label: Internal document marking
      severity: high
```

Rule files load automatically — they're parsed, never executed, so they need no `--allow-local-plugins`.

**Python analyzers** for anything else:

```bash
imgintel plugins new anpr        # scaffolds an installable package
cd imgintel-anpr && pip install -e .
imgintel plugins validate anpr --sample photo.jpg
```

```python
from imgintel.plugins import Analyzer, Cost, Finding, Severity, require_api

require_api("1.0")


class AspectRatioAnalyzer(Analyzer):
    name = "aspect"
    version = "0.1.0"
    description = "Flags unusual aspect ratios"
    requires = ("fileinfo",)
    cost = Cost.CHEAP

    def analyze(self, ctx):
        ratio = ctx.data("fileinfo").get("aspect_ratio")
        findings = []
        if ratio and (ratio > 3 or ratio < 0.33):
            findings.append(Finding("composition", "Unusual aspect ratio", ratio, Severity.LOW))
        return self.ok({"aspect_ratio": ratio}, findings)
```

Import from `imgintel.plugins`, never `imgintel.core.*`. The SDK has its own `API_VERSION`, so those names keep working while internals stay free to move.

`validate --sample` runs your analyzer twice with the decode path instrumented — the only way to catch a `CHEAP` analyzer that secretly decodes pixels, or one that isn't deterministic. It found four real defects in the built-ins the first time it ran.

Full contract: **[docs/PLUGINS.md](docs/PLUGINS.md)**.

## Output

JSON is canonical and versioned. Everything else — terminal, CSV, HTML, evidence bundles — is a projection of it.

```json
{
  "tool":    { "name": "imgintel", "version": "0.1.0", "schema_version": "1.0" },
  "target":  { "filename": "photo.jpg", "size_bytes": 2847362 },
  "run":     { "profile": "standard", "duration_ms": 84.2 },
  "summary": { "findings_total": 22, "by_severity": { "high": 1, "medium": 4 } },
  "findings":  [ { "analyzer": "gps", "category": "location", "severity": "high", ... } ],
  "analyzers": { "gps": { "status": "ok", "data": { "latitude": 48.8584, ... } } }
}
```

Note the split: `analyzers.*.data` is each analyzer's rich structure, while `findings` is a flat normalized list. Renderers consume **only** `findings`, which is why a new analyzer needs no output-code changes.

`imgintel schema` prints the JSON Schema.

---

## On manipulation detection

**imgintel never says an image is fake.** The `tamper` analyzer reports independent indicators, each with a confidence and an explanation of what could produce it innocently. That's not hedging — it's the difference between a forensic tool and a liability.

Error Level Analysis in particular is widely misread as a manipulation detector. It is [not reliable as a standalone test](https://ieeexplore.ieee.org/document/7412439/): a bright ELA region is far more often texture or a recent recompression than an edit. It's one signal among seven, weighted accordingly.

What carries weight is **agreement between indicators that measure different things**. On a splice, three fire together — a localised recompression ghost pinpointing the donor's quality, a mismatched noise floor in the same place, and an error-level anomaly. On a clean image, none fire. There's deliberately no combined "manipulation score": these indicators aren't independent enough for that arithmetic to mean anything.

All of them are gated on content type — none apply to a screenshot, where uneven compression is simply how the image was made.

## On faces

Face **detection** — locating and counting — is ungated and useful: "how many people are in these 10,000 images", "what should I blur before sharing".

Face **identification** is different. It's processing of biometric data under GDPR Art. 9, Illinois BIPA and equivalents, and requires *both* a lawful basis recorded in the case file **and** `--allow-biometrics` on that run. Neither alone is enough: a basis recorded months ago shouldn't silently authorise today's run, and a flag shouldn't manufacture authority.

**Enrolment is gated identically to matching** — creating a biometric template is the same category of processing as comparing one.

When the gate refuses, it refuses: nothing is embedded, and the reason is recorded in the findings document and evidence bundle. That check runs *before* dependencies and models, so the record shows the governing reason ("no lawful basis recorded") rather than an incidental one ("the model is missing").

## Development

```bash
pip install -e ".[dev,all]"
pytest                          # 380 tests, all fixtures generated — no binary assets
ruff check src tests
imgintel plugins validate --sample photo.jpg
```

Design decisions and the reasoning behind them: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Licence

MIT — and every model choice was made to keep it that way:

| Model | Licence | Why this one |
|---|---|---|
| PP-OCR | Apache-2.0 | Weights ship in the wheel, no download |
| YOLOX-S | Apache-2.0 | **Not Ultralytics YOLO**  |
| YuNet | MIT | **Not InsightFace**, whose code is MIT but whose pretrained models are non-commercial research only |
| SFace | Apache-2.0 | Same, for face embeddings |

Both alternatives are more accurate. Neither is usable in a tool you intend to distribute without accepting an obligation most people don't notice until late.
