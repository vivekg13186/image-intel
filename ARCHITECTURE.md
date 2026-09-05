# imgintel — Architecture & Build Plan

A cross-platform CLI that extracts intelligence from images. Python 3.11+, local models only (fully offline by default), plugin-extensible.

---

## 1. The core idea

Every one of your 30 features is the same shape: *look at an image, emit structured findings*. So build **one small engine** and make all 30 features plugins on it.

```
input(s) ──► AnalysisContext ──► Analyzer DAG ──► FindingsDocument ──► Renderers
             (decode once,       (30+ plugins,     (one versioned      (JSON/CSV/
              share everything)   run in dep        JSON schema)        HTML/evidence
                                  order, parallel                       bundle)
                                  where possible)
```

Three rules that make this work:

1. **Decode once.** Loading a 40MP JPEG, converting to grayscale, and downscaling costs more than most analyzers. `AnalysisContext` computes each derived form lazily and caches it. OCR, blur, and object detection all reuse the same decoded array.
2. **Every analyzer returns the same envelope.** Never let an analyzer write freeform into the output. Uniform envelope = one schema, one renderer, one error path, one way to add feature #31.
3. **Everything is optional.** Analyzers declare their dependencies. Missing Tesseract doesn't break EXIF extraction — it marks the OCR analyzer `unavailable` with a reason and moves on. This is what lets you ship Phase 1 before you own a single model file.

---

## 2. Module layout

```
imgintel/
├── cli.py                    # Typer app — thin, argument parsing only
├── core/
│   ├── analyzer.py           # Analyzer ABC, AnalyzerResult, Cost enum
│   ├── context.py            # AnalysisContext — lazy shared image state
│   ├── registry.py           # discovery: builtin + entry-points + local dir
│   ├── pipeline.py           # DAG topo-sort, execution, timing, isolation
│   ├── schema.py             # pydantic models → FindingsDocument (versioned)
│   ├── modelstore.py         # ONNX model download / cache / sha256 verify
│   └── profiles.py           # quick / standard / deep presets
├── analyzers/                # one file per feature — all 30 live here
├── store/
│   ├── db.py                 # sqlite: cases, images, hashes, embeddings
│   ├── vector.py             # sqlite-vec (or faiss) entity index
│   └── bktree.py             # BK-tree over pHashes for fast dupe queries
├── report/
│   ├── json_out.py  csv_out.py  html_out.py
│   └── evidence.py           # chain-of-custody bundle + manifest
├── runner/batch.py           # process pool, resumable job table
└── plugins/                  # SDK helpers exposed to third parties
```

`cli.py` should never contain analysis logic. If you can't run the whole pipeline from a Python function with no CLI involved, batch mode and tests will fight you later.

---

## 3. The two interfaces everything hangs off

### AnalysisContext — shared, lazy, cached

```python
@dataclass
class AnalysisContext:
    path: Path
    case: CaseConfig
    allow_network: bool = False

    @cached_property
    def raw(self) -> bytes: ...                    # file bytes
    @cached_property
    def pil(self) -> Image.Image: ...              # Pillow, EXIF-oriented
    @cached_property
    def rgb(self) -> np.ndarray: ...               # HxWx3 uint8
    @cached_property
    def gray(self) -> np.ndarray: ...
    @cached_property
    def small(self) -> np.ndarray: ...             # longest side 1024 — feed models this
    @cached_property
    def jpeg_tables(self) -> QuantTables | None: ...  # for tamper + quality
    @cached_property
    def exif(self) -> dict: ...                    # populated by the exif analyzer

    results: dict[str, AnalyzerResult] = field(default_factory=dict)
```

Downstream analyzers read upstream output via `ctx.results["ocr"].data`. That's the only coupling between plugins — no imports across `analyzers/`.

### Analyzer — the plugin contract

```python
class Analyzer(ABC):
    name: str                          # "gps", "ocr", "tamper"
    version: str                       # bump when output shape changes
    requires: tuple[str, ...] = ()     # analyzers that must run first
    cost: Cost = Cost.CHEAP            # CHEAP | MEDIUM | HEAVY
    needs_network: bool = False
    needs_models: tuple[str, ...] = () # modelstore keys

    def available(self) -> Availability:
        """Check binaries/models/imports. Never raise. Return a reason string."""

    @abstractmethod
    def analyze(self, ctx: AnalysisContext) -> AnalyzerResult: ...
```

```python
@dataclass
class AnalyzerResult:
    analyzer: str
    version: str
    status: Literal["ok", "skipped", "unavailable", "error"]
    data: dict          # the actual findings — analyzer-specific
    findings: list[Finding] = ()   # normalized, human-facing highlights
    duration_ms: float = 0
    error: str | None = None
```

`data` is the analyzer's own rich structure. `findings` is a *flat, normalized* list — `{severity, category, label, value, confidence, bbox?}` — and it is what the HTML report, the CSV, and the summarizer consume. This split is the single most valuable design decision in the whole tool: adding an analyzer requires zero renderer changes.

The pipeline topologically sorts by `requires`, runs `CHEAP` analyzers inline and `HEAVY` ones in a thread pool (ONNX Runtime releases the GIL), catches every exception per analyzer, and records timing.

---

## 4. Feature → implementation

### Tier A — metadata & pixels, no models (fast, ships first)

| Feature | How |
|---|---|
| **File info** | `os.stat` + `filetype`/`python-magic` for real magic-byte type + Pillow for dimensions/mode/frames/animation. **Flag extension↔magic mismatch** — a `.jpg` that's really a PNG is itself an intel signal. |
| **EXIF metadata** | `pyexiftool` wrapping the `exiftool` binary. Do not use Pillow's EXIF for this — exiftool reads XMP, IPTC, ICC, MakerNotes (camera serial, lens, shutter count, owner name), which is where the good intel is. Fall back to `exifread` when the binary is missing. |
| **GPS / location** | Parse GPSLatitude/Longitude/Altitude/Timestamp/ImgDirection rationals → decimal degrees. Offline reverse geocode with a bundled GeoNames cities dump + KD-tree (`reverse_geocoder`). Timezone from coords with `timezonefinder`. **Cross-check** `DateTimeOriginal` (local) against `GPSTimeStamp` (UTC) — a mismatched offset betrays edited timestamps or a fabricated location. |
| **Hashing** | sha256/sha1/md5 over file bytes **and separately over decoded pixel bytes**. The pixel hash survives metadata stripping and re-containering, so it links files that byte hashes don't. |
| **Perceptual hashing** | `imagehash`: pHash (DCT, robust to scale/JPEG), dHash (fast), aHash, wHash, colorhash. Store all as hex — different hashes fail on different transforms, so keep several. |
| **Colour analysis** | k-means on LAB-space pixels for a dominant palette, channel histograms, colour-cast estimate, `is_grayscale`, Hasler-Süsstrunk colourfulness, embedded ICC profile name. |
| **Quality analysis** | Estimate JPEG quality from the quantization tables, exposure clipping percentages, resolution class, noise estimate (MAD of a Laplacian residual), banding detection. |
| **Blur detection** | Variance-of-Laplacian globally **and per tile**. The tile map matters: a sharp image with one blurred rectangle is an attempted redaction, not a bad photo. Separate motion blur from defocus by looking for directional energy concentration in the FFT magnitude spectrum. |

### Tier B — local ONNX / native models

| Feature | How |
|---|---|
| **OCR** | PaddleOCR (PP-OCRv5) as primary — best multilingual and layout handling. Tesseract as the light fallback for small installs. Return **per-block boxes + confidence**, never a single text blob: boxes are required for PII localization and redaction. |
| **Object detection** | ONNX Runtime + **RF-DETR or YOLOX (Apache-2.0)** on COCO weights. Avoid Ultralytics YOLO unless you accept AGPL-3.0 or buy the enterprise licence — this is a real trap. |
| **Face detection** | SCRFD or RetinaFace via ONNX, detection + 5-point landmarks + quality score. Keep detection strictly separate from recognition (Tier C) so users can enable one without the other. |
| **Scene classification** | CLIP ViT-B/32 (MIT) zero-shot against a scene-label list, rather than Places365. Zero-shot means you extend the label set by editing a YAML file instead of retraining. |
| **Vehicle detection** | Filter the COCO detector to car/truck/bus/motorcycle, then a plate-region detector → crop → OCR. Plate *format validation* is jurisdiction-specific; keep it in a separate rules file so it's user-extensible. |
| **Logo / brand** | No good permissive general logo model exists. Hybrid: (a) CLIP zero-shot against a brand-name list for common marks, (b) ORB/SIFT + FLANN matching against a **user-supplied reference logo folder**. (b) is the one that actually works for a specific investigation. |
| **QR / barcode** | `zxing-cpp` (ships self-contained wheels, more symbologies) over `pyzbar` (needs a system libzbar). Then **parse the payload**: URL → domain + TLD + IDN-homograph check, WiFi config, vCard, `geo:` URI, crypto address. The payload is the intel; the decode is just plumbing. |
| **Text language** | `lingua-py`, not `langdetect` — lingua is calibrated for short strings and doesn't produce confident garbage on 3-word signs. Run it **per OCR block**, not on the concatenated text. |
| **Sensitive info** | Regex + validator pass over OCR output: emails, phones (`phonenumbers`), cards (Luhn), IBAN, national IDs, API keys/JWTs (pattern + Shannon entropy), IPs, crypto addresses, postal addresses. Plus visual signals: face present, document/screen detected, ID-card aspect+layout. Carry bboxes through so `imgintel redact` becomes a two-line follow-up feature. |

### Tier C — comparison, database, correlation

| Feature | How |
|---|---|
| **Image similarity** | Two tiers. Fast: Hamming distance on pHash (near-duplicates, crops, recompression). Semantic: cosine distance on CLIP or DINOv2 embeddings (same scene, different shot). Report both — they answer different questions. |
| **Duplicate detection** | Exact via sha256. Near via a **BK-tree** over pHashes — gives you Hamming-radius queries in O(log n) instead of the O(n²) pairwise comparison that kills you at 100k images. |
| **People from DB** | Face crop → ArcFace-style embedding → `sqlite-vec` index of enrolled identities → cosine match with **both a threshold and a margin check** (top-1 must beat top-2 by a gap, or return "inconclusive"). Store enrollment provenance per identity. |
| **Objects/pets/vehicles from DB** | Identical pattern, different embedder: crop the detection → CLIP/DINOv2 embed → match against enrolled reference crops. Write **one generic `EntityIndex`** parameterized by embedder; faces and objects are then the same 40 lines of code. |
| **Reverse image search** | No local option — this is the one genuinely network-bound feature. Pluggable providers (TinEye API, Bing Visual Search, Yandex). Gate behind `--allow-network`, and record in the evidence log that the image left the machine. Note that scraping consumer endpoints violates their ToS; prefer paid APIs. |
| **Geolocation estimation** | A **ranked hypothesis list**, never a single point. Sources in confidence order: (1) EXIF GPS, (2) OCR'd text — street signs, business names, phone country codes, plate formats → geocode, (3) landmark match, (4) StreetCLIP/GeoCLIP coarse country/region prior, (5) sun-elevation from shadow angles + timestamp → latitude band. Emit each hypothesis with its supporting evidence chain. |
| **Landmark detection** | The hardest to do locally and honestly. Practical scope: CLIP zero-shot over a curated landmark list, plus a DELG/DOLG embedding index over a **top-N landmarks subset** of Google Landmarks v2 with FAISS. Ship a few thousand landmarks well rather than pretending to global coverage. |

### Tier D — tampering, summarization, output

**Tampering / manipulation detection** deserves its own note. Build it as a *panel of independent indicators*, each with its own confidence, and **never emit a binary "fake / authentic" verdict** — that's the difference between a forensic tool and a liability.

- Error Level Analysis (re-save at known quality, diff)
- JPEG ghost sweep (recompress at q=1..99, find anomalous minima per region)
- Double-JPEG detection via DCT coefficient histogram periodicity
- Quantization-table fingerprint vs. a known camera/software table database — matching Photoshop's tables when EXIF claims a Canon is a strong signal
- Noise/PRNU inconsistency map across regions
- Copy-move via keypoint clustering (ORB matches with consistent affine offset)
- CFA/demosaicing artifact consistency
- Metadata contradictions: Software field, missing MakerNotes on a "camera original", **embedded thumbnail vs. main image mismatch** (the classic — thumbnails often survive edits)

**Image summarization** — default to a **deterministic, template-driven** rollup over the normalized `findings` list. Offline, reproducible, no hallucination. That is the right default for forensic output.

**AI-based analysis** — optionally a local VLM (Qwen2.5-VL or Moondream via llama.cpp/ONNX) for a descriptive caption. Keep it in a separate `ai.*` namespace in the JSON, clearly labeled model-generated, so it can never be mistaken for a measured finding.

**Reports** — JSON is canonical: pydantic → versioned JSON Schema, with `schema_version` at the root. CSV is a configurable flattening of selected JSON paths. JSONL for streaming batch output. HTML is Jinja2 producing a **single self-contained file** with base64-embedded thumbnails and an annotated overlay image (boxes for faces/objects/OCR/PII).

**Evidence export** — a bundle directory containing: the original file bit-for-bit unmodified, a sha256 manifest, `findings.json`, `report.html`, and a `provenance.json` recording tool version, every model file's sha256, timestamps, operator, and the exact command line. Optionally minisign the manifest. Append-only case log. This is what turns a script into something usable in an actual investigation.

---

## 5. Plugin system — three layers

1. **Built-in** — `imgintel.analyzers` package scanned at import.
2. **Installable** — `importlib.metadata.entry_points(group="imgintel.analyzers")`, so anyone can `pip install imgintel-anpr` and it appears automatically. This is the standard, dependency-free Python plugin mechanism; use it rather than inventing one.
3. **Drop-in** — `.py` files in `~/.imgintel/plugins/` loaded by path, for one-off case-specific analyzers. Load these **only** when `--allow-local-plugins` is passed; arbitrary code execution from a directory should be explicit.

Plus a **declarative** escape hatch: a YAML analyzer for the common "match a regex against OCR text" or "flag when EXIF field X equals Y" rule, so simple custom checks need no Python at all. In practice this covers most user extensions.

Minimum viable third-party plugin:

```python
from imgintel.plugins import Analyzer, AnalyzerResult, Finding

class RedactionCheck(Analyzer):
    name, version, requires = "redaction", "1.0", ("blur", "ocr")

    def analyze(self, ctx):
        tiles = ctx.results["blur"].data["tile_map"]
        suspicious = [t for t in tiles if t["variance"] < 15]
        return AnalyzerResult(
            analyzer=self.name, version=self.version, status="ok",
            data={"regions": suspicious},
            findings=[Finding(severity="medium", category="redaction",
                              label="Selective blur region", confidence=0.7,
                              bbox=t["bbox"]) for t in suspicious],
        )
```

---

## 6. Cross-platform specifics

The things that will actually break on Windows/macOS:

- **Prefer pure wheels.** `zxing-cpp` over `pyzbar` (libzbar is a system-library headache on Windows). `opencv-python-headless` over `opencv-python` (no GUI/Qt deps).
- **Native binaries are optional, never required.** `exiftool` (Perl) and `tesseract` are external. Detect them, degrade gracefully, and give per-OS install instructions in `imgintel doctor`. Optionally vendor `exiftool.exe` for Windows.
- **ONNX Runtime** has wheels for Windows/macOS(arm64+x64)/Linux. Default to the CPU EP; auto-detect CoreML on Apple Silicon, DirectML on Windows, CUDA where present. Never hard-require a GPU.
- **Paths and dirs:** `pathlib` everywhere, `platformdirs` for config/cache/model locations. Handle case-insensitive filesystems and long Windows paths.
- **Distribution:** `uv tool install imgintel` or `pipx`. Extras keep the base install small: `imgintel[ocr]`, `[detect]`, `[faces]`, `[forensics]`, `[all]`.
- **Models are never bundled in the wheel.** `imgintel models pull` downloads to the platform cache dir and verifies sha256. Keeps the package a few MB instead of 2 GB, and makes model provenance auditable for evidence.

---

## 7. CLI surface

```
imgintel analyze IMG  [--profile quick|standard|deep] [--analyzers a,b,c]
                      [--offline] [--json out.json] [--html report.html]
imgintel batch DIR    --recursive --workers 8 --resume --out results/
imgintel compare A B
imgintel dedupe DIR   --threshold 8
imgintel db enroll    --identity "Jane Doe" --images ./refs/
imgintel db search    IMG --top 10
imgintel report       findings.json --html out.html
imgintel evidence     export findings.json --case CASE-001 --operator vivek
imgintel models       pull [--all] | list | verify
imgintel plugins      list
imgintel doctor       # environment + dependency + model check
```

`--profile` is the most important UX decision: `quick` = metadata tier only (milliseconds), `standard` = + hashes/OCR/objects/barcodes, `deep` = everything including tampering and geo-estimation. Without profiles, users pay the 30-second deep cost for a question that EXIF alone answers.

`doctor` should be the second command you build. Most support burden on a tool like this is "why is OCR not working" — a command that prints exactly what's installed, missing, and how to fix it per-OS pays for itself immediately.

---

## 8. Build order

Each phase leaves you with a tool that is genuinely useful on its own. Resist reordering to get to the exciting ML early — the engine has to be right first, and every ML feature is trivially bolted on once it is.

| Phase | Contents | Why here |
|---|---|---|
| **0 ✅** | Analyzer ABC, context, registry, DAG pipeline, JSON schema, CLI skeleton, `doctor` | No dependencies, no models. Prove the plumbing before it carries weight. |
| **1 ✅** | File info, EXIF, GPS, hashes, perceptual hashes, colour, quality, blur | All CPU, all pure-Python deps. **This alone is already a shippable OSINT tool.** |
| **2 ✅** | JSON/CSV/HTML reports, batch runner, compare, dedupe (BK-tree), evidence export | Makes Phase 1 usable at scale. Locks the output schema before 20 more analyzers depend on it. |
| **3 ✅** | Model store, ONNX runtime wiring, OCR, barcode/QR, language, sensitive-info | First real "intelligence" tier. Model-store plumbing is exercised by the easiest models. |
| **4 ✅** | Object detection, face detection, scene classification, vehicle, logo | Reuses the Phase-3 model store. Purely additive. |
| **5 ✅** | Entity DB, embeddings, people-from-DB, objects-from-DB, similarity search | Needs Phase 4's detectors to produce crops. |
| **6 ✅** | Tampering suite, geolocation estimation, reverse search, summarization | Hardest, most nuanced, most legally sensitive. Deliberately last. |
| **7 ✅** | Plugin SDK docs, entry points, declarative YAML analyzers, packaging polish | Formalize the interface only after 25 real analyzers have stress-tested it. |

A reasonable first-week target is Phases 0–2: `imgintel analyze photo.jpg --html report.html` producing a real report with EXIF, GPS, a map link, hashes, and quality metrics.

**All seven phases are built** — see [README.md](README.md) for what they do and how to run them. Five invariants emerged while building them, and anything added later should preserve all five:

- **`quick` is header-only** (no pixel decode), enforced by a regression test. Any `CHEAP` analyzer added later must respect that or the profile stops meaning anything.
- **Shared derivations live on the context.** Anything two analyzers both need belongs on `AnalysisContext` as a `cached_property`, not in one of them. `sharpness` moved there once `blur` and `quality` both wanted it; ONNX sessions follow the same rule, one per process.
- **All bounding boxes are in analysis space** (`ctx.small`, longest edge 1024). OCR, barcodes, blur tiles and the HTML overlay all agree because of this, with no rescaling anywhere. Phase 4 detectors must emit boxes in the same space.
- **Absolute *and* relative thresholds.** Every "this region differs from the rest" check needs both, or hard-edged graphics defeat it — a QR code has such extreme edge energy that a quarter of the frame reads as "soft" purely by ratio while being razor sharp.
- **Photographic assumptions are gated on `quality.content_type`.** Exposure, noise and dynamic-range findings are meaningless on a screenshot or document scan, and firing them there buries the real findings.

Two more things learned the hard way, both now fixed in code:

- **Per-process warm state is not optional.** Constructing `TimezoneFinder` per image cost 340 ms and made the `quick` profile 250× slower than it should be. Model sessions have the same shape; the batch pool initializer exists for exactly this.
- **One perceptual hash is not enough.** Each fails on a different transform: a plain resize of high-frequency content lands 16 bits away in pHash and 4 in dHash. `compare` and `dedupe` link on the closest of several hashes, which is the reason for computing several in the first place.

Phase 4 added three more:

- **Weights are not the interesting part; the path around them is.** Session construction, letterboxing, anchor-free decode, class-aware NMS and coordinate mapping are where the bugs live, and a letterbox off by the pad offset produces boxes that drift toward a corner rather than an obvious crash. `tests/fakemodel.py` builds a genuine ONNX graph that emits detections chosen by the test, so the whole path is verified against boxes known to the pixel — no 36 MB download required in CI.
- **Never claim verification that did not happen.** The detection models are shipped `sha256=None` because imgintel has not verified those specific published files; recording a guessed hash would reject every legitimate download, and accepting anything silently would make findings unreproducible. The third option — install, report *unpinned*, offer `models pin` — is the honest one, and evidence bundles record `pinned: false` so the difference survives into the record.
- **Not every capability needs a model.** Scene classification is derived from object evidence via an editable rule table rather than a 350 MB CLIP; logo detection matches user-supplied reference images with ORB and a homography check, needing no weights at all. Both are less accurate than a dedicated network and both show their reasoning, which for this tool is the better trade.

Phase 5 added three more, all about the difference between a tool that finds things and one that produces evidence:

- **Permission is checked before capability.** `Analyzer.precheck()` runs before dependencies, models and `available()`. Without that ordering, a face-identification run with no lawful basis reported "the model is missing" — true, but the wrong answer to "did you run face matching, and why not". The refusal an evidence bundle records has to be the *governing* one.
- **The authorisation and the intent are separate, and both are required.** The lawful basis lives in the case file, travels with the investigation, and answers "were you entitled to do this". `--allow-biometrics` is per-run and answers "did you mean to do it now". Neither alone permits identification, because a basis recorded months ago should not silently authorise today's run, and a flag should not manufacture authority. Enrolment is gated identically to matching: creating a biometric template is the same category of processing as comparing one.
- **An identification tool must be able to say "I don't know".** Matching uses a threshold *and* a margin over the runner-up. A threshold alone answers "does this resemble Alice?", which across a thousand enrolled people is nearly always yes for someone; the margin asks whether it resembles Alice *distinctly more* than anyone else, and returns `inconclusive` when it does not. Near-misses stay in the shortlist and in the summary CSV rather than being resolved by taking the higher score.

Phase 6 added four, mostly learned by getting the forensics wrong first:

- **A local minimum is not a global one.** The JPEG-ghost technique looks for a dip in a region's recompression curve. My first implementation took the `argmin`, which always lands on the image's *current* quality — recompressing at what it already is changes it least — so it recovered the last save and could never see a splice. Only an interior-local-minimum search finds a region carried over from an earlier, harder compression. The corrected version pinpoints a q=35 donor inside a q=95 host, and names the tiles.
- **Sweep ranges must cover what people actually do.** The ghost sweep started at q=50, so a first save at q=45 — completely ordinary — produced "no prior compression". A false negative that looks identical to a clean image is the worst possible failure mode for this analyzer.
- **A robust statistic that divides by zero is not robust.** Outlier tiles are scored by median absolute deviation, which is right for images. But when most tiles are identical the MAD is zero, and the code returned "no outliers" for the case where a single region differs starkly from a uniform background — the clearest outlier there is. It now falls back to the mean deviation.
- **Fixtures must have the statistics of the thing being measured.** The smooth periodic gradients used elsewhere in the suite are so self-similar that copy-move detection trips on them, and their DCT coefficients do not follow the Benford-like law the double-compression test depends on. Phase 6 fixtures synthesise 1/f noise instead, which is the standard model of natural-image statistics; on that content, clean images produce 4 coincidental keypoint pairs and a real clone produces 75.

Phase 7 added two, and confirmed the value of deferring it:

- **A public surface has to exist, or "public" is a fiction.** Until Phase 7 the docs told plugin authors to import from `imgintel.core.analyzer` — an internal path that had already changed twice, once to add `after` and once to add `precheck`. `imgintel.plugins` is now the promise, with an `API_VERSION` independent of imgintel's own and a `require_api()` that fails at import rather than mysteriously mid-run. Deferring the formalisation until twenty-five analyzers had stressed the interface was right: the shape that got frozen includes `precheck` and `after`, neither of which existed when the interface was first sketched.
- **A validator earns its keep immediately, including against its author.** `imgintel plugins validate --sample` found four real defects in the built-ins within a minute of working: a `MEDIUM` analyzer that never touches pixels, a medium-severity finding with no explanation, and three analyzers reporting unavailability with no actionable hint. It also caught its own worst bug — restoring a patched `cached_property` by *rebuilding* it skips `__set_name__`, leaving the descriptor nameless and corrupting `AnalysisContext` for the rest of the process. That surfaced as three unrelated tests failing only when run after the validator, which is exactly the kind of fault a suite catches and a manual check never would.

---

## 9. Traps worth knowing before you start

**Licensing.** Three specific ones that bite:
- **Ultralytics YOLO is AGPL-3.0** — using it in a distributed tool obliges you to release your source under AGPL, or buy an enterprise licence. Use RF-DETR or YOLOX (Apache-2.0) instead.
- **InsightFace**: the *code* is MIT, but the *pretrained models* are licensed for non-commercial research only. This catches many projects.
- Model weights carry their own licences independent of the library. Record each model's licence in the model store manifest.

**Legal/ethical.** Face matching against a database and geolocation estimation are the two features with real regulatory exposure (GDPR Art. 9 biometrics, Illinois BIPA, and equivalents). Gate them behind explicit opt-in flags, record a lawful-basis note in the case config, and keep them out of the `deep` profile default. Cheap to build in now, expensive to retrofit.

**Forensic honesty.** Tampering detection produces *indicators*, not verdicts. ELA in particular is widely misread — it's [known to be unreliable as a standalone test](https://ieeexplore.ieee.org/document/7412439/). Always report the indicator, its confidence, and what could explain it benignly. A tool that says "MANIPULATED" will eventually be wrong in a way that matters.

**Performance.** In batch mode, model loading dominates. Load models **once per worker process**, not once per image — with a `ProcessPoolExecutor` and an initializer that warms the ONNX sessions. Getting this wrong makes batch 50× slower and it's an easy mistake to make.

---

## Sources

- [PaddleOCR vs Tesseract vs EasyOCR benchmark](https://www.codesota.com/ocr/paddleocr-vs-tesseract)
- [Best open source OCR tools 2026](https://unstract.com/blog/best-opensource-ocr-tools/)
- [Ultralytics licensing](https://github.com/ultralytics/ultralytics)
- [Permissive Ultralytics alternatives (RF-DETR, YOLOX)](https://www.lightly.ai/blog/best-ultralytics-alternatives-in-2026)
- [InsightFace](https://github.com/deepinsight/insightface)
- [An evaluation of Error Level Analysis in image forensics](https://ieeexplore.ieee.org/document/7412439/)
- [ForensicLens — reference forensic technique implementations](https://github.com/makalin/ForensicLens)
