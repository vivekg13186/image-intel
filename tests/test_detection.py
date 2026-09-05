"""Object and face detection: inference plumbing, analyzers, the model workflow."""

from __future__ import annotations

import numpy as np
import pytest

from imgintel.core.deps import has_module
from imgintel.core.engine import analyze_image
from imgintel.util.coco import COCO_CLASSES, infer_scenes
from imgintel.util.onnx import (
    Letterbox,
    batched_nms,
    decode_yolox,
    letterbox,
    nms,
    to_nchw,
)


def _needs(module: str):
    return pytest.mark.skipif(not has_module(module), reason=f"{module} not installed")


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer}: {block.error}"
    return block.data


def labels(doc) -> set[str]:
    return {f.label for f in doc.findings}


# -- letterbox -------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "target"),
    [((480, 640), (640, 640)), ((640, 480), (640, 640)), ((100, 1000), (416, 416))],
)
def test_letterbox_preserves_aspect_ratio(src, target):
    image = np.zeros((*src, 3), dtype=np.uint8)
    padded, transform = letterbox(image, target)

    assert padded.shape[:2] == (target[1], target[0])
    src_h, src_w = src
    assert transform.scale == pytest.approx(min(target[0] / src_w, target[1] / src_h))


def test_letterbox_roundtrip_is_exact():
    """The bug this guards: boxes that drift toward a corner.

    Mapping a box forward through the transform and back must return it, or
    every detection is quietly displaced by the padding offset.
    """
    image = np.zeros((450, 800, 3), dtype=np.uint8)
    _, transform = letterbox(image, (640, 640))

    source_boxes = np.array(
        [[0, 0, 100, 80], [400, 200, 560, 340], [0, 0, 800, 450]], dtype=np.float32
    )
    network = source_boxes * transform.scale
    recovered = transform.to_source(network)
    assert np.allclose(recovered, source_boxes, atol=1e-3)


def test_letterbox_clips_to_source_bounds():
    transform = Letterbox(
        scale=0.5, pad_x=0, pad_y=0, input_width=640, input_height=640,
        source_width=200, source_height=100,
    )
    out = transform.to_source(np.array([[-50, -50, 900, 900]], dtype=np.float32))
    assert out[0].tolist() == [0.0, 0.0, 200.0, 100.0]


def test_letterbox_content_is_top_left():
    """YOLOX pads bottom-right; centring would shift every box by half the pad."""
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    padded, _ = letterbox(image, (640, 640), fill=0)
    assert padded[0, 0].tolist() == [255, 255, 255]
    assert padded[-1, -1].tolist() == [0, 0, 0]


def test_to_nchw_shape_and_scaling():
    image = np.full((4, 6, 3), 255, dtype=np.uint8)
    assert to_nchw(image).shape == (1, 3, 4, 6)
    assert to_nchw(image, normalize=False).max() == 255.0
    assert to_nchw(image, normalize=True).max() == pytest.approx(1.0)


# -- NMS -------------------------------------------------------------------


def test_nms_keeps_the_best_of_an_overlapping_pair():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [500, 500, 600, 600]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_nms_keeps_distinct_boxes():
    boxes = np.array([[0, 0, 50, 50], [200, 200, 250, 250]], np.float32)
    assert sorted(nms(boxes, np.array([0.9, 0.8], np.float32), 0.5)) == [0, 1]


def test_nms_handles_empty_input():
    assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5) == []


def test_class_aware_nms_does_not_suppress_across_classes():
    """The person standing in front of the car must survive."""
    boxes = np.array([[0, 0, 100, 100], [2, 2, 98, 98]], np.float32)
    scores = np.array([0.9, 0.85], np.float32)

    same_class = np.array([0, 0], np.int32)
    assert len(batched_nms(boxes, scores, same_class, 0.5)) == 1

    different = np.array([0, 2], np.int32)
    assert len(batched_nms(boxes, scores, different, 0.5)) == 2


# -- YOLOX decode ----------------------------------------------------------


def test_decode_recovers_an_encoded_box():
    """Encode a known box into a raw head, decode it, get the box back."""
    from tests.fakemodel import build_head

    size = (416, 416)
    wanted = (50.0, 60.0, 150.0, 200.0)
    head = build_head(*size, [{"bbox": wanted, "class_id": 2, "score": 0.9, "anchor": 0}])

    boxes, scores, class_ids = decode_yolox(head, size)
    best = int(scores.argmax())

    assert np.allclose(boxes[best], wanted, atol=0.5)
    assert class_ids[best] == 2
    assert scores[best] == pytest.approx(0.9, abs=0.01)


def test_decode_uses_objectness_times_class_score():
    """Class score alone would flood the output with background detections."""
    from tests.fakemodel import NUM_CLASSES, total_anchors

    size = (416, 416)
    head = np.zeros((1, total_anchors(*size), 5 + NUM_CLASSES), np.float32)
    head[0, 0, 4] = 0.5           # objectness
    head[0, 0, 5 + 3] = 0.6       # class score
    head[0, 0, 2:4] = 0.0         # unit box, avoids exp overflow

    _, scores, class_ids = decode_yolox(head, size)
    assert scores[0] == pytest.approx(0.30, abs=1e-4)
    assert class_ids[0] == 3


def test_decode_rejects_a_mismatched_anchor_count():
    from tests.fakemodel import NUM_CLASSES

    head = np.zeros((1, 17, 5 + NUM_CLASSES), np.float32)
    with pytest.raises(ValueError, match="anchors"):
        decode_yolox(head, (416, 416))


def test_anchor_grid_covers_every_stride():
    from tests.fakemodel import total_anchors

    assert total_anchors(640, 640) == 52 * 52 + 40 * 40 + 20 * 20 + (80 * 80 - 52 * 52)


# -- full inference path with a real ONNX graph ----------------------------


@_needs("onnx")
@_needs("onnxruntime")
def test_end_to_end_detection_against_a_constructed_model(images, tmp_path, monkeypatch):
    """Everything except the weights: session, letterbox, decode, NMS, mapping.

    Real YOLOX weights are 36 MB and not always reachable; this exercises the
    identical code path with boxes we chose, so the expected output is known
    exactly rather than eyeballed.
    """
    from tests.fakemodel import build_head, write_constant_model

    size = (416, 416)
    # Two objects: a person and a car, placed where we can check them.
    wanted = [
        {"bbox": (40.0, 50.0, 140.0, 250.0), "class_id": 0, "score": 0.92, "anchor": 0},
        {"bbox": (200.0, 120.0, 360.0, 240.0), "class_id": 2, "score": 0.81, "anchor": 5},
    ]
    head = build_head(*size, wanted)
    model_path = write_constant_model(tmp_path / "fake_yolox.onnx", head, size)

    import imgintel.analyzers.objects as objects_mod
    from imgintel.core.modelstore import ModelStatus

    monkeypatch.setattr(objects_mod, "DEFAULT_INPUT", size)
    monkeypatch.setattr(
        "imgintel.core.modelstore.status",
        lambda key, verify=False: ModelStatus(
            key=key, description="fake", kind="download", licence="test",
            present=True, path=str(model_path),
        ),
    )
    monkeypatch.setattr(
        objects_mod.ObjectAnalyzer, "available", lambda self: __import__(
            "imgintel.core.analyzer", fromlist=["Availability"]
        ).Availability.yes()
    )

    doc = analyze_image(images["photo"], only=["objects"])
    data = data_of(doc, "objects")

    assert data["count"] == 2
    found = {d["label"]: d for d in data["detections"]}
    assert set(found) == {"person", "car"}
    assert found["person"]["confidence"] == pytest.approx(0.92, abs=0.01)

    # Boxes must land in analysis space, scaled out of the 416x416 network.
    width, height = data["analysis_dimensions"]
    for det in data["detections"]:
        x0, y0, x1, y1 = det["bbox"]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height

    # The person box is taller than wide and left of the car — geometry that
    # survives the letterbox round trip only if the mapping is right.
    person, car = found["person"], found["car"]
    assert person["bbox"][3] - person["bbox"][1] > person["bbox"][2] - person["bbox"][0]
    assert person["bbox"][0] < car["bbox"][0]

    assert "1 person detected" in " ".join(labels(doc)) or "people" in " ".join(labels(doc))


# -- scene inference -------------------------------------------------------


def test_scene_rules_rank_street_from_strong_cues():
    scenes = infer_scenes(["car", "person", "traffic light", "stop sign"])
    assert scenes[0]["scene"] == "street / traffic"
    assert "traffic light" in scenes[0]["evidence"]


def test_scene_rules_need_more_than_one_weak_cue():
    assert infer_scenes(["chair"]) == []


def test_scene_rules_carry_their_evidence():
    scenes = infer_scenes(["laptop", "keyboard", "mouse", "cup"])
    assert scenes[0]["scene"] == "workplace / desk"
    assert set(scenes[0]["evidence"]) >= {"laptop", "keyboard", "mouse"}


def test_scene_rules_reference_only_real_coco_classes():
    """A typo in the rules would silently never match."""
    from imgintel.util.coco import SCENE_RULES

    known = set(COCO_CLASSES)
    for scene, rule in SCENE_RULES.items():
        unknown = (set(rule["strong"]) | set(rule["weak"])) - known
        assert not unknown, f"{scene} references non-COCO classes: {unknown}"


def test_empty_detections_produce_no_scene():
    assert infer_scenes([]) == []


# -- vehicles --------------------------------------------------------------


def test_plate_region_sits_low_and_central():
    from imgintel.analyzers.vehicles import _plate_region

    vehicle = {"bbox": [100, 100, 400, 300], "area_fraction": 0.2}
    region = _plate_region(vehicle)
    assert region is not None
    x0, y0, x1, y1 = region
    assert y0 > 100 + (300 - 100) * 0.5, "plate band must be in the lower half"
    assert x0 > 100 and x1 < 400, "plate band must be inset horizontally"


def test_plate_region_rejects_tiny_vehicles():
    from imgintel.analyzers.vehicles import _plate_region

    assert _plate_region({"bbox": [0, 0, 20, 15], "area_fraction": 0.0001}) is None


def test_plate_region_rejects_wrong_aspect():
    """A tall narrow box is a side-on vehicle with no visible plate."""
    from imgintel.analyzers.vehicles import _plate_region

    assert _plate_region({"bbox": [0, 0, 60, 400], "area_fraction": 0.3}) is None


def test_plate_text_matched_by_box_centre():
    from imgintel.analyzers.vehicles import _text_in

    blocks = [
        {"text": "AB12 CDE", "bbox": [110, 260, 190, 280]},
        {"text": "elsewhere", "bbox": [900, 900, 950, 920]},
    ]
    assert _text_in([100, 250, 300, 290], blocks) == "AB12 CDE"


# -- logo matching (needs no downloaded weights) ---------------------------


@_needs("cv2")
def test_logo_matches_a_planted_reference(images, tmp_path, monkeypatch):
    """The one Phase 4 analyzer that is fully verifiable without model weights."""
    import cv2
    from PIL import Image

    import imgintel.analyzers.logo as logo_mod

    # A textured mark with plenty of corners for ORB to find.
    rng = np.random.default_rng(11)
    mark = (rng.integers(0, 255, (120, 160, 3))).astype(np.uint8)
    mark = cv2.GaussianBlur(mark, (3, 3), 0)

    refs = tmp_path / "logos"
    refs.mkdir()
    Image.fromarray(mark).save(refs / "acme.png")

    # Paste the mark into a larger scene.
    scene = np.asarray(Image.open(images["photo"]).convert("RGB")).copy()
    scene[200:320, 300:460] = mark
    scene_path = tmp_path / "scene.png"
    Image.fromarray(scene).save(scene_path)

    monkeypatch.setattr(logo_mod, "reference_dir", lambda: refs)
    logo_mod.load_references.cache_clear()

    doc = analyze_image(scene_path, only=["logo"])
    data = data_of(doc, "logo")

    assert data["reference_count"] == 1
    assert data["matches"], "planted logo should be found"
    match = data["matches"][0]
    assert match["name"] == "acme"
    assert match["inliers"] >= logo_mod.MIN_INLIERS
    assert any("acme" in f.label for f in doc.findings)


@_needs("cv2")
def test_logo_does_not_match_an_absent_reference(images, tmp_path, monkeypatch):
    import cv2
    from PIL import Image

    import imgintel.analyzers.logo as logo_mod

    rng = np.random.default_rng(12)
    mark = cv2.GaussianBlur((rng.integers(0, 255, (120, 160, 3))).astype(np.uint8), (3, 3), 0)
    refs = tmp_path / "logos"
    refs.mkdir()
    Image.fromarray(mark).save(refs / "absent.png")

    monkeypatch.setattr(logo_mod, "reference_dir", lambda: refs)
    logo_mod.load_references.cache_clear()

    doc = analyze_image(images["photo"], only=["logo"])
    assert not data_of(doc, "logo")["matches"]


@_needs("cv2")
def test_logo_unavailable_without_references(tmp_path, monkeypatch):
    import imgintel.analyzers.logo as logo_mod

    monkeypatch.setattr(logo_mod, "reference_dir", lambda: tmp_path / "nothing")
    logo_mod.load_references.cache_clear()

    availability = logo_mod.LogoAnalyzer().available()
    assert not availability.ok
    assert "reference" in availability.reason.lower()


# -- model workflow --------------------------------------------------------


def _with_hash(spec, digest: str):
    """Copy a ModelSpec with a different pinned digest."""
    from dataclasses import replace

    return replace(spec, sha256=digest)


def test_import_rejects_a_hash_mismatch(tmp_path, monkeypatch):
    """An unverified model makes every finding it produces unreproducible."""
    from imgintel.core import modelstore

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(
        modelstore,
        "MODELS",
        {**modelstore.MODELS, "detect": _with_hash(modelstore.MODELS["detect"], "0" * 64)},
    )
    fake = tmp_path / "yolox_s.onnx"
    fake.write_bytes(b"not the real weights")

    with pytest.raises(modelstore.DownloadError, match="pinned hash"):
        modelstore.import_file(fake)
    assert not (tmp_path / "cache" / "yolox_s.onnx").exists(), "must not install a bad file"


def test_import_installs_and_pins(tmp_path, monkeypatch):
    from imgintel.core import modelstore

    cache = tmp_path / "cache"
    monkeypatch.setattr(modelstore, "cache_dir", lambda: cache)
    monkeypatch.setenv(modelstore.MODEL_DIR_ENV, "")

    payload = b"pretend onnx bytes"
    source = tmp_path / "yolox_s.onnx"
    source.write_bytes(payload)

    status = modelstore.import_file(source)
    assert status.present
    assert (cache / "yolox_s.onnx").read_bytes() == payload
    # Unpinned by default, and it says so rather than implying verification.
    assert status.verified is None
    assert "unpinned" in status.note

    pinned = modelstore.pin("detect")
    assert pinned.verified is True
    assert modelstore.expected_sha256("detect") == pinned.sha256


def test_import_rejects_an_unrecognised_file(tmp_path, monkeypatch):
    from imgintel.core import modelstore

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "cache")
    stray = tmp_path / "something_else.onnx"
    stray.write_bytes(b"x")

    with pytest.raises(modelstore.DownloadError, match="does not match any known model"):
        modelstore.import_file(stray)


def test_env_var_adds_a_search_directory(tmp_path, monkeypatch):
    """Downloading is not always possible; a folder of files must be enough."""
    from imgintel.core import modelstore

    external = tmp_path / "external"
    external.mkdir()
    (external / "yolox_s.onnx").write_bytes(b"weights")

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "empty-cache")
    monkeypatch.setenv(modelstore.MODEL_DIR_ENV, str(external))

    assert external in modelstore.search_dirs()
    assert modelstore.find_model("detect") == external / "yolox_s.onnx"
    assert modelstore.status("detect").present


# -- TLS and download integrity --------------------------------------------


def test_secure_context_verifies_and_loads_certificates():
    import ssl

    from imgintel.core.modelstore import ssl_context

    context = ssl_context()
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.get_ca_certs(), "a CA bundle must actually be loaded"


def test_insecure_context_is_explicitly_unverified():
    import ssl

    from imgintel.core.modelstore import ssl_context

    context = ssl_context(insecure=True)
    assert context.check_hostname is False
    assert context.verify_mode == ssl.CERT_NONE


def test_certificate_failure_gives_actionable_guidance(tmp_path, monkeypatch):
    """A trust-store problem must not read as "the server is down"."""
    import ssl
    import urllib.error

    from imgintel.core import modelstore

    def explode(*args, **kwargs):
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError(1, "certificate verify failed")
        )

    monkeypatch.setattr(modelstore.urllib.request, "urlopen", explode)
    with pytest.raises(modelstore.CertificateError) as caught:
        modelstore._download("https://example.invalid/m.onnx", tmp_path / "x", timeout=1)

    message = str(caught.value)
    assert "certifi" in message
    assert "models import" in message, "the offline route must be offered"
    assert "--insecure" in message


def test_certificate_error_survives_the_mirror_loop(tmp_path, monkeypatch):
    """The specific fix must not be buried under a generic 'could not fetch'."""
    from imgintel.core import modelstore

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "cache")

    def explode(*args, **kwargs):
        raise modelstore.CertificateError("cert bad")

    monkeypatch.setattr(modelstore, "_download", explode)
    with pytest.raises(modelstore.CertificateError):
        modelstore.pull("face")


def test_insecure_download_is_recorded_in_provenance(tmp_path, monkeypatch):
    """"Fetched over an unauthenticated connection" must survive into the record."""
    from imgintel.core import modelstore

    cache = tmp_path / "cache"
    monkeypatch.setattr(modelstore, "cache_dir", lambda: cache)
    monkeypatch.setenv(modelstore.MODEL_DIR_ENV, "")

    captured = {}

    def fake_download(url, target, *, timeout, progress=None, insecure=False):
        captured["insecure"] = insecure
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")

    monkeypatch.setattr(modelstore, "_download", fake_download)
    modelstore.pull("face", insecure=True)

    assert captured["insecure"] is True
    receipt = modelstore.load_receipts()["face"]
    assert receipt["tls_verified"] is False
    assert modelstore.installed_models()["face"]["tls_verified"] is False


def test_secure_download_records_verified_tls(tmp_path, monkeypatch):
    from imgintel.core import modelstore

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setenv(modelstore.MODEL_DIR_ENV, "")
    monkeypatch.setattr(
        modelstore,
        "_download",
        lambda url, target, *, timeout, progress=None, insecure=False: target.write_bytes(b"w"),
    )
    modelstore.pull("face")
    assert modelstore.load_receipts()["face"]["tls_verified"] is True


def test_provenance_records_where_a_model_came_from(tmp_path, monkeypatch):
    from imgintel.core import modelstore

    monkeypatch.setattr(modelstore, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setenv(modelstore.MODEL_DIR_ENV, "")
    monkeypatch.setattr(
        modelstore,
        "_download",
        lambda url, target, *, timeout, progress=None, insecure=False: target.write_bytes(b"w"),
    )
    modelstore.pull("face")

    record = modelstore.installed_models()["face"]
    assert record["source_url"].startswith("https://")
    assert record["fetched_utc"]
    assert record["pinned"] is False


def test_missing_model_gives_both_recovery_routes():
    from imgintel.core.modelstore import MODELS, status

    if status("detect").present:
        pytest.skip("detect model is installed")
    note = status("detect").note
    assert "models pull" in note
    assert "models import" in note
    assert MODELS["detect"].licence == "Apache-2.0"


# -- availability ----------------------------------------------------------


def test_detection_analyzers_are_registered(registry):
    for name in ("objects", "faces", "vehicles", "scene", "logo"):
        assert name in registry, f"{name} not registered"


def test_detection_analyzers_report_missing_models_not_crashes(registry, images):
    """A missing model must never be an exception."""
    doc = analyze_image(images["photo"], registry=registry, profile="deep")
    for name in ("objects", "faces"):
        block = doc.analyzers.get(name)
        if block is None:
            continue
        assert block.status in ("ok", "unavailable", "skipped")
        if block.status != "ok":
            assert block.error


def test_derived_analyzers_are_dropped_from_light_profiles(registry):
    """scene and vehicles need objects, which is HEAVY — so they are deep-only."""
    from imgintel.core.engine import plan

    for profile in ("quick", "standard"):
        names = {a.name for a in plan(registry, profile=profile)}
        assert "scene" not in names
        assert "vehicles" not in names
    deep = {a.name for a in plan(registry, profile="deep")}
    assert {"objects", "faces", "scene", "vehicles"} <= deep
