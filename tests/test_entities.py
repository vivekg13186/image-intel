"""The entity database: enrolment, identity matching, and the biometric gate."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from imgintel.core.context import CaseConfig
from imgintel.core.deps import has_module
from imgintel.core.engine import analyze_image
from imgintel.store.entities import (
    DEFAULT_MARGIN,
    DEFAULT_THRESHOLD,
    EntityStore,
    normalise,
)


def _needs(module: str):
    return pytest.mark.skipif(not has_module(module), reason=f"{module} not installed")


def data_of(doc, analyzer: str) -> dict:
    block = doc.analyzers[analyzer]
    assert block.status == "ok", f"{analyzer}: {block.error}"
    return block.data


def authorised(**kwargs) -> CaseConfig:
    return CaseConfig(
        case_id="C-1",
        operator="tester",
        biometric_lawful_basis="Warrant 2026/114",
        biometric_authorised_by="Supervising Officer",
        **kwargs,
    )


def unit(*values: float) -> np.ndarray:
    """A normalised vector, padded to SFace's 128 dimensions."""
    vector = np.zeros(128, dtype=np.float32)
    vector[: len(values)] = values
    return normalise(vector)


# -- the gate --------------------------------------------------------------
#
# These are the most important tests in the project. Identification must refuse
# rather than warn, and the refusal must be recorded.


def test_no_lawful_basis_refuses_even_with_the_run_flag():
    case = CaseConfig(case_id="C-1", operator="tester")
    reason = case.biometric_refusal(run_flag=True)
    assert reason is not None
    assert "lawful basis" in reason


def test_lawful_basis_alone_is_not_enough():
    """Authorisation must never be inherited silently from a config file."""
    reason = authorised().biometric_refusal(run_flag=False)
    assert reason is not None
    assert "--allow-biometrics" in reason


def test_both_conditions_permit_identification():
    assert authorised().biometric_refusal(run_flag=True) is None


def test_expired_authorisation_refuses():
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    case = authorised(biometric_expires=yesterday)
    reason = case.biometric_refusal(run_flag=True)
    assert reason is not None
    assert "expired" in reason
    assert not case.biometrics_authorized


def test_unexpired_authorisation_permits():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert authorised(biometric_expires=tomorrow).biometric_refusal(run_flag=True) is None


def test_unparseable_expiry_fails_closed():
    """An authorisation record that cannot be read is not an authorisation."""
    case = authorised(biometric_expires="whenever")
    assert case.biometric_authorisation_expired
    assert case.biometric_refusal(run_flag=True) is not None


def test_facedb_skips_with_the_reason_rather_than_running(images, tmp_path):
    doc = analyze_image(
        images["photo"],
        only=["facedb"],
        entity_db=tmp_path / "e.db",
        allow_biometrics=True,  # but no lawful basis
        case=CaseConfig(case_id="C-1"),
    )
    block = doc.analyzers["facedb"]
    assert block.status == "skipped"
    assert "lawful basis" in block.error
    assert not block.data, "nothing may be computed when the gate refuses"


def test_refusal_is_recorded_in_evidence_provenance(images, tmp_path):
    """The bundle must show that identification was not performed, and why."""
    from imgintel.report.evidence import build_provenance

    doc = analyze_image(images["photo"], only=["facedb"], case=CaseConfig(case_id="C-1"))
    provenance = build_provenance(doc, images["photo"])

    biometrics = provenance["biometrics"]
    assert biometrics["performed"] is False
    assert biometrics["not_performed_because"]
    assert biometrics["lawful_basis"] is None


def test_authorisation_travels_into_the_document(images, tmp_path):
    doc = analyze_image(
        images["photo"], only=["fileinfo"], case=authorised(), allow_biometrics=True
    )
    assert doc.case.biometric_lawful_basis == "Warrant 2026/114"
    assert doc.case.biometric_authorised_by == "Supervising Officer"
    assert doc.run.allow_biometrics is True


# -- entity store ----------------------------------------------------------


def test_entities_and_references_roundtrip(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        alice = store.add_entity("Alice", "person", notes="subject 1")
        store.add_reference(alice, unit(1, 0, 0), embedder="test")
        store.add_reference(alice, unit(0.9, 0.1, 0), embedder="test")

        entities = store.entities("person")
        assert len(entities) == 1
        assert entities[0].name == "Alice"
        assert entities[0].reference_count == 2
        assert entities[0].notes == "subject 1"


def test_adding_the_same_entity_twice_reuses_it(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        first = store.add_entity("Alice", "person")
        second = store.add_entity("Alice", "person")
        assert first == second
        assert store.count("person") == 1


def test_same_name_different_kinds_are_distinct(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        person = store.add_entity("Rex", "person")
        obj = store.add_entity("Rex", "object")
        assert person != obj


def test_forget_removes_references_too(tmp_path):
    path = tmp_path / "e.db"
    with EntityStore(path) as store:
        alice = store.add_entity("Alice", "person")
        store.add_reference(alice, unit(1, 0), embedder="test")
        assert store.forget("Alice") == 1
        assert store.count("person") == 0
        assert store.stats()["refs"] == 0, "cascade must remove the vectors"


def test_audit_log_records_enrolment_and_search(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        store.log("enrol:person", "Alice", operator="tester", case_id="C-1")
        store.log("search:person", "1 face")
        actions = [entry["action"] for entry in store.audit_log()]
    assert "enrol:person" in actions
    assert "search:person" in actions


def test_vectors_are_normalised_on_storage(tmp_path):
    """Cosine similarity is a dot product only if everything is unit length."""
    with EntityStore(tmp_path / "e.db") as store:
        alice = store.add_entity("Alice", "person")
        store.add_reference(alice, np.array([3.0, 4.0] + [0.0] * 126), embedder="test")
        match, _ = store.search(np.array([3.0, 4.0] + [0.0] * 126), kind="person",
                                embedder="test", threshold=0.1)
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


# -- matching: threshold and margin ----------------------------------------


def _store_with(tmp_path, people: dict[str, np.ndarray]) -> EntityStore:
    store = EntityStore(tmp_path / "e.db")
    for name, vector in people.items():
        entity = store.add_entity(name, "person")
        store.add_reference(entity, vector, embedder="test")
    return store


def test_clear_match_is_conclusive(tmp_path):
    with _store_with(tmp_path, {"Alice": unit(1, 0, 0), "Bob": unit(0, 1, 0)}) as store:
        match, shortlist = store.search(unit(0.99, 0.01, 0), kind="person", embedder="test")
    assert match.verdict == "match"
    assert match.entity == "Alice"
    assert match.margin > DEFAULT_MARGIN
    assert len(shortlist) == 2


def test_two_similar_candidates_are_inconclusive(tmp_path):
    """The test that matters: an identification tool must be able to say "I don't know".

    Both candidates clear the similarity threshold, so a threshold-only design
    would confidently name whichever scored a hair higher.
    """
    with _store_with(
        tmp_path, {"Alice": unit(1, 0, 0), "Alicia": unit(0.999, 0.045, 0)}
    ) as store:
        match, shortlist = store.search(unit(1, 0, 0), kind="person", embedder="test")

    assert match.verdict == "inconclusive"
    assert match.margin < DEFAULT_MARGIN
    assert match.runner_up is not None
    assert len(shortlist) == 2, "near-misses must stay visible to the analyst"


def test_below_threshold_is_no_match(tmp_path):
    with _store_with(tmp_path, {"Alice": unit(1, 0, 0)}) as store:
        match, _ = store.search(unit(0, 1, 0), kind="person", embedder="test")
    assert match.verdict == "no match"
    assert match.similarity < DEFAULT_THRESHOLD


def test_a_single_enrolled_identity_can_still_match(tmp_path):
    """With no runner-up the margin is undefined, not zero."""
    with _store_with(tmp_path, {"Alice": unit(1, 0, 0)}) as store:
        match, _ = store.search(unit(1, 0, 0), kind="person", embedder="test")
    assert match.verdict == "match"
    assert match.runner_up is None


def test_multiple_references_rank_as_one_candidate(tmp_path):
    """Five photos of Alice are one candidate, not five.

    Otherwise they fill the shortlist and hide the runner-up that the margin
    test depends on — turning an inconclusive result into a confident one.
    """
    with EntityStore(tmp_path / "e.db") as store:
        alice = store.add_entity("Alice", "person")
        for offset in (0.0, 0.01, 0.02, 0.03, 0.04):
            store.add_reference(alice, unit(1 - offset, offset, 0), embedder="test")
        bob = store.add_entity("Bob", "person")
        store.add_reference(bob, unit(0.98, 0.03, 0), embedder="test")

        match, shortlist = store.search(unit(1, 0, 0), kind="person", embedder="test")

    assert [c["entity"] for c in shortlist] == ["Alice", "Bob"]
    assert match.runner_up == "Bob"


def test_empty_database_returns_no_candidates(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        match, shortlist = store.search(unit(1, 0), kind="person", embedder="test")
    assert match is None
    assert shortlist == []


def test_dimension_mismatch_is_an_explicit_error(tmp_path):
    """Swapping embedding models must not silently produce garbage matches."""
    with EntityStore(tmp_path / "e.db") as store:
        entity = store.add_entity("Alice", "person")
        store.add_reference(entity, np.ones(128, dtype=np.float32), embedder="test")
        with pytest.raises(ValueError, match="dimensional"):
            store.search(np.ones(64, dtype=np.float32), kind="person", embedder="test")


def test_kinds_do_not_cross_contaminate(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        obj = store.add_entity("Suitcase", "object")
        store.add_reference(obj, unit(1, 0, 0), embedder="test")
        match, _ = store.search(unit(1, 0, 0), kind="person", embedder="test")
    assert match is None


def test_embedders_do_not_cross_contaminate(tmp_path):
    with EntityStore(tmp_path / "e.db") as store:
        entity = store.add_entity("Alice", "person")
        store.add_reference(entity, unit(1, 0, 0), embedder="sface-2021dec")
        match, _ = store.search(unit(1, 0, 0), kind="person", embedder="some-other-model")
    assert match is None


# -- facedb analyzer -------------------------------------------------------


def test_facedb_requires_an_entity_database(images):
    doc = analyze_image(
        images["photo"], only=["facedb"], allow_biometrics=True, case=authorised()
    )
    block = doc.analyzers["facedb"]
    assert block.status == "skipped"
    assert "entity database" in block.error


def test_facedb_reports_a_missing_database_file(images, tmp_path):
    doc = analyze_image(
        images["photo"],
        only=["facedb"],
        entity_db=tmp_path / "nope.db",
        allow_biometrics=True,
        case=authorised(),
    )
    assert "not found" in doc.analyzers["facedb"].error


def test_facedb_depends_on_face_detection(registry):
    analyzer = registry.get("facedb")
    assert "faces" in analyzer.requires


def test_facedb_is_deep_only(registry):
    from imgintel.core.engine import plan

    assert "facedb" not in {a.name for a in plan(registry, profile="standard")}


# -- object matching -------------------------------------------------------


@_needs("cv2")
def test_enrolled_object_is_matched(images, tmp_path):
    """The ORB path is fully verifiable — it needs no downloaded weights."""
    import cv2
    from PIL import Image

    from imgintel.analyzers.objectdb import OBJECT_EMBEDDER, encode_descriptors
    from imgintel.util.orbmatch import describe

    rng = np.random.default_rng(21)
    mark = cv2.GaussianBlur((rng.integers(0, 255, (140, 180, 3))).astype(np.uint8), (3, 3), 0)
    gray = cv2.cvtColor(mark, cv2.COLOR_RGB2GRAY)

    db = tmp_path / "e.db"
    with EntityStore(db) as store:
        entity = store.add_entity("blue-holdall", "object", notes="seized bag")
        keypoints, descriptors = describe(gray)
        store.add_raw_reference(
            entity,
            encode_descriptors(keypoints, descriptors, gray.shape[:2]),
            embedder=OBJECT_EMBEDDER,
        )

    scene = np.asarray(Image.open(images["photo"]).convert("RGB")).copy()
    scene[150:290, 200:380] = mark
    scene_path = tmp_path / "scene.png"
    Image.fromarray(scene).save(scene_path)

    doc = analyze_image(scene_path, only=["objectdb"], entity_db=db)
    data = data_of(doc, "objectdb")

    assert data["references_loaded"] == 1
    assert data["matches"], "enrolled object should be found"
    assert data["matches"][0]["name"] == "blue-holdall"
    assert any("blue-holdall" in f.label for f in doc.findings)


@_needs("cv2")
def test_absent_object_is_not_matched(images, tmp_path):
    import cv2

    from imgintel.analyzers.objectdb import OBJECT_EMBEDDER, encode_descriptors
    from imgintel.util.orbmatch import describe

    rng = np.random.default_rng(22)
    other = cv2.cvtColor(
        cv2.GaussianBlur((rng.integers(0, 255, (140, 180, 3))).astype(np.uint8), (3, 3), 0),
        cv2.COLOR_RGB2GRAY,
    )
    db = tmp_path / "e.db"
    with EntityStore(db) as store:
        entity = store.add_entity("elsewhere", "object")
        keypoints, descriptors = describe(other)
        store.add_raw_reference(
            entity,
            encode_descriptors(keypoints, descriptors, other.shape[:2]),
            embedder=OBJECT_EMBEDDER,
        )

    doc = analyze_image(images["photo"], only=["objectdb"], entity_db=db)
    assert not data_of(doc, "objectdb")["matches"]


@_needs("cv2")
def test_object_matching_needs_no_biometric_authorisation(images, tmp_path):
    """A photograph of a suitcase is not biometric data."""
    db = tmp_path / "e.db"
    EntityStore(db).close()
    doc = analyze_image(images["photo"], only=["objectdb"], entity_db=db, case=CaseConfig())
    assert doc.analyzers["objectdb"].status == "ok"


def test_descriptor_encoding_roundtrips():
    from imgintel.analyzers.objectdb import _decode_descriptors, encode_descriptors

    class FakeKeypoint:
        def __init__(self, x, y, size):
            self.pt = (x, y)
            self.size = size

    keypoints = [FakeKeypoint(1.0, 2.0, 7.0), FakeKeypoint(3.0, 4.0, 9.0)]
    descriptors = np.arange(64, dtype=np.uint8).reshape(2, 32)

    decoded = _decode_descriptors(encode_descriptors(keypoints, descriptors, (100, 200)))
    assert decoded is not None
    out_descriptors, shape, points = decoded
    assert np.array_equal(out_descriptors, descriptors)
    assert shape == (100, 200)
    assert points[0] == [1.0, 2.0, 7.0]


def test_corrupt_descriptor_blob_is_ignored_not_fatal():
    from imgintel.analyzers.objectdb import _decode_descriptors

    assert _decode_descriptors(b"nonsense") is None


# -- case file -------------------------------------------------------------


def test_case_file_roundtrips(tmp_path):
    from imgintel.core import casefile

    path = tmp_path / "case.json"
    casefile.save(authorised(biometric_expires="2030-01-01"), path)
    loaded = casefile.load(path)

    assert loaded.case_id == "C-1"
    assert loaded.biometric_lawful_basis == "Warrant 2026/114"
    assert loaded.biometrics_authorized


def test_missing_case_file_is_an_empty_config(tmp_path):
    from imgintel.core import casefile

    case = casefile.load(tmp_path / "absent.json")
    assert case.case_id is None
    assert not case.biometrics_authorized


def test_corrupt_case_file_does_not_grant_authorisation(tmp_path):
    """Failing closed matters more than reporting the parse error."""
    from imgintel.core import casefile

    path = tmp_path / "case.json"
    path.write_text("{ not json")
    assert not casefile.load(path).biometrics_authorized


def test_cli_flags_override_the_case_file(tmp_path):
    from imgintel.core import casefile

    path = tmp_path / "case.json"
    casefile.save(authorised(), path)
    merged = casefile.merge(casefile.load(path), operator="someone-else", case_id=None)

    assert merged.operator == "someone-else"
    assert merged.case_id == "C-1", "None must not clear an existing value"
    assert merged.biometric_lawful_basis == "Warrant 2026/114"


# -- flat record -----------------------------------------------------------


def test_flat_columns_cover_entity_matching():
    from imgintel.report.flatten import FLAT_COLUMNS

    for column in ("identities", "identity_verdicts", "matched_objects"):
        assert column in FLAT_COLUMNS


def test_inconclusive_verdicts_appear_in_the_summary_row():
    """A row showing only confirmed matches would hide every near-miss."""
    from imgintel.core.schema import AnalyzerBlock, FindingsDocument, TargetInfo
    from imgintel.report.flatten import flatten

    doc = FindingsDocument(target=TargetInfo(path="/x.jpg", filename="x.jpg"))
    doc.analyzers["facedb"] = AnalyzerBlock(
        analyzer="facedb",
        version="1.0.0",
        status="ok",
        data={
            "matches": [
                {"entity": "Alice", "verdict": "match"},
                {"entity": "Bob", "verdict": "inconclusive"},
            ]
        },
    )
    row = flatten(doc)
    assert row["identities"] == "Alice"
    assert "Bob:inconclusive" in row["identity_verdicts"]
