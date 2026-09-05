"""Phase 2: batch execution, the sqlite index, and duplicate detection."""

from __future__ import annotations

import pytest

from imgintel.core.context import CaseConfig
from imgintel.report.flatten import FLAT_COLUMNS
from imgintel.runner import BatchConfig, find_images, run_batch
from imgintel.store.bktree import BKTree, cluster, cluster_multi
from imgintel.store.db import open_index
from imgintel.util.imgmath import hamming_hex

QUICK = BatchConfig(only=["hashes", "perceptual", "fileinfo"], workers=1)


# -- discovery -------------------------------------------------------------


def test_finds_images_recursively(corpus):
    found = find_images(corpus)
    names = {p.name for p in found}
    assert "sharp.jpg" in names
    assert "photo_small.jpg" in names, "must descend into subdirectories"


def test_non_recursive_stays_shallow(corpus):
    shallow = {p.name for p in find_images(corpus, recursive=False)}
    assert "photo_small.jpg" not in shallow


def test_ignores_non_images(corpus, tmp_path):
    (corpus / "notes.txt").write_text("not an image")
    assert not any(p.suffix == ".txt" for p in find_images(corpus))


def test_ordering_is_deterministic(corpus):
    assert find_images(corpus) == find_images(corpus)


# -- execution -------------------------------------------------------------


def test_every_input_appears_in_the_output(corpus):
    """Silent omission is how an investigator concludes a file was clean."""
    images = find_images(corpus)
    result = run_batch(images, QUICK)
    assert result.total == len(images)
    assert {r["path_input"] for r in result.rows} == {str(p) for p in images}


def test_undecodable_file_is_recorded_not_dropped(corpus):
    result = run_batch(find_images(corpus), QUICK)
    broken = next(r for r in result.rows if r["filename"] == "broken.jpg")
    # fileinfo still succeeds on it, so it is a row with findings, not a crash.
    assert broken["findings_high"] >= 1


def test_rows_use_the_canonical_columns(corpus):
    result = run_batch(find_images(corpus)[:2], QUICK)
    for row in result.rows:
        assert set(FLAT_COLUMNS) <= set(row)


def test_parallel_and_serial_agree(corpus):
    """Multiprocessing must not change results — only how fast they arrive."""
    images = find_images(corpus)
    serial = run_batch(images, BatchConfig(only=["hashes", "fileinfo"], workers=1))
    parallel = run_batch(images, BatchConfig(only=["hashes", "fileinfo"], workers=3))

    def digest(result):
        return {r["path_input"]: r["sha256"] for r in result.rows}

    assert digest(serial) == digest(parallel)
    assert parallel.workers > 1


def test_resume_skips_completed_paths(corpus):
    images = find_images(corpus)
    done = {str(images[0]), str(images[1])}
    result = run_batch(images, QUICK, already_done=done)
    assert result.skipped == 2
    assert result.total == len(images) - 2


def test_case_metadata_reaches_every_document(corpus):
    config = BatchConfig(
        only=["hashes"], workers=1, case=CaseConfig(case_id="C-9", operator="tester")
    )
    result = run_batch(find_images(corpus)[:2], config)
    assert all(d.case.case_id == "C-9" for d in result.documents)


def test_progress_callback_fires_once_per_image(corpus):
    seen = []
    images = find_images(corpus)[:3]
    run_batch(images, QUICK, on_result=lambda p, d, e: seen.append(p))
    assert len(seen) == len(images)


# -- index -----------------------------------------------------------------


def test_index_roundtrips_rows(corpus, tmp_path):
    result = run_batch(find_images(corpus), QUICK)
    with open_index(tmp_path) as index:
        index.upsert_many(result.rows)
        assert index.count() == result.total
        stored = {r["path"] for r in index.rows()}
    assert stored == {r["path"] for r in result.rows}


def test_index_upsert_is_idempotent(corpus, tmp_path):
    result = run_batch(find_images(corpus)[:3], QUICK)
    with open_index(tmp_path) as index:
        index.upsert_many(result.rows)
        index.upsert_many(result.rows)
        assert index.count() == 3


def test_index_tracks_job_state(tmp_path):
    with open_index(tmp_path) as index:
        index.mark("/a.jpg", "done")
        index.mark("/b.jpg", "failed", "boom")
        assert index.completed() == {"/a.jpg"}
        assert index.stats()["failed"] == 1


def test_index_finds_exact_duplicates(corpus, tmp_path):
    result = run_batch(find_images(corpus), QUICK)
    with open_index(tmp_path) as index:
        index.upsert_many(result.rows)
        groups = index.exact_duplicates("sha256")
    names = {tuple(sorted(p.rsplit("/", 1)[-1] for p in paths)) for _, paths in groups}
    assert ("sharp.jpg", "sharp_copy.jpg") in names


def test_index_rejects_unknown_columns(tmp_path):
    with open_index(tmp_path) as index, pytest.raises(ValueError, match="unknown column"):
        index.hashes("; DROP TABLE images")


def test_index_survives_reopening(corpus, tmp_path):
    result = run_batch(find_images(corpus)[:2], QUICK)
    with open_index(tmp_path) as index:
        index.upsert_many(result.rows)
    with open_index(tmp_path) as reopened:
        assert reopened.count() == 2


# -- BK-tree ---------------------------------------------------------------


def test_bktree_finds_exact_and_near_matches():
    tree = BKTree([("00000000", "a"), ("00000001", "b"), ("ffffffff", "c")])
    assert [p for _, p in tree.within("00000000", 0)] == ["a"]
    assert {p for _, p in tree.within("00000000", 1)} == {"a", "b"}
    assert {p for _, p in tree.within("00000000", 64)} == {"a", "b", "c"}


def test_bktree_matches_brute_force():
    """The pruning must not change the answer, only the cost."""
    import random

    rng = random.Random(4)
    keys = [f"{rng.getrandbits(64):016x}" for _ in range(300)]
    tree = BKTree((k, i) for i, k in enumerate(keys))
    probe = keys[0]
    for radius in (0, 4, 12, 24):
        expected = {i for i, k in enumerate(keys) if hamming_hex(probe, k) <= radius}
        assert {i for _, i in tree.within(probe, radius)} == expected


def test_bktree_rejects_mixed_hash_widths():
    tree = BKTree([("0000", "a")])
    with pytest.raises(ValueError, match="width"):
        tree.add("00000000", "b")


def test_bktree_handles_duplicate_keys():
    tree = BKTree([("abcd", "a"), ("abcd", "b")])
    assert {p for _, p in tree.within("abcd", 0)} == {"a", "b"}
    assert len(tree) == 2


def test_cluster_groups_transitively():
    """Single linkage: resize then recompress must stay one cluster."""
    entries = [("0000000000000000", "a"), ("0000000000000003", "b"),
               ("000000000000000f", "c"), ("ffffffffffffffff", "z")]
    groups = cluster(entries, radius=2)
    assert len(groups) == 1
    assert {p for _, p in groups[0]} == {"a", "b", "c"}


def test_cluster_multi_links_on_any_hash():
    """The reason several hashes are computed: each fails on a different edit.

    Here pHash puts the pair far apart while dHash sees them as near-identical,
    which is exactly what a resize of high-frequency content does in practice.
    """
    far, near = "0f0f0f0f0f0f0f0f", "0000000000000001"
    hash_sets = [["0000000000000000", "0000000000000000"], [far, near]]
    assert not cluster_multi([[h[0]] for h in hash_sets], ["a", "b"], radius=4)
    groups = cluster_multi(hash_sets, ["a", "b"], radius=4)
    assert len(groups) == 1
    assert {p for _, p in groups[0]} == {"a", "b"}


def test_near_duplicates_found_in_a_real_corpus(corpus):
    """End to end: a resize and a recompression cluster with their original."""
    result = run_batch(find_images(corpus), QUICK)
    usable = [r for r in result.rows if r.get("phash")]
    groups = cluster_multi(
        [[r["phash"], r["dhash"]] for r in usable],
        [r["filename"] for r in usable],
        radius=8,
    )
    families = [{p for _, p in g} for g in groups]
    assert any(
        {"photo.jpg", "photo_small.jpg", "photo_q40.jpg"} <= family for family in families
    ), f"resize and recompress should cluster with the original; got {families}"


def test_cluster_ignores_singletons():
    assert cluster([("0000", "a"), ("ffff", "b")], radius=1) == []
