"""Tests for cluster and embedding persistence in PipelineState.

The bug these cover: cluster ids used to survive only inside motif_labels.json
label records, so a motif that was clustered but never labelled lost its
assignment on save — which is most of them on a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panel_art.pipeline_state import PipelineState


@pytest.fixture
def state(analysis_dir: Path) -> PipelineState:
    ps = PipelineState()
    ps.load_from_disk(
        annotated_dir=analysis_dir / "annotated",
        panels_dir=analysis_dir / "panels",
        labels_path=analysis_dir / "motif_labels.json",
    )
    return ps


# ── Clusters ─────────────────────────────────────────────────────────────────

def test_clusters_of_unlabelled_motifs_survive_a_save(state: PipelineState, tmp_path: Path):
    """The regression: no label must not mean no saved cluster."""
    for motif in state.motifs:
        motif.label = None                       # nothing is labelled
        motif.cluster = 5

    path = state.save_clusters(tmp_path / "clusters.json")
    assignments = json.loads(path.read_text())["assignments"]
    assert len(assignments) == len(state.motifs)
    assert set(assignments.values()) == {5}

    # And labels remain empty — a cluster is not a label.
    labels_path, written = state.save_all_labels(tmp_path / "motif_labels.json")
    assert written == 0
    assert json.loads(labels_path.read_text()) == {}


def test_clusters_round_trip(state: PipelineState, tmp_path: Path):
    original = {m.motif_key: m.cluster for m in state.motifs}
    path = state.save_clusters(tmp_path / "clusters.json",
                               params={"min_cluster_size": 3})

    for motif in state.motifs:                   # simulate a fresh kernel
        motif.cluster = -1
    applied = state.load_clusters(path)

    assert applied == sum(1 for c in original.values() if c >= 0)
    assert {m.motif_key: m.cluster for m in state.motifs} == original
    assert json.loads(path.read_text())["params"]["min_cluster_size"] == 3


def test_saved_clusters_record_their_run(state: PipelineState, tmp_path: Path):
    data = json.loads(
        state.save_clusters(tmp_path / "c.json", params={"preprocess": "edges"}).read_text())
    assert data["params"]["preprocess"] == "edges"
    assert data["n_clusters"] == 3
    assert data["generated_at"]


def test_unclustered_and_excluded_motifs_are_not_saved(state: PipelineState, tmp_path: Path):
    state.motifs[0].cluster = -1                 # noise
    state.motifs[1].included = False             # rejected detection
    excluded_key = state.motifs[1].motif_key

    assignments = json.loads(
        state.save_clusters(tmp_path / "c.json").read_text())["assignments"]
    assert state.motifs[0].motif_key not in assignments
    assert excluded_key not in assignments


def test_load_clusters_accepts_a_bare_mapping(state: PipelineState, tmp_path: Path):
    """The shape the interpretation CLI's --clusters override uses."""
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"panel_a/1": 9, "panel_b/0": 9}))
    assert state.load_clusters(path) == 2
    assert state.motif_by_key("panel_a/1").cluster == 9


def test_load_clusters_ignores_keys_that_no_longer_exist(state: PipelineState, tmp_path: Path):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"assignments": {"panel_a/1": 4, "gone/99": 7}}))
    assert state.load_clusters(path) == 1


def test_load_clusters_on_a_missing_file_is_a_no_op(state: PipelineState, tmp_path: Path):
    assert state.load_clusters(tmp_path / "nope.json") == 0


def test_load_from_disk_restores_clusters(analysis_dir: Path, tmp_path: Path):
    ps = PipelineState()
    ps.load_from_disk(annotated_dir=analysis_dir / "annotated",
                      panels_dir=analysis_dir / "panels",
                      labels_path=analysis_dir / "motif_labels.json")
    for motif in ps.motifs:
        motif.cluster = 11
        motif.label = None
    clusters = ps.save_clusters(analysis_dir / "clusters.json")

    fresh = PipelineState()
    fresh.load_from_disk(annotated_dir=analysis_dir / "annotated",
                         panels_dir=analysis_dir / "panels",
                         labels_path=analysis_dir / "motif_labels.json",
                         clusters_path=clusters)
    assert {m.cluster for m in fresh.motifs} == {11}


def test_clusters_file_wins_over_stale_ids_in_labels(analysis_dir: Path):
    """clusters.json is the newer record, so it must load after labels."""
    ps = PipelineState()
    ps.load_from_disk(annotated_dir=analysis_dir / "annotated",
                      panels_dir=analysis_dir / "panels",
                      labels_path=analysis_dir / "motif_labels.json")
    assert ps.motif_by_key("panel_a/1").cluster == 0          # from motif_labels.json

    (analysis_dir / "clusters.json").write_text(
        json.dumps({"assignments": {"panel_a/1": 42}}))
    fresh = PipelineState()
    fresh.load_from_disk(annotated_dir=analysis_dir / "annotated",
                         panels_dir=analysis_dir / "panels",
                         labels_path=analysis_dir / "motif_labels.json",
                         clusters_path=analysis_dir / "clusters.json")
    assert fresh.motif_by_key("panel_a/1").cluster == 42


# ── Embeddings ───────────────────────────────────────────────────────────────

def test_embeddings_round_trip(state: PipelineState, tmp_path: Path):
    keys = [m.motif_key for m in state.included_motifs()]
    state.embeddings = np.arange(len(keys) * 4, dtype=float).reshape(len(keys), 4)
    state.save_embeddings(tmp_path / "e.npy", tmp_path / "e.txt", keys)

    fresh = PipelineState()
    restored = fresh.load_embeddings(tmp_path / "e.npy", tmp_path / "e.txt")
    assert restored == keys
    assert np.array_equal(fresh.embeddings, state.embeddings)


def test_saving_embeddings_with_mismatched_keys_raises(state: PipelineState, tmp_path: Path):
    state.embeddings = np.zeros((3, 4))
    with pytest.raises(ValueError, match="keys"):
        state.save_embeddings(tmp_path / "e.npy", tmp_path / "e.txt", ["only/one"])


def test_saving_embeddings_before_computing_them_raises(state: PipelineState, tmp_path: Path):
    with pytest.raises(ValueError, match="no embeddings"):
        state.save_embeddings(tmp_path / "e.npy", tmp_path / "e.txt")


def test_a_stale_cache_is_rejected_rather_than_guessed(state: PipelineState, tmp_path: Path):
    """Row count and key count disagreeing means the crops changed — recompute."""
    np.save(tmp_path / "e.npy", np.zeros((5, 4)))
    (tmp_path / "e.txt").write_text("panel_a/1\npanel_a/2")

    fresh = PipelineState()
    assert fresh.load_embeddings(tmp_path / "e.npy", tmp_path / "e.txt") == []
    assert fresh.embeddings is None


def test_loading_an_absent_cache_is_a_no_op(tmp_path: Path):
    fresh = PipelineState()
    assert fresh.load_embeddings(tmp_path / "nope.npy", tmp_path / "nope.txt") == []


def test_save_embeddings_accepts_an_explicit_matrix(state: PipelineState, tmp_path: Path):
    """The notebook holds its working copy in cell state, not on PipelineState."""
    keys = [m.motif_key for m in state.included_motifs()]
    working = np.ones((len(keys), 3))
    assert state.embeddings is None                  # nothing mirrored onto the state

    state.save_embeddings(tmp_path / "e.npy", tmp_path / "e.txt", keys,
                          embeddings=working)
    fresh = PipelineState()
    assert fresh.load_embeddings(tmp_path / "e.npy", tmp_path / "e.txt") == keys
    assert np.array_equal(fresh.embeddings, working)


# ── Label file: two writers, one file ────────────────────────────────────────

def test_one_motif_gets_exactly_one_key(state: PipelineState, tmp_path: Path):
    """The notebook and the script must not each write their own entry."""
    from panel_art.pipeline_state import key_to_motif, motif_label_key
    motif = state.motif_by_key("panel_a/1")

    notebook_key = motif_label_key(motif.panel_stem, motif.index)
    legacy_key = (f"../../frobenius_artifacts/analysis/motifs_norm/"
                  f"{motif.panel_stem}/{motif.index:03d}_motif_iou0.942.png")
    assert key_to_motif(notebook_key) == key_to_motif(legacy_key) == "panel_a/1"


def test_saving_replaces_a_legacy_duplicate_rather_than_adding_one(
        state: PipelineState, tmp_path: Path):
    from panel_art.pipeline_state import key_to_motif
    path = tmp_path / "motif_labels.json"
    path.write_text(json.dumps({
        "../../frobenius_artifacts/analysis/motifs_norm/panel_a/001_motif_iou0.942.png":
            {"label": "old_form", "cluster": 0, "source": "llm"},
    }))

    state.save_label(state.motif_by_key("panel_a/1"), "new_form", source="human")
    state.save_all_labels(path, only_changed=True)

    saved = json.loads(path.read_text())
    for_motif = [k for k in saved if key_to_motif(k) == "panel_a/1"]
    assert len(for_motif) == 1                       # not two
    assert saved[for_motif[0]]["label"] == "new_form"


def test_saving_preserves_entries_this_session_never_loaded(
        state: PipelineState, tmp_path: Path):
    """label_motifs.py may have written entries after the kernel started."""
    path = tmp_path / "motif_labels.json"
    path.write_text(json.dumps({
        "../../frobenius_artifacts/analysis/motifs_norm/other_panel/000_motif.png":
            {"label": "written_by_the_script", "cluster": 3, "source": "cluster-brief"},
    }))

    state.save_label(state.motif_by_key("panel_a/1"), "edited_here", source="human")
    state.save_all_labels(path, only_changed=True)

    saved = json.loads(path.read_text())
    assert any(v["label"] == "written_by_the_script" for v in saved.values())
    assert any(v["label"] == "edited_here" for v in saved.values())


def test_only_changed_writes_just_this_sessions_edits(state: PipelineState, tmp_path: Path):
    path = tmp_path / "motif_labels.json"
    assert all(not m.dirty for m in state.motifs)     # loading marks nothing dirty

    state.save_label(state.motif_by_key("panel_a/1"), "touched", source="human")
    _, written = state.save_all_labels(path, only_changed=True)
    assert written == 1

    # And the flag clears, so a second save is a no-op rather than a rewrite.
    _, again = state.save_all_labels(path, only_changed=True)
    assert again == 0


def test_a_stale_in_memory_label_cannot_clobber_a_newer_one_on_disk(
        state: PipelineState, tmp_path: Path):
    """The real hazard: the kernel holds an old copy while the script improves it."""
    from panel_art.pipeline_state import motif_label_key
    path = tmp_path / "motif_labels.json"
    key = motif_label_key("panel_a", 2)
    path.write_text(json.dumps({key: {"label": "improved_on_disk",
                                      "cluster": 0, "source": "human"}}))

    # This session edited a *different* motif and never touched panel_a/2.
    state.save_label(state.motif_by_key("panel_a/1"), "edited_here", source="human")
    state.save_all_labels(path, only_changed=True)

    assert json.loads(path.read_text())[key]["label"] == "improved_on_disk"


def test_save_all_labels_without_only_changed_still_merges(state: PipelineState, tmp_path: Path):
    path = tmp_path / "motif_labels.json"
    path.write_text(json.dumps({
        "../../frobenius_artifacts/analysis/motifs_norm/elsewhere/000_motif.png":
            {"label": "keep_me", "cluster": 1, "source": "human"}}))

    _, written = state.save_all_labels(path)
    saved = json.loads(path.read_text())
    assert written == 7                              # every labelled motif in memory
    assert any(v["label"] == "keep_me" for v in saved.values())
