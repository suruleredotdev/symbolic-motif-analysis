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

    # And the old path would have dropped every one of them.
    labels_path = state.save_all_labels(tmp_path / "motif_labels.json")
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
