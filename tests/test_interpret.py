"""Tests for panel_art.interpret — loading, statistics, prompts, persistence.

Everything here runs without an API key: the Claude passes are exercised
through a stub client, so prompt assembly and response handling are covered
without network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panel_art.interpret import (
    Corpus,
    InterpretationStore,
    Interpreter,
    MotifView,
    build_cluster_prompt,
    build_corpus_prompt,
    build_panel_prompt,
    cluster_context_lines,
    compute_cluster_stats,
    corpus_scale,
    load_corpus,
    parse_motif_key,
    render_clusters_markdown,
    render_panel_markdown,
)


# ── Key parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected", [
    ("../../analysis/motifs_norm/panel_a/003_motif_iou0.900.png", "panel_a/3"),
    ("analysis/motifs/EBA-B_00425_panel_00/012_register_iou0.8.png",
     "EBA-B_00425_panel_00/12"),
    ("nonsense.png", None),
])
def test_parse_motif_key(path, expected):
    assert parse_motif_key(path) == expected


# ── Corpus loading ───────────────────────────────────────────────────────────

def test_load_corpus_reads_panels_motifs_and_labels(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)

    assert corpus.panel_stems() == ["panel_a", "panel_b"]
    assert len(corpus.motifs) == 7
    assert corpus.panels["panel_a"].width == 400
    assert corpus.panels["panel_a"].height == 900

    figure = corpus.by_key("panel_a/1")
    assert figure.label == "standing_figure"
    assert figure.cluster == 0
    assert figure.label_source == "human"


def test_load_corpus_groups_clusters_and_excludes_noise(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    clusters = corpus.clusters()

    assert sorted(clusters) == [0, 1, 2]
    assert len(clusters[0]) == 4          # two on panel_a, two on panel_b
    assert len(clusters[2]) == 1

    corpus.motifs[0].cluster = -1
    assert -1 not in corpus.clusters()
    assert -1 in corpus.clusters(include_noise=True)


def test_load_corpus_attaches_embeddings(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)

    assert corpus.embeddings is not None
    assert corpus.embeddings.shape == (7, 8)
    assert corpus.embedding_for("panel_a/1") is not None
    assert corpus.embedding_for("does/not-exist") is None


def test_embeddings_join_across_the_cropped_stem_suffix(analysis_dir: Path, tmp_path: Path):
    """Crop paths may carry a `_cropped` stem that the detections side strips."""
    npy, txt = tmp_path / "c.npy", tmp_path / "c.txt"
    np.save(npy, np.ones((2, 4)))
    txt.write_text("\n".join([
        "../../analysis/motifs_norm/panel_a_cropped/000_motif_iou0.9.png",
        "../../analysis/motifs_norm/panel_a_cropped/001_motif_iou0.9.png",
    ]))
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)

    assert corpus.embedding_for("panel_a/0") is not None      # suffix bridged
    assert corpus.embedding_for("panel_a/1") is not None
    assert corpus.embedding_for("panel_b/0") is None          # genuinely absent


def test_embedding_coverage_reports_the_join_rate(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    assert load_corpus(analysis_dir, npy, txt).embedding_coverage() == (7, 7)

    empty = load_corpus(analysis_dir)
    assert empty.embedding_coverage() == (0, 7)


def test_embedding_coverage_detects_a_stale_embedding_run(analysis_dir: Path, tmp_path: Path):
    npy, txt = tmp_path / "stale.npy", tmp_path / "stale.txt"
    np.save(npy, np.ones((1, 4)))
    txt.write_text("../../analysis/motifs_norm/some_other_panel/000_motif_iou0.9.png")

    matched, total = load_corpus(analysis_dir, npy, txt).embedding_coverage()
    assert (matched, total) == (0, 7)


def test_mismatched_embeddings_and_paths_raise(analysis_dir: Path, tmp_path: Path):
    npy = tmp_path / "bad.npy"
    txt = tmp_path / "bad.txt"
    np.save(npy, np.zeros((3, 4)))
    txt.write_text("../../analysis/motifs_norm/panel_a/000_motif_iou0.9.png")

    with pytest.raises(ValueError, match="mismatch"):
        load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)


def test_cluster_overrides_are_applied(analysis_dir: Path, tmp_path: Path):
    overrides = tmp_path / "clusters.json"
    overrides.write_text(json.dumps({
        "../../analysis/motifs_norm/panel_a/001_motif_iou0.900.png": 7,
        "panel_b/2": {"cluster": 8},
    }))
    corpus = load_corpus(analysis_dir, clusters_path=overrides)

    assert corpus.by_key("panel_a/1").cluster == 7
    assert corpus.by_key("panel_b/2").cluster == 8
    assert corpus.by_key("panel_a/2").cluster == 0        # untouched


def test_corpus_scale_summary(analysis_dir: Path):
    scale = corpus_scale(load_corpus(analysis_dir))
    assert scale == {"panels": 2, "motifs": 7, "clusters": 3,
                     "unclustered": 0, "labelled": 7}


# ── Layout integration ───────────────────────────────────────────────────────

def test_layout_for_panel_uses_real_panel_dimensions(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    layout = corpus.layout_for("panel_a")

    assert layout.panel_width == 400 and layout.panel_height == 900
    assert layout.by_key("panel_a/0").is_field is True     # the border detection
    assert ("panel_a/1", "panel_a/2") in layout.mirror_pairs
    assert len(layout.registers) == 2


def test_crop_stays_inside_the_panel(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    crop = corpus.crop(corpus.by_key("panel_a/0"), padding=100)
    assert crop.width <= 400 and crop.height <= 900


# ── Cluster statistics ───────────────────────────────────────────────────────

def test_cluster_stats_without_embeddings(analysis_dir: Path):
    stats = compute_cluster_stats(load_corpus(analysis_dir))

    assert stats[0].size == 4
    assert stats[0].panel_spread == 2
    assert stats[0].label_counts == {"standing_figure": 4}
    assert stats[0].cohesion is None
    assert stats[0].neighbour_clusters == []
    assert len(stats[0].exemplar_keys) == 4


def test_cluster_stats_with_embeddings(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)
    stats = compute_cluster_stats(corpus, exemplars=2, neighbours=2)

    # Fixture clusters are near-orthogonal, so cohesion within one is high.
    assert stats[0].cohesion is not None
    assert stats[0].cohesion > 0.9
    assert len(stats[0].exemplar_keys) == 2
    assert all(key in {m.key for m in corpus.clusters()[0]}
               for key in stats[0].exemplar_keys)

    # Every cluster gets ranked kin, and similarity is descending.
    kin = stats[0].neighbour_clusters
    assert len(kin) == 2
    assert [c for c, _ in kin] == sorted({1, 2})
    assert kin[0][1] >= kin[1][1]


def test_singleton_cluster_has_no_cohesion(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)
    stats = compute_cluster_stats(corpus)
    assert stats[2].size == 1
    assert stats[2].cohesion is None                     # undefined for one member
    assert stats[2].exemplar_keys == ["panel_a/0"]


def test_cluster_stats_serialise(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    stats = compute_cluster_stats(load_corpus(analysis_dir, npy, txt))
    assert json.loads(json.dumps(stats[0].as_dict()))["cluster_id"] == 0


# ── Prompt assembly ──────────────────────────────────────────────────────────

def test_cluster_prompt_carries_the_evidence(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)
    stats = compute_cluster_stats(corpus)
    prompt = build_cluster_prompt(stats[0], corpus.clusters()[0], exemplar_count=2)

    assert "cluster 0" in prompt
    assert "4 motifs across 2 panel(s)" in prompt
    assert "Embedding cohesion" in prompt
    assert "Nearest other families" in prompt
    assert "standing_figure" in prompt
    assert "closest to the cluster centroid" in prompt


def test_cluster_prompt_without_notes_says_so(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    for motif in corpus.motifs:
        motif.label = motif.description = motif.iconography = None
    stats = compute_cluster_stats(corpus)
    prompt = build_cluster_prompt(stats[0], corpus.clusters()[0], exemplar_count=1)
    assert "No per-motif notes exist" in prompt


def test_panel_prompt_includes_layout_motifs_and_cluster_context(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)
    motifs = corpus.motifs_for_panel("panel_a")
    briefs = {"0": {"name": "flanking_figure",
                    "visual_definition": "Upright frontal figure",
                    "iconographic_reading": "attendant"}}
    context = cluster_context_lines(motifs, briefs, stats)
    prompt = build_panel_prompt(corpus.layout_for("panel_a"), motifs, context)

    assert "SPATIAL STRUCTURE" in prompt
    assert "REGISTERS" in prompt
    assert "MOTIFS ON THIS PANEL" in prompt
    assert "MOTIF FAMILIES PRESENT" in prompt
    assert "flanking_figure" in prompt
    assert "outlined and numbered" in prompt


def test_cluster_context_reports_cross_panel_reach(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)

    shared = cluster_context_lines(corpus.motifs_for_panel("panel_a"), {}, stats)
    assert any("also appears on 1 other panel" in line for line in shared)
    assert any("unique to this panel" in line for line in shared)


def test_cluster_context_accepts_int_or_str_brief_keys(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)
    motifs = corpus.motifs_for_panel("panel_b")

    for briefs in ({"0": {"name": "int_or_str"}}, {0: {"name": "int_or_str"}}):
        lines = cluster_context_lines(motifs, briefs, stats)
        assert any("int_or_str" in line for line in lines)


def test_corpus_prompt_joins_both_earlier_passes():
    briefs = {"0": {"name": "interlace", "visual_definition": "woven bands",
                    "variation": "density varies", "distribution_note": "borders",
                    "iconographic_reading": "unclear", "confidence": "medium",
                    "relation_to_neighbours": "close to cluster 3"}}
    readings = {"panel_a": {"title": "Two attendants", "summary": "A two-register panel",
                            "composition": "symmetrical", "narrative": "an audience scene",
                            "cross_panel_links": ["shares the interlace border"],
                            "confidence": "low"}}
    prompt = build_corpus_prompt(briefs, readings, {"panels": 2, "motifs": 7,
                                                    "clusters": 3, "unclustered": 0})

    assert "MOTIF FAMILIES" in prompt and "PANEL READINGS" in prompt
    assert "interlace" in prompt and "Two attendants" in prompt
    assert "shares the interlace border" in prompt
    assert "Markdown essay" in prompt


# ── Persistence ──────────────────────────────────────────────────────────────

def test_store_round_trips_clusters_and_panels(tmp_path: Path, analysis_dir: Path):
    store = InterpretationStore(tmp_path / "interpretation")
    store.ensure_dirs()

    store.save_clusters({"1": {"name": "b"}, "0": {"name": "a"}})
    loaded = store.load_clusters()
    assert list(loaded) == ["0", "1"]                 # numerically ordered on disk

    reading = {"panel_stem": "panel_a", "title": "T", "summary": "S",
               "register_readings": [{"register": 0, "motifs": [1, 2], "reading": "R"}],
               "composition": "C", "narrative": "N", "cross_panel_links": ["L"],
               "uncertainties": ["U"], "confidence": "medium"}
    store.save_panel("panel_a", reading)
    assert store.load_panels()["panel_a"]["title"] == "T"
    assert (store.panels_dir / "panel_a.md").exists()

    layout = load_corpus(analysis_dir).layout_for("panel_a")
    path = store.save_layout("panel_a", layout)
    assert json.loads(path.read_text())["panel_stem"] == "panel_a"

    store.save_corpus("# Synthesis")
    assert store.corpus_path.read_text() == "# Synthesis"


def test_store_load_is_empty_before_anything_is_written(tmp_path: Path):
    store = InterpretationStore(tmp_path / "nothing-here")
    assert store.load_clusters() == {}
    assert store.load_panels() == {}


def test_render_panel_markdown_covers_all_sections():
    md = render_panel_markdown({
        "panel_stem": "panel_a", "title": "Two attendants", "summary": "S",
        "register_readings": [{"register": 0, "motifs": [1, 2], "reading": "top band"}],
        "composition": "symmetrical", "narrative": "an audience scene",
        "cross_panel_links": ["link"], "uncertainties": ["doubt"],
        "confidence": "medium", "generated_at": "2026-08-06", "model": "claude-opus-5",
    })
    assert md.startswith("# Two attendants")
    assert "### Register 0 (#1, #2)" in md
    assert "## Composition" in md and "## Reading" in md
    assert "- link" in md and "- doubt" in md
    assert "claude-opus-5" in md


def test_render_clusters_markdown():
    md = render_clusters_markdown({"0": {
        "name": "interlace", "visual_definition": "woven bands",
        "confidence": "high", "open_questions": ["is it one family?"],
        "stats": {"size": 4, "panel_spread": 2, "cohesion": 0.812},
    }})
    assert "## Cluster 0 — interlace" in md
    assert "cohesion 0.812" in md
    assert "- is it one family?" in md


# ── Interpreter, against a stub client ───────────────────────────────────────

class _StubStream:
    def __init__(self, message, events=()):
        self._message = message
        self._events = list(events)
        self.iterated = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        self.iterated = True
        return iter(self._events)

    def get_final_message(self):
        return self._message


class StubMessages:
    """Records requests and replays canned responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        payload = self.responses.pop(0)
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=payload)],
            stop_reason="end_turn",
            stop_details=None,
        )
        return _StubStream(message)


class StubClient:
    def __init__(self, responses):
        self.messages = StubMessages(responses)


def test_cluster_brief_sends_images_and_returns_parsed_json(analysis_dir: Path, embeddings):
    npy, txt = embeddings
    corpus = load_corpus(analysis_dir, embeddings_path=npy, paths_path=txt)
    stats = compute_cluster_stats(corpus, exemplars=2)

    client = StubClient([json.dumps({"name": "flanking_figure", "confidence": "medium"})])
    brief = Interpreter(client=client).cluster_brief(corpus, stats[0])

    assert brief["name"] == "flanking_figure"
    assert brief["cluster_id"] == 0
    assert brief["stats"]["size"] == 4
    assert brief["model"] == "claude-opus-5"

    request = client.messages.requests[0]
    content = request["messages"][0]["content"]
    assert sum(1 for block in content if block["type"] == "image") == 2
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["thinking"] == {"type": "adaptive"}
    # No sampling parameters — they are rejected on current models.
    assert "temperature" not in request and "top_p" not in request


def test_panel_reading_attaches_layout_and_annotated_image(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)

    client = StubClient([json.dumps({"title": "Two attendants", "confidence": "low"})])
    reading = Interpreter(client=client).panel_reading(
        corpus, "panel_a", {"0": {"name": "flanking_figure"}}, stats)

    assert reading["panel_stem"] == "panel_a"
    assert reading["title"] == "Two attendants"
    assert len(reading["layout"]["registers"]) == 2

    content = client.messages.requests[0]["messages"][0]["content"]
    assert sum(1 for block in content if block["type"] == "image") == 1
    assert "flanking_figure" in content[-1]["text"]


def test_corpus_synthesis_returns_markdown_text():
    client = StubClient(["# Synthesis\n\nThe collection shares…"])
    markdown = Interpreter(client=client).corpus_synthesis(
        {"0": {"name": "interlace"}}, {"panel_a": {"title": "T"}},
        {"panels": 1, "motifs": 4, "clusters": 1, "unclustered": 0})

    assert markdown.startswith("# Synthesis")
    # The essay pass is free-form: no schema is imposed.
    assert "format" not in client.messages.requests[0]["output_config"]


def test_effort_is_passed_through():
    client = StubClient(["ok"])
    Interpreter(client=client, effort="max").corpus_synthesis({}, {}, {})
    assert client.messages.requests[0]["output_config"]["effort"] == "max"


def test_heartbeat_reports_liveness_during_a_long_call():
    """A silent socket during adaptive thinking must still show progress."""
    events = [
        SimpleNamespace(type="content_block_start",
                        content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="content_block_delta"),
        SimpleNamespace(type="content_block_start",
                        content_block=SimpleNamespace(type="text")),
    ]

    class EventfulMessages(StubMessages):
        def stream(self, **kwargs):
            self.requests.append(kwargs)
            message = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=self.responses.pop(0))],
                stop_reason="end_turn", stop_details=None)
            self.last_stream = _StubStream(message, events)
            return self.last_stream

    client = StubClient([])
    client.messages = EventfulMessages(["# Synthesis"])

    ticks: list[str] = []
    # heartbeat=0 makes every event due, so the tick path runs deterministically.
    interpreter = Interpreter(client=client, heartbeat=0.0, on_progress=ticks.append)
    assert interpreter.corpus_synthesis({}, {}, {}) == "# Synthesis"

    assert client.messages.last_stream.iterated is True
    assert len(ticks) == len(events)
    assert "thinking…" in ticks[0]
    assert "writing text…" in ticks[-1]


def test_stream_is_not_iterated_without_a_progress_callback():
    client = StubClient([])

    class EventfulMessages(StubMessages):
        def stream(self, **kwargs):
            self.requests.append(kwargs)
            message = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=self.responses.pop(0))],
                stop_reason="end_turn", stop_details=None)
            self.last_stream = _StubStream(message, [SimpleNamespace(type="x")])
            return self.last_stream

    client.messages = EventfulMessages(["ok"])
    Interpreter(client=client).corpus_synthesis({}, {}, {})
    assert client.messages.last_stream.iterated is False


def test_unparseable_structured_response_raises(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)
    client = StubClient(["not json at all"])

    with pytest.raises(RuntimeError, match="could not parse"):
        Interpreter(client=client).cluster_brief(corpus, stats[0])


def test_refusal_is_surfaced_not_swallowed(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)

    class RefusingMessages(StubMessages):
        def stream(self, **kwargs):
            self.requests.append(kwargs)
            message = SimpleNamespace(
                content=[], stop_reason="refusal",
                stop_details=SimpleNamespace(category="cyber"),
            )
            return _StubStream(message)

    client = StubClient([])
    client.messages = RefusingMessages([])

    with pytest.raises(RuntimeError, match="declined"):
        Interpreter(client=client).cluster_brief(corpus, stats[0])
