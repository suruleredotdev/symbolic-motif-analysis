"""Tests for what the interpretation prompts ask for, and how their output lands.

The prompts are the project's method written down, so the parts that encode a
decision — the thesis they test, reading a motif as one element of a board,
putting the story before the pipeline's bookkeeping, protecting human labels —
are asserted here rather than left to drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from panel_art.interpret import (
    CLUSTER_SCHEMA,
    CORPUS_ESSAY_BRIEF,
    MOTIF_READING_SCHEMA,
    PANEL_SCHEMA,
    SYSTEM_PROMPT,
    InterpretationStore,
    MotifView,
    build_cluster_prompt,
    build_corpus_prompt,
    build_direct_prompt,
    build_panel_prompt,
    compute_cluster_stats,
    load_corpus,
    render_panel_markdown,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_label_cli():
    spec = importlib.util.spec_from_file_location(
        "label_motifs", REPO_ROOT / "scripts" / "label_motifs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["label_motifs"] = module
    spec.loader.exec_module(module)
    return module


label_cli = _load_label_cli()


# ── The shared system prompt ─────────────────────────────────────────────────

def test_system_prompt_states_role_thesis_and_method():
    for needle in (
        "anthropologist, art historian",          # who is reading
        "Rosetta Stone", "Dresden Codex", "àrokò",  # symbolic systems to reason from
        "Panels as chronicles", "Pattern as number",  # the two hypotheses
        "Kiriji", "Modákẹ́kẹ́",                       # historical frame for war / rule
        "ouroboros", "Horse", "Tortoise", "Monkey", "Crocodile",  # known symbols
        "far from watertight",                    # Aronson's caution
        "Never describe a single motif as a group",
    ):
        assert needle.replace("\n", " ") in SYSTEM_PROMPT.replace("\n", " "), needle


def test_system_prompt_asks_for_story_over_bookkeeping():
    text = SYSTEM_PROMPT.replace("\n", " ")
    assert "No pipeline statistics" in text
    assert "Do not retitle or dismiss the collection" in text
    assert "motif → panel → story" in text


# ── Schemas ──────────────────────────────────────────────────────────────────

def test_cluster_briefs_carry_a_single_motif_gloss():
    assert "motif_gloss" in CLUSTER_SCHEMA["required"]
    assert "members" in CLUSTER_SCHEMA["properties"]["motif_gloss"]["description"]


def test_panel_readings_classify_and_test_the_thesis():
    for key in ("object_type", "panel_type", "hypotheses"):
        assert key in PANEL_SCHEMA["required"]
    assert "chronicle_of_rulers" in PANEL_SCHEMA["properties"]["panel_type"]["enum"]


def test_motif_reading_places_the_motif_on_its_board():
    assert "role_on_board" in MOTIF_READING_SCHEMA["required"]
    assert "where it sits" in MOTIF_READING_SCHEMA["properties"]["description"]["description"]


# ── Pass prompts ─────────────────────────────────────────────────────────────

def test_cluster_prompt_asks_for_the_gloss(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    stats = compute_cluster_stats(corpus)
    prompt = build_cluster_prompt(stats[0], corpus.clusters()[0], exemplar_count=1)
    assert "motif_gloss" in prompt


def test_panel_prompt_treats_detections_as_parts_of_one_object(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    prompt = build_panel_prompt(corpus.layout_for("panel_a"),
                                corpus.motifs_for_panel("panel_a"), [])
    assert "parts of one object" in prompt
    assert "panel_a" in prompt
    assert "hypotheses" in prompt


def test_corpus_prompt_puts_the_story_first_and_the_counts_last():
    readings = {"panel_a": {"title": "A tray", "object_type": "opon Ifa",
                            "panel_type": "divination_instrument", "summary": "s",
                            "composition": "c", "narrative": "n",
                            "hypotheses": "weakens the chronicle reading"}}
    prompt = build_corpus_prompt({}, readings, {"panels": 1, "motifs": 3,
                                                "clusters": 0, "unclustered": 3})
    assert "CORPUS SCALE" not in prompt
    assert prompt.index("PANEL READINGS") < prompt.index("DATA NOTE")
    assert "opon Ifa / divination_instrument" in prompt
    assert "weakens the chronicle reading" in prompt
    assert CORPUS_ESSAY_BRIEF in prompt
    assert "The story in brief" in prompt


def test_direct_prompt_shares_the_essay_brief(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    prompt = build_direct_prompt(corpus, compute_cluster_stats(corpus))
    assert CORPUS_ESSAY_BRIEF in prompt
    assert prompt.index("── panel_a ──") < prompt.index("DATA NOTE")


# ── Rendering ────────────────────────────────────────────────────────────────

def test_panel_markdown_shows_type_and_thesis_check():
    md = render_panel_markdown({"title": "T", "panel_stem": "p", "confidence": "low",
                                "object_type": "door leaf",
                                "panel_type": "chronicle_of_rulers",
                                "summary": "s", "hypotheses": "supports it"})
    assert "door leaf · chronicle of rulers" in md
    assert "## Against the thesis" in md and "supports it" in md


# ── Label provenance and the per-motif pass ──────────────────────────────────

def test_llm_edited_labels_count_as_human():
    edited = MotifView(key="p/0", panel_stem="p", index=0, bbox={},
                       label="x", label_source="llm-edited")
    assert edited.is_human_labelled is True


def test_refresh_generated_leaves_llm_edited_labels_alone():
    edited = MotifView(key="p/0", panel_stem="p", index=0, bbox={},
                       label="x", label_source="llm-edited")
    drafted = MotifView(key="p/1", panel_stem="p", index=1, bbox={},
                        label="y", label_source="cluster-brief")
    assert label_cli.should_write(edited, overwrite=False, refresh_generated=True) is False
    assert label_cli.should_write(drafted, overwrite=False, refresh_generated=True) is True


def test_from_briefs_writes_the_single_motif_gloss(analysis_dir: Path, tmp_path: Path):
    corpus = load_corpus(analysis_dir)
    store = InterpretationStore(tmp_path / "interpretation")
    store.save_clusters({"0": {"name": "standing_figure",
                               "visual_definition": "Each member is an upright figure.",
                               "motif_gloss": "An upright frontal figure.",
                               "iconographic_reading": "attendant"}})
    labels: dict = {}
    written = label_cli.run_from_briefs(corpus, store, labels, corpus.motifs,
                                        overwrite=False, dry_run=False,
                                        refresh_generated=True)
    assert written > 0
    descriptions = {v["description"] for v in labels.values()}
    assert descriptions == {"An upright frontal figure."}


def test_motif_prompt_reads_one_element_in_context(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    motif = corpus.motifs_for_panel("panel_a")[1]
    prompt = label_cli.build_motif_prompt(corpus, motif, {
        "name": "standing_figure", "motif_gloss": "An upright figure.",
        "visual_definition": "Each member is upright."})
    assert "ONE element" in prompt
    assert "OTHER MOTIFS ON THE SAME OBJECT" in prompt
    assert "An upright figure." in prompt and "Each member" not in prompt
