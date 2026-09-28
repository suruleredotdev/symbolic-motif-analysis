"""Cross-panel comparison: pair selection, prompt, storage and corpus feed."""

from __future__ import annotations

import json
from pathlib import Path

from panel_art.compare import (
    COMPARISON_SCHEMA,
    RELATIONS,
    _tokens,
    build_comparison_prompt,
    catalogue_affinity,
    catalogue_number,
    comparison_lines,
    pair_key,
    render_comparisons_markdown,
    score_pairs,
    select_pairs,
)
from panel_art.interpret import (
    InterpretationStore,
    Interpreter,
    build_corpus_prompt,
    load_corpus,
)

SAME_WORKSHOP = {"a": "panel_a", "b": "panel_b", "relation": "same_workshop",
                 "evidence": "Same fowl and rifleman.", "confidence": "medium",
                 "motif_matches": [{"a": [1], "b": [0], "note": "rifleman"}],
                 "differences": "Rim cropped.", "significance": "One workshop.", "score": 0.9}
UNRELATED = {"a": "panel_a", "b": "panel_c", "relation": "unrelated",
             "evidence": "Nothing shared.", "confidence": "high", "motif_matches": [],
             "differences": "", "significance": ""}


def test_catalogue_numbers_and_affinity():
    assert catalogue_number("KBA_19153_Yoruba_q221664_i1_panel_00") == ("KBA", 19153)
    assert catalogue_number("FoA_04-5580_Modakeke_(Ife)_q48630_i3_panel_00") == ("FoA_04", 5580)
    assert catalogue_number("EBA-Div_00302_q166558_i1_panel_00") == ("EBA-Div", 302)
    # Crops of one photograph; neighbouring numbers; different series.
    assert catalogue_affinity("EBA-Div_00311_Ife_q1_i1_panel_00",
                              "EBA-Div_00311_Ife_q1_i1_panel_02") == 1.0
    assert catalogue_affinity("KBA_17553_q1_i1_panel_00", "KBA_17556_q2_i1_panel_00") == 0.6
    assert catalogue_affinity("KBA_10299_q1_i1_panel_00", "FoA_04-5588_q2_i1_panel_00") == 0.0
    assert catalogue_affinity("KBA_10274_q1_i1_panel_00", "KBA_19157_q2_i1_panel_00") == 0.0


def test_tokens_drop_archive_identifiers():
    # A reading that cites another panel must not match it on the citation.
    toks = _tokens("Head and foot of the tray KBA_17556 (#3), a pillow-shaped tray")
    assert "kba" not in toks and "pillow" in toks and "tray" in toks


def test_pairs_are_scored_and_the_shared_family_pair_is_selected(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    scored = score_pairs(corpus)
    assert [pair_key(p.a, p.b) for p in scored] == ["panel_a|panel_b"]
    assert "shared families 0, 1" in scored[0].reasons

    chosen = select_pairs(corpus, min_score=0.0)
    assert [p.key for p in chosen] == ["panel_a|panel_b"]
    assert select_pairs(corpus, min_score=0.0, only=["panel_x"]) == []


def test_drawing_and_photo_pair_is_found_on_rare_words_alone(analysis_dir: Path, tmp_path: Path):
    # Three panels, no shared families: the pair that shares a rare word
    # ("pillow") in its readings must be chosen over the one sharing "tray".
    corpus = load_corpus(analysis_dir)
    for m in corpus.motifs:
        m.cluster = -1
    corpus.panels["panel_c"] = corpus.panels["panel_b"]
    from dataclasses import replace
    extra = [replace(m, panel_stem="panel_c", key=f"panel_c/{m.index}")
             for m in corpus.motifs_for_panel("panel_b")]
    corpus.motifs.extend(extra)
    readings = {
        "panel_a": {"title": "Photograph of a pillow-shaped tray with drummers", "panel_type": "x"},
        "panel_b": {"title": "Survey drawing of a pillow-shaped tray, drummers at the rim",
                    "panel_type": "x"},
        "panel_c": {"title": "Round tray with serpents", "panel_type": "x"},
    }
    chosen = select_pairs(corpus, readings, per_panel=1, min_score=0.0)
    assert chosen[0].key == "panel_a|panel_b"


def test_comparison_prompt_names_both_objects_and_allows_unrelated(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    prompt = build_comparison_prompt(
        "panel_a", "panel_b", corpus.motifs_for_panel("panel_a"),
        corpus.motifs_for_panel("panel_b"), {"title": "A door", "panel_type": "war_and_conflict"},
        None, ["shared families 0"])
    assert "OBJECT A: panel_a" in prompt and "OBJECT B: panel_b" in prompt
    assert "Reading so far: A door" in prompt
    assert "'unrelated'" in prompt and "same object recorded twice" in prompt
    assert COMPARISON_SCHEMA["properties"]["relation"]["enum"] == RELATIONS


def test_compare_panels_sends_both_images_and_records_the_pair(analysis_dir: Path):
    corpus = load_corpus(analysis_dir)
    sent = {}

    class Stub(Interpreter):
        def _json_call(self, content, max_tokens, schema):
            sent["content"], sent["schema"] = content, schema
            return {k: v for k, v in SAME_WORKSHOP.items() if k not in ("a", "b", "score")}

    interp = Stub(client=object(), model="test-model")
    result = interp.compare_panels(corpus, "panel_a", "panel_b", reasons=["why"])
    images = [c for c in sent["content"] if c["type"] == "image"]
    assert len(images) == 2
    assert sent["schema"] is COMPARISON_SCHEMA
    assert (result["a"], result["b"], result["model"], result["reasons"]) == \
        ("panel_a", "panel_b", "test-model", ["why"])


def test_store_round_trip_writes_markdown(tmp_path: Path):
    store = InterpretationStore(tmp_path / "interpretation")
    store.save_comparisons({"panel_a|panel_b": SAME_WORKSHOP})
    assert store.load_comparisons()["panel_a|panel_b"]["relation"] == "same_workshop"
    md = (tmp_path / "interpretation" / "comparisons.md").read_text()
    assert "## panel_a ↔ panel_b" in md and "A #1 ↔ B #0: rifleman" in md


def test_corpus_prompt_carries_comparisons_but_not_unrelated_ones():
    comps = {"panel_a|panel_b": SAME_WORKSHOP, "panel_a|panel_c": UNRELATED}
    prompt = build_corpus_prompt({}, {"panel_a": {"title": "A"}}, {}, comps)
    assert "CROSS-PANEL COMPARISONS" in prompt
    assert "panel_a ↔ panel_b: same workshop" in prompt
    assert "panel_c" not in prompt
    # Comparisons sit with the evidence, before the data note and the brief.
    assert prompt.index("CROSS-PANEL") < prompt.index("DATA NOTE")
    assert comparison_lines({"x": UNRELATED}) == []
    assert "CROSS-PANEL" not in build_corpus_prompt({}, {}, {}, None)


def test_rendered_markdown_orders_closest_relations_first():
    same = {**SAME_WORKSHOP, "a": "p1", "b": "p2", "relation": "same_object"}
    md = render_comparisons_markdown({"x": SAME_WORKSHOP, "y": same})
    assert md.index("p1 ↔ p2") < md.index("panel_a ↔ panel_b")
