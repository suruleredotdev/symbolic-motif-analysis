"""Tests for the interpretation site exporter — no API key, no browser."""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_exporter():
    spec = importlib.util.spec_from_file_location(
        "export_interpretation_site", REPO_ROOT / "scripts" / "export_interpretation_site.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_interpretation_site"] = module
    spec.loader.exec_module(module)
    return module


site = _load_exporter()


@pytest.fixture
def interpreted(analysis_dir: Path) -> Path:
    """The fixture analysis dir, with a partial interpretation written into it."""
    out = analysis_dir / "interpretation"
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "clusters.json").write_text(json.dumps({
        "0": {"name": "standing_attendant", "visual_definition": "Upright frontal figure",
              "iconographic_reading": "attendant", "confidence": "medium"},
    }))
    # Deliberately only one of the two panels — the page must tolerate partial work.
    (out / "panels" / "panel_a.json").write_text(json.dumps({
        "panel_stem": "panel_a", "title": "Two attendants", "summary": "A two-register panel",
        "composition": "symmetrical", "narrative": "an audience scene", "confidence": "low",
        "register_readings": [{"register": 0, "motifs": [1, 2], "reading": "the upper band"}],
        "cross_panel_links": ["shares the interlace"], "uncertainties": ["#0 may be the frame"],
    }))
    (out / "corpus.md").write_text("# Synthesis\n\nThe collection shares **one** vocabulary.\n")
    return analysis_dir


# ── Markdown ─────────────────────────────────────────────────────────────────

def test_markdown_renders_the_shapes_the_essay_uses():
    html = site.markdown_to_html(
        "# Title\n\nA paragraph with **bold**, *italic* and `code`.\n\n"
        "## Section\n\n- one\n- two\n\n1. first\n2. second\n\n---\n\nTail.\n")
    for fragment in ("<h1>Title</h1>", "<strong>bold</strong>", "<em>italic</em>",
                     "<code>code</code>", "<h2>Section</h2>", "<ul>", "<li>one</li>",
                     "<ol>", "<li>first</li>", "<hr>", "<p>Tail.</p>"):
        assert fragment in html, fragment


def test_markdown_escapes_html_before_adding_tags():
    html = site.markdown_to_html("A <script>alert(1)</script> & an ampersand.")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "&amp;" in html


def test_markdown_joins_wrapped_lines_into_one_paragraph():
    html = site.markdown_to_html("one line\ncontinued here\n\nsecond para")
    assert "<p>one line continued here</p>" in html
    assert html.count("<p>") == 2


def test_empty_markdown_is_empty():
    assert site.markdown_to_html("   \n  ") == ""


# ── Payload ──────────────────────────────────────────────────────────────────

def _payload(analysis_dir: Path, **kwargs):
    from panel_art.interpret import InterpretationStore, load_corpus
    corpus = load_corpus(analysis_dir, **{k: v for k, v in kwargs.items()
                                          if k in {"embeddings_path", "paths_path"}})
    store = InterpretationStore(analysis_dir / "interpretation")
    return site.build_payload(corpus, store, corpus.panel_stems(),
                              max_dim=200, crop_dim=48, quality=60,
                              crops_per_family=3,
                              motif_crops=kwargs.get("motif_crops", True),
                              verbose=False)


def test_payload_carries_geometry_labels_and_reading(interpreted: Path):
    payload = _payload(interpreted)

    panel = next(p for p in payload["panels"] if p["stem"] == "panel_a")
    assert panel["width"] == 400 and panel["height"] == 900
    assert panel["image"].startswith("data:image/jpeg;base64,")
    assert len(panel["registers"]) == 2
    assert panel["reading"]["title"] == "Two attendants"

    motif = next(m for m in panel["motifs"] if m["index"] == 1)
    assert motif["key"] == "panel_a/1"
    assert motif["panel_stem"] == "panel_a"
    assert motif["label"] == "standing_figure"
    assert motif["cluster"] == 0
    assert motif["zone"] == "upper left"
    assert motif["register"] == 0
    assert motif["is_field"] is False
    assert (motif["x"], motif["y"], motif["w"], motif["h"]) == (40, 80, 90, 120)

    field = next(m for m in panel["motifs"] if m["index"] == 0)
    assert field["is_field"] is True


def test_every_motif_gets_one_shared_crop(interpreted: Path):
    payload = _payload(interpreted)
    keys = {m["key"] for p in payload["panels"] for m in p["motifs"]}

    assert set(payload["crops"]) == keys
    assert all(src.startswith("data:image/jpeg;base64,") for src in payload["crops"].values())


def test_no_motif_crops_keeps_only_the_family_exemplars(interpreted: Path):
    payload = _payload(interpreted, motif_crops=False)
    exemplars = {k for f in payload["families"].values() for k in f["exemplars"]}

    assert exemplars                                   # families still have faces
    assert set(payload["crops"]) == exemplars          # and nothing else was embedded


def test_reading_order_is_motif_indices_not_keys(interpreted: Path):
    panel = next(p for p in _payload(interpreted)["panels"] if p["stem"] == "panel_a")
    assert panel["reading_order"] == [1, 2, 3, 0]      # registers, then the field
    assert all(isinstance(i, int) for i in panel["reading_order"])


def test_panels_without_a_reading_still_render(interpreted: Path):
    panel = next(p for p in _payload(interpreted)["panels"] if p["stem"] == "panel_b")
    assert panel["reading"] is None
    assert panel["motifs"] and panel["registers"]     # geometry survives regardless
    assert panel["title"] == "panel_b"


def test_families_merge_briefs_with_statistics(interpreted: Path, embeddings):
    npy, txt = embeddings
    payload = _payload(interpreted, embeddings_path=npy, paths_path=txt)

    briefed = payload["families"]["0"]
    assert briefed["name"] == "standing_attendant"
    assert briefed["size"] == 4 and briefed["panel_spread"] == 2
    assert briefed["cohesion"] > 0.9
    # Exemplars point into the shared crop index rather than carrying their own
    # copy of the bytes.
    assert briefed["exemplars"]
    assert all(key in payload["crops"] for key in briefed["exemplars"])

    unbriefed = payload["families"]["1"]               # no brief written for it
    assert unbriefed["name"] is None
    assert unbriefed["size"] == 2                      # statistics still present


def test_scale_reports_how_much_has_been_read(interpreted: Path):
    scale = _payload(interpreted)["scale"]
    assert scale["panels"] == 2 and scale["motifs"] == 7
    assert scale["readings"] == 1                      # only panel_a was read


def test_synthesis_is_rendered_into_the_payload(interpreted: Path):
    payload = _payload(interpreted)
    assert "<strong>one</strong>" in payload["synthesis"]
    # The sidebar's corpus card opens with the essay's first paragraph.
    assert payload["synthesis_lead"].startswith("<p>")
    assert payload["synthesis_lead"].endswith("</p>")
    assert "<h1>" not in payload["synthesis_lead"]


# ── Cross-references ─────────────────────────────────────────────────────────

def test_aliases_cover_the_spellings_a_reading_actually_uses():
    refs = site.reference_aliases([
        "EBA-Div_00302_Ife_q166566_i1_panel_00",
        "EBA-Div_00302_Ife_q166566_i1_panel_01",
        "FoA_04-5580_Benin_panel_00",
    ])
    # Full stem → exactly its own plate.
    assert refs["EBA-Div_00302_Ife_q166566_i1_panel_00"] == \
        ["EBA-Div_00302_Ife_q166566_i1_panel_00"]
    # Object id → every plate cut from it.
    assert refs["EBA-Div_00302_Ife_q166566_i1"] == [
        "EBA-Div_00302_Ife_q166566_i1_panel_00",
        "EBA-Div_00302_Ife_q166566_i1_panel_01",
    ]
    # The bare catalogue number, which is how the prose usually names it.
    assert refs["EBA-Div_00302"] == [
        "EBA-Div_00302_Ife_q166566_i1_panel_00",
        "EBA-Div_00302_Ife_q166566_i1_panel_01",
    ]
    assert refs["FoA_04-5580"] == ["FoA_04-5580_Benin_panel_00"]


def test_aliases_refuse_tokens_that_would_linkify_prose():
    refs = site.reference_aliases(["figure_panel_00", "of_1"])
    assert "figure" not in refs           # no digit
    assert "of_1" not in refs             # too short to be an identifier


def test_object_id_drops_only_the_panel_suffix():
    assert site.object_id("EBA-Div_00302_Ife_panel_07") == "EBA-Div_00302_Ife"
    assert site.object_id("no_suffix_here") == "no_suffix_here"


def test_payload_refs_and_objects_reach_the_page(interpreted: Path):
    payload = _payload(interpreted)
    assert {p["object"] for p in payload["panels"]} == {"panel_a", "panel_b"}
    assert payload["refs"] == {}          # fixture stems carry no catalogue digits


# ── Page ─────────────────────────────────────────────────────────────────────

def test_rendered_page_loads_no_external_assets(interpreted: Path):
    from panel_art.site_template import render
    html = render(_payload(interpreted))

    # The artifact CSP blocks every external host, so nothing may be *fetched*.
    assert not re.search(r'src="(?!data:)https?://', html)
    assert not re.search(r"<link[^>]+href=\"https?://", html)
    assert not re.search(r"url\(\s*[\"']?https?://", html)
    assert "@import" not in html
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    # Navigation is another matter: the masthead mark links home, per the
    # surulere.dev design system.
    assert 'href="https://surulere.dev"' in html

    assert "<script>" in html and "const DATA = {" in html
    assert html.count("</script>") == 1        # payload didn't break out


def test_page_defines_colours_for_both_themes(interpreted: Path):
    from panel_art.site_template import render
    # The layout's own stylesheet, not the document shell's ground-paint block.
    css = render(_payload(interpreted)).split('<div id="layout">', 1)[1].split("</style>")[0]

    # Every token redefined for dark must first exist on bare :root, or the
    # un-stamped "system" state renders one theme's text on the other's ground.
    root_block = css.split(":root {", 1)[1].split("}", 1)[0]
    light_tokens = set(re.findall(r"(--[\w-]+):", root_block))
    for selector in (':root[data-theme="dark"] {', "#layout.DARK {", "#layout.LIGHT {"):
        block = css.split(selector, 1)[1].split("}", 1)[0]
        tokens = set(re.findall(r"(--[\w-]+):", block))
        assert tokens, f"no theme tokens under {selector}"
        assert tokens <= light_tokens, (selector, tokens - light_tokens)

    # The toggle's two states must cover the same tokens, or cycling the theme
    # would leave one of them stuck on the other's value.
    dark = set(re.findall(r"(--[\w-]+):", css.split("#layout.DARK {", 1)[1].split("}", 1)[0]))
    light = set(re.findall(r"(--[\w-]+):", css.split("#layout.LIGHT {", 1)[1].split("}", 1)[0]))
    assert dark == light

    assert "background-color: var(--bg-color)" in css   # the page paints its own ground


def test_page_carries_the_surulere_chrome(interpreted: Path):
    from panel_art.site_template import render
    html = render(_payload(interpreted))

    assert "Symbolic Motif Analysis" in html
    assert 'class="brand-mark"' in html                 # the mark, inlined
    assert 'id="menu-bar"' in html and 'id="sidebar"' in html
    assert 'id="statusbar"' in html and "SURULERE.DEV" in html
    assert 'id="crumbs"' in html
    assert "data:image/svg+xml," in html                # background tiles, inlined


def test_cli_writes_a_page(interpreted: Path, tmp_path: Path, capsys):
    out = tmp_path / "site.html"
    code = site.main(["--analysis-dir", str(interpreted), "--out", str(out),
                      "--max-dim", "200", "--crop-dim", "48"])
    assert code == 0
    assert out.exists() and out.stat().st_size > 5_000
    assert "2 plates, 1 with readings" in capsys.readouterr().out


def _payload_from_page(html: str) -> dict:
    """Pull the embedded data island back out of a rendered page."""
    match = re.search(r"const DATA = (\{.*\});", html)
    assert match, "no data island found in the page"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_cli_can_limit_to_one_plate(interpreted: Path, tmp_path: Path):
    out = tmp_path / "one.html"
    site.main(["--analysis-dir", str(interpreted), "--out", str(out),
               "--panels", "panel_b", "--max-dim", "200"])

    payload = _payload_from_page(out.read_text())
    assert [p["stem"] for p in payload["panels"]] == ["panel_b"]
    # Families still span the whole corpus — a family is not panel-scoped, so
    # its exemplar crops may well come from a plate that was not exported.
    assert payload["families"]["0"]["panel_spread"] == 2
    # …and every exemplar the page will try to draw is in the crop index,
    # including those from the plate that was left out.
    for family in payload["families"].values():
        assert all(key in payload["crops"] for key in family["exemplars"])


def test_cli_rejects_a_missing_analysis_dir(tmp_path: Path, capsys):
    assert site.main(["--analysis-dir", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().out


def test_embedded_images_are_real_jpegs(interpreted: Path):
    panel = _payload(interpreted)["panels"][0]
    raw = base64.b64decode(panel["image"].split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"                  # JPEG SOI marker


def test_standalone_page_is_a_whole_document(interpreted: Path):
    from panel_art.site_template import render
    html = render(_payload(interpreted))

    # A file opened off disk gets no <head> from anyone, and without the
    # viewport meta every mobile breakpoint in the stylesheet is inert.
    assert html.startswith("<!doctype html>")
    assert 'name="viewport"' in html and 'charset="utf-8"' in html
    assert "<title>Symbolic Motif Analysis</title>" in html
    assert 'rel="icon" href="data:image/svg+xml,' in html


def test_fragment_mode_brings_no_head_of_its_own(interpreted: Path):
    from panel_art.site_template import render
    html = render(_payload(interpreted), standalone=False)

    assert "<!doctype" not in html.lower() and "<head>" not in html
    assert html.lstrip().startswith('<div id="layout">')


def test_cli_can_emit_a_fragment(interpreted: Path, tmp_path: Path):
    out = tmp_path / "frag.html"
    assert site.main(["--analysis-dir", str(interpreted), "--out", str(out),
                      "--fragment", "--max-dim", "200"]) == 0
    assert "<!doctype" not in out.read_text().lower()


def test_register_members_are_keys_the_page_can_resolve(interpreted: Path):
    """`Register.members` holds placement keys, not indices.

    The page turns each one into a chip labelled with the motif's number, so a
    member that is not in the motif index would render an unclickable chip.
    """
    payload = _payload(interpreted)
    keys = {m["key"] for p in payload["panels"] for m in p["motifs"]}

    members = [key for p in payload["panels"]
               for reg in p["registers"] for key in reg["members"]]
    assert members
    assert all("/" in key for key in members)          # keys, not bare indices
    assert set(members) <= keys
