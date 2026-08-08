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
                              crops_per_family=3, verbose=False)


def test_payload_carries_geometry_labels_and_reading(interpreted: Path):
    payload = _payload(interpreted)

    panel = next(p for p in payload["panels"] if p["stem"] == "panel_a")
    assert panel["width"] == 400 and panel["height"] == 900
    assert panel["image"].startswith("data:image/jpeg;base64,")
    assert len(panel["registers"]) == 2
    assert panel["reading"]["title"] == "Two attendants"

    motif = next(m for m in panel["motifs"] if m["index"] == 1)
    assert motif["label"] == "standing_figure"
    assert motif["cluster"] == 0
    assert motif["zone"] == "upper left"
    assert motif["register"] == 0
    assert motif["is_field"] is False
    assert (motif["x"], motif["y"], motif["w"], motif["h"]) == (40, 80, 90, 120)

    field = next(m for m in panel["motifs"] if m["index"] == 0)
    assert field["is_field"] is True


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
    assert briefed["crops"] and briefed["crops"][0]["src"].startswith("data:image/jpeg")

    unbriefed = payload["families"]["1"]               # no brief written for it
    assert unbriefed["name"] is None
    assert unbriefed["size"] == 2                      # statistics still present


def test_scale_reports_how_much_has_been_read(interpreted: Path):
    scale = _payload(interpreted)["scale"]
    assert scale["panels"] == 2 and scale["motifs"] == 7
    assert scale["readings"] == 1                      # only panel_a was read


def test_synthesis_is_rendered_into_the_payload(interpreted: Path):
    assert "<strong>one</strong>" in _payload(interpreted)["synthesis"]


# ── Page ─────────────────────────────────────────────────────────────────────

def test_rendered_page_is_self_contained(interpreted: Path):
    from panel_art.site_template import render
    html = render(_payload(interpreted))

    assert "http://" not in html and "https://" not in html   # no external assets
    assert "<script>" in html and "const DATA = {" in html
    assert html.count("</script>") == 1                        # payload didn't break out


def test_page_defines_colours_for_both_themes(interpreted: Path):
    from panel_art.site_template import render
    css = render(_payload(interpreted)).split("</style>")[0]

    # Every token redefined for dark must first exist on bare :root, or the
    # un-stamped "system" state renders one theme's text on the other's ground.
    root_block = css.split(":root {", 1)[1].split("}", 1)[0]
    light_tokens = set(re.findall(r"(--[\w-]+):", root_block))
    dark_block = css.split(':root[data-theme="dark"] {', 1)[1].split("}", 1)[0]
    dark_tokens = set(re.findall(r"(--[\w-]+):", dark_block))

    assert dark_tokens, "no dark theme tokens found"
    assert dark_tokens <= light_tokens, dark_tokens - light_tokens
    assert "background: var(--board)" in css      # body paints its own ground


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


def test_cli_rejects_a_missing_analysis_dir(tmp_path: Path, capsys):
    assert site.main(["--analysis-dir", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().out


def test_embedded_images_are_real_jpegs(interpreted: Path):
    panel = _payload(interpreted)["panels"][0]
    raw = base64.b64decode(panel["image"].split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"                  # JPEG SOI marker
