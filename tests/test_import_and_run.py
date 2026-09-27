"""import_readings.py and run_interpretation.py — no API key needed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


importer = _load("import_readings")
runner = _load("run_interpretation")

PANEL = {
    "title": "Test panel",
    "object_type": "door leaf",
    "panel_type": "chronicle_of_rulers",
    "summary": "Two figures over one.",
    "register_readings": [{"register": 0, "motifs": [1, 2], "reading": "A pair."}],
    "composition": "Mirrored.",
    "narrative": "A pair of attendants over a seated figure.",
    "hypotheses": "Neutral.",
    "cross_panel_links": [],
    "uncertainties": [],
    "confidence": "medium",
}


def _reading(label, bbox=None, **extra):
    r = {"label": label, "description": f"{label} seen", "iconography": "meaning",
         "confidence": "low", **extra}
    if bbox:
        r["bbox"] = dict(zip("xywh", bbox))
    return r


def _write(dir_: Path, stem: str, data: dict) -> None:
    dir_.mkdir(exist_ok=True)
    (dir_ / f"{stem}.json").write_text(json.dumps(data))


def _labels(analysis_dir: Path) -> dict:
    labels = json.loads((analysis_dir / "motif_labels.json").read_text())
    return {k.split("motifs_norm/")[1].rsplit("_motif", 1)[0]: v for k, v in labels.items()}


def test_import_writes_labels_panels_and_corpus(analysis_dir: Path, tmp_path: Path):
    readings = tmp_path / "readings"
    _write(readings, "panel_a", {"motifs": {"2": _reading("attendant", role_on_board="right of pair")},
                                 "panel": PANEL})
    essay = tmp_path / "corpus.md"
    essay.write_text("# The story\n")

    assert importer.main(["--analysis-dir", str(analysis_dir), "--readings", str(readings),
                          "--corpus", str(essay), "--model", "test"]) == 0

    label = _labels(analysis_dir)["panel_a/002"]
    assert (label["label"], label["source"], label["notes"]) == ("attendant", "llm", "confidence: low")
    assert label["description"] == "attendant seen right of pair"
    saved = json.loads((analysis_dir / "interpretation/panels/panel_a.json").read_text())
    assert saved["model"] == "test" and saved["layout"]["panel_stem"] == "panel_a"
    assert (analysis_dir / "interpretation/panels/panel_a.md").exists()
    assert (analysis_dir / "interpretation/corpus.md").read_text() == "# The story\n"


def test_import_never_touches_a_human_label(analysis_dir: Path, tmp_path: Path):
    readings = tmp_path / "readings"
    _write(readings, "panel_a", {"motifs": {"1": _reading("replacement")}})
    before = _labels(analysis_dir)["panel_a/001"]

    importer.main(["--analysis-dir", str(analysis_dir), "--readings", str(readings)])
    assert _labels(analysis_dir)["panel_a/001"] == before

    importer.main(["--analysis-dir", str(analysis_dir), "--readings", str(readings), "--overwrite"])
    assert _labels(analysis_dir)["panel_a/001"]["label"] == "replacement"


def test_import_follows_a_box_that_was_reindexed(analysis_dir: Path, tmp_path: Path):
    # The reading says #0, but its box is panel_b's #2 — the boxes were edited
    # after the reading was made. It must land on #2, and a box that no longer
    # exists must be reported rather than written anywhere.
    readings = tmp_path / "readings"
    _write(readings, "panel_b", {"motifs": {
        "0": _reading("moved", bbox=(140, 600, 120, 150)),
        "1": _reading("gone", bbox=(5, 5, 10, 10)),
    }})
    labels = {}
    from panel_art.interpret import load_corpus
    stats = importer.import_readings(load_corpus(analysis_dir), json.loads(json.dumps(
        {"panel_b": json.loads((readings / "panel_b.json").read_text())})), labels,
        None, "test", overwrite=True)
    written = {k.split("motifs_norm/")[1]: v["label"] for k, v in labels.items()}
    assert written == {"panel_b/002_motif.png": "moved"}
    assert stats["unmatched"] == 1


def test_import_keeps_duplicate_boxes_on_their_own_index(analysis_dir: Path):
    from panel_art.interpret import load_corpus
    corpus = load_corpus(analysis_dir)
    motifs = corpus.motifs_for_panel("panel_a")
    motifs[2].bbox = dict(motifs[1].bbox)             # two identical boxes
    for key in ("1", "2"):
        match = importer.match_motif(motifs, key, _reading("x", bbox=[motifs[1].bbox[k] for k in "xywh"]))
        assert match.index == int(key)


def test_import_rejects_a_panel_reading_with_a_bad_type(analysis_dir: Path, tmp_path: Path):
    readings = tmp_path / "readings"
    _write(readings, "panel_a", {"panel": {**PANEL, "panel_type": "novel"}})
    importer.main(["--analysis-dir", str(analysis_dir), "--readings", str(readings)])
    assert not (analysis_dir / "interpretation/panels/panel_a.json").exists()


def test_runner_orders_steps_so_labels_precede_panels(tmp_path: Path):
    args = runner.build_parser().parse_args(["--analysis-dir", str(tmp_path)])
    steps = runner.plan(args)
    assert [s for s, *_ in steps] == ["clusters", "labels", "panels", "compare", "corpus", "site"]
    labels_argv = dict((s, argv) for s, _, argv in steps)["labels"]
    assert "--per-motif" in labels_argv and "--refresh-generated" in labels_argv
    assert "--overwrite" not in labels_argv


def test_runner_dry_run_skips_the_site_and_passes_the_flag(tmp_path: Path):
    args = runner.build_parser().parse_args(
        ["--analysis-dir", str(tmp_path), "--dry-run", "--panels", "p1"])
    steps = runner.plan(args)
    assert "site" not in [s for s, *_ in steps]
    assert all("--dry-run" in argv for _, _, argv in steps)
    assert "p1" in dict((s, a) for s, _, a in steps)["panels"]


def test_runner_can_rebuild_just_the_site(analysis_dir: Path, tmp_path: Path):
    out = analysis_dir / "interpretation" / "site.html"
    assert runner.main(["--analysis-dir", str(analysis_dir), "--only", "site"]) == 0
    assert out.exists()
