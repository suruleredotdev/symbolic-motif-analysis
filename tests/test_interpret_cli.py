"""End-to-end tests for scripts/interpret_motifs.py with the API stubbed out."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_cli():
    """Import the CLI by path — `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "interpret_motifs", REPO_ROOT / "scripts" / "interpret_motifs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["interpret_motifs"] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


class FakeInterpreter:
    """Stands in for the real Interpreter; counts calls so resume can be checked."""

    calls: dict[str, int] = {}

    def __init__(self, *args, **kwargs):
        FakeInterpreter.calls = {"cluster": 0, "panel": 0, "corpus": 0}

    def cluster_brief(self, corpus, stats, **kwargs):
        FakeInterpreter.calls["cluster"] += 1
        return {"name": f"family_{stats.cluster_id}", "confidence": "medium",
                "cluster_id": stats.cluster_id, "stats": stats.as_dict()}

    def panel_reading(self, corpus, stem, briefs=None, cluster_stats=None, **kwargs):
        FakeInterpreter.calls["panel"] += 1
        return {"panel_stem": stem, "title": f"Reading of {stem}",
                "summary": "s", "register_readings": [], "composition": "c",
                "narrative": "n", "cross_panel_links": [], "uncertainties": [],
                "confidence": "low", "layout": corpus.layout_for(stem).as_dict()}

    def corpus_synthesis(self, briefs, readings, scale, **kwargs):
        FakeInterpreter.calls["corpus"] += 1
        return f"# Synthesis\n\n{len(briefs)} families, {len(readings)} panels."


@pytest.fixture
def stub_interpreter(monkeypatch):
    monkeypatch.setattr(cli, "Interpreter", FakeInterpreter)
    return FakeInterpreter


def _args(analysis_dir: Path, *extra: str) -> list[str]:
    return ["--analysis-dir", str(analysis_dir), *extra]


def test_dry_run_makes_no_calls_but_writes_layouts(analysis_dir: Path, stub_interpreter, capsys):
    assert cli.main(_args(analysis_dir, "--stage", "all", "--dry-run")) == 0

    out = capsys.readouterr().out
    assert "no API calls made" in out
    assert "MOTIF FAMILY: cluster 0" in out
    assert "SPATIAL STRUCTURE" in out

    interpretation = analysis_dir / "interpretation"
    assert (interpretation / "layouts" / "panel_a.json").exists()
    assert not (interpretation / "clusters.json").exists()
    assert FakeInterpreter.calls == {}          # never constructed


def test_full_run_writes_all_three_stages(analysis_dir: Path, embeddings, stub_interpreter):
    npy, txt = embeddings
    code = cli.main(_args(analysis_dir, "--stage", "all",
                          "--embeddings", str(npy), "--paths", str(txt)))
    assert code == 0

    interpretation = analysis_dir / "interpretation"
    briefs = json.loads((interpretation / "clusters.json").read_text())
    assert set(briefs) == {"0", "1", "2"}
    assert briefs["0"]["name"] == "family_0"
    assert briefs["0"]["stats"]["cohesion"] is not None       # embeddings were used

    assert (interpretation / "clusters.md").exists()
    assert (interpretation / "panels" / "panel_a.json").exists()
    assert (interpretation / "panels" / "panel_a.md").exists()
    assert "3 families, 2 panels" in (interpretation / "corpus.md").read_text()

    assert FakeInterpreter.calls == {"cluster": 3, "panel": 2, "corpus": 1}


def test_resume_skips_work_already_on_disk(analysis_dir: Path, stub_interpreter):
    cli.main(_args(analysis_dir, "--stage", "all"))
    assert FakeInterpreter.calls["cluster"] == 3

    cli.main(_args(analysis_dir, "--stage", "all", "--resume"))
    # Clusters and panels are already written; only the synthesis re-runs.
    assert FakeInterpreter.calls == {"cluster": 0, "panel": 0, "corpus": 1}


def test_panels_stage_can_be_limited_to_one_panel(analysis_dir: Path, stub_interpreter):
    assert cli.main(_args(analysis_dir, "--stage", "panels", "--panels", "panel_b")) == 0

    panels_dir = analysis_dir / "interpretation" / "panels"
    assert (panels_dir / "panel_b.json").exists()
    assert not (panels_dir / "panel_a.json").exists()
    assert FakeInterpreter.calls["panel"] == 1


def test_panel_stage_feeds_cluster_briefs_forward(analysis_dir: Path, stub_interpreter, monkeypatch):
    seen: dict = {}

    def capture(self, corpus, stem, briefs=None, cluster_stats=None, **kwargs):
        seen[stem] = briefs
        return {"panel_stem": stem, "title": "t", "confidence": "low"}

    monkeypatch.setattr(FakeInterpreter, "panel_reading", capture)
    cli.main(_args(analysis_dir, "--stage", "all"))

    assert seen["panel_a"]["0"]["name"] == "family_0"


def test_a_failing_cluster_does_not_sink_the_run(analysis_dir: Path, monkeypatch, capsys):
    class Flaky(FakeInterpreter):
        def cluster_brief(self, corpus, stats, **kwargs):
            if stats.cluster_id == 1:
                raise RuntimeError("boom")
            return super().cluster_brief(corpus, stats, **kwargs)

    monkeypatch.setattr(cli, "Interpreter", Flaky)
    assert cli.main(_args(analysis_dir, "--stage", "clusters")) == 0

    assert "FAILED after 0s: boom" in capsys.readouterr().out
    briefs = json.loads((analysis_dir / "interpretation" / "clusters.json").read_text())
    assert set(briefs) == {"0", "2"}          # the other two still landed


def test_missing_analysis_dir_exits_nonzero(tmp_path: Path, capsys):
    assert cli.main(["--analysis-dir", str(tmp_path / "nope")]) == 1
    assert "not found" in capsys.readouterr().out


def test_missing_credentials_are_reported_clearly(analysis_dir: Path, monkeypatch, capsys):
    """Without a usable client the run stops with an actionable message, not a traceback."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(_args(analysis_dir, "--stage", "clusters")) == 1

    out = capsys.readouterr().out
    assert out.count("ERROR:") == 1
    # Either the SDK is absent or the key is — both name the fix.
    assert "ANTHROPIC_API_KEY" in out or "pip install anthropic" in out
