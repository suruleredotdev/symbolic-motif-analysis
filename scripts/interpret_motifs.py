#!/usr/bin/env python3
"""
interpret_motifs.py — Phase 6 driver: turn the cluster analysis into an interpretation.

Runs the three passes in `panel_art/interpret.py` over an analysis directory
produced by the pipeline, and writes the results to `<analysis>/interpretation/`.

Usage:
  # Everything, resuming any passes already on disk
  python3 scripts/interpret_motifs.py \\
      --analysis-dir frobenius_artifacts/analysis \\
      --embeddings motif_embeddings_edges.npy \\
      --paths      motif_paths_edges.txt \\
      --stage all --resume

  # Inspect what would be sent — no API key needed, no calls made
  python3 scripts/interpret_motifs.py --analysis-dir ... --stage all --dry-run

  # Just the cluster briefs, or just one panel
  python3 scripts/interpret_motifs.py --analysis-dir ... --stage clusters
  python3 scripts/interpret_motifs.py --analysis-dir ... --stage panels \\
      --panels EBA-Div_00311_Ife_q166566_i1_panel_00

Stages depend on one another: `panels` reads `clusters.json` if it exists (and
warns if it does not, since the readings are much thinner without it), and
`corpus` reads both. `--stage all` runs them in order in a single invocation.

Environment:
  ANTHROPIC_API_KEY  — required unless --dry-run

Dependencies:
  pip install anthropic pillow numpy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as `python3 scripts/interpret_motifs.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panel_art.interpret import (  # noqa: E402
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    Corpus,
    InterpretationStore,
    Interpreter,
    build_cluster_prompt,
    build_corpus_prompt,
    build_panel_prompt,
    cluster_context_lines,
    compute_cluster_stats,
    corpus_scale,
    load_corpus,
    render_clusters_markdown,
)
from panel_art.layout import render_layout_text  # noqa: E402


# ── Stages ───────────────────────────────────────────────────────────────────

def run_clusters(
    corpus: Corpus,
    stats: dict,
    store: InterpretationStore,
    interpreter: Interpreter | None,
    resume: bool,
    dry_run: bool,
    delay: float,
    exemplars: int,
) -> dict[str, dict]:
    briefs = store.load_clusters() if resume else {}
    todo = [cid for cid in sorted(stats) if str(cid) not in briefs]

    print(f"\n{'═' * 68}")
    print(f"Stage 1 — cluster briefs: {len(todo)} to generate "
          f"({len(briefs)} already on disk)")

    for n, cid in enumerate(todo, start=1):
        st = stats[cid]
        print(f"\n  [{n}/{len(todo)}] cluster {cid} — {st.size} motifs, "
              f"{st.panel_spread} panel(s)"
              + (f", cohesion {st.cohesion:.3f}" if st.cohesion is not None else ""))

        if dry_run:
            members = corpus.clusters().get(cid, [])
            print(_boxed(build_cluster_prompt(st, members, len(st.exemplar_keys))))
            continue

        assert interpreter is not None
        try:
            brief = interpreter.cluster_brief(corpus, st, max_dim=512)
        except Exception as exc:                       # keep going; one bad cluster
            print(f"    FAILED: {exc}")                # shouldn't sink the run
            continue

        briefs[str(cid)] = brief
        store.save_clusters(briefs)                    # checkpoint after each call
        print(f"    → {brief.get('name', '?')} [{brief.get('confidence', '?')}]")
        if delay and n < len(todo):
            time.sleep(delay)

    if not dry_run and briefs:
        store.save_clusters(briefs)
        (store.out_dir / "clusters.md").write_text(
            render_clusters_markdown(briefs), encoding="utf-8")
        print(f"\n  Wrote {store.clusters_path} (+ clusters.md)")

    return briefs


def run_panels(
    corpus: Corpus,
    stats: dict,
    store: InterpretationStore,
    interpreter: Interpreter | None,
    briefs: dict[str, dict],
    only: list[str] | None,
    resume: bool,
    dry_run: bool,
    delay: float,
) -> dict[str, dict]:
    readings = store.load_panels() if resume else {}
    stems = only or corpus.panel_stems()
    stems = [s for s in stems if corpus.motifs_for_panel(s)]
    todo = [s for s in stems if not (resume and s in readings)]

    print(f"\n{'═' * 68}")
    print(f"Stage 2 — panel readings: {len(todo)} to generate "
          f"({len(readings)} already on disk)")
    if not briefs:
        print("  NOTE: no cluster briefs available — readings will name families by id only."
              + (" (Expected in a dry run: stage 1 generates nothing to feed forward.)"
                 if dry_run else " Run --stage clusters first for corpus-aware readings."))

    for n, stem in enumerate(todo, start=1):
        motifs = corpus.motifs_for_panel(stem)
        layout = corpus.layout_for(stem)
        store.save_layout(stem, layout)                # deterministic, always useful

        print(f"\n  [{n}/{len(todo)}] {stem} — {len(motifs)} motifs, "
              f"{len(layout.registers)} register(s)")

        if dry_run:
            context = cluster_context_lines(motifs, briefs, stats)
            print(_boxed(build_panel_prompt(layout, motifs, context)))
            continue

        assert interpreter is not None
        try:
            reading = interpreter.panel_reading(corpus, stem, briefs, stats)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            continue

        readings[stem] = reading
        store.save_panel(stem, reading)
        print(f"    → {reading.get('title', '?')} [{reading.get('confidence', '?')}]")
        if delay and n < len(todo):
            time.sleep(delay)

    return readings


def run_corpus(
    corpus: Corpus,
    store: InterpretationStore,
    interpreter: Interpreter | None,
    briefs: dict[str, dict],
    readings: dict[str, dict],
    dry_run: bool,
) -> str | None:
    scale = corpus_scale(corpus)

    print(f"\n{'═' * 68}")
    print(f"Stage 3 — corpus synthesis: {len(briefs)} families, "
          f"{len(readings)} panel readings")

    if not briefs and not readings:
        print("  Nothing to synthesise — run the clusters and panels stages first.")
        return None

    if dry_run:
        print(_boxed(build_corpus_prompt(briefs, readings, scale)))
        return None

    assert interpreter is not None
    markdown = interpreter.corpus_synthesis(briefs, readings, scale)
    path = store.save_corpus(markdown)
    print(f"\n  Wrote {path} ({len(markdown):,} characters)")
    return markdown


def _boxed(text: str, width: int = 68) -> str:
    rule = "─" * width
    return f"    ┌{rule}\n" + "\n".join(f"    │ {line}" for line in text.splitlines()) + f"\n    └{rule}"


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate cluster, panel, and corpus interpretations from pipeline output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("frobenius_artifacts/analysis"),
                   help="Directory containing panels/, annotated/, motif_labels.json")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: <analysis-dir>/interpretation)")
    p.add_argument("--embeddings", type=Path, default=None,
                   help="motif_embeddings_*.npy — enables cohesion, centroid exemplars, "
                        "and cluster adjacency")
    p.add_argument("--paths", type=Path, default=None,
                   help="motif_paths_*.txt matching --embeddings")
    p.add_argument("--clusters", type=Path, default=None,
                   help="Optional JSON overriding cluster assignments "
                        "(motif key or crop path → cluster id)")

    p.add_argument("--stage", choices=["clusters", "panels", "corpus", "all"], default="all",
                   help="Which pass(es) to run")
    p.add_argument("--panels", nargs="*", metavar="STEM", default=None,
                   help="Limit the panels stage to these panel stems")

    p.add_argument("--model", default=DEFAULT_MODEL, help="Claude model to use")
    p.add_argument("--effort", default=DEFAULT_EFFORT,
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="Reasoning effort — interpretation quality scales with this")
    p.add_argument("--exemplars", type=int, default=4,
                   help="Exemplar crops sent per cluster brief")

    p.add_argument("--resume", action="store_true",
                   help="Skip clusters and panels already written to the output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the layouts and prompts that would be sent; make no API calls")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Seconds to wait between API calls")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    analysis_dir = args.analysis_dir
    if not analysis_dir.exists():
        print(f"ERROR: analysis directory not found: {analysis_dir}")
        return 1

    out_dir = args.out_dir or (analysis_dir / "interpretation")
    store = InterpretationStore(out_dir)
    store.ensure_dirs()

    print(f"Loading corpus from {analysis_dir}")
    corpus = load_corpus(
        analysis_dir,
        embeddings_path=args.embeddings,
        paths_path=args.paths,
        clusters_path=args.clusters,
    )
    scale = corpus_scale(corpus)
    print(f"  {scale['panels']} panels, {scale['motifs']} motifs, "
          f"{scale['clusters']} clusters, {scale['unclustered']} unclustered, "
          f"{scale['labelled']} labelled")

    if not corpus.motifs:
        print("ERROR: no approved motifs found — run the segment/review stages first.")
        return 1
    if corpus.embeddings is None:
        print("  NOTE: no embeddings supplied — cluster cohesion, centroid exemplars, "
              "and family adjacency will be unavailable.")
    else:
        matched, total = corpus.embedding_coverage()
        pct = (100 * matched / total) if total else 0.0
        print(f"  Embeddings joined to {matched}/{total} motifs ({pct:.0f}%)")
        if matched == 0:
            print("  WARNING: nothing joined — the embeddings were almost certainly "
                  "computed from a different run than the approved bboxes on disk. "
                  "Re-run the embedding notebook, or drop --embeddings.")
        elif pct < 60:
            print("  WARNING: low join rate — clusters below the threshold fall back "
                  "to arbitrary exemplars and report no cohesion.")

    stats = compute_cluster_stats(corpus, exemplars=args.exemplars)

    interpreter = None
    if not args.dry_run:
        try:
            interpreter = Interpreter(model=args.model, effort=args.effort)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1

    stages = ["clusters", "panels", "corpus"] if args.stage == "all" else [args.stage]

    briefs = store.load_clusters()
    readings = store.load_panels()

    if "clusters" in stages:
        briefs = run_clusters(corpus, stats, store, interpreter,
                              args.resume, args.dry_run, args.delay, args.exemplars)
    if "panels" in stages:
        readings = run_panels(corpus, stats, store, interpreter, briefs,
                              args.panels, args.resume, args.dry_run, args.delay)
    if "corpus" in stages:
        run_corpus(corpus, store, interpreter, briefs, readings, args.dry_run)

    print(f"\n{'═' * 68}")
    print("Dry run complete — no API calls made." if args.dry_run
          else f"Interpretation written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
