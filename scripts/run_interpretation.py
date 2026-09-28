#!/usr/bin/env python3
"""
run_interpretation.py — the whole interpretation in one command.

Runs the existing scripts in the order their inputs depend on each other:

  1. interpret_motifs.py --stage clusters    family briefs (with motif_gloss)
  2. label_motifs.py --per-motif --refresh-generated
                                             one reading per motif, in the
                                             context of its board; labels by a
                                             person (human, llm-edited) are kept
  3. interpret_motifs.py --stage panels      one reading per object
  4. interpret_motifs.py --stage compare     likely pairs of objects side by side:
                                             same object, copies, one workshop
  5. interpret_motifs.py --stage corpus      the collection essay
  6. export_interpretation_site.py           the self-contained site.html

Labels come before panels because the panel prompt quotes every motif's
label; comparisons follow the panels because pairs are picked partly from
the readings; the essay comes last because it reads all of them.

Usage:
  export ANTHROPIC_API_KEY=...
  python3 scripts/run_interpretation.py --analysis-dir frobenius_artifacts/analysis \\
      --embeddings motif_embeddings_edges.npy --paths motif_paths_edges.txt

  # Print every prompt that would be sent; no key needed, nothing written
  python3 scripts/run_interpretation.py --analysis-dir ... --dry-run

  # Pick up after an interruption (skips briefs and panels already written)
  python3 scripts/run_interpretation.py --analysis-dir ... --resume

  # Rebuild only the site from what is already on disk
  python3 scripts/run_interpretation.py --analysis-dir ... --only site
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = ["clusters", "labels", "panels", "compare", "corpus", "site"]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def cached_embeddings(analysis_dir: Path) -> tuple[Path, Path] | None:
    """The embedding matrix motif_pipeline.ipynb Stage 2 saves beside clusters.json.

    Its "Save Clusters" button writes clusters.json and this cache from one
    Compute Embeddings run, so the two always describe the same boxes — which
    makes it the right default when no --embeddings are given.
    """
    npy = analysis_dir / "embeddings_cache.npy"
    keys = analysis_dir / "embeddings_cache_keys.txt"
    return (npy, keys) if npy.exists() and keys.exists() else None


def plan(args) -> list[tuple[str, str, list[str]]]:
    """(step, script, argv) for each step to run — separated out so it is testable."""
    common = ["--analysis-dir", str(args.analysis_dir)]
    if not (args.embeddings and args.paths):
        cached = cached_embeddings(args.analysis_dir)
        if cached:
            args.embeddings, args.paths = cached
    emb = []
    if args.embeddings and args.paths:
        emb = ["--embeddings", str(args.embeddings), "--paths", str(args.paths)]
    panels = ["--panels", *args.panels] if args.panels else []
    dry = ["--dry-run"] if args.dry_run else []
    resume = ["--resume"] if args.resume else []
    model = ["--model", args.model] if args.model else []

    steps = {
        "clusters": ("interpret_motifs",
                     common + emb + ["--stage", "clusters"] + resume + dry + model),
        "labels": ("label_motifs",
                   common + emb + ["--per-motif", "--refresh-generated"] + panels + dry + model),
        "panels": ("interpret_motifs",
                   common + emb + ["--stage", "panels"] + panels + resume + dry + model),
        "compare": ("interpret_motifs",
                    common + emb + ["--stage", "compare", "--max-pairs", str(args.max_pairs)]
                    + panels + resume + dry + model),
        "corpus": ("interpret_motifs",
                   common + emb + ["--stage", "corpus"] + dry + model),
        "site": ("export_interpretation_site", common + emb + panels),
    }
    chosen = args.only or [s for s in STEPS if s not in (args.skip or [])]
    if args.dry_run:
        chosen = [s for s in chosen if s != "site"]      # the export has no dry run
    return [(s, *steps[s]) for s in STEPS if s in chosen]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run clusters → labels → panels → compare → corpus → site in one go",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("frobenius_artifacts/analysis"))
    p.add_argument("--embeddings", type=Path, default=None,
                   help="Default: <analysis-dir>/embeddings_cache.npy, which the "
                        "pipeline notebook's Save Clusters writes, if present")
    p.add_argument("--paths", type=Path, default=None,
                   help="Default: <analysis-dir>/embeddings_cache_keys.txt")
    p.add_argument("--panels", nargs="*", metavar="STEM", default=None,
                   help="Limit labels, panels and the site to these panel stems")
    p.add_argument("--model", default=None, help="Override each script's default model")
    p.add_argument("--max-pairs", type=int, default=100,
                   help="Cap on cross-panel comparisons (one API call each)")
    p.add_argument("--only", nargs="+", choices=STEPS, default=None,
                   help="Run just these steps (still in pipeline order)")
    p.add_argument("--skip", nargs="+", choices=STEPS, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip briefs, panel readings and comparisons already on disk")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the prompts each step would send; no key needed")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = plan(args)
    for n, (step, script, script_argv) in enumerate(steps, start=1):
        print(f"\n{'█' * 68}\n  [{n}/{len(steps)}] {step}: {script}.py {' '.join(script_argv)}"
              f"\n{'█' * 68}")
        code = _script(script).main(script_argv)
        if code:
            print(f"\nStopped: step '{step}' exited with {code}.")
            return code
    print("\nDone." + ("" if args.dry_run else
                       f" Site: {args.analysis_dir / 'interpretation' / 'site.html'}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
