#!/usr/bin/env python3
"""
label_motifs.py — bootstrap labels across the whole corpus, then refine.

Two modes, cheapest first.

  --from-briefs   FREE, instant, no API. Propagates each cluster brief's name
                  and visual definition onto its unlabelled members. The briefs
                  already characterise the family from centroid exemplars, so
                  this gives the entire corpus a coherent v1 in one pass, with
                  labels that agree with each other by construction.

  --per-motif     One API call per unlabelled motif: the crop, its panel
                  context, and its family, in exchange for a label specific to
                  that motif rather than to its family.

Start with --from-briefs, read the result, and spend --per-motif only where
the family label is genuinely too coarse.

Human labels are never overwritten without --overwrite. Everything written
records its provenance (`cluster-brief` / `llm` / `human`), so a later pass —
and the interpretation prompts — can tell what a person asserted from what a
model guessed.

Usage:
  # v1 across the corpus, no API calls
  python3 scripts/label_motifs.py --analysis-dir analysis/ --from-briefs

  # refine one family, or one panel
  python3 scripts/label_motifs.py --analysis-dir analysis/ --per-motif --clusters 3
  python3 scripts/label_motifs.py --analysis-dir analysis/ --per-motif --panels <stem>

Environment:
  ANTHROPIC_API_KEY  — required for --per-motif only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panel_art.interpret import (  # noqa: E402
    DEFAULT_MODEL,
    Corpus,
    InterpretationStore,
    Interpreter,
    SYSTEM_PROMPT,
    compute_cluster_stats,
    encode_image,
    load_corpus,
)
from panel_art.layout import render_layout_text  # noqa: E402

GENERATED_SOURCES = {"llm", "cluster-brief", "llm-edited"}

MOTIF_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string",
                  "description": "2-4 words, snake_case, naming what this motif is"},
        "description": {"type": "string",
                        "description": "One sentence on what is visually present"},
        "iconography": {"type": "string",
                        "description": "Probable significance, or 'unclear'"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["label", "description", "iconography", "confidence"],
    "additionalProperties": False,
}


# ── Label store ──────────────────────────────────────────────────────────────

def load_labels(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def label_key(motif) -> str:
    """The crop-path key motif_labels.json uses, matching PipelineState."""
    return (f"../../frobenius_artifacts/analysis/motifs_norm/"
            f"{motif.panel_stem}/{motif.index:03d}_motif_iou1.000.png")


def write_label(labels: dict, motif, label: str, description: str,
                iconography: str, source: str, notes: str = "") -> None:
    labels[label_key(motif)] = {
        "label": label,
        "description": description,
        "iconography": iconography,
        "notes": notes,
        "cluster": motif.cluster,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def should_write(motif, overwrite: bool) -> bool:
    """Never clobber a human label unless asked; generated labels are fair game.

    A generated label is a placeholder — refreshing it from a newer brief, or
    upgrading a family-inherited one to a per-motif reading, is the point. A
    human label is an assertion, so it stands until --overwrite says otherwise.
    """
    if not motif.label:
        return True
    if overwrite:
        return True
    return (motif.label_source or "human") in GENERATED_SOURCES


# ── Mode 1: propagate cluster briefs (no API) ────────────────────────────────

def run_from_briefs(corpus: Corpus, store: InterpretationStore, labels: dict,
                    targets: list, overwrite: bool, dry_run: bool) -> int:
    briefs = store.load_clusters()
    if not briefs:
        print("  No cluster briefs found. Run:\n"
              "    python3 scripts/interpret_motifs.py --analysis-dir <dir> --stage clusters")
        return 0

    written = 0
    by_cluster: dict[int, list] = {}
    for motif in targets:
        if motif.cluster >= 0:
            by_cluster.setdefault(motif.cluster, []).append(motif)

    for cid in sorted(by_cluster):
        brief = briefs.get(str(cid))
        members = by_cluster[cid]
        if not brief or not brief.get("name"):
            print(f"  cluster {cid}: no brief — skipped ({len(members)} motifs)")
            continue

        name = brief["name"]
        description = brief.get("visual_definition", "")
        icon = brief.get("iconographic_reading", "") or "unclear"
        print(f"  cluster {cid} -> {name}  ({len(members)} motifs)")

        for motif in members:
            if not should_write(motif, overwrite):
                continue
            if not dry_run:
                # `source: cluster-brief` already marks this as inherited and
                # provisional. Notes is the human's channel — leave it clear so
                # a correction doesn't have to delete our disclaimer first.
                write_label(labels, motif, name, description, icon,
                            source="cluster-brief")
            written += 1
    return written


# ── Mode 2: per-motif labelling ──────────────────────────────────────────────

def build_motif_prompt(corpus: Corpus, motif, brief: dict | None) -> str:
    layout = corpus.layout_for(motif.panel_stem)
    placement = next((p for p in layout.placements if p.key == motif.key), None)

    lines = [
        f"Label motif #{motif.index} on panel {motif.panel_stem}.",
        "",
        "The first image is the motif crop; the second is the whole panel for context.",
        "",
        "WHERE IT SITS:",
    ]
    if placement is not None:
        lines.append(f"  {placement.zone}, covering {placement.area_fraction * 100:.1f}% "
                     f"of the panel, in "
                     + (f"register {placement.register}" if placement.register >= 0
                        else "the field/ground rather than a register"))
    lines.append("")
    lines.append("PANEL STRUCTURE:")
    lines.append(render_layout_text(layout, max_relations=8))

    if brief and brief.get("name"):
        lines += ["", f"ITS FAMILY (cluster {motif.cluster}, characterised across the corpus):",
                  f"  {brief['name']} — {brief.get('visual_definition', '')}",
                  f"  Iconography: {brief.get('iconographic_reading', 'unclear')}",
                  "",
                  "Say where this motif departs from the family description, if it does."]

    lines += ["", "Give a label for this motif specifically. Prefer the family's "
                  "vocabulary where it fits, so labels stay comparable across the corpus."]
    return "\n".join(lines)


def run_per_motif(corpus: Corpus, store: InterpretationStore, labels: dict,
                  labels_path: Path, targets: list,
                  interpreter: Interpreter | None,
                  overwrite: bool, dry_run: bool, delay: float) -> int:
    briefs = store.load_clusters()
    todo = [m for m in targets if should_write(m, overwrite)]
    kept = len(targets) - len(todo)
    print(f"  {len(todo)} motifs to label"
          + (f" ({kept} kept — labelled by a person)" if kept else ""))

    written = 0
    for n, motif in enumerate(todo, start=1):
        brief = briefs.get(str(motif.cluster))
        prompt = build_motif_prompt(corpus, motif, brief)
        print(f"\n  [{n}/{len(todo)}] {motif.panel_stem} #{motif.index} "
              f"(cluster {motif.cluster})")

        if dry_run:
            print("    ┌" + "─" * 64)
            print("\n".join("    │ " + l for l in prompt.splitlines()))
            print("    └" + "─" * 64)
            continue

        assert interpreter is not None
        started = time.monotonic()
        try:
            content = [
                {"type": "text", "text": "Motif crop:"},
                encode_image(corpus.crop(motif, padding=8), max_dim=512),
                {"type": "text", "text": "Whole panel:"},
                encode_image(corpus.panel_image(motif.panel_stem), max_dim=800),
                {"type": "text", "text": prompt},
            ]
            data = interpreter._json_call(content, 3000, MOTIF_LABEL_SCHEMA)
        except Exception as exc:
            print(f"    FAILED after {time.monotonic() - started:.0f}s: {exc}")
            continue

        write_label(labels, motif, data.get("label", ""), data.get("description", ""),
                    data.get("iconography", ""), source="llm",
                    notes=f"confidence: {data.get('confidence', '?')}")
        written += 1
        print(f"    → {data.get('label', '?')} [{data.get('confidence', '?')}] "
              f"({time.monotonic() - started:.0f}s)")

        if written % 10 == 0:                    # checkpoint; runs get interrupted
            _flush(labels, labels_path)
        if delay and n < len(todo):
            time.sleep(delay)
    return written


def _flush(labels: dict, labels_path: Path) -> None:
    labels_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bootstrap and refine motif labels across the corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("frobenius_artifacts/analysis"))
    p.add_argument("--labels", type=Path, default=None,
                   help="Default: <analysis-dir>/motif_labels.json")
    p.add_argument("--embeddings", type=Path, default=None)
    p.add_argument("--paths", type=Path, default=None)

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--from-briefs", action="store_true",
                      help="Propagate cluster brief names onto members (no API calls)")
    mode.add_argument("--per-motif", action="store_true",
                      help="One API call per unlabelled motif")

    p.add_argument("--panels", nargs="*", metavar="STEM", default=None)
    p.add_argument("--clusters", nargs="*", type=int, metavar="ID", default=None,
                   help="Limit to these cluster ids")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing labels, including human ones")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="low",
                   choices=["low", "medium", "high", "xhigh", "max"],
                   help="A single motif label is a small task; low is usually right")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delay", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.analysis_dir.exists():
        print(f"ERROR: analysis directory not found: {args.analysis_dir}")
        return 1
    if not args.from_briefs and not args.per_motif:
        print("Choose a mode: --from-briefs (free) or --per-motif (one call each).")
        return 1

    labels_path = args.labels or (args.analysis_dir / "motif_labels.json")
    store = InterpretationStore(args.analysis_dir / "interpretation")

    print(f"Loading corpus from {args.analysis_dir}")
    corpus = load_corpus(args.analysis_dir, embeddings_path=args.embeddings,
                         paths_path=args.paths)

    targets = [m for m in corpus.motifs
               if (not args.panels or m.panel_stem in args.panels)
               and (args.clusters is None or m.cluster in args.clusters)]
    labelled = sum(1 for m in corpus.motifs if m.label)
    human = sum(1 for m in corpus.motifs
                if m.label and (m.label_source or "human") not in GENERATED_SOURCES)
    print(f"  {len(corpus.motifs)} motifs, {labelled} labelled "
          f"({human} by a person), {len(targets)} in scope")

    labels = load_labels(labels_path)

    print(f"\n{'═' * 68}")
    if args.from_briefs:
        print("Seeding labels from cluster briefs (no API calls)")
        written = run_from_briefs(corpus, store, labels, targets,
                                  args.overwrite, args.dry_run)
    else:
        interpreter = None
        if not args.dry_run:
            try:
                interpreter = Interpreter(model=args.model, effort=args.effort,
                                          on_progress=lambda m: print(f"      … {m}"))
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                return 1
        print(f"Labelling per motif (model={args.model}, effort={args.effort})")
        written = run_per_motif(corpus, store, labels, labels_path, targets,
                                interpreter, args.overwrite, args.dry_run,
                                args.delay)

    print(f"\n{'═' * 68}")
    if args.dry_run:
        print(f"Dry run — {written} labels would be written, nothing saved.")
        return 0

    _flush(labels, labels_path)
    print(f"Wrote {written} labels to {labels_path}")
    if written:
        print("\nCluster briefs written before this are now out of date — refresh them:")
        print(f"  python3 scripts/interpret_motifs.py --analysis-dir {args.analysis_dir} "
              "--stage clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
