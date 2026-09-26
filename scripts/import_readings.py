#!/usr/bin/env python3
"""
import_readings.py — write motif and panel readings made outside the API run
into an analysis directory, through the same writers the pipeline uses.

`interpret_motifs.py` and `label_motifs.py --per-motif` produce readings by
calling the API. This script is for readings produced any other way — written
in a Claude session that had no API key, produced by a batch job, or corrected
by hand in bulk — so they land in exactly the files and formats a pipeline run
would have written:

  <analysis>/motif_labels.json                   (via label_motifs.write_label)
  <analysis>/interpretation/panels/<stem>.json    (via InterpretationStore)
  <analysis>/interpretation/panels/<stem>.md
  <analysis>/interpretation/corpus.md

Readings directory: one `<panel_stem>.json` per panel:

  {
    "motifs": {
      "3": {"label": "...", "description": "...", "iconography": "...",
            "confidence": "medium", "role_on_board": "... (optional)",
            "bbox": {"x": 0, "y": 0, "w": 0, "h": 0}   (optional, recommended)},
      ...
    },
    "panel": { ...the PANEL_SCHEMA fields... }        (optional)
  }

Motifs are matched to the current approved boxes by `bbox` when it is given,
so a reading still lands on the right motif after the boxes have been edited
and re-indexed; a reading whose box no longer exists is reported and skipped.
Without `bbox` the key is taken as the motif index.

Labels by a person (source `human` or `llm-edited`) are never replaced unless
--overwrite is given, the same rule `label_motifs.py --refresh-generated`
follows.

Usage:
  python3 scripts/import_readings.py --analysis-dir frobenius_artifacts/analysis \\
      --readings path/to/readings --corpus path/to/corpus.md \\
      --model "in-session reading, prompts v2"
  # Check what would change without writing anything
  python3 scripts/import_readings.py ... --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from panel_art.interpret import (  # noqa: E402
    CONFIDENCE,
    PANEL_SCHEMA,
    InterpretationStore,
    load_corpus,
)


def _load_label_cli():
    spec = importlib.util.spec_from_file_location(
        "label_motifs", REPO_ROOT / "scripts" / "label_motifs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("label_motifs", module)
    spec.loader.exec_module(module)
    return module


label_cli = _load_label_cli()

# A reading's box must overlap the approved box this much to count as the same
# motif. Boxes that were only nudged still match; a different motif does not.
MIN_IOU = 0.8


def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def match_motif(motifs: list, key: str, reading: dict):
    """The approved motif this reading describes, or None."""
    same_index = next((m for m in motifs if m.index == int(key)), None)
    box = reading.get("bbox")
    if not box:
        return same_index
    # The index wins while its box still matches — that also keeps duplicate
    # boxes apart. Only a re-indexed motif is looked up by its box.
    if same_index is not None and iou(same_index.bbox, box) >= MIN_IOU:
        return same_index
    best = max(motifs, key=lambda m: iou(m.bbox, box), default=None)
    return best if best is not None and iou(best.bbox, box) >= MIN_IOU else None


def panel_problems(panel: dict) -> list[str]:
    problems = [f"missing '{k}'" for k in PANEL_SCHEMA.get("required", []) if k not in panel]
    allowed = PANEL_SCHEMA["properties"]["panel_type"].get("enum")
    if allowed and panel.get("panel_type") not in allowed:
        problems.append(f"panel_type {panel.get('panel_type')!r} not in {allowed}")
    if panel.get("confidence") not in CONFIDENCE["enum"]:
        problems.append(f"confidence {panel.get('confidence')!r} not in {CONFIDENCE['enum']}")
    return problems


def import_readings(corpus, readings: dict[str, dict], labels: dict,
                    store: InterpretationStore | None, model: str,
                    overwrite: bool = False, dry_run: bool = False) -> dict:
    """Apply readings to `labels` (in place) and, unless dry_run, save panels.

    Returns counts plus the list of problems, so callers and tests can check it.
    """
    stats = {"labels": 0, "protected": 0, "unmatched": 0, "panels": 0, "problems": []}
    now = datetime.now(timezone.utc).isoformat()

    for stem, data in sorted(readings.items()):
        if stem not in corpus.panels:
            stats["problems"].append(f"{stem}: no such panel in the analysis directory")
            continue
        motifs = corpus.motifs_for_panel(stem)

        for key, reading in (data.get("motifs") or {}).items():
            motif = match_motif(motifs, key, reading)
            if motif is None:
                stats["unmatched"] += 1
                stats["problems"].append(f"{stem} #{key}: no approved box matches this reading")
                continue
            if not label_cli.should_write(motif, overwrite, refresh_generated=True):
                stats["protected"] += 1
                continue
            description = " ".join(x for x in (reading.get("description", ""),
                                                reading.get("role_on_board", "")) if x)
            label_cli.write_label(labels, motif, reading.get("label", ""), description,
                                  reading.get("iconography", ""), source="llm",
                                  notes=f"confidence: {reading.get('confidence', '?')}")
            stats["labels"] += 1

        panel = data.get("panel")
        if panel:
            problems = panel_problems(panel)
            if problems:
                stats["problems"].append(f"{stem}: panel reading skipped — " + "; ".join(problems))
                continue
            reading = dict(panel)
            reading["panel_stem"] = stem
            reading["layout"] = corpus.layout_for(stem).as_dict()
            reading["generated_at"] = now
            reading["model"] = model
            if not dry_run and store is not None:
                store.save_panel(stem, reading)
            stats["panels"] += 1
    return stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import externally produced readings through the pipeline's writers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("frobenius_artifacts/analysis"))
    p.add_argument("--readings", type=Path, required=True,
                   help="Directory of <panel_stem>.json reading files")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Markdown essay to save as interpretation/corpus.md")
    p.add_argument("--labels", type=Path, default=None,
                   help="Default: <analysis-dir>/motif_labels.json")
    p.add_argument("--model", default="imported",
                   help="Recorded as the readings' `model` field: say where they came from")
    p.add_argument("--overwrite", action="store_true",
                   help="Also replace labels written by a person (human, llm-edited)")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.analysis_dir.exists():
        print(f"ERROR: analysis directory not found: {args.analysis_dir}")
        return 1
    files = sorted(args.readings.glob("*.json"))
    if not files:
        print(f"ERROR: no reading files in {args.readings}")
        return 1

    labels_path = args.labels or (args.analysis_dir / "motif_labels.json")
    store = InterpretationStore(args.analysis_dir / "interpretation")
    corpus = load_corpus(args.analysis_dir)
    labels = label_cli.load_labels(labels_path)
    readings = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in files}

    stats = import_readings(corpus, readings, labels, store, args.model,
                            overwrite=args.overwrite, dry_run=args.dry_run)
    for problem in stats["problems"]:
        print(f"  ! {problem}")
    print(f"{stats['labels']} labels, {stats['panels']} panel readings; "
          f"{stats['protected']} human labels left alone, {stats['unmatched']} unmatched")

    if args.dry_run:
        print("Dry run — nothing written.")
        return 0
    label_cli._flush(labels, labels_path)
    print(f"Wrote {labels_path}")
    if args.corpus:
        path = store.save_corpus(args.corpus.read_text(encoding="utf-8"))
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
