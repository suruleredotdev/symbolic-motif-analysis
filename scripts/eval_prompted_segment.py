#!/usr/bin/env python3
"""
eval_prompted_segment.py — score SAM Generate against boxes a person approved.

A panel's `_approved.json` is the ground truth. The boxes that existed before
the review session (by default the ones SAM's automatic pass produced) are fed
back in as `existing`, exactly as the notebook does, and every other approved
box is a target the generator should find. A candidate is a hit when it
overlaps a target at IoU >= --iou.

    uv run python scripts/eval_prompted_segment.py FoA_04-5947_q48640_i1_panel_00
    uv run python scripts/eval_prompted_segment.py <stem> --legacy   # old defaults

Prints one row per candidate in the order the notebook would show them, then
precision (hits / candidates), recall (targets found / targets) and how many
of the first 5 candidates were hits — the queue is reviewed top-down, so the
head of the list matters most.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panel_art import motif_segment as ms  # noqa: E402


def _box(b: dict) -> list[int]:
    return [b["x"], b["y"], b["w"], b["h"]]


def load_templates(annotated: Path, exclude_stem: str) -> list[dict]:
    """Manual boxes from every other panel — what PS.manual_templates() returns."""
    out = []
    for p in sorted(annotated.glob("*_approved.json")):
        if p.name.startswith(exclude_stem):
            continue
        for d in json.loads(p.read_text()):
            if d.get("source") == "manual":
                out.append(d["bbox"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stem")
    ap.add_argument("--root", default="frobenius_artifacts/analysis")
    ap.add_argument("--existing-source", default="sam_auto",
                    help="approved boxes with this source count as already present")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--near-iou", type=float, default=0.4,
                    help="looser match: the right motif, box needs tightening")
    ap.add_argument("--seed", type=int, default=0,
                    help="treat the first N reviewed boxes as already accepted — "
                         "the second SAM Generate after a few Adds")
    ap.add_argument("--legacy", action="store_true",
                    help="run with the pre-2026-09 defaults (grid probes, 3%% edge floor)")
    ap.add_argument("--json", help="also write per-candidate results here")
    args = ap.parse_args(argv)

    root = Path(args.root)
    approved = json.loads((root / "annotated" / f"{args.stem}_approved.json").read_text())
    existing = [d["bbox"] for d in approved if d.get("source") == args.existing_source]
    targets = [d["bbox"] for d in approved if d.get("source") != args.existing_source]
    img = np.array(Image.open(root / "panels" / f"{args.stem}.png").convert("RGB"))
    templates = load_templates(root / "annotated", args.stem)
    if args.seed:
        existing, targets = existing + targets[:args.seed], targets[args.seed:]

    kw = dict(ms.LEGACY_PROMPT_PARAMS) if args.legacy else {"panel_templates": existing}
    dets = ms.prompted_segment(img, existing, approved_templates=templates or None,
                               verbose=True, **kw)
    if args.legacy:
        # The notebook ranked legacy candidates by 0.5·score + 0.3·edge + 0.2·novelty.
        gray = ms.cv2.cvtColor(img, ms.cv2.COLOR_RGB2GRAY)
        for d in dets:
            ed = ms._edge_density(gray, d.segmentation) if d.segmentation is not None else 0
            d.edge_density = ed
            d.rank_score = 0.5 * d.predicted_iou + 0.3 * ed + 0.2 * d.novelty
    dets.sort(key=lambda d: d.rank_score, reverse=True)

    found: set[int] = set()
    near_found: set[int] = set()
    rows = []
    for rank, d in enumerate(dets):
        ious = [ms._iou(_box(d.bbox), _box(t)) for t in targets]
        best = int(np.argmax(ious)) if ious else -1
        hit = bool(ious) and ious[best] >= args.iou
        near = bool(ious) and ious[best] >= args.near_iou
        if hit:
            found.add(best)
        if near:
            near_found.add(best)
        rows.append({"rank": rank, "bbox": d.bbox, "hit": hit, "near": near,
                     "best_iou": round(ious[best], 3) if ious else 0.0,
                     "edge": round(d.edge_density, 3), "rank_score": round(d.rank_score, 3),
                     "sam": round(d.predicted_iou, 3)})
        b = d.bbox
        tag = "HIT " if hit else ("near" if near else "    ")
        print(f"  {rank:2d} {tag} {b['w']:3d}x{b['h']:<3d} "
              f"@({b['x']},{b['y']})  iou={rows[-1]['best_iou']:.2f} "
              f"edge={d.edge_density:.2f} rank={d.rank_score:.2f} sam={d.predicted_iou:.2f}")

    n = len(dets)
    hits = sum(r["hit"] for r in rows)
    nears = sum(r["near"] for r in rows)
    top5 = sum(r["hit"] for r in rows[:5])
    print(f"\n{args.stem}  {'legacy' if args.legacy else 'current'} defaults"
          f"{f', {args.seed} boxes already accepted' if args.seed else ''}")
    print(f"  candidates={n}  hits={hits}  precision={hits / n if n else 0:.0%}"
          f"  (with near misses: {nears / n if n else 0:.0%})")
    print(f"  targets found={len(found)}/{len(targets)}  recall={len(found) / len(targets) if targets else 0:.0%}"
          f"  (with near misses: {len(near_found) / len(targets) if targets else 0:.0%})")
    print(f"  hits in first 5={top5}/5")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
