#!/usr/bin/env python3
"""
extract_crops.py — crop detected bounding boxes from panel images.

Reads  frobenius_artifacts/analysis/detections.json
Writes frobenius_artifacts/analysis/motifs/<panel_stem>/<index>_<scale>.png

Usage:
    uv run --project src/python python src/python/extract_crops.py

Options:
    --detections PATH        Path to detections JSON  (default: auto-resolved)
    --panels-dir PATH        Directory containing panel PNGs  (default: auto-resolved)
    --out-dir PATH           Output root directory  (default: auto-resolved)
    --padding INT            Extra pixels to add on each side of the bbox  (default: 4)
    --scale SCALE            Only export detections of this scale (motif|register|all)
    --min-iou FLOAT          Skip detections with pred_iou below this threshold
    --filter-containment     Drop detections whose bbox is mostly inside a larger one
    --containment-threshold  Fraction of smaller bbox that must overlap to suppress it
                             (default: 0.80)
    --dry-run                Print what would be written without writing anything
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


# ── Repo-relative defaults ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # src/python/
_REPO = _HERE.parent.parent                      # repo root

DEFAULT_DETECTIONS = _REPO / "frobenius_artifacts/analysis/detections.json"
DEFAULT_PANELS_DIR = _REPO / "frobenius_artifacts/analysis/panels"
DEFAULT_OUT_DIR    = _REPO / "frobenius_artifacts/analysis/motifs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    p.add_argument("--panels-dir", type=Path, default=DEFAULT_PANELS_DIR)
    p.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--padding",    type=int,  default=4,
                   help="Extra pixels on each side of the bounding box")
    p.add_argument("--scale",      default="all",
                   choices=["motif", "register", "all"],
                   help="Only export this detection scale (default: all)")
    p.add_argument("--min-iou",    type=float, default=0.0,
                   help="Skip detections below this pred_iou")
    p.add_argument("--filter-containment", action="store_true",
                   help="Remove detections mostly contained within a larger detection")
    p.add_argument("--containment-threshold", type=float, default=0.80,
                   help="Fraction of smaller bbox covered by larger to trigger removal "
                        "(default: 0.80)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print planned crops without writing files")
    return p.parse_args()


def _containment_ratio(a: dict, b: dict) -> float:
    """
    Fraction of the *smaller* bbox area that is covered by the intersection of a and b.
    Returns 0 if the boxes don't overlap.
    """
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter    = (ix2 - ix1) * (iy2 - iy1)
    min_area = min(a["w"] * a["h"], b["w"] * b["h"])
    return inter / min_area if min_area > 0 else 0.0


def filter_containment(detections: list, threshold: float) -> list:
    """
    Remove detections whose bbox is >= threshold fraction contained within a
    larger detection.  Larger bboxes (by area) always survive; smaller ones
    that are mostly inside a larger one are suppressed.

    Returns the filtered list; preserves original ordering of survivors.
    """
    n       = len(detections)
    areas   = [d["bbox"]["w"] * d["bbox"]["h"] for d in detections]
    # Indices sorted largest → smallest
    by_size = sorted(range(n), key=lambda i: -areas[i])
    suppress = [False] * n

    for pos, i in enumerate(by_size):
        if suppress[i]:
            continue
        for j in by_size[pos + 1:]:    # j is always smaller than i
            if suppress[j]:
                continue
            if _containment_ratio(detections[i]["bbox"], detections[j]["bbox"]) >= threshold:
                suppress[j] = True

    kept    = [d for d, s in zip(detections, suppress) if not s]
    removed = n - len(kept)
    return kept, removed


def crop_and_save(img: Image.Image, bbox: dict, padding: int,
                  out_path: Path, dry_run: bool) -> bool:
    """
    Crop bbox (x, y, w, h) from img with optional padding, save to out_path.
    Returns True if the crop was non-empty and saved (or would be).
    """
    iw, ih = img.size
    x = max(0, bbox["x"] - padding)
    y = max(0, bbox["y"] - padding)
    x2 = min(iw, bbox["x"] + bbox["w"] + padding)
    y2 = min(ih, bbox["y"] + bbox["h"] + padding)

    if x2 <= x or y2 <= y:
        print(f"  SKIP (degenerate bbox after padding): {out_path.name}", file=sys.stderr)
        return False

    if dry_run:
        print(f"  [dry-run] {out_path}  ({x},{y})→({x2},{y2})")
        return True

    crop = img.crop((x, y, x2, y2))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True


def main():
    args = parse_args()

    if not args.detections.exists():
        sys.exit(f"Detections file not found: {args.detections}")
    if not args.panels_dir.exists():
        sys.exit(f"Panels directory not found: {args.panels_dir}")

    with open(args.detections) as f:
        data = json.load(f)

    total_written = 0
    total_skipped = 0

    for panel in data["panels"]:
        filename  = panel["filename"]
        panel_path = args.panels_dir / filename

        if not panel_path.exists():
            print(f"WARNING: panel image not found, skipping — {panel_path}", file=sys.stderr)
            continue

        img = Image.open(panel_path).convert("RGB")
        stem = panel_path.stem                        # filename without .png
        out_panel_dir = args.out_dir / stem

        detections = panel["detections"]
        to_export = [
            d for d in detections
            if (args.scale == "all" or d["scale"] == args.scale)
            and d["pred_iou"] >= args.min_iou
        ]

        n_containment_removed = 0
        if args.filter_containment and len(to_export) > 1:
            to_export, n_containment_removed = filter_containment(
                to_export, args.containment_threshold
            )

        suffix = ""
        if args.filter_containment:
            suffix = f", -{n_containment_removed} sub-crops"
        print(f"\n{filename}  ({len(to_export)}/{len(detections)} detections{suffix})")

        for det in to_export:
            idx   = det["index"]
            scale = det["scale"]
            iou   = det["pred_iou"]
            out_name = f"{idx:03d}_{scale}_iou{iou:.3f}.png"
            out_path = out_panel_dir / out_name

            ok = crop_and_save(img, det["bbox"], args.padding, out_path, args.dry_run)
            if ok:
                total_written += 1
                if not args.dry_run:
                    print(f"  wrote {out_path.relative_to(args.out_dir)}")
            else:
                total_skipped += 1

    print(f"\n{'[dry-run] would write' if args.dry_run else 'wrote'} "
          f"{total_written} crop(s), skipped {total_skipped}")


if __name__ == "__main__":
    main()
