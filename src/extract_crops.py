#!/usr/bin/env python3
"""
extract_crops.py — crop detected bounding boxes from panel images.

Reads  frobenius_artifacts/analysis/detections.json
Writes frobenius_artifacts/analysis/motifs/<panel_stem>/<index>_<scale>.png

Usage:
    uv run --project src/python python src/python/extract_crops.py

Options:
    --detections PATH   Path to detections JSON  (default: auto-resolved)
    --panels-dir PATH   Directory containing panel PNGs  (default: auto-resolved)
    --out-dir PATH      Output root directory  (default: auto-resolved)
    --padding INT       Extra pixels to add on each side of the bbox  (default: 4)
    --scale SCALE       Only export detections of this scale (motif|register|all)
    --min-iou FLOAT     Skip detections with pred_iou below this threshold
    --dry-run           Print what would be written without writing anything
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
    p.add_argument("--dry-run",    action="store_true",
                   help="Print planned crops without writing files")
    return p.parse_args()


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

        print(f"\n{filename}  ({len(to_export)}/{len(detections)} detections)")

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
