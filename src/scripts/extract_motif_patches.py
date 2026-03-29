"""
extract_motif_patches.py

Runs SAM 2 automatic mask generation on one or more source images to detect
distinct motif regions (carved figures, geometric patterns, relief elements).

For each image it outputs:
  - <stem>_annotated.jpg  — original image with bounding boxes overlaid,
                            labelled with area%, pred_iou, stability score
  - <stem>_patches/       — individual cropped patch files
  - <stem>_patches.json   — metadata for every kept patch

Usage:
  # Single image
  python3 extract_motif_patches.py path/to/image.png

  # Multiple images
  python3 extract_motif_patches.py img1.png img2.png img3.png

  # All images in a directory
  python3 extract_motif_patches.py --dir path/to/images/

  # Tune filtering thresholds
  python3 extract_motif_patches.py image.png \\
      --min-area 0.01 --max-area 0.60 \\
      --iou-thresh 0.82 --stability-thresh 0.88 \\
      --pad 0.08

  # Dry run — annotated image only, no patch files
  python3 extract_motif_patches.py image.png --annotate-only

Dependencies:
  pip install segment-anything-2 opencv-python-headless pillow numpy torch
  # SAM 2 checkpoint — download one of:
  #   https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
  #   https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
  # Set SAM2_CHECKPOINT env var or pass --checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ─── SAM 2 import (graceful error if not installed) ──────────────────────────

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "sam2_hiera_large.pt")
DEFAULT_MODEL_CFG  = os.environ.get("SAM2_MODEL_CFG",  "sam2_hiera_l.yaml")

# Patch filtering defaults
DEFAULT_MIN_AREA       = 0.015   # minimum fraction of image area
DEFAULT_MAX_AREA       = 0.55    # maximum fraction of image area
DEFAULT_MAX_ASPECT     = 4.5     # max(w,h) / min(w,h)
DEFAULT_IOU_THRESH     = 0.82    # SAM predicted IoU quality threshold
DEFAULT_STABILITY      = 0.86    # SAM stability score threshold
DEFAULT_PAD            = 0.08    # fractional padding added around each bbox
DEFAULT_NMS_IOU        = 0.72    # overlap above which smaller box is dropped

# Annotation colours (BGR for OpenCV, RGB for PIL)
PALETTE = [
    (255,  80,  80), (255, 165,  50), ( 80, 200,  80),
    ( 50, 150, 255), (200,  80, 255), ( 50, 220, 200),
    (255, 220,  50), (200, 200, 200),
]


# ─── Filtering helpers ────────────────────────────────────────────────────────

def _bbox_iou(a: list[int], b: list[int]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def filter_masks(
    masks: list[dict],
    img_area: int,
    min_area: float,
    max_area: float,
    max_aspect: float,
    iou_thresh: float,
    stability_thresh: float,
    nms_iou: float,
) -> list[dict]:
    """
    Apply quality + geometry filters to SAM masks, then NMS to remove
    heavily overlapping duplicates (keep the one with higher predicted_iou).
    """
    kept = []
    for m in masks:
        x, y, w, h = m["bbox"]
        area_ratio = m["area"] / img_area

        if area_ratio < min_area or area_ratio > max_area:
            continue
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        if m["predicted_iou"] < iou_thresh:
            continue
        if m["stability_score"] < stability_thresh:
            continue
        kept.append(m)

    # Sort by predicted_iou descending so NMS keeps higher-quality boxes
    kept.sort(key=lambda m: m["predicted_iou"], reverse=True)

    # Greedy NMS
    final = []
    suppressed = set()
    for i, m in enumerate(kept):
        if i in suppressed:
            continue
        final.append(m)
        for j in range(i + 1, len(kept)):
            if j not in suppressed and _bbox_iou(m["bbox"], kept[j]["bbox"]) > nms_iou:
                suppressed.add(j)

    return final


# ─── Annotation ───────────────────────────────────────────────────────────────

def annotate_image(
    image_path: Path,
    masks: list[dict],
    img_w: int,
    img_h: int,
    out_path: Path,
) -> None:
    """
    Draw colour-coded bounding boxes on the image, each labelled with:
      #N  area=X.X%  iou=0.XX  stab=0.XX
    """
    img = cv2.imread(str(image_path))
    if img is None:
        img_pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    overlay = img.copy()
    img_area = img_w * img_h

    # Semi-transparent mask fills
    for idx, m in enumerate(masks):
        colour = PALETTE[idx % len(PALETTE)]
        seg = m["segmentation"]  # boolean HxW array
        colour_img = np.zeros_like(img)
        colour_img[:] = (colour[2], colour[1], colour[0])  # BGR
        mask3 = np.stack([seg, seg, seg], axis=-1)
        overlay = np.where(mask3, (overlay * 0.55 + colour_img * 0.45).astype(np.uint8), overlay)

    img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    # Bounding boxes + labels
    font_scale = max(0.35, min(img_w, img_h) / 1200)
    thickness  = max(1, int(min(img_w, img_h) / 500))

    for idx, m in enumerate(masks):
        colour = PALETTE[idx % len(PALETTE)]
        bgr    = (colour[2], colour[1], colour[0])
        x, y, w, h = [int(v) for v in m["bbox"]]
        area_pct   = m["area"] / img_area * 100

        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, thickness + 1)

        label = (
            f"#{idx+1} "
            f"area={area_pct:.1f}% "
            f"iou={m['predicted_iou']:.2f} "
            f"stab={m['stability_score']:.2f}"
        )

        # Background pill for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ly = max(y - 4, th + 4)
        cv2.rectangle(img, (x, ly - th - baseline - 2), (x + tw + 4, ly + 2), bgr, -1)
        cv2.putText(img, label, (x + 2, ly - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  Annotated → {out_path}")


# ─── Patch extraction ─────────────────────────────────────────────────────────

def extract_patch(
    image_path: Path,
    bbox: list[int],
    img_w: int,
    img_h: int,
    pad: float,
    out_path: Path,
) -> dict:
    """
    Crop a padded region from the source image and save as PNG.
    Returns the actual pixel bbox used (after padding + clamping).
    """
    x, y, w, h = bbox
    pad_px_x = int(w * pad)
    pad_px_y = int(h * pad)
    x1 = max(0, x - pad_px_x)
    y1 = max(0, y - pad_px_y)
    x2 = min(img_w, x + w + pad_px_x)
    y2 = min(img_h, y + h + pad_px_y)

    img = Image.open(image_path).convert("RGB")
    patch = img.crop((x1, y1, x2, y2))
    patch.save(str(out_path), "PNG")

    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


# ─── Per-image pipeline ───────────────────────────────────────────────────────

def process_image(
    image_path: Path,
    generator: "SAM2AutomaticMaskGenerator",
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict]:
    """
    Run SAM 2 on one image, filter masks, save annotated image and patches.
    Returns a list of patch metadata dicts.
    """
    print(f"\n{'─'*60}")
    print(f"Image: {image_path.name}")

    # Load image
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    img_h, img_w = img_np.shape[:2]
    img_area = img_w * img_h
    print(f"  Size: {img_w}×{img_h}  ({img_area:,} px²)")

    # Run SAM 2
    print("  Running SAM 2 auto-mask generation…")
    all_masks = generator.generate(img_np)
    print(f"  Raw masks: {len(all_masks)}")

    # Filter
    kept = filter_masks(
        all_masks,
        img_area,
        min_area      = args.min_area,
        max_area      = args.max_area,
        max_aspect    = args.max_aspect,
        iou_thresh    = args.iou_thresh,
        stability_thresh = args.stability_thresh,
        nms_iou       = args.nms_iou,
    )
    print(f"  Kept after filtering: {len(kept)}")
    for i, m in enumerate(kept):
        area_pct = m["area"] / img_area * 100
        print(f"    #{i+1:2d}  bbox={m['bbox']}  "
              f"area={area_pct:.1f}%  iou={m['predicted_iou']:.3f}  "
              f"stab={m['stability_score']:.3f}")

    # Annotated image
    stem       = image_path.stem
    annotated  = out_dir / f"{stem}_annotated.jpg"
    annotate_image(image_path, kept, img_w, img_h, annotated)

    if args.annotate_only:
        return []

    # Extract patch files
    patches_dir = out_dir / f"{stem}_patches"
    patches_dir.mkdir(exist_ok=True)

    patch_meta = []
    for idx, m in enumerate(kept):
        patch_filename = f"{stem}_patch_{idx+1:03d}.png"
        patch_path     = patches_dir / patch_filename
        padded_bbox    = extract_patch(
            image_path, m["bbox"], img_w, img_h, args.pad, patch_path
        )
        area_ratio = m["area"] / img_area

        patch_meta.append({
            "patch_index":     idx + 1,
            "patch_file":      str(patch_path.relative_to(out_dir)),
            "source_image":    image_path.name,
            "source_path":     str(image_path),
            "bbox_original":   {"x": m["bbox"][0], "y": m["bbox"][1],
                                 "w": m["bbox"][2], "h": m["bbox"][3]},
            "bbox_padded":     padded_bbox,
            "bbox_relative":   {
                "x": m["bbox"][0] / img_w,
                "y": m["bbox"][1] / img_h,
                "w": m["bbox"][2] / img_w,
                "h": m["bbox"][3] / img_h,
            },
            "area_ratio":      round(area_ratio, 5),
            "area_pct":        round(area_ratio * 100, 2),
            "predicted_iou":   round(m["predicted_iou"], 4),
            "stability_score": round(m["stability_score"], 4),
            "image_size":      {"w": img_w, "h": img_h},
        })

    # Save metadata JSON
    meta_path = out_dir / f"{stem}_patches.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(patch_meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata → {meta_path}")
    print(f"  Patches  → {patches_dir}/ ({len(patch_meta)} files)")

    return patch_meta


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SAM 2 automatic motif extraction from carved panel illustrations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="*", metavar="IMAGE",
                   help="Path(s) to image file(s)")
    p.add_argument("--dir", metavar="DIR",
                   help="Process all PNG/JPG images in this directory")
    p.add_argument("--out-dir", metavar="OUT_DIR",
                   help="Output directory (default: same dir as input image)")

    # SAM 2 model
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Path to SAM 2 .pt checkpoint file")
    p.add_argument("--model-cfg",  default=DEFAULT_MODEL_CFG,
                   help="SAM 2 model config YAML (e.g. sam2_hiera_l.yaml)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Inference device")

    # SAM generator params
    p.add_argument("--points-per-side", type=int,   default=32)
    p.add_argument("--iou-thresh",      type=float, default=DEFAULT_IOU_THRESH,
                   help="SAM predicted_iou threshold")
    p.add_argument("--stability-thresh",type=float, default=DEFAULT_STABILITY,
                   help="SAM stability_score threshold")

    # Geometry filters
    p.add_argument("--min-area",   type=float, default=DEFAULT_MIN_AREA,
                   help="Min patch area as fraction of image area")
    p.add_argument("--max-area",   type=float, default=DEFAULT_MAX_AREA,
                   help="Max patch area as fraction of image area")
    p.add_argument("--max-aspect", type=float, default=DEFAULT_MAX_ASPECT,
                   help="Max aspect ratio (long side / short side)")
    p.add_argument("--nms-iou",    type=float, default=DEFAULT_NMS_IOU,
                   help="IoU threshold for NMS deduplication")
    p.add_argument("--pad",        type=float, default=DEFAULT_PAD,
                   help="Fractional padding added around each bbox")

    p.add_argument("--annotate-only", action="store_true",
                   help="Only produce annotated image, skip patch extraction")
    return p


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    if not SAM2_AVAILABLE:
        print(
            "ERROR: SAM 2 is not installed.\n"
            "Install with:  pip install segment-anything-2\n"
            "Then download a checkpoint:\n"
            "  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt\n"
            "Set SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt"
        )
        sys.exit(1)

    parser = build_parser()
    args   = parser.parse_args()

    # Collect image paths
    image_paths: list[Path] = []
    for s in args.images:
        p = Path(s)
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        image_paths.append(p)

    if args.dir:
        d = Path(args.dir)
        image_paths += sorted(d.glob("*.png")) + sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg"))

    if not image_paths:
        parser.error("No images provided. Use positional args or --dir.")

    # Build SAM 2 generator
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    sam2_model = build_sam2(args.model_cfg, args.checkpoint, device=device)
    generator  = SAM2AutomaticMaskGenerator(
        model                      = sam2_model,
        points_per_side            = args.points_per_side,
        pred_iou_thresh            = args.iou_thresh,
        stability_score_thresh     = args.stability_thresh,
        min_mask_region_area       = 400,   # absolute px² — drops sub-pixel noise
        output_mode                = "binary_mask",
    )

    # Process each image
    all_meta: list[dict] = []
    for image_path in image_paths:
        out_dir = Path(args.out_dir) if args.out_dir else image_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = process_image(image_path, generator, args, out_dir)
        all_meta.extend(meta)

    # Summary
    print(f"\n{'═'*60}")
    print(f"Processed {len(image_paths)} image(s), extracted {len(all_meta)} patches total")

    if all_meta and not args.annotate_only:
        summary_path = (Path(args.out_dir) if args.out_dir else image_paths[0].parent) / "extraction_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"total_patches": len(all_meta), "patches": all_meta}, f, indent=2, ensure_ascii=False)
        print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
