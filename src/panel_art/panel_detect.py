"""
panel_detect.py — Phase 2: split multi-panel images into individual ROI crops

Frobenius archive photos often show 2–3 physical door panels side-by-side
against a plain studio background. This module finds each panel's bounding
box and crops them out for independent analysis.

Strategy: traditional CV (no neural detector needed)
  1. Otsu threshold on inverted grayscale → binary mask of dark objects vs
     light background (the studio backdrop is always near-white)
  2. Morphological closing → fills holes within carved-surface texture
  3. connectedComponentsWithStats → per-blob bounding rects
  4. Filter by aspect ratio and minimum area
  5. Return axis-aligned crops (minAreaRect used internally to detect rotation,
     but final crops are axis-aligned for downstream compatibility)

For illustration images (EBA-B), which often have a single drawing on a
cream/grey paper field, this same approach works — it isolates the drawing
from the paper surround.

CLI usage:
  uv run python -m panel_art.panel_detect <image> [<image> ...]
  uv run python -m panel_art.panel_detect --out-dir /path/to/panels <image>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PanelROI:
    """Axis-aligned bounding box for one detected panel."""
    x: int
    y: int
    w: int
    h: int
    area_fraction: float   # fraction of total image area
    index: int             # 0-based, left-to-right order

    @property
    def aspect_ratio(self) -> float:
        return self.h / max(self.w, 1)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "x": self.x, "y": self.y, "w": self.w, "h": self.h,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "area_fraction": round(self.area_fraction, 4),
        }


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_panels(
    img_gray: np.ndarray,
    min_area_fraction: float = 0.04,
    max_area_fraction: float = 0.95,
    min_aspect: float = 0.8,    # h/w — panels are at least as tall as wide
    max_aspect: float = 15.0,   # very tall narrow panels are fine
    closing_px: int = 12,       # smaller kernel → less merging of nearby panels
) -> list[PanelROI]:
    """
    Find individual door-panel ROIs in a grayscale image.

    Parameters
    ----------
    min_area_fraction : blobs smaller than this fraction of total pixels are
                        discarded (removes noise, frame edges, small labels).
    max_area_fraction : blobs larger than this fraction are discarded
                        (prevents the whole image from being returned as one
                        blob when the background threshold fails).
    min_aspect        : minimum h/w ratio. Door panels are tall; very wide blobs
                        are usually spurious (paper edges, horizontal shadows).
    max_aspect        : maximum h/w ratio.
    closing_px        : morphological closing kernel size in pixels. Large enough
                        to bridge the inter-element gaps in dense carvings.
    """
    h, w = img_gray.shape
    total = h * w

    # ── Step 1: Otsu threshold ─────────────────────────────────────────────
    # Invert so panels (dark objects) become white blobs on black background.
    # THRESH_BINARY_INV + THRESH_OTSU finds the threshold automatically.
    _, binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Step 2: Morphological closing ─────────────────────────────────────
    # Close gaps within carved surfaces (the texture creates many small holes).
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_px, closing_px)
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Fill any remaining internal holes with floodfill from corners
    filled = closed.copy()
    cv2.floodFill(filled, None, (0, 0), 255)         # from top-left
    cv2.floodFill(filled, None, (w - 1, 0), 255)     # from top-right
    cv2.floodFill(filled, None, (0, h - 1), 255)     # from bottom-left
    cv2.floodFill(filled, None, (w - 1, h - 1), 255) # from bottom-right
    # Invert back and merge with original closed mask
    filled_inv = cv2.bitwise_not(filled)
    merged = cv2.bitwise_or(closed, filled_inv)

    # ── Step 3: Connected components ───────────────────────────────────────
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        merged, connectivity=8
    )

    rois: list[PanelROI] = []
    for label in range(1, num_labels):  # skip label 0 (background)
        sx = int(stats[label, cv2.CC_STAT_LEFT])
        sy = int(stats[label, cv2.CC_STAT_TOP])
        sw = int(stats[label, cv2.CC_STAT_WIDTH])
        sh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        frac = area / total
        aspect = sh / max(sw, 1)

        if frac < min_area_fraction:
            continue
        if frac > max_area_fraction:
            continue
        if aspect < min_aspect:
            continue
        if aspect > max_aspect:
            continue

        rois.append(PanelROI(
            x=sx, y=sy, w=sw, h=sh,
            area_fraction=frac,
            index=0,  # will be sorted and re-indexed below
        ))

    # Sort left-to-right by x position, then assign stable indices
    rois.sort(key=lambda r: r.x)
    for i, roi in enumerate(rois):
        roi.index = i

    # ── Step 4: Try to split wide blobs using vertical projection ─────────
    # If two panels are physically touching, closing will merge them into one
    # wide blob. The column-wise sum of dark pixels (vertical projection) will
    # have a valley at the gap between panels. Split blobs whose aspect ratio
    # is suspiciously wide (h/w < 1.5) at detected projection valleys.
    rois = _split_wide_blobs(rois, merged, min_aspect=min_aspect)

    # Re-sort and re-index after any splits
    rois.sort(key=lambda r: r.x)
    for i, roi in enumerate(rois):
        roi.index = i

    return rois


def _split_wide_blobs(
    rois: list[PanelROI],
    binary_mask: np.ndarray,
    min_aspect: float = 0.8,
    valley_relative_threshold: float = 0.80,
) -> list[PanelROI]:
    """
    Attempt to split merged blobs using vertical projection valleys.

    When two adjacent panels are touching, morphological closing merges them.
    The column-wise sum of foreground pixels has a local dip at the gap.
    We detect significant relative valleys — where the profile drops below
    80% of the surrounding peaks — and split the blob at those points.

    Only applied to large blobs (> 15% of image) that might be merged panels.

    Parameters
    ----------
    valley_relative_threshold : a valley is significant if its value is below
        this fraction of the profile's central peak. 0.80 means a 20% dip.
    """
    result = []
    for roi in rois:
        # Only attempt splitting on large blobs
        if roi.area_fraction < 0.15:
            result.append(roi)
            continue

        # Column-wise projection within the blob's bounding box
        region = binary_mask[roi.y: roi.y + roi.h, roi.x: roi.x + roi.w]
        col_sum = region.sum(axis=0).astype(np.float64)
        col_sum_norm = col_sum / (col_sum.max() + 1e-6)

        # Smooth to suppress carved-texture noise
        kernel_width = max(3, roi.w // 30)
        if kernel_width % 2 == 0:
            kernel_width += 1
        smoothed = cv2.GaussianBlur(
            col_sum_norm.reshape(1, -1).astype(np.float32),
            (1, kernel_width), 0,
        ).flatten()

        # Reference peak = max of the central 60% of the profile
        # (avoids being thrown off by edge zeros)
        centre_start = roi.w // 5
        centre_end = roi.w - roi.w // 5
        peak_val = float(smoothed[centre_start:centre_end].max())
        threshold = peak_val * valley_relative_threshold

        # Find valley columns
        valleys = np.where(smoothed < threshold)[0]
        # Ignore leading/trailing zeros (image border)
        interior_start = int(np.argmax(smoothed > 0.1))
        interior_end = roi.w - int(np.argmax(smoothed[::-1] > 0.1))
        valleys = valleys[(valleys > interior_start) & (valleys < interior_end)]

        if len(valleys) == 0:
            result.append(roi)
            continue

        # Group consecutive valley columns into contiguous gap ranges
        gaps = []
        start = int(valleys[0])
        for i in range(1, len(valleys)):
            if int(valleys[i]) > int(valleys[i - 1]) + 3:
                gaps.append((start, int(valleys[i - 1])))
                start = int(valleys[i])
        gaps.append((start, int(valleys[-1])))

        # Split at gap midpoints
        split_xs = sorted((a + b) // 2 for a, b in gaps)
        split_xs = [0] + split_xs + [roi.w]

        # Minimum width for a valid sub-panel: 10% of the parent blob width
        # (avoids retaining thin edge-gap slivers as "panels")
        min_sub_w = max(60, roi.w // 10)

        sub_rois = []
        for j in range(len(split_xs) - 1):
            x_start = roi.x + split_xs[j]
            x_end = roi.x + split_xs[j + 1]
            sw = x_end - x_start
            if sw < min_sub_w:
                continue
            frac = (sw * roi.h) / (binary_mask.shape[0] * binary_mask.shape[1])
            sub_rois.append(PanelROI(
                x=x_start, y=roi.y, w=sw, h=roi.h,
                area_fraction=frac,
                index=0,
            ))

        # Accept split only if all sub-blobs would pass the aspect ratio filter
        if len(sub_rois) > 1 and all(s.aspect_ratio >= min_aspect for s in sub_rois):
            result.extend(sub_rois)
        else:
            result.append(roi)

    return result


# ── Cropping ──────────────────────────────────────────────────────────────────

def crop_panels(
    image_path: str | Path,
    out_dir: str | Path,
    pad_fraction: float = 0.02,
    **detect_kwargs,
) -> list[dict]:
    """
    Detect panels in an image, crop each one, and save as PNG.

    Parameters
    ----------
    pad_fraction : fractional padding to add around each bounding box
                   (prevents clipping the panel edges).

    Returns a list of metadata dicts, one per saved crop.
    """
    path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    ih, iw = img_gray.shape

    rois = detect_panels(img_gray, **detect_kwargs)

    # Fallback: if nothing passes the filters, treat the whole image as one panel.
    # This handles single-object photos with unusual aspect ratios or very dark
    # backgrounds that confuse the Otsu threshold.
    if not rois:
        ih, iw = img_gray.shape
        rois = [PanelROI(x=0, y=0, w=iw, h=ih,
                         area_fraction=1.0, index=0)]

    results = []
    for roi in rois:
        # Add padding
        pad_x = int(roi.w * pad_fraction)
        pad_y = int(roi.h * pad_fraction)
        x1 = max(0, roi.x - pad_x)
        y1 = max(0, roi.y - pad_y)
        x2 = min(iw, roi.x + roi.w + pad_x)
        y2 = min(ih, roi.y + roi.h + pad_y)

        crop = img.crop((x1, y1, x2, y2))

        suffix = f"_panel_{roi.index:02d}.png"
        out_path = out_dir / (path.stem + suffix)
        crop.save(str(out_path), "PNG")

        results.append({
            "source": str(path),
            "panel_index": roi.index,
            "out_path": str(out_path),
            "bbox": {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
            "area_fraction": roi.area_fraction,
            "aspect_ratio": roi.aspect_ratio,
        })

    return results


# ── Debug annotation ──────────────────────────────────────────────────────────

def annotate_panels(
    image_path: str | Path,
    rois: list[PanelROI],
    out_path: str | Path,
) -> None:
    """Draw bounding boxes on the image and save for inspection."""
    img = cv2.imread(str(image_path))
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    colours = [
        (255, 80, 80), (80, 200, 80), (50, 150, 255),
        (200, 80, 255), (255, 165, 50),
    ]
    for roi in rois:
        colour = colours[roi.index % len(colours)]
        cv2.rectangle(img, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h),
                      colour, 3)
        label = f"panel {roi.index}  {roi.area_fraction*100:.1f}%  AR={roi.aspect_ratio:.1f}"
        cv2.putText(img, label, (roi.x + 4, roi.y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2: detect and crop individual panels from multi-panel images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="+", metavar="IMAGE")
    p.add_argument("--out-dir", default="frobenius_artifacts/analysis/panels")
    p.add_argument("--annotate", action="store_true",
                   help="Also save annotated version showing bounding boxes")
    p.add_argument("--min-area", type=float, default=0.04,
                   dest="min_area_fraction")
    p.add_argument("--min-aspect", type=float, default=0.8)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    for img_path in args.images:
        path = Path(img_path)
        img = Image.open(path).convert("RGB")
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        rois = detect_panels(
            img_gray,
            min_area_fraction=args.min_area_fraction,
            min_aspect=args.min_aspect,
        )

        print(f"{path.name}: {len(rois)} panel(s) detected")
        for r in rois:
            print(f"  panel {r.index}: {r.w}×{r.h}  "
                  f"area={r.area_fraction*100:.1f}%  AR={r.aspect_ratio:.2f}")

        results = crop_panels(img_path, args.out_dir,
                              min_area_fraction=args.min_area_fraction,
                              min_aspect=args.min_aspect)
        for res in results:
            print(f"  → {Path(res['out_path']).name}")

        if args.annotate:
            ann_path = Path(args.out_dir) / (path.stem + "_panels_annotated.jpg")
            annotate_panels(img_path, rois, ann_path)
            print(f"  → {ann_path.name}  (annotated)")

        print()


if __name__ == "__main__":
    main(sys.argv[1:])
