#!/usr/bin/env python3
"""
normalize_motifs.py — normalise motif crops to a medium-agnostic representation.

Reads  frobenius_artifacts/analysis/motifs/<panel>/<crop>.png
Writes frobenius_artifacts/analysis/motifs_norm/<panel>/<crop>.png

Two modes:

  sketch (default)
    Shadow removal + contrast normalisation + soft DoG edge emphasis.
    Retains tonal information; looks like a well-scanned photograph.

  lines
    Isolates carved grooves as dark strokes on a white ground — a true
    bitmap / technical-drawing look.  Pipeline:
      retinex → CLAHE → bilateral → adaptive threshold → morphological clean.
    Works uniformly across photographs of relief boards, B&W archival scans,
    and ink drawings because adaptive thresholding is local (handles residual
    brightness gradients) and the threshold width parameter is tuned to the
    groove width rather than pixel transitions.

Usage:
    uv run --project src/python python src/python/normalize_motifs.py
    uv run --project src/python python src/python/normalize_motifs.py --mode lines

Options:
    --in-dir PATH          Source motifs root   (default: auto-resolved)
    --out-dir PATH         Output root          (default: auto-resolved)
    --mode MODE            sketch | lines       (default: sketch)

  Shared parameters (both modes):
    --shadow-sigma FLOAT   Retinex blur sigma; larger removes broader shadows
                           (default: 40)
    --clahe-clip FLOAT     CLAHE clipLimit  (default: 2.5)
    --bilateral-d INT      Bilateral filter diameter  (default: 7)

  sketch-only parameters:
    --dog-sigma-fine F     DoG fine-scale sigma   (default: 1.5)
    --dog-sigma-coarse F   DoG coarse-scale sigma (default: 8.0)
    --dog-weight FLOAT     DoG blend strength 0–1 (default: 0.7)

  lines-only parameters:
    --block-size INT       Adaptive threshold neighbourhood (odd, default: 31)
    --adapt-c FLOAT        Threshold constant subtracted from local mean
                           (default: 8); higher = fewer lines detected
    --morph-open INT       Morphological opening radius for noise cleanup
                           (default: 1; 0 = disabled)

    --dry-run              Print planned outputs without writing anything
    --force                Re-write files that already exist
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Repo-relative defaults ────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_REPO     = _HERE.parent.parent
_ANALYSIS = _REPO / "frobenius_artifacts/analysis"

DEFAULT_IN_DIR  = _ANALYSIS / "motifs"
DEFAULT_OUT_DIR = _ANALYSIS / "motifs_norm"


# ── Shared preprocessing steps ────────────────────────────────────────────────

def _shared_preprocess(
    img_rgb: np.ndarray,
    shadow_sigma: float,
    clahe_clip: float,
    bilateral_d: int,
) -> np.ndarray:
    """
    Steps shared by both modes:
      greyscale → percentile-stretch → retinex → CLAHE → bilateral smooth
    Returns a uint8 greyscale array.
    """
    # Greyscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Percentile stretch
    lo, hi = np.percentile(gray, [2, 98])
    if hi > lo:
        gray = np.clip((gray - lo) / (hi - lo) * 255.0, 0, 255)
    gray = gray.astype(np.uint8)

    # Retinex — divide by large-sigma blur to cancel shadow gradients
    gray_f  = gray.astype(np.float32)
    illum   = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=shadow_sigma)
    retinex = np.clip(gray_f / (illum + 1.0) * 128.0, 0, 255).astype(np.uint8)

    # CLAHE
    clahe  = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
    eq     = clahe.apply(retinex)

    # Bilateral smooth
    smooth = cv2.bilateralFilter(eq, d=bilateral_d,
                                 sigmaColor=35, sigmaSpace=35)
    return smooth, clahe   # return clahe object so callers can reuse it


# ── Mode: sketch ──────────────────────────────────────────────────────────────

def normalize_crop_sketch(
    img_rgb: np.ndarray,
    shadow_sigma: float     = 40.0,
    clahe_clip: float       = 2.5,
    bilateral_d: int        = 7,
    dog_sigma_fine: float   = 1.5,
    dog_sigma_coarse: float = 8.0,
    dog_weight: float       = 0.7,
) -> np.ndarray:
    """
    Soft normalisation: tonal range equalised, shadows removed, edges gently
    emphasised via DoG.  Retains photographic texture.
    """
    smooth, clahe = _shared_preprocess(img_rgb, shadow_sigma, clahe_clip, bilateral_d)

    g_fine   = cv2.GaussianBlur(smooth, (0, 0), sigmaX=dog_sigma_fine).astype(np.float32)
    g_coarse = cv2.GaussianBlur(smooth, (0, 0), sigmaX=dog_sigma_coarse).astype(np.float32)
    dog      = g_fine - g_coarse
    blended  = np.clip(smooth.astype(np.float32) - dog_weight * dog, 0, 255).astype(np.uint8)
    result   = clahe.apply(blended)

    return np.stack([result, result, result], axis=-1)


# ── Mode: lines ───────────────────────────────────────────────────────────────

def normalize_crop_lines(
    img_rgb: np.ndarray,
    shadow_sigma: float = 40.0,
    clahe_clip: float   = 2.5,
    bilateral_d: int    = 7,
    block_size: int     = 31,
    adapt_c: float      = 8.0,
    morph_open: int     = 1,
) -> np.ndarray:
    """
    Bitmap / line-isolation mode: carved grooves → dark strokes on white ground.

    Adaptive Gaussian thresholding is local (block_size × block_size window),
    so it handles any residual brightness gradient after retinex and produces
    uniform-width line strokes regardless of surface reflectance variation.

    block_size controls the neighbourhood — should be larger than the widest
    flat carved face and smaller than the motif itself.
    adapt_c is subtracted from the local mean before thresholding; higher
    values reject more of the low-contrast noise and keep only strong grooves.
    """
    smooth, _ = _shared_preprocess(img_rgb, shadow_sigma, clahe_clip, bilateral_d)

    # Ensure block_size is odd (required by OpenCV)
    bs = block_size if block_size % 2 == 1 else block_size + 1

    # Adaptive threshold: pixels below local_mean - adapt_c → black (groove)
    # THRESH_BINARY produces white background, black lines
    binary = cv2.adaptiveThreshold(
        smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs, adapt_c,
    )

    # Morphological opening: remove isolated noise pixels smaller than the
    # structuring element without affecting the groove strokes themselves
    if morph_open > 0:
        r      = morph_open
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return np.stack([binary, binary, binary], axis=-1)


# ── Unified entry point ───────────────────────────────────────────────────────

def normalize_crop(img_rgb: np.ndarray, mode: str = "sketch", **kwargs) -> np.ndarray:
    """
    Dispatch to normalize_crop_sketch or normalize_crop_lines based on mode.
    Extra kwargs are forwarded to the chosen function.
    """
    if mode == "sketch":
        return normalize_crop_sketch(img_rgb, **kwargs)
    elif mode == "lines":
        return normalize_crop_lines(img_rgb, **kwargs)
    else:
        raise ValueError(f"Unknown mode {mode!r} — expected 'sketch' or 'lines'")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir",           type=Path,  default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir",          type=Path,  default=DEFAULT_OUT_DIR)
    p.add_argument("--mode",             default="sketch",
                   choices=["sketch", "lines"],
                   help="Output mode: sketch (soft) or lines (bitmap, default: sketch)")
    # shared
    p.add_argument("--shadow-sigma",     type=float, default=40.0)
    p.add_argument("--clahe-clip",       type=float, default=2.5)
    p.add_argument("--bilateral-d",      type=int,   default=7)
    # sketch
    p.add_argument("--dog-sigma-fine",   type=float, default=1.5)
    p.add_argument("--dog-sigma-coarse", type=float, default=8.0)
    p.add_argument("--dog-weight",       type=float, default=0.7)
    # lines
    p.add_argument("--block-size",       type=int,   default=31,
                   help="Adaptive threshold block size (odd, default: 31)")
    p.add_argument("--adapt-c",          type=float, default=8.0,
                   help="Threshold constant (default: 8)")
    p.add_argument("--morph-open",       type=int,   default=1,
                   help="Morphological open radius; 0=disabled (default: 1)")
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--force",            action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.in_dir.exists():
        sys.exit(f"Input directory not found: {args.in_dir}")

    crops = sorted(args.in_dir.rglob("*.png"))
    if not crops:
        sys.exit(f"No PNG crops found in: {args.in_dir}")

    print(f"Source : {args.in_dir}  ({len(crops)} crops)")
    print(f"Output : {args.out_dir}")
    print(f"Mode   : {args.mode}")
    if args.mode == "sketch":
        print(f"Params : shadow_sigma={args.shadow_sigma}  clahe_clip={args.clahe_clip}"
              f"  bilateral_d={args.bilateral_d}"
              f"  dog=({args.dog_sigma_fine},{args.dog_sigma_coarse})×{args.dog_weight}")
    else:
        print(f"Params : shadow_sigma={args.shadow_sigma}  clahe_clip={args.clahe_clip}"
              f"  bilateral_d={args.bilateral_d}"
              f"  block_size={args.block_size}  adapt_c={args.adapt_c}"
              f"  morph_open={args.morph_open}")
    if args.dry_run:
        print("[dry-run — nothing will be written]\n")

    written = skipped = 0

    for src in crops:
        rel = src.relative_to(args.in_dir)
        dst = args.out_dir / rel

        if dst.exists() and not args.force and not args.dry_run:
            skipped += 1
            continue

        if args.dry_run:
            print(f"  [dry-run] {rel}")
            written += 1
            continue

        img_rgb = np.array(Image.open(src).convert("RGB"))

        if args.mode == "sketch":
            out_arr = normalize_crop_sketch(
                img_rgb,
                shadow_sigma    = args.shadow_sigma,
                clahe_clip      = args.clahe_clip,
                bilateral_d     = args.bilateral_d,
                dog_sigma_fine  = args.dog_sigma_fine,
                dog_sigma_coarse= args.dog_sigma_coarse,
                dog_weight      = args.dog_weight,
            )
        else:
            out_arr = normalize_crop_lines(
                img_rgb,
                shadow_sigma = args.shadow_sigma,
                clahe_clip   = args.clahe_clip,
                bilateral_d  = args.bilateral_d,
                block_size   = args.block_size,
                adapt_c      = args.adapt_c,
                morph_open   = args.morph_open,
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_arr).save(dst)
        written += 1

    action = "[dry-run] would write" if args.dry_run else "wrote"
    print(f"\n{action} {written} crop(s)"
          + (f", skipped {skipped} existing" if skipped else ""))


if __name__ == "__main__":
    main()
