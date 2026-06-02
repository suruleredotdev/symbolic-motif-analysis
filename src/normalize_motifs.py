#!/usr/bin/env python3
"""
normalize_motifs.py — normalise motif crops to a medium-agnostic representation.

Reads  frobenius_artifacts/analysis/motifs/<panel>/<crop>.png
Writes frobenius_artifacts/analysis/motifs_norm/<panel>/<crop>.png

The pipeline converts all crops — colour photographs of relief boards, B&W
archival scans, and ink/pencil drawings — into a common illustration-like
visual space so that CLIP embeddings reflect carved structure rather than
photographic artefacts (shadows, grain, tint, exposure).

Pipeline (applied to every crop):
  1. Greyscale          — removes all colour/tint variation
  2. Percentile stretch — clips [2nd, 98th] percentile → 0–255;
                          handles underexposed / blown-out images
  3. Retinex            — divides by a large-sigma Gaussian blur to cancel
                          slowly-varying shadow gradients from directional
                          lighting on 3D relief surfaces; preserves the
                          high-frequency carved-edge structure
  4. CLAHE              — local contrast equalisation
  5. Bilateral smooth   — removes photographic grain while preserving edges
  6. DoG blend          — Difference-of-Gaussians at carved-edge scale;
                          brightens flat areas, darkens carved lines →
                          illustration-like appearance
  7. Final CLAHE        — second local-contrast pass on the blended result

Usage:
    uv run --project src/python python src/python/normalize_motifs.py

Options:
    --in-dir PATH         Source motifs root   (default: auto-resolved)
    --out-dir PATH        Output root          (default: auto-resolved)
    --shadow-sigma FLOAT  Retinex blur sigma — controls shadow scale removed;
                          larger = removes broader shadow gradients (default: 40)
    --clahe-clip FLOAT    CLAHE clipLimit      (default: 2.5)
    --bilateral-d INT     Bilateral filter diameter  (default: 7)
    --dog-sigma-fine F    DoG fine-scale sigma  (default: 1.5)
    --dog-sigma-coarse F  DoG coarse-scale sigma (default: 8.0)
    --dog-weight FLOAT    DoG blend strength 0–1  (default: 0.7)
    --dry-run             Print what would be written without writing anything
    --force               Re-write files that already exist
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Repo-relative defaults ────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent          # src/python/
_REPO     = _HERE.parent.parent                      # repo root
_ANALYSIS = _REPO / "frobenius_artifacts/analysis"

DEFAULT_IN_DIR  = _ANALYSIS / "motifs"
DEFAULT_OUT_DIR = _ANALYSIS / "motifs_norm"


# ── Normalisation pipeline ────────────────────────────────────────────────────

def normalize_crop(
    img_rgb: np.ndarray,
    shadow_sigma: float  = 40.0,
    clahe_clip: float    = 2.5,
    bilateral_d: int     = 7,
    dog_sigma_fine: float   = 1.5,
    dog_sigma_coarse: float = 8.0,
    dog_weight: float    = 0.7,
) -> np.ndarray:
    """
    Apply the full normalisation pipeline to one RGB crop.
    Returns a 3-channel uint8 array (channels are identical; kept 3-ch for
    direct CLIP compatibility).
    """
    # 1. Greyscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 2. Percentile stretch — robust to extreme highlights/shadows
    lo, hi = np.percentile(gray, [2, 98])
    if hi > lo:
        gray = np.clip((gray - lo) / (hi - lo) * 255.0, 0, 255)
    gray = gray.astype(np.uint8)

    # 3. Retinex shadow removal
    #    Divide by a large Gaussian blur (the estimated local illumination).
    #    This cancels slowly-varying cast shadows from directional lighting
    #    across 3D relief surfaces, while preserving the high-frequency
    #    carved-edge structure.  Scale back to 0-255 via the midpoint 128.
    gray_f  = gray.astype(np.float32)
    illum   = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=shadow_sigma)
    retinex = np.clip(gray_f / (illum + 1.0) * 128.0, 0, 255).astype(np.uint8)

    # 4. CLAHE — local contrast equalisation
    clahe   = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
    eq      = clahe.apply(retinex)

    # 5. Bilateral smooth — denoise while preserving edge sharpness
    smooth  = cv2.bilateralFilter(eq, d=bilateral_d,
                                  sigmaColor=35, sigmaSpace=35)

    # 6. DoG blend — emphasise carved-edge scale lines
    #    sigma_fine  ≈ width of a single carved line (1–2 px after stretch)
    #    sigma_coarse ≈ width of the illuminated face between lines (6–10 px)
    #    dog is positive at line edges → subtracting it darkens those edges
    g_fine   = cv2.GaussianBlur(smooth, (0, 0), sigmaX=dog_sigma_fine).astype(np.float32)
    g_coarse = cv2.GaussianBlur(smooth, (0, 0), sigmaX=dog_sigma_coarse).astype(np.float32)
    dog      = g_fine - g_coarse          # positive at edges / lines
    blended  = smooth.astype(np.float32) - dog_weight * dog
    blended  = np.clip(blended, 0, 255).astype(np.uint8)

    # 7. Final CLAHE pass on blended result
    result = clahe.apply(blended)

    return np.stack([result, result, result], axis=-1)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir",           type=Path,  default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir",          type=Path,  default=DEFAULT_OUT_DIR)
    p.add_argument("--shadow-sigma",     type=float, default=40.0,
                   help="Retinex blur sigma for shadow removal (default: 40)")
    p.add_argument("--clahe-clip",       type=float, default=2.5,
                   help="CLAHE clipLimit (default: 2.5)")
    p.add_argument("--bilateral-d",      type=int,   default=7,
                   help="Bilateral filter diameter (default: 7)")
    p.add_argument("--dog-sigma-fine",   type=float, default=1.5,
                   help="DoG fine-scale sigma (default: 1.5)")
    p.add_argument("--dog-sigma-coarse", type=float, default=8.0,
                   help="DoG coarse-scale sigma (default: 8.0)")
    p.add_argument("--dog-weight",       type=float, default=0.7,
                   help="DoG blend strength 0–1 (default: 0.7)")
    p.add_argument("--dry-run",          action="store_true",
                   help="Print planned outputs without writing anything")
    p.add_argument("--force",            action="store_true",
                   help="Overwrite files that already exist")
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
    print(f"Params : shadow_sigma={args.shadow_sigma}  clahe_clip={args.clahe_clip}"
          f"  bilateral_d={args.bilateral_d}"
          f"  dog=({args.dog_sigma_fine},{args.dog_sigma_coarse})×{args.dog_weight}")
    if args.dry_run:
        print("[dry-run mode — nothing will be written]\n")

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
        out_arr = normalize_crop(
            img_rgb,
            shadow_sigma=args.shadow_sigma,
            clahe_clip=args.clahe_clip,
            bilateral_d=args.bilateral_d,
            dog_sigma_fine=args.dog_sigma_fine,
            dog_sigma_coarse=args.dog_sigma_coarse,
            dog_weight=args.dog_weight,
        )

        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_arr).save(dst)
        written += 1

    action = "[dry-run] would write" if args.dry_run else "wrote"
    print(f"\n{action} {written} crop(s)"
          + (f", skipped {skipped} existing" if skipped else ""))


if __name__ == "__main__":
    main()
