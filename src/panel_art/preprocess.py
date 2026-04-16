"""
preprocess.py — Phase 1: image boundary-ification

Converts source images (aged photographs or hand-drawn illustrations) into
clean line-art suitable for motif segmentation.

Two paths:
  Photo (FoA_*)        → grayscale → CLAHE → bilateral filter → XDoG
  Illustration (EBA-*) → grayscale → Otsu threshold → morphological clean

Image type is detected automatically by looking at the brightness distribution:
illustrations on paper have a high fraction of near-white pixels and very sharp
edges (high Laplacian variance), while aged photographs of carved wood have
a more uniform mid-tone histogram.

XDoG reference:
  Winnemöller et al., "XDoG: An eXtended difference-of-Gaussians compendium
  including advanced image stylization", Computers & Graphics, 2012.

CLI usage:
  uv run python -m panel_art.preprocess <image> [<image> ...]
  uv run python -m panel_art.preprocess --out-dir /path/to/out <image>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Type detection ────────────────────────────────────────────────────────────

def detect_image_type(img_gray: np.ndarray) -> str:
    """
    Return 'photo' or 'illustration' by inspecting the brightness distribution.

    Illustrations (pen-and-ink on paper) have:
      • > 40 % of pixels brighter than 220 (white paper background)
      • High Laplacian variance (crisp ink strokes)

    Photos of carved wood have:
      • Mostly mid-tone pixels (grey wood surface)
      • Lower, more distributed Laplacian variance
    """
    h, w = img_gray.shape
    total = h * w

    white_fraction = float(np.sum(img_gray > 220)) / total

    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    lap_var = float(lap.var())

    is_illustration = white_fraction > 0.40 and lap_var > 400
    return "illustration" if is_illustration else "photo"


# ── Photo path: XDoG ──────────────────────────────────────────────────────────

def xdog(
    img_gray: np.ndarray,
    sigma1: float = 0.5,
    sigma2: float = 5.0,
    epsilon: float = 0.98,
    phi: float = 200.0,
    tau: float = 0.01,
) -> np.ndarray:
    """
    Extended Difference of Gaussians.

    Produces illustration-quality closed-contour line drawings from photos.
    Returns a uint8 image where lines are dark (0) on a white (255) background.

    Parameters
    ----------
    sigma1  : fine-detail Gaussian std. Smaller = finer lines.
    sigma2  : coarse-structure Gaussian std. sigma2 / sigma1 controls the
              scale of suppressed texture (ratio ~10 works well for carved wood).
    epsilon : mixing ratio. 1.0 = pure DoG; < 1.0 retains some coarse structure.
    phi     : sharpness of the soft threshold (tanh slope). Higher = harder lines.
    tau     : threshold centre for soft step. Keep near 0 — the diff values from
              G(σ1) - ε·G(σ2) are in the range [-0.3, 0.3], so tau=0.01 means
              "treat slight positive differences as background, everything else
              as a line".

    How the math works
    ------------------
    At a smooth background pixel: G(σ1) ≈ G(σ2), diff ≈ (1-ε)·G ≈ 0.02 > tau
    → result = 1.0 → output 255 (white).

    At a carved-groove edge pixel: G(σ1) < G(σ2) in the groove,
    diff becomes negative → result → 0 → output 0 (dark line).

    The tanh roll-off between 1 and 0 gives smooth, anti-aliased strokes
    rather than the jagged binary edges that Canny produces.
    """
    f = img_gray.astype(np.float64) / 255.0

    g1 = cv2.GaussianBlur(f, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(f, (0, 0), sigma2)

    diff = g1 - epsilon * g2

    # result = 1 (white/background) where diff >= tau;
    # rolls toward 0 (dark/line) as diff drops below tau.
    result = np.where(
        diff >= tau,
        np.ones_like(diff),
        1.0 + np.tanh(phi * (diff - tau)),
    )
    result = np.clip(result, 0.0, 1.0)

    # result=1 → 255 (white background), result=0 → 0 (dark line). No inversion.
    return (result * 255).astype(np.uint8)


def preprocess_photo(img_gray: np.ndarray) -> np.ndarray:
    """
    Full photo → line-art pipeline.

    Steps:
      1. CLAHE — normalises contrast across the image (handles uneven lighting
         common in 1910 field photography).
      2. Bilateral filter — smooths texture noise while preserving sharp
         carved-relief edges (σ_color=75, σ_space=75 is a good starting point).
      3. XDoG — extracts clean boundary strokes.
    """
    # 1. CLAHE: clip_limit=2 to avoid over-amplifying grain
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    # 2. Bilateral filter: d=9 pixel diameter; preserves edges while smoothing
    #    within-region texture noise
    smooth = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. XDoG
    return xdog(smooth)


# ── Illustration path ─────────────────────────────────────────────────────────

def preprocess_illustration(img_gray: np.ndarray) -> np.ndarray:
    """
    Illustration → clean binary image.

    EBA-B drawings are already line-art, so we skip XDoG and just:
      1. Otsu threshold → binary (dark ink on white)
      2. Morphological close → fill tiny gaps in ink strokes

    Returns dark lines on white background (same convention as photo path).
    """
    # Otsu's threshold
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close small gaps (3×3 kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closed


# ── Top-level entry point ─────────────────────────────────────────────────────

def preprocess(image_path: str | Path) -> tuple[np.ndarray, str]:
    """
    Load an image, auto-detect its type, and return the line-art result.

    Returns
    -------
    (line_art_uint8, image_type)
        line_art_uint8 : H×W uint8 array, dark lines on white background
        image_type     : 'photo' or 'illustration'
    """
    path = Path(image_path)
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    # Convert to grayscale (weighted luminance)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    image_type = detect_image_type(img_gray)

    if image_type == "photo":
        line_art = preprocess_photo(img_gray)
    else:
        line_art = preprocess_illustration(img_gray)

    return line_art, image_type


def preprocess_to_file(
    image_path: str | Path,
    out_dir: str | Path,
) -> dict:
    """
    Run preprocess() and save result as PNG.

    Returns a dict with: source, out_path, image_type, shape.
    """
    path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_art, image_type = preprocess(path)

    out_path = out_dir / f"{path.stem}_lineart.png"
    cv2.imwrite(str(out_path), line_art)

    return {
        "source": str(path),
        "out_path": str(out_path),
        "image_type": image_type,
        "shape": line_art.shape,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: convert panel images to XDoG line-art",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="+", metavar="IMAGE", help="Input image path(s)")
    p.add_argument(
        "--out-dir",
        default="frobenius_artifacts/analysis/line_art",
        help="Directory for output PNG files",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    print(f"Output dir: {args.out_dir}")
    print()

    for img_path in args.images:
        result = preprocess_to_file(img_path, args.out_dir)
        print(
            f"[{result['image_type']:>12s}]  {Path(img_path).name}"
            f"  →  {Path(result['out_path']).name}"
            f"  shape={result['shape']}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
