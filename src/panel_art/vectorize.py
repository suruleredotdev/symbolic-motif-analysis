"""
vectorize.py — Phase 4: motif region → normalised SVG

Converts each detected motif bounding box into a compact, line-focused SVG
that can be compared by shape similarity.

Pipeline per motif:
  1. Crop the motif region from the Phase 1 line-art image
  2. Otsu threshold → binary (dark lines on white background)
  3. OpenCV findContours → list of polyline contours
  4. Ramer-Douglas-Peucker simplification (reduces noise / point count)
  5. svgwrite → <path> elements, stroke-only, no fill
  6. Normalise viewBox to 100×100 (scale/position invariant)

Why OpenCV contours + svgwrite over pypotrace:
  pypotrace needs the potrace *development libraries* (not just the binary) to
  build, and they're not always available on macOS without extra Homebrew steps.
  OpenCV contour→SVG gives us direct control over simplification and produces
  compact, stroke-only paths that are ideal for shape comparison.

CLI usage:
  uv run python -m panel_art.vectorize \\
    --panel   frobenius_artifacts/analysis/panels/FoA_..._panel_01.png \\
    --lineart frobenius_artifacts/analysis/line_art/FoA_..._panel_01_lineart.png \\
    --detections frobenius_artifacts/analysis/annotated/FoA_..._panel_01_detections.json \\
    --out-dir frobenius_artifacts/analysis/motifs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import svgwrite

# ── Defaults ──────────────────────────────────────────────────────────────────

VIEWBOX_SIZE = 100          # normalised coordinate space (100×100 units)
DEFAULT_EPSILON = 1.5       # RDP simplification tolerance in pixels
DEFAULT_MIN_CONTOUR_AREA = 4  # ignore contours smaller than this (px²)


# ── Core conversion ───────────────────────────────────────────────────────────

def region_to_svg(
    lineart: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    epsilon: float = DEFAULT_EPSILON,
    min_contour_area: float = DEFAULT_MIN_CONTOUR_AREA,
    viewbox_size: int = VIEWBOX_SIZE,
) -> str:
    """
    Crop [x,y,w,h] from the line-art image and return a normalised SVG string.

    Parameters
    ----------
    lineart          : H×W uint8 greyscale line-art from Phase 1 (white bg,
                       dark lines). Can be the full panel image — we crop here.
    x, y, w, h       : bounding box of the motif within `lineart`
    epsilon          : Ramer-Douglas-Peucker tolerance (px). Higher = simpler
                       paths, fewer points. 1.5 works well for ~500px motifs.
    min_contour_area : contours smaller than this (px²) are discarded as noise
    viewbox_size     : normalised output coordinate space (100 = 100×100 units)

    Returns an SVG string with <path> elements for each contour.
    Stroke is black (#000), fill is none. ViewBox is 0 0 100 100.
    """
    if w <= 0 or h <= 0:
        return _empty_svg(viewbox_size)

    # 1. Crop and invert (findContours expects dark background, white objects)
    crop = lineart[y: y + h, x: x + w]
    if crop.size == 0:
        return _empty_svg(viewbox_size)

    # Line-art has dark lines (0) on white (255); invert so lines are white
    inverted = cv2.bitwise_not(crop)

    # Threshold to binary (should already be near-binary from Phase 1)
    _, binary = cv2.threshold(inverted, 127, 255, cv2.THRESH_BINARY)

    # 2. Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 3. Filter and simplify
    scale_x = viewbox_size / w
    scale_y = viewbox_size / h

    drawing = svgwrite.Drawing(size=(f"{viewbox_size}", f"{viewbox_size}"))
    drawing.viewbox(0, 0, viewbox_size, viewbox_size)

    path_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        # RDP simplification — closed=True treats the contour as a closed curve
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(simplified) < 2:
            continue

        # Scale points to viewbox coordinates
        pts = [
            (float(pt[0][0] * scale_x), float(pt[0][1] * scale_y))
            for pt in simplified
        ]

        # Build SVG path string: M x,y L x,y ... Z
        d = f"M {pts[0][0]:.2f},{pts[0][1]:.2f} "
        d += " ".join(f"L {p[0]:.2f},{p[1]:.2f}" for p in pts[1:])
        d += " Z"

        drawing.add(drawing.path(d=d, fill="none", stroke="#000",
                                 stroke_width=0.5))
        path_count += 1

    if path_count == 0:
        return _empty_svg(viewbox_size)

    return drawing.tostring()


def _empty_svg(size: int) -> str:
    """Return a minimal empty SVG (used when no contours are found)."""
    d = svgwrite.Drawing(size=(f"{size}", f"{size}"))
    d.viewbox(0, 0, size, size)
    return d.tostring()


# ── Batch processing ──────────────────────────────────────────────────────────

def vectorize_detections(
    panel_stem: str,
    lineart_path: str | Path,
    detections: list[dict],
    out_dir: str | Path,
    **kwargs,
) -> list[dict]:
    """
    For each detection bounding box, produce one SVG file.

    Parameters
    ----------
    panel_stem   : filename stem used as prefix for output SVG names
    lineart_path : Phase 1 output — full-panel line-art image
    detections   : list of detection dicts from Phase 3
                   (each must have 'bbox': {x, y, w, h} and 'index')
    out_dir      : directory to write SVG files into

    Returns the input detection list enriched with 'svg_path' keys.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lineart = cv2.imread(str(lineart_path), cv2.IMREAD_GRAYSCALE)
    if lineart is None:
        raise FileNotFoundError(f"Line-art image not found: {lineart_path}")

    results = []
    for det in detections:
        bbox = det["bbox"]
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        idx = det["index"]

        svg_str = region_to_svg(lineart, x, y, w, h, **kwargs)

        out_path = out_dir / f"{panel_stem}_motif_{idx:03d}.svg"
        out_path.write_text(svg_str, encoding="utf-8")

        results.append({**det, "svg_path": str(out_path)})

    return results


def vectorize_from_files(
    panel_path: str | Path,
    lineart_path: str | Path,
    detections_json: str | Path,
    out_dir: str | Path,
    **kwargs,
) -> list[dict]:
    """Convenience wrapper: load detection JSON, run vectorize_detections."""
    with open(detections_json) as f:
        detections = json.load(f)

    stem = Path(panel_path).stem
    return vectorize_detections(stem, lineart_path, detections, out_dir, **kwargs)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 4: convert motif bounding boxes to normalised SVGs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--panel",       required=True, metavar="PNG",
                   help="Panel crop image (Phase 2 output)")
    p.add_argument("--lineart",     required=True, metavar="PNG",
                   help="Line-art image for this panel (Phase 1 output)")
    p.add_argument("--detections",  required=True, metavar="JSON",
                   help="Detections JSON (Phase 3 output)")
    p.add_argument("--out-dir",
                   default="frobenius_artifacts/analysis/motifs")
    p.add_argument("--epsilon",     type=float, default=DEFAULT_EPSILON,
                   help="RDP simplification tolerance (pixels)")
    p.add_argument("--min-area",    type=float, default=DEFAULT_MIN_CONTOUR_AREA,
                   help="Minimum contour area to keep (px²)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    results = vectorize_from_files(
        panel_path=args.panel,
        lineart_path=args.lineart,
        detections_json=args.detections,
        out_dir=args.out_dir,
        epsilon=args.epsilon,
        min_contour_area=args.min_area,
    )

    print(f"Vectorised {len(results)} motifs → {args.out_dir}/")
    sizes = []
    for r in results:
        sz = Path(r["svg_path"]).stat().st_size
        sizes.append(sz)
        print(f"  {Path(r['svg_path']).name}  {sz//1024}KB  scale={r['scale']}")

    if sizes:
        print(f"\nSize range: {min(sizes)//1024}KB – {max(sizes)//1024}KB")


if __name__ == "__main__":
    main(sys.argv[1:])
