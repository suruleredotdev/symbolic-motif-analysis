"""
pipeline.py — end-to-end orchestration of Phases 1–5

Processes only images listed in frobenius_panel_art.json (the curated 108
panel-art records). Images in the raw image directory that are not referenced
by that metadata file — expedition field photos, village scenes, portraits —
are silently skipped.

Full pipeline per image:
  Phase 1: preprocess whole image → line-art PNG
  Phase 2: detect panels → individual panel crops
  For each panel crop:
    Phase 1: preprocess crop → line-art PNG
    Phase 3: SAM segmentation → detections JSON + annotated JPEG
    Phase 4: vectorize detections → SVG files
  Phase 5: cluster all SVGs across the collection → clusters.json

Output tree (all under <out_dir>/):
  line_art/          Phase 1 PNGs
  panels/            Phase 2 panel crops
  annotated/         Phase 3 annotated JPEGs + detections JSONs
  motifs/            Phase 4 SVG files
  clusters/          Phase 5 clusters.json + similarity_graph.json

CLI usage:
  # Process all panel-art images
  uv run python -m panel_art.pipeline \\
    --image-dir frobenius_artifacts/images/ \\
    --metadata  src/typescript/backend/lib/data/frobenius_panel_art.json \\
    --img-manifest frobenius_artifacts/images/manifest.json \\
    --checkpoint src/python/sam_vit_b_01ec64.pth \\
    --out-dir   frobenius_artifacts/analysis/

  # Single image (useful for testing)
  uv run python -m panel_art.pipeline \\
    --images frobenius_artifacts/images/FoA_04-5578_Modakeke_(Ife)_q48628_i1.png \\
    --metadata src/typescript/backend/lib/data/frobenius_panel_art.json \\
    --img-manifest frobenius_artifacts/images/manifest.json \\
    --checkpoint src/python/sam_vit_b_01ec64.pth \\
    --out-dir frobenius_artifacts/analysis/

  # Skip clustering (useful when iterating on Phases 1–4)
  uv run python -m panel_art.pipeline ... --no-cluster
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .preprocess import preprocess_to_file
from .panel_detect import crop_panels
from .motif_segment import load_generator, segment_to_files
from .vectorize import vectorize_detections
from .similarity import build_feature_matrix, cluster_motifs, annotate_clusters


# ── Metadata loading ──────────────────────────────────────────────────────────

def load_allowed_images(
    metadata_json: str | Path,
    img_manifest: str | Path,
    image_dir: str | Path,
) -> list[dict]:
    """
    Return a list of dicts with keys:
      path           — absolute Path to the local image file
      registration   — registration number (e.g. "FoA 04-5578")
      record         — full metadata record from frobenius_panel_art.json

    Only images that are:
      (a) referenced in frobenius_panel_art.json, AND
      (b) present on disk
    are included. Non-panel-art images in the directory are excluded.
    """
    with open(metadata_json) as f:
        panel_art = json.load(f)
    with open(img_manifest) as f:
        manifest = json.load(f)

    reg_to_record = {r.get("registration_number", ""): r
                     for r in panel_art.get("records", [])}
    reg_to_filename = {m["registration_number"]: m["filename"]
                       for m in manifest}

    image_dir = Path(image_dir)
    allowed = []
    for reg, record in reg_to_record.items():
        filename = reg_to_filename.get(reg)
        if not filename:
            continue
        path = image_dir / filename
        if not path.exists():
            continue
        allowed.append({
            "path": path,
            "registration": reg,
            "record": record,
        })

    return allowed


# ── Per-image processing ──────────────────────────────────────────────────────

def process_image(
    img_entry: dict,
    out_dir: Path,
    sam_generator,
    verbose: bool = True,
) -> list[str]:
    """
    Run Phases 1–4 for a single source image.
    Returns a list of SVG paths produced.
    """
    src_path = img_entry["path"]
    reg = img_entry["registration"]
    record = img_entry["record"]
    cats = record.get("categories", [])

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {src_path.name}")
        print(f"  {reg}  categories={cats}")

    svg_paths: list[str] = []
    t0 = time.time()

    # ── Phase 1: whole-image line-art ─────────────────────────────────────
    lineart_dir = out_dir / "line_art"
    la_result = preprocess_to_file(src_path, lineart_dir)
    full_lineart = Path(la_result["out_path"])
    if verbose:
        print(f"  Phase1 [{la_result['image_type']}] → {full_lineart.name}")

    # ── Phase 2: panel detection + crops ─────────────────────────────────
    panel_dir = out_dir / "panels"
    panels = crop_panels(src_path, panel_dir)

    if not panels:
        # crop_panels fallback returns whole image; this shouldn't happen
        # but guard anyway
        if verbose:
            print("  Phase2: no panels detected")
        return svg_paths

    if verbose:
        print(f"  Phase2: {len(panels)} panel(s)")

    # ── Phases 1b, 3, 4 per panel ─────────────────────────────────────────
    annotated_dir = out_dir / "annotated"
    motifs_dir = out_dir / "motifs"

    for panel_info in panels:
        panel_path = Path(panel_info["out_path"])
        panel_stem = panel_path.stem

        # Phase 1b: line-art for this panel crop
        la_panel = preprocess_to_file(panel_path, lineart_dir)
        panel_lineart = Path(la_panel["out_path"])

        # Phase 3: SAM segmentation
        detections = segment_to_files(panel_path, annotated_dir, sam_generator)
        if not detections:
            if verbose:
                print(f"    {panel_stem}: 0 detections — skipping")
            continue

        if verbose:
            by_scale = {}
            for d in detections:
                by_scale[d["scale"]] = by_scale.get(d["scale"], 0) + 1
            print(f"    {panel_stem}: {len(detections)} detections "
                  + str(by_scale))

        # Phase 4: vectorize
        det_json = annotated_dir / f"{panel_stem}_detections.json"
        enriched = vectorize_detections(
            panel_stem, panel_lineart, detections, motifs_dir
        )
        for e in enriched:
            svg_paths.append(e["svg_path"])

    elapsed = time.time() - t0
    if verbose:
        print(f"  → {len(svg_paths)} SVGs  ({elapsed:.1f}s)")

    return svg_paths


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    metadata_json: str | Path,
    img_manifest: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    checkpoint: str | Path,
    no_cluster: bool = False,
    images: list[str | Path] | None = None,
    verbose: bool = True,
) -> None:
    """
    Main entry point. Processes the curated panel-art image set end-to-end.

    Parameters
    ----------
    images : if provided, process only these specific image paths (still must
             be in the panel-art metadata). Used for testing individual images.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build allowlist from metadata
    allowed = load_allowed_images(metadata_json, img_manifest, image_dir)
    if verbose:
        print(f"Panel-art images to process: {len(allowed)}")

    # If specific images requested, filter to those
    if images:
        requested = {Path(p).name for p in images}
        allowed = [e for e in allowed if e["path"].name in requested]
        if verbose:
            print(f"Filtered to requested images: {len(allowed)}")

    if not allowed:
        print("No matching panel-art images found. Check --image-dir and --metadata.")
        return

    # Load SAM generator once (shared across all images)
    generator = load_generator(checkpoint=str(checkpoint))

    all_svg_paths: list[str] = []
    for img_entry in allowed:
        svgs = process_image(img_entry, out_dir, generator, verbose=verbose)
        all_svg_paths.extend(svgs)

    print(f"\n{'═'*60}")
    print(f"Total SVGs produced: {len(all_svg_paths)}")

    if no_cluster or not all_svg_paths:
        return

    # ── Phase 5: cluster ─────────────────────────────────────────────────
    print("\nPhase 5: computing features and clustering…")
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    features = build_feature_matrix(all_svg_paths)
    labels = cluster_motifs(features)
    n_clusters = int(labels.max()) + 1
    noise = int((labels == -1).sum())
    print(f"  Clusters: {n_clusters}  Noise: {noise} "
          f"({noise/max(len(labels),1)*100:.1f}%)")

    result = annotate_clusters(
        all_svg_paths, labels,
        metadata_json=metadata_json,
        images_manifest=img_manifest,
    )
    out_path = clusters_dir / "clusters.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Frobenius panel art motif analysis — full pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input sources
    p.add_argument("--image-dir", default="frobenius_artifacts/images/",
                   help="Directory containing downloaded panel images")
    p.add_argument("--images", nargs="*", metavar="IMAGE",
                   help="Process only these specific image files")
    p.add_argument("--metadata",
                   default="src/typescript/backend/lib/data/frobenius_panel_art.json",
                   help="frobenius_panel_art.json — curated allowlist")
    p.add_argument("--img-manifest",
                   default="frobenius_artifacts/images/manifest.json",
                   help="images/manifest.json (registration_number → filename)")

    # SAM
    p.add_argument("--checkpoint",
                   default="src/python/sam_vit_b_01ec64.pth",
                   help="SAM checkpoint .pth file")

    # Output
    p.add_argument("--out-dir", default="frobenius_artifacts/analysis/",
                   help="Root output directory")
    p.add_argument("--no-cluster", action="store_true",
                   help="Skip Phase 5 clustering")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_pipeline(
        metadata_json=args.metadata,
        img_manifest=args.img_manifest,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        checkpoint=args.checkpoint,
        no_cluster=args.no_cluster,
        images=args.images,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
