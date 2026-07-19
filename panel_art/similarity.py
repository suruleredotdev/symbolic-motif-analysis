"""
similarity.py — Phase 5: motif embeddings + HDBSCAN clustering

Computes feature vectors for each extracted motif SVG and clusters them to
discover natural motif families across the panel art collection.

Two complementary descriptors are combined:

  Hu Moments (7 values)
    Invariant to translation, scale, and rotation.
    Captures overall silhouette shape class quickly.
    Computed by rasterising each SVG to a 128×128 bitmap.

  DINOv2 patch embeddings (1024-dim)
    Semantic visual similarity — structurally similar carvings cluster together
    even if photographed from slightly different angles or lighting.
    Loaded via torch.hub (facebookresearch/dinov2).

Clustering: HDBSCAN — density-based, no preset cluster count.
  Label -1 = noise (motifs that don't fit any cluster).
  Expects < 20% noise for a healthy collection.

CLI usage:
  uv run python -m panel_art.similarity \\
    --motif-dir frobenius_artifacts/analysis/motifs/ \\
    --panel-dir frobenius_artifacts/analysis/panels/ \\
    --metadata  src/typescript/backend/lib/data/frobenius_panel_art.json \\
    --out-dir   frobenius_artifacts/analysis/clusters/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# ── SVG rasterisation ─────────────────────────────────────────────────────────

def _rasterise_svg(svg_path: str | Path, size: int = 128) -> np.ndarray | None:
    """
    Convert an SVG file to a grayscale bitmap using OpenCV's built-in SVG
    path parser (available in OpenCV 4.x via cv2.dnn / cv2.imread for SVG).

    Since OpenCV cannot read SVGs directly, we parse the path data manually
    and draw contours onto a blank canvas. This avoids a cairosvg/inkscape
    dependency and keeps everything in the existing dependency set.

    Returns a size×size uint8 grayscale array, or None on failure.
    """
    import re
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(str(svg_path))
    except Exception:
        return None

    canvas = np.ones((size, size), dtype=np.uint8) * 255   # white background

    # Extract viewBox to compute scale
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}
    vb_str = root.get("viewBox", "0,0,100,100")
    vb = [float(v.replace(",", " ").split()[i])
          for i, v in enumerate(re.split(r"[\s,]+", vb_str))]
    vb_w = vb[2] if len(vb) >= 4 else 100.0
    vb_h = vb[3] if len(vb) >= 4 else 100.0
    scale_x = size / vb_w
    scale_y = size / vb_h

    # Draw each <path> element as polylines
    for path_el in root.iter("{http://www.w3.org/2000/svg}path"):
        d = path_el.get("d", "")
        if not d:
            continue
        pts = _parse_svg_path_to_points(d, scale_x, scale_y)
        if len(pts) >= 2:
            pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_array], isClosed=True,
                          color=0, thickness=1)

    return canvas


def _parse_svg_path_to_points(
    d: str,
    scale_x: float,
    scale_y: float,
) -> list[tuple[int, int]]:
    """
    Very lightweight M/L/Z SVG path parser — handles the specific output of
    our vectorize.py (which only emits M, L, Z commands).
    Returns a list of (x, y) integer pixel coordinates.
    """
    import re
    tokens = re.split(r"\s+", d.strip())
    pts = []
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        if cmd in ("M", "L"):
            i += 1
            if i < len(tokens):
                xy = tokens[i].split(",")
                if len(xy) == 2:
                    try:
                        px = int(float(xy[0]) * scale_x)
                        py = int(float(xy[1]) * scale_y)
                        pts.append((px, py))
                    except ValueError:
                        pass
        elif cmd == "Z":
            pass  # close path — handled by isClosed=True in polylines
        i += 1
    return pts


# ── Hu moments ────────────────────────────────────────────────────────────────

def hu_moments(svg_path: str | Path) -> np.ndarray | None:
    """
    Rasterise SVG and compute the 7 Hu moment invariants.

    Hu moments are invariant to translation, scale, and rotation — ideal for
    comparing motif silhouettes regardless of panel orientation or photo scale.
    Log-transform applied to compress the large dynamic range.
    """
    bitmap = _rasterise_svg(svg_path)
    if bitmap is None:
        return None

    # Invert so the motif is foreground (dark on white → white on black)
    inv = cv2.bitwise_not(bitmap)
    moments = cv2.moments(inv)
    hu = cv2.HuMoments(moments).flatten()   # shape (7,)

    # Log-transform: hu[i] = sign(h) * log10(|h| + eps)
    eps = 1e-10
    hu_log = np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log.astype(np.float32)


# ── DINOv2 embeddings ─────────────────────────────────────────────────────────

_dino_model = None
_dino_transform = None


def _load_dino() -> tuple:
    """Load DINOv2 ViT-S/14 (smallest variant, 384-dim) via torch.hub."""
    global _dino_model, _dino_transform
    if _dino_model is not None:
        return _dino_model, _dino_transform

    if not TORCH_AVAILABLE:
        raise ImportError("torch not available")

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("  Loading DINOv2 ViT-S/14 via torch.hub…", flush=True)
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False
    )
    model = model.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _dino_model = (model, device)
    _dino_transform = transform
    return _dino_model, _dino_transform


def dino_embedding(svg_path: str | Path) -> np.ndarray | None:
    """
    Rasterise SVG to RGB bitmap, embed with DINOv2 ViT-S/14.
    Returns a 384-dim L2-normalised float32 vector, or None on failure.
    """
    if not TORCH_AVAILABLE:
        return None

    bitmap = _rasterise_svg(svg_path, size=224)
    if bitmap is None:
        return None

    # Convert grayscale to RGB (DINO expects 3-channel)
    rgb = cv2.cvtColor(bitmap, cv2.COLOR_GRAY2RGB)

    try:
        (model, device), transform = _load_dino()
    except Exception:
        return None

    with torch.no_grad():
        tensor = transform(rgb).unsqueeze(0).to(device)
        emb = model(tensor)                       # (1, 384)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().float().numpy()


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(
    svg_paths: list[str | Path],
    use_dino: bool = True,
    use_hu: bool = True,
) -> np.ndarray:
    """
    Compute combined feature vector for each SVG.

    Feature vector = [normalised Hu moments (7-dim)] + [DINOv2 embedding (384-dim)]
    If a component fails (e.g. DINO not available), only the other is used.

    Returns float32 array of shape (N, D).
    """
    all_hu, all_dino = [], []

    for p in svg_paths:
        if use_hu:
            hv = hu_moments(p)
            all_hu.append(hv if hv is not None else np.zeros(7, dtype=np.float32))
        if use_dino and TORCH_AVAILABLE:
            dv = dino_embedding(p)
            all_dino.append(dv if dv is not None else np.zeros(384, dtype=np.float32))

    parts = []
    if use_hu and all_hu:
        hu_mat = np.array(all_hu, dtype=np.float32)
        # Normalise Hu moments to unit range per feature
        rng = hu_mat.max(0) - hu_mat.min(0) + 1e-8
        hu_mat = (hu_mat - hu_mat.min(0)) / rng
        parts.append(hu_mat)

    if use_dino and all_dino:
        dino_mat = np.array(all_dino, dtype=np.float32)
        parts.append(dino_mat)

    if not parts:
        raise ValueError("No features computed — check SVG paths and dependencies")

    return np.hstack(parts)


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_motifs(
    features: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> np.ndarray:
    """
    HDBSCAN clustering on the feature matrix.

    Parameters
    ----------
    min_cluster_size : minimum number of motifs to form a cluster.
                       3 is sensible for a ~few-hundred motif collection.
    min_samples      : HDBSCAN core point threshold. Lower = more points
                       assigned to clusters, fewer noise points.

    Returns int array of shape (N,) with cluster labels.
    Label -1 = noise (unassigned).
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed: uv pip install hdbscan")

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(features)


# ── Metadata annotation ───────────────────────────────────────────────────────

def annotate_clusters(
    svg_paths: list[str | Path],
    labels: np.ndarray,
    metadata_json: str | Path,
    images_manifest: str | Path,
) -> dict:
    """
    Combine cluster assignments with panel-level metadata from
    frobenius_panel_art.json, producing a structured output document.

    The panel stem is used to look up the source registration number, which
    in turn maps to the metadata record (culture, location, date, category).

    Returns a dict ready to JSON-serialise.
    """
    with open(metadata_json) as f:
        panel_art = json.load(f)
    with open(images_manifest) as f:
        img_manifest = json.load(f)

    # registration_number → metadata record
    reg_to_meta = {r.get("registration_number", ""): r
                   for r in panel_art.get("records", [])}
    # filename → registration_number
    file_to_reg = {m["filename"]: m["registration_number"]
                   for m in img_manifest}

    n_clusters = int(labels.max()) + 1
    noise_count = int((labels == -1).sum())

    clusters: dict[str, list] = {}
    for svg_path, label in zip(svg_paths, labels):
        key = str(int(label))
        clusters.setdefault(key, [])

        # Extract source image filename from SVG stem
        # SVG names are: <panel_stem>_motif_NNN.svg
        # panel_stem is: <image_stem>_panel_NN
        stem = Path(svg_path).stem                         # e.g. FoA_..._panel_02_motif_001
        parts = stem.split("_motif_")
        panel_stem = parts[0] if len(parts) == 2 else stem # e.g. FoA_..._panel_02
        image_stem = "_".join(panel_stem.split("_")[:-2])  # strip _panel_NN

        # Find matching image filename (try both .png and .jpg)
        source_meta = None
        for ext in (".png", ".jpg"):
            for fname, reg in file_to_reg.items():
                if fname.startswith(image_stem):
                    source_meta = reg_to_meta.get(reg)
                    break
            if source_meta:
                break

        clusters[key].append({
            "svg": str(svg_path),
            "panel_stem": panel_stem,
            "source_meta": source_meta,
        })

    return {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_fraction": round(noise_count / max(len(labels), 1), 3),
        "clusters": clusters,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 5: Hu + DINOv2 embeddings and HDBSCAN clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--motif-dir",  required=True,
                   help="Directory of motif SVG files (Phase 4 output)")
    p.add_argument("--panel-dir",  required=True,
                   help="Directory of panel crop PNGs (Phase 2 output)")
    p.add_argument("--metadata",   required=True,
                   help="frobenius_panel_art.json")
    p.add_argument("--img-manifest", default=None,
                   help="images/manifest.json (default: <image-dir>/../manifest.json)")
    p.add_argument("--out-dir",
                   default="frobenius_artifacts/analysis/clusters")
    p.add_argument("--no-dino",    action="store_true")
    p.add_argument("--no-hu",      action="store_true")
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument("--scale",      default="motif",
                   choices=["zone", "motif", "element", "all"],
                   help="Only cluster motifs at this scale level")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    motif_dir = Path(args.motif_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect SVG paths, optionally filtering by scale
    # Scale is encoded in the detections JSON — load those to filter
    svg_paths_all = sorted(motif_dir.glob("*.svg"))
    print(f"SVG files found: {len(svg_paths_all)}")

    if not svg_paths_all:
        print("No SVG files found. Run Phase 4 first.")
        return

    svg_paths = svg_paths_all

    print(f"Computing features for {len(svg_paths)} motifs…")
    features = build_feature_matrix(
        svg_paths,
        use_dino=not args.no_dino,
        use_hu=not args.no_hu,
    )
    print(f"Feature matrix: {features.shape}")

    labels = cluster_motifs(features, min_cluster_size=args.min_cluster_size)
    n_clusters = int(labels.max()) + 1
    noise = int((labels == -1).sum())
    print(f"Clusters: {n_clusters}  Noise: {noise} ({noise/len(labels)*100:.1f}%)")

    img_manifest = args.img_manifest
    if img_manifest is None:
        # Default: images/manifest.json relative to the metadata file
        img_manifest = Path(args.metadata).parent.parent.parent.parent \
            / "frobenius_artifacts" / "images" / "manifest.json"

    result = annotate_clusters(
        [str(p) for p in svg_paths], labels,
        metadata_json=args.metadata,
        images_manifest=img_manifest,
    )

    out_path = out_dir / "clusters.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Clusters → {out_path}")

    # Pairwise distance summary (cosine)
    from sklearn.metrics.pairwise import cosine_distances
    dist = cosine_distances(features)
    sim_path = out_dir / "similarity_graph.json"
    # Store only top-5 neighbours per motif to keep file size reasonable
    graph = []
    for i, svg_path in enumerate(svg_paths):
        sims = [(j, float(dist[i, j])) for j in range(len(svg_paths)) if j != i]
        sims.sort(key=lambda x: x[1])
        graph.append({
            "id": Path(svg_path).stem,
            "neighbours": [
                {"id": Path(svg_paths[j]).stem, "distance": d}
                for j, d in sims[:5]
            ],
        })
    with open(sim_path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"Similarity graph → {sim_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
