"""
embed_motif_patches.py

Generates CLIP (ViT-L/14) and DINOv2 (ViT-L/14) embeddings for extracted
motif patch images and writes the results to a JSON file.

Input:  one or more *_patches.json files (output of extract_motif_patches.py)
        OR a directory of patch PNG files
Output: <stem>_embeddings.json  — per-patch embedding vectors + metadata

Usage:
  # From a patches.json manifest
  python3 embed_motif_patches.py path/to/image_patches.json

  # From a directory of patch PNGs
  python3 embed_motif_patches.py --patch-dir path/to/image_patches/

  # Multiple manifests
  python3 embed_motif_patches.py *_patches.json

  # Disable one model if not needed
  python3 embed_motif_patches.py patches.json --no-dino
  python3 embed_motif_patches.py patches.json --no-clip

  # Run a quick similarity demo after embedding
  python3 embed_motif_patches.py patches.json --similarity-demo

Dependencies:
  pip install open_clip_torch torch torchvision pillow numpy
  # DINOv2 is loaded via torch.hub (downloads ~1.2 GB on first run)
  # CLIP ViT-L/14 weights download ~890 MB on first run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# ─── Model config ─────────────────────────────────────────────────────────────

CLIP_MODEL    = "ViT-L-14"
CLIP_PRETRAIN = "openai"          # or "laion2b_s32b_b82k" for open weights
CLIP_DIM      = 768

DINO_REPO     = "facebookresearch/dinov2"
DINO_MODEL    = "dinov2_vitl14"   # 1024-dim; use dinov2_vitb14 for smaller/faster
DINO_DIM      = 1024

DINO_TRANSFORM = None   # built lazily


# ─── Device ───────────────────────────────────────────────────────────────────

def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if choice == "mps" or (
        choice == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


# ─── CLIP ─────────────────────────────────────────────────────────────────────

class CLIPEmbedder:
    """
    Wraps open_clip ViT-L/14.

    Produces 768-dim L2-normalised vectors that live in the same space as
    CLIP text embeddings, so text-to-image cosine queries work directly.
    """

    def __init__(self, device: torch.device):
        if not CLIP_AVAILABLE:
            raise ImportError("open_clip_torch not installed.  pip install open_clip_torch")
        print(f"  Loading CLIP {CLIP_MODEL} ({CLIP_PRETRAIN})…")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAIN
        )
        self.model = self.model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.device = device

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> list[float]:
        """Single image → 768-dim normalised vector."""
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().float().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Text string → 768-dim normalised vector (same space as images)."""
        tokens = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().float().tolist()

    @torch.no_grad()
    def embed_batch(self, imgs: list[Image.Image]) -> list[list[float]]:
        tensors = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)
        embs = self.model.encode_image(tensors)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().float().tolist()


# ─── DINOv2 ───────────────────────────────────────────────────────────────────

class DINOv2Embedder:
    """
    Wraps facebookresearch/dinov2 ViT-L/14.

    Produces 1024-dim vectors optimised for visual structural similarity —
    two visually similar but semantically different regions will still be
    close in this space, unlike CLIP.
    """

    # Standard ImageNet normalisation (what DINOv2 was trained with)
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(self, device: torch.device):
        print(f"  Loading DINOv2 {DINO_MODEL} via torch.hub…")
        self.model = torch.hub.load(DINO_REPO, DINO_MODEL, verbose=False)
        self.model = self.model.to(device).eval()
        self.device = device
        self._transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=self._MEAN, std=self._STD),
        ])

    @torch.no_grad()
    def embed_image(self, img: Image.Image) -> list[float]:
        """Single image → 1024-dim vector (not normalised by default)."""
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        emb = self.model(tensor)                  # [1, 1024]
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].cpu().float().tolist()

    @torch.no_grad()
    def embed_batch(self, imgs: list[Image.Image]) -> list[list[float]]:
        tensors = torch.stack([self._transform(img) for img in imgs]).to(self.device)
        embs = self.model(tensors)
        embs = embs / embs.norm(dim=-1, keepdim=True)
        return embs.cpu().float().tolist()


# ─── Similarity helper ────────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def top_k_similar(
    query_idx: int,
    embeddings: list[list[float]],
    k: int = 5,
) -> list[tuple[int, float]]:
    """Return indices and scores of the k most similar patches to query_idx."""
    q = np.array(embeddings[query_idx], dtype=np.float32)
    mat = np.array(embeddings, dtype=np.float32)
    scores = mat @ q   # all vectors already normalised → dot = cosine
    scores[query_idx] = -1   # exclude self
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_idx]


# ─── Patch loading ────────────────────────────────────────────────────────────

def load_patches_from_manifest(manifest_path: Path) -> list[dict]:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    patches = data if isinstance(data, list) else data.get("patches", [])
    return patches


def load_patches_from_dir(patch_dir: Path) -> list[dict]:
    files = sorted(patch_dir.glob("*.png")) + sorted(patch_dir.glob("*.jpg"))
    return [{"patch_file": str(p), "source_image": p.name} for p in files]


# ─── Main embedding loop ──────────────────────────────────────────────────────

def embed_patches(
    patches: list[dict],
    clip_model: CLIPEmbedder | None,
    dino_model: DINOv2Embedder | None,
    base_dir: Path,
    batch_size: int = 8,
) -> list[dict]:
    """
    Add 'clip_embedding' and/or 'dino_embedding' keys to each patch dict.
    Processes in batches for efficiency.
    """
    results = [dict(p) for p in patches]

    def _load(patch: dict) -> Image.Image | None:
        pf = patch.get("patch_file", "")
        # patch_file may be relative to base_dir or absolute
        candidates = [
            Path(pf),
            base_dir / pf,
        ]
        for c in candidates:
            if c.exists():
                return Image.open(c).convert("RGB")
        print(f"    WARNING: patch file not found: {pf}")
        return None

    # Process in batches
    for start in range(0, len(patches), batch_size):
        batch_patches = patches[start:start + batch_size]
        batch_imgs: list[Image.Image | None] = [_load(p) for p in batch_patches]
        valid_imgs  = [(i, img) for i, img in enumerate(batch_imgs) if img is not None]

        if not valid_imgs:
            continue

        indices, imgs = zip(*valid_imgs)

        if clip_model:
            t0 = time.time()
            clip_embs = clip_model.embed_batch(list(imgs))
            for i, emb in zip(indices, clip_embs):
                results[start + i]["clip_embedding"] = emb
            elapsed = time.time() - t0
            print(f"  CLIP  batch [{start+1}–{start+len(batch_patches)}] "
                  f"({len(valid_imgs)} imgs) {elapsed:.2f}s")

        if dino_model:
            t0 = time.time()
            dino_embs = dino_model.embed_batch(list(imgs))
            for i, emb in zip(indices, dino_embs):
                results[start + i]["dino_embedding"] = emb
            elapsed = time.time() - t0
            print(f"  DINOv2 batch [{start+1}–{start+len(batch_patches)}] "
                  f"({len(valid_imgs)} imgs) {elapsed:.2f}s")

    return results


# ─── Similarity demo ──────────────────────────────────────────────────────────

def run_similarity_demo(results: list[dict]) -> None:
    """
    For the first few patches, print nearest neighbours under both models.
    Useful for quickly comparing CLIP vs DINOv2 similarity judgements.
    """
    clip_embs = [r.get("clip_embedding") for r in results]
    dino_embs = [r.get("dino_embedding") for r in results]

    has_clip = all(e is not None for e in clip_embs)
    has_dino = all(e is not None for e in dino_embs)

    print(f"\n{'─'*60}")
    print("Similarity demo (top-3 neighbours per model)")
    print("─"*60)

    for qi in range(min(3, len(results))):
        q = results[qi]
        src = Path(q.get("patch_file", f"patch_{qi}")).name
        print(f"\nQuery #{qi+1}: {src}")

        if has_clip:
            nns = top_k_similar(qi, clip_embs, k=3)
            print("  CLIP nearest:")
            for idx, score in nns:
                name = Path(results[idx].get("patch_file", f"patch_{idx}")).name
                print(f"    #{idx+1:3d} {name}  score={score:.4f}")

        if has_dino:
            nns = top_k_similar(qi, dino_embs, k=3)
            print("  DINOv2 nearest:")
            for idx, score in nns:
                name = Path(results[idx].get("patch_file", f"patch_{idx}")).name
                print(f"    #{idx+1:3d} {name}  score={score:.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLIP + DINOv2 embeddings for extracted motif patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("manifests", nargs="*", metavar="PATCHES_JSON",
                   help="*_patches.json manifest file(s) from extract_motif_patches.py")
    p.add_argument("--patch-dir", metavar="DIR",
                   help="Directory of patch PNG files (alternative to manifest)")
    p.add_argument("--out-dir", metavar="OUT_DIR",
                   help="Output directory (default: same as input)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--clip-model",    default=CLIP_MODEL,
                   help="open_clip model name")
    p.add_argument("--clip-pretrain", default=CLIP_PRETRAIN,
                   help="open_clip pretrained weights name")
    p.add_argument("--dino-model",    default=DINO_MODEL,
                   choices=["dinov2_vits14", "dinov2_vitb14",
                            "dinov2_vitl14", "dinov2_vitg14"])
    p.add_argument("--no-clip",  action="store_true", help="Skip CLIP embedding")
    p.add_argument("--no-dino",  action="store_true", help="Skip DINOv2 embedding")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--similarity-demo", action="store_true",
                   help="Print nearest-neighbour demo table after embedding")
    return p


def main() -> None:
    if not TORCH_AVAILABLE:
        print("ERROR: torch not installed.  pip install torch torchvision")
        sys.exit(1)

    parser = build_parser()
    args   = parser.parse_args()

    # Collect inputs
    manifest_paths: list[Path] = []
    patch_collections: list[tuple[list[dict], Path]] = []  # (patches, base_dir)

    if args.patch_dir:
        d = Path(args.patch_dir)
        patches = load_patches_from_dir(d)
        patch_collections.append((patches, d))

    for s in args.manifests:
        mp = Path(s)
        if not mp.exists():
            print(f"WARNING: {mp} not found, skipping")
            continue
        manifest_paths.append(mp)
        patches = load_patches_from_manifest(mp)
        patch_collections.append((patches, mp.parent))

    if not patch_collections:
        parser.error("No input provided.  Pass a *_patches.json file or --patch-dir.")

    # Load models
    device = resolve_device(args.device)
    print(f"\nDevice: {device}")
    print("Loading models:")

    clip_model = None
    dino_model = None

    if not args.no_clip:
        clip_model = CLIPEmbedder(device)
    if not args.no_dino:
        dino_model = DINOv2Embedder(device)

    # Process each collection
    for patches, base_dir in patch_collections:
        print(f"\n{'─'*60}")
        print(f"Processing {len(patches)} patches from {base_dir}")

        results = embed_patches(
            patches, clip_model, dino_model, base_dir, args.batch_size
        )

        # Write output (strip heavy embedding arrays for printing)
        out_dir = Path(args.out_dir) if args.out_dir else base_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = "patches" if args.patch_dir else Path(patches[0].get("source_image", "patches")).stem
        # Derive stem from manifest or source image name
        if manifest_paths:
            stem = manifest_paths[0].stem.replace("_patches", "")

        out_path = out_dir / f"{stem}_embeddings.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nWrote {len(results)} embedding records → {out_path}")
        dims = []
        if not args.no_clip:
            dims.append(f"CLIP {CLIP_DIM}d")
        if not args.no_dino:
            dims.append(f"DINOv2 {DINO_DIM}d")
        print(f"Embedding dimensions: {', '.join(dims)}")

        if args.similarity_demo:
            run_similarity_demo(results)


if __name__ == "__main__":
    main()
