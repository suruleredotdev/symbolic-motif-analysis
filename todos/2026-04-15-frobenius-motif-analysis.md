# TODO — Frobenius Motif Analysis Pipeline

**Date:** 2026-04-15
**Plan:** `plans/2026-04-15-frobenius-motif-analysis.md`
**Branch:** `claude/panel-art-dataset-XCZOw`

Status key: ` ` not started · `~` in progress · `x` done · `!` blocked

---

## Phase 0 · Project Setup

- [ ] Create `src/python/pyproject.toml` (UV project with `panel_art` package)
- [ ] Add dependencies: opencv-python, Pillow, numpy, scikit-image, torch, torchvision, sam2, pypotrace, svgpathtools, scikit-learn, hdbscan, matplotlib
- [ ] Create `src/python/panel_art/__init__.py` and package skeleton
- [ ] Add `frobenius_artifacts/analysis/` to `.gitignore`
- [ ] Confirm `potrace` binary available: `potrace --version`

**Validation:** `uv run python -c "import cv2, PIL, numpy, skimage, torch; print('all imports ok')`

```
# result: (not yet run)
```

---

## Phase 1 · Preprocessing (`preprocess.py`)

- [ ] Implement `detect_image_type(img)` — photo vs illustration via Laplacian variance
- [ ] Implement `to_grayscale_clahe(img)` — grayscale + contrast normalisation
- [ ] Implement `xdog(img, sigma1, sigma2, epsilon, tau)` — XDoG line-art extraction
- [ ] Implement `binarise_illustration(img)` — clean path for already-line-art EBA-B images
- [ ] Implement `preprocess(image_path) -> np.ndarray` — top-level function, routes by type
- [ ] Write output PNGs to `frobenius_artifacts/analysis/line_art/`

**Validation:** Run on 3 representative images — one photo (FoA), one pen-and-ink (EBA-B ink), one watercolour (EBA-B colour). Inspect output PNGs manually.

```
# Validation command:
# uv run python -m panel_art.preprocess \
#   frobenius_artifacts/images/FoA_04-5578_Modakeke_\(Ife\)_q48628_i1.png \
#   frobenius_artifacts/images/EBA-B_00425_Ibadan_q97912_i1.png \
#   frobenius_artifacts/images/EBA-B_00426_Ibadan_q97913_i2.png

# result: (not yet run)
```

---

## Phase 2 · Panel Detection (`panel_detect.py`)

- [ ] Implement `detect_panels(img) -> list[Rect]` — Otsu threshold + connected components
- [ ] Handle rotated panels via `minAreaRect`
- [ ] Filter by aspect ratio (0.1–0.8) and minimum area (5% of image)
- [ ] Implement `crop_panels(image_path, out_dir)` — save each ROI crop to `analysis/panels/`

**Validation:** Run on the 3-panel photo `FoA_04-5578` and a single-panel photo. Confirm correct count of panels extracted and no spurious crops.

```
# Validation command:
# uv run python -m panel_art.panel_detect \
#   frobenius_artifacts/images/FoA_04-5578_Modakeke_\(Ife\)_q48628_i1.png

# Expected: 3 panels detected
# result: (not yet run)
```

---

## Phase 3 · Motif Segmentation (`motif_segment.py`)

- [ ] Set up SAM-2 tiny model loading (local checkpoint or HuggingFace hub)
- [ ] Implement `auto_segment(panel_img) -> list[Mask]` — SAM-2 automatic mask generation
- [ ] Implement `classify_scale(mask, panel_area) -> str` — zone / motif / element
- [ ] Implement `nms(masks, iou_threshold=0.3) -> list[Mask]` — non-max suppression
- [ ] Implement `segment_panel(panel_img) -> list[Detection]` — full pipeline per panel
- [ ] Write annotated images to `analysis/annotated/` (bounding boxes coloured by scale)

**Validation:** Run on 2 door panels. Inspect annotated output — boxes should align with visually distinct carved motif regions. Count of motifs per panel should be plausible (5–50).

```
# Validation command:
# uv run python -m panel_art.motif_segment \
#   frobenius_artifacts/analysis/panels/FoA_04-5578_panel_0.png

# Expected: ~10-30 motif detections at mixed scales
# result: (not yet run)
```

---

## Phase 4 · Vectorisation (`vectorize.py`)

- [ ] Confirm `pypotrace` importable and `potrace` binary accessible
- [ ] Implement `motif_to_svg(panel_img, edge_img, bbox) -> str` — crop, threshold, potrace
- [ ] Implement `normalise_svg(svg_str) -> str` — rewrite to 100×100 viewBox, stroke-only
- [ ] Implement `vectorize_all(detections, panel_img, edge_img, out_dir)` — batch over detections
- [ ] Save SVGs as `analysis/motifs/<panel_id>_<motif_idx>.svg`

**Validation:** Open 5 SVG outputs in browser. Lines should form recognisable outlines of the motif shape, not noise. File sizes should be < 20KB each.

```
# Validation command:
# uv run python -m panel_art.vectorize \
#   frobenius_artifacts/analysis/panels/FoA_04-5578_panel_0.png \
#   frobenius_artifacts/analysis/line_art/FoA_04-5578_panel_0_edges.png \
#   frobenius_artifacts/analysis/annotated/FoA_04-5578_panel_0_detections.json

# Expected: SVG files in analysis/motifs/, visually recognisable shapes
# result: (not yet run)
```

---

## Phase 5 · Similarity & Clustering (`similarity.py`)

- [ ] Implement `hu_moments(svg_path) -> np.ndarray` — rasterise SVG, compute Hu moments
- [ ] Implement `dino_embedding(img_crop) -> np.ndarray` — DINO patch features (reuse dinohash)
- [ ] Implement `build_feature_matrix(motif_dir) -> (np.ndarray, list[str])` — all motifs
- [ ] Implement `cluster(features) -> np.ndarray` — HDBSCAN labels
- [ ] Implement `annotate_clusters(labels, motif_ids, metadata_json) -> dict` — attach panel metadata
- [ ] Write `analysis/clusters/clusters.json` and `similarity_graph.json`

**Validation:** Check cluster count (expect 5–20 natural clusters across 73 images). Spot-check: motifs in the same cluster should be visually similar. Confirm noise points (label=-1) are < 20% of total.

```
# Validation command:
# uv run python -m panel_art.similarity \
#   frobenius_artifacts/analysis/motifs/ \
#   src/typescript/backend/lib/data/frobenius_panel_art.json

# Expected: clusters.json with N clusters, <20% noise
# result: (not yet run)
```

---

## Phase 6 · End-to-End Pipeline (`pipeline.py`)

- [ ] Implement `run_pipeline(image_path, metadata_json, out_dir)` — calls phases 1–5 in order
- [ ] Add CLI: `uv run python -m panel_art.pipeline <image> [--metadata <json>] [--out <dir>]`
- [ ] Run full pipeline on all 73 images
- [ ] Write final `analysis/clusters/clusters.json` across all images

**Validation:** Run on all 73 images. No crashes. Output directory populated with expected structure. Final cluster report printed to stdout.

```
# Validation command:
# uv run python -m panel_art.pipeline \
#   frobenius_artifacts/images/ \
#   --metadata src/typescript/backend/lib/data/frobenius_panel_art.json \
#   --out frobenius_artifacts/analysis/

# Expected: all phases complete, summary printed
# result: (not yet run)
```

---

## Notes & Decisions Log

| Date | Note |
|---|---|
| 2026-04-15 | SAM-2: use `sam2-hiera-tiny` locally; HuggingFace Inference API as fallback |
| 2026-04-15 | `potrace` installed via `brew install potrace` |
| 2026-04-15 | motif_illustration images (EBA-B) already line-art — Phase 1 routes them to binarise path, skipping XDoG |
| 2026-04-15 | Non-axis-aligned panels: use `minAreaRect` in Phase 2 |
