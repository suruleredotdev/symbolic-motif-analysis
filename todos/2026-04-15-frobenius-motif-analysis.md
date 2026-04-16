# TODO — Frobenius Motif Analysis Pipeline

**Date:** 2026-04-15
**Plan:** `plans/2026-04-15-frobenius-motif-analysis.md`
**Branch:** `claude/panel-art-dataset-XCZOw`

Status key: ` ` not started · `~` in progress · `x` done · `!` blocked

---

## Phase 0 · Project Setup

- [x] Create `src/python/pyproject.toml` (UV project with `panel_art` package)
- [x] Add dependencies: opencv-python-headless, Pillow, numpy<2 (torch ABI), scikit-image, torch==2.2.x (last Intel Mac build), torchvision, open-clip-torch, svgwrite, svgpathtools, scikit-learn, hdbscan, matplotlib, anthropic
- [x] Create `src/python/panel_art/__init__.py` and package skeleton
- [x] Add `frobenius_artifacts/analysis/` and `.venv/` to `.gitignore`
- [x] Confirm `potrace` binary available: `potrace --version`

**Validation result (2026-04-15):**
```
cv2: 4.11.0  numpy: 1.26.4  torch: 2.2.2  mps available: True  svgwrite: 1.4.3
all imports ok
potrace 1.16 — ok
```
Notes: Python pinned to 3.12 (torch 2.2.x has no py3.13 Intel Mac wheel);
numpy pinned <2 (torch 2.2.x compiled against numpy 1.x ABI);
SAM-2 must be installed separately via git (not on PyPI).

---

## Phase 1 · Preprocessing (`preprocess.py`)

- [x] Implement `detect_image_type(img)` — photo vs illustration via white-pixel fraction + Laplacian variance
- [x] Implement `to_grayscale_clahe` + bilateral filter (embedded in `preprocess_photo`)
- [x] Implement `xdog(img, sigma1, sigma2, epsilon, phi, tau)` — XDoG line-art extraction
- [x] Implement `binarise_illustration(img)` — Otsu + morphological close for EBA-B images
- [x] Implement `preprocess(image_path) -> (np.ndarray, str)` — routes by detected type
- [x] Implement `preprocess_to_file` + CLI `__main__`
- [x] Write output PNGs to `frobenius_artifacts/analysis/line_art/`

**Validation result (2026-04-15):**
```
[       photo]  FoA_04-5578_Modakeke_(Ife)_q48628_i1.png  →  _lineart.png  shape=(1531, 977)
[       photo]  EBA-B_00425_Ibadan_q97912_i1.png           →  _lineart.png  shape=(1398, 992)
[illustration]  EBA-B_00426_Ibadan_q97913_i2.png           →  _lineart.png  shape=(673, 1000)
```
Visual inspection: FoA photo → clean pen-and-ink style line art (excellent);
EBA-B pen-and-ink → also good through XDoG path (white-pixel fraction too low
for the hatched drawing to trip the illustration detector, but XDoG handles it);
EBA-B watercolour → correctly identified as illustration, clean binary output.

Bug fixed: original tau=0.95 suppressed all output (diff values are in [-0.3, 0.3],
not [0,1]); corrected to tau=0.01. Output inversion also corrected.

---

## Phase 2 · Panel Detection (`panel_detect.py`)

- [x] Implement `detect_panels(img)` — Otsu + morphological closing + connected components
- [x] Handle touching panels via `_split_wide_blobs` (vertical projection valleys, relative 80% threshold)
- [x] Filter by minimum sub-panel width (max(60, blob_w // 10)) to remove edge slivers
- [x] Fallback: 0 detections → treat whole image as single panel
- [x] Implement `crop_panels(image_path, out_dir)` — save ROI crops
- [x] Implement `annotate_panels` + `--annotate` CLI flag

**Validation result (2026-04-15):**
```
FoA_04-5578 (3 physical panels): 3 panels detected ✓
  panel 0: 162×1462  AR=9.02   (left narrow knotwork panel)
  panel 1: 380×1462  AR=3.85   (central tall knotwork panel)
  panel 2: 274×993   AR=3.62   (right figurative panel)

FoA_04-5042 (village scene photo, no panels): 0 detected → fallback to whole image ✓
EBA-B_00425 (ink drawing, one full-page illustration): 0 detected → fallback ✓
```
Notes: The left two panels were physically touching; projection-valley splitting at
80% of peak value correctly separated them. Min-width filter (60px) removes the
28px gap-edge slivers produced by the split.

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

- [ ] Implement `motif_to_svg(panel_img, edge_img, bbox) -> str` — crop, threshold, OpenCV contours → svgwrite paths
- [ ] Implement `normalise_svg(svg_str) -> str` — rewrite to 100×100 viewBox, stroke-only
- [ ] Implement `vectorize_all(detections, panel_img, edge_img, out_dir)` — batch over detections
- [ ] Save SVGs as `analysis/motifs/<panel_id>_<motif_idx>.svg`

**Validation:** Open 5 SVG outputs in browser. Lines should form recognisable outlines of the motif shape, not noise. File sizes should be < 20KB each.

```
# result: (not yet run)
```

---

## Phase 5 · Similarity & Clustering (`similarity.py`)

- [ ] Implement `hu_moments(svg_path) -> np.ndarray` — rasterise SVG, compute Hu moments
- [ ] Implement `dino_embedding(img_crop) -> np.ndarray` — DINOv2 patch features
- [ ] Implement `build_feature_matrix(motif_dir) -> (np.ndarray, list[str])` — all motifs
- [ ] Implement `cluster(features) -> np.ndarray` — HDBSCAN labels
- [ ] Implement `annotate_clusters(labels, motif_ids, metadata_json) -> dict` — attach panel metadata
- [ ] Write `analysis/clusters/clusters.json` and `similarity_graph.json`

**Validation:** Check cluster count (expect 5–20). Spot-check: motifs in the same cluster should be visually similar. Confirm noise points (label=-1) < 20%.

```
# result: (not yet run)
```

---

## Phase 6 · End-to-End Pipeline (`pipeline.py`)

- [ ] Implement `run_pipeline(image_path, metadata_json, out_dir)` — phases 1–5 in order
- [ ] Add CLI: `panel-art <image_dir> [--metadata <json>] [--out <dir>]`
- [ ] Run full pipeline on all 73 images
- [ ] Write final `analysis/clusters/clusters.json` across all images

```
# result: (not yet run)
```

---

## Notes & Decisions Log

| Date | Note |
|---|---|
| 2026-04-15 | SAM-2: use `sam2-hiera-tiny` locally; HuggingFace Inference API as fallback |
| 2026-04-15 | `potrace` installed via `brew install potrace` |
| 2026-04-15 | motif_illustration images (EBA-B) already line-art — Phase 1 routes them to binarise path; XDoG also works on them if mis-detected |
| 2026-04-15 | Non-axis-aligned panels: projection-valley split handles touching panels; minAreaRect not needed |
| 2026-04-15 | torch pinned to 2.2.x (last Intel Mac / x86_64 wheel); numpy pinned <2; Python 3.12 |
| 2026-04-15 | Panel detection: valley_relative_threshold=0.80 + min_sub_w=max(60, w//10) gives clean 3-panel split |
