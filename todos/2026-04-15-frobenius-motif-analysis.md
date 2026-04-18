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
SAM-2 requires torch ≥ 2.5.1 — incompatible with Intel Mac. Using SAM-1 (segment-anything==1.0) instead.
SAM-1 also cannot use MPS (float64 ops unsupported) — forced to CPU.

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

- [x] SAM-1 (`segment-anything==1.0`) used — SAM-2 requires torch≥2.5, incompatible with Intel Mac
- [x] Downloaded ViT-B checkpoint (369MB) to `src/python/sam_vit_b_01ec64.pth`
- [x] Implement `load_generator` — SAM model + checkpoint loading, cached per process
- [x] Implement `segment_panel(img, generator)` — auto-mask + filter_and_nms
- [x] Implement `classify_scale` — zone/motif/element by area fraction
- [x] Implement `segment_to_files` — annotated JPEG + detections JSON
- [x] CLI `__main__`
- [x] Forced CPU device (SAM-1 uses float64 ops; MPS doesn't support float64)

**Validation result (2026-04-16):**
```
panel_01 (dense knotwork):    5 detections: zone=1, motif=3, element=1
panel_02 (figurative panel): 42 detections: zone=1, motif=26, element=15
```
Visual inspection: bounding boxes align with carved motif registers. Zone box
covers the full panel, motif boxes capture individual figures and geometric
cells, element boxes pick up sub-details. Detection quality is good.

---

## Phase 4 · Vectorisation (`vectorize.py`)

- [x] Implement `region_to_svg(lineart, x, y, w, h)` — OpenCV contours → svgwrite paths
- [x] RDP simplification (epsilon=1.5) — reduces noise, keeps meaningful shape boundaries
- [x] Normalise to 100×100 viewBox, stroke-only, no fill
- [x] Implement `vectorize_detections` + `vectorize_from_files`
- [x] CLI `__main__`

**Validation result (2026-04-16):**
```
42 motifs vectorised for panel_02
Size range: 0KB – 108KB (zone); individual motifs: 1–23KB ✓
SVG structure: M/L/Z paths in 0–100 coordinate space, fill=none ✓
```
Notes: pypotrace (C bindings) avoided — uses pure OpenCV+svgwrite.
Zone-level SVGs are large (108KB) due to full-panel detail.
Element-level SVGs are sometimes 0KB (too small for meaningful contours — expected).

---

## Phase 5 · Similarity & Clustering (`similarity.py`)

- [x] Implement `_rasterise_svg` — lightweight M/L/Z SVG parser + OpenCV renderer
- [x] Implement `hu_moments(svg_path)` — 7 log-Hu moment invariants
- [x] Implement `dino_embedding(svg_path)` — DINOv2 ViT-S/14 via torch.hub (384-dim)
- [x] Implement `build_feature_matrix` — combined Hu + DINO features
- [x] Implement `cluster_motifs` — HDBSCAN
- [x] Implement `annotate_clusters` — attach frobenius_panel_art.json metadata
- [x] CLI: writes `clusters.json` + `similarity_graph.json`

**Validation:** Not yet run on full set — pipeline end-to-end test first.

```
# result: (not yet run on full set)
```

---

## Phase 6 · End-to-End Pipeline (`pipeline.py`)

- [x] `load_allowed_images` — uses frobenius_panel_art.json as allowlist;
      non-panel-art images (village scenes etc.) are excluded automatically
- [x] `process_image` — chains Phases 1→2→1b→3→4 per image
- [x] `run_pipeline` — calls Phase 5 clustering after all images processed
- [x] CLI with `--images` for single-image testing, `--no-cluster` flag

**Validation result (2026-04-16) — single image end-to-end:**
```
Input: FoA_04-5578 (door_panel, 3 physical panels)
Output: 3 panels → 75 SVGs in 143s

  panel_00 (narrow left):      28 detections (motif=15, element=13)
  panel_01 (central knotwork):  5 detections (zone=1, motif=3, element=1)
  panel_02 (figurative right): 42 detections (zone=1, motif=26, element=15)

Allowlist filtering: 108 panel-art records → 40 FoA images found locally
  (EBA-B images in the manifest use a different registration scheme;
   their frobenius_panel_art.json records need registration_number matching — see Notes)
```

---

## Remaining work

- [ ] Run full pipeline on all 40 FoA panel-art images
- [ ] Run Phase 5 clustering on all collected SVGs and inspect cluster quality
- [ ] Verify EBA-B image registration_number matching in load_allowed_images
      (currently only 40 of 108 records resolve to local files — 68 EBA-B records
       likely use a different registration scheme than the manifest.json)
- [ ] Review cluster output and tune HDBSCAN parameters if noise > 20%

---

## Notes & Decisions Log

| Date | Note |
|---|---|
| 2026-04-15 | SAM-2 requires torch≥2.5 — incompatible with Intel Mac. Using SAM-1 (segment-anything==1.0) |
| 2026-04-15 | SAM-1 float64 ops incompatible with MPS — forced CPU inference |
| 2026-04-15 | `potrace` installed via `brew install potrace` (not used — using OpenCV+svgwrite instead) |
| 2026-04-15 | motif_illustration images (EBA-B) already line-art — Phase 1 routes them to binarise path; XDoG also works on them |
| 2026-04-15 | torch pinned to 2.2.x (last Intel Mac / x86_64 wheel); numpy pinned <2; Python 3.12 |
| 2026-04-15 | Panel detection: valley_relative_threshold=0.80 + min_sub_w=max(60, w//10) gives clean 3-panel split |
| 2026-04-16 | Pipeline uses frobenius_panel_art.json as allowlist — non-panel-art images (village scenes etc.) automatically excluded |
| 2026-04-16 | 108 panel-art records but only 40 resolve to local FoA files; EBA-B registration scheme mismatch needs investigation |
