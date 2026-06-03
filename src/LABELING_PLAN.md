# motif_labeling.ipynb — Design Plan

Jupyter notebook for interactive labeling, clustering review, and LLM-assisted
description of extracted motifs from the Frobenius panel art collection.

---

## Goals

- Browse motifs grouped by cluster; navigate freely through the full set
- View each motif **in context** of its source panel image with numbered bbox overlays
- Select **multiple** motifs to see their panel context at once
- Get **LLM-generated label suggestions** using both original and normalised crop images
- Edit and save labels to a persistent JSON file
- Produce a **static HTML export** (via nbconvert) that captures the labelled state for sharing

---

## Architecture

New notebook: `src/python/motif_labeling.ipynb`

**Reads from:**
- `src/python/motif_embeddings_*.npy` + `motif_paths_*.txt` — embeddings + paths from `motif_similarity.ipynb`
- `frobenius_artifacts/analysis/motifs_norm/<panel>/` — normalised crops (for embedding display)
- `frobenius_artifacts/analysis/motifs/<panel>/` — original crops (for LLM and context view)
- `frobenius_artifacts/analysis/panels/` — full panel PNGs (for in-context view)
- `frobenius_artifacts/analysis/annotated/<panel>_{approved,detections}.json` — bbox coordinates
- `src/typescript/backend/lib/data/frobenius_panel_art.json` — artifact metadata (location, collection, categories)

**Writes to:**
- `frobenius_artifacts/analysis/motif_labels.json` — persistent label store

**Path convention:** `motifs_norm` → `motifs` substitution gives the original crop path for any normalised path.

---

## Cell layout

### Cell 1 — Imports & paths
Standard setup: pathlib, numpy, PIL, ipywidgets, anthropic, sklearn cosine_similarity.
`ANTHROPIC_API_KEY` read from env; LLM cell degrades gracefully if absent.

### Cell 2 — Data load

```python
# Loads embeddings .npy + paths .txt (inherits PREPROCESS_MODE from motif_similarity)
# Derives per-record dict:
records = [{
    "path_norm": str,        # motifs_norm/…/NNN_scale_iouX.png (relative)
    "path_orig": str,        # motifs/… same filename
    "panel_stem": str,       # e.g. EBA-Div_00311_Ife_q166566_i1_panel_00
    "index": int,            # NNN — matches bbox index in detections JSON
    "scale": str,            # motif | register
    "iou": float,
    "cluster": int,          # from motif_similarity HDBSCAN labels
    "xy": (float, float),    # t-SNE coords
    "bbox": dict,            # {x, y, w, h} — loaded from annotated/
    "panel_png": Path,       # panels/<stem>.png
    "artifact_meta": dict,   # from frobenius_panel_art.json (nullable)
}]

sim_matrix = cosine_similarity(embeddings)   # (N, N), computed once
nn_indices  = argsort(-sim_matrix, axis=1)[:, 1:7]  # top-6 per motif, excl. self
```

### Cell 3 — Label store

```python
# Loads analysis/motif_labels.json or creates empty {}
# Keys = path_norm (relative), survives directory moves

def get_label(rec) -> dict | None
def save_label(rec, label, description, notes, source="human")
    # writes immediately; "source" = "human" | "llm-accepted" | "llm-edited"

def label_progress() -> (int labeled, int total)
```

### Cell 4 — Cluster gallery

Scrollable grid of cluster strips. Each cluster is a titled HBox of thumbnail
cards. Each card has:
- Thumbnail image (normalised, 64px)
- Checkbox (for multi-select)
- Coloured dot if already labeled

Clicking a card (not the checkbox) sets the **active motif** (drives cells 5–6).
Checking the checkbox adds/removes from the **multi-select set** (drives cell 5 panel view).

```
Cluster 0 (14)   [✓]🖼  [ ]🖼  [✓]🖼  [ ]🖼  …
Cluster 1 (9)    [ ]🖼  [ ]🖼  …
Noise (47)       [ ]🖼  [ ]🖼  …
```

Keyboard shortcut: `Shift+click` to range-select within a cluster.

### Cell 5 — Context viewer

**Single-select mode** (one active motif, nothing else checked):

```
┌─ zoom crop (2×) ──────┐  ┌─ panel with bbox ──────────────────────┐
│                        │  │                                         │
│  Original crop at 2×   │  │  Full panel scaled to ~450px wide       │
│  with 30px padding     │  │  Lime rectangle around active motif     │
│                        │  │                                         │
└────────────────────────┘  └─────────────────────────────────────────┘
```

Layout is `HBox` with `flex-wrap: wrap` so it collapses to vertical on narrow
screens. The panel is capped at 600px wide to avoid excessive scrolling.

Below: **Nearest neighbours** — 6 thumbnails (original crops) with cosine sim score.
Click any NN thumbnail to navigate to it.

**Multi-select mode** (≥2 motifs checked):

Groups selections by panel. For each panel that has ≥1 selected motif:

```
Panel: EBA-Div_00311_Ife_q166566_i1_panel_00   [3 motifs selected]
┌─ panel image with numbered lime bboxes ────────────────────────────┐
│  ①  ②  ③  drawn with lime outlines, white filled number labels   │
└────────────────────────────────────────────────────────────────────┘
┌──┐ ┌──┐ ┌──┐
│①│ │②│ │③│   ← original crops, 80px, beneath the panel
└──┘ └──┘ └──┘

Panel: KBA_10298_q148403_i1_panel_00   [1 motif selected]
┌─ panel image ──┐
│  ④             │
└────────────────┘
┌──┐
│④│
└──┘
```

Numbers are global across all selected motifs (1…N in the order they appear in
`records`). The panel image is drawn with PIL in-kernel; all boxes for that
panel are drawn in one pass.

### Cell 6 — Label editor (always single-active-motif)

```
Active motif: EBA-Div_00311…/000_motif_iou0.942.png   Cluster 2

Label         [geometric_knot_pattern                       ]
Description   [Interlaced rope-work filling upper register  ]
Notes         [cf. KBA_10274 panel_00/002                   ]

[Get LLM Suggestion ✨]    [Save ✓]

[← Prev]    [Next →]    [Next Unlabeled →]
```

- **Get LLM Suggestion**: fires the multimodal Claude call (see Cell 7),
  fills fields with suggestion; user edits before saving
- **Save**: writes to label store, marks thumbnail dot, advances to next
- **← Prev / Next →**: navigate through all records regardless of label status
- **Next Unlabeled →**: skip to next record without a label

Navigation wraps at boundaries.

### Cell 7 — LLM suggestion (fires on button press)

Sends a multimodal request to `claude-opus-4-6`.

**Payload:**

1. Original crop image (base64 PNG)
2. Normalised (lines) crop image (base64 PNG)
3. Top 3 nearest-neighbour original crops (base64 PNG each)

**Prompt template:**

```
You are describing carved motifs from Yoruba door panels and architectural
carvings in the Frobenius Institute archive.

This motif comes from:
  Artifact: {registration_number}
  Location: {location}
  Collection: {collection}
  Object type: {categories}
  Cluster: {cluster_id} ({cluster_size} visually similar motifs)

Image 1: the original photographic/illustrative crop (use this to understand
  medium, age, and cultural context).
Image 2: the normalised line-art version (use this to understand structure
  and geometry more clearly).
Images 3–5: three visually nearest neighbours from the same collection.

{If any NNs are already labeled: "Nearest labelled motifs: X, Y, Z."}

Please provide:
1. label: 2–4 words, snake_case (e.g. "interlaced_knotwork", "standing_figure",
   "geometric_diamond_grid")
2. description: one sentence describing what is visually present — shapes,
   figures, patterns, composition
3. iconography: one sentence on probable symbolic or iconographic significance
   (say "unclear" if not determinable from the image)
```

**Response parsed** into `{label, description, iconography}` and pre-filled into
the editor. Full LLM response stored as `llm_reasoning` in the label record.

Source field: `"llm"` if saved without editing, `"llm-edited"` if text was
modified before saving.

Degrades gracefully: if `ANTHROPIC_API_KEY` not set, button shows an error
message rather than throwing.

### Cell 8 — Progress & export (static, nbconvert-friendly)

Runs once to produce shareable summary output:

- **Coverage bar**: N/total labeled (matplotlib horizontal bar)
- **Label frequency table**: top-20 labels, count + example thumbnail
- **Cluster-label heatmap**: seaborn `heatmap(cluster × label_cooccurrence)`
- **Per-cluster summary**: for each cluster, list its most common labels

This cell renders fully in the nbconvert HTML export.

---

## Label file schema

```json
{
  "../../frobenius_artifacts/analysis/motifs_norm/EBA-Div_.../000_motif_iou0.778.png": {
    "label": "full_figure_humanoid",
    "description": "Standing figure with outstretched arms, crosshatched body fill",
    "notes": "Similar to FoA_04-5578 panel_00/003",
    "iconography": "Likely Eshu/Elegba figure based on posture and door placement",
    "cluster": 2,
    "source": "llm-edited",
    "llm_suggestion": "carved_human_figure",
    "llm_reasoning": "Upright bilateral posture, limb articulation visible...",
    "timestamp": "2026-06-03T14:22:11"
  }
}
```

Keys are the same relative paths as `motif_paths_*.txt` — stable across notebook runs.

---

## What works in nbconvert HTML export

| Feature | Live Jupyter | nbconvert HTML |
|---|---|---|
| Cluster gallery (last rendered state) | ✓ interactive | ✓ static snapshot |
| In-context panel view | ✓ | ✓ (last active) |
| Multi-select panel overview | ✓ | ✓ (last selection) |
| Nearest neighbour strip | ✓ | ✓ |
| Label progress bar + heatmap | ✓ | ✓ |
| LLM suggestion button | ✓ | — |
| Label editor + save | ✓ | — |

For HTML snapshots showing the full labelled collection, run Cell 8 after
labeling a batch, then `bash src/python/export_html.sh motif_labeling`.

---

## Implementation order

1. **Cell 2** — data load + bbox lookup (foundation; validates panel/bbox wiring)
2. **Cell 5** — single-select context view (in-panel zoom; most impactful to get right)
3. **Cell 4** — cluster gallery + active-motif selection
4. **Cell 3 + 6** — label store + editor (nav buttons, save)
5. **Cell 5 multi-select** — panel overview with numbered boxes
6. **Cell 7** — LLM suggestion (add last; works without it)
7. **Cell 8** — progress / export cell

---

## Dependencies already in pyproject.toml

`anthropic`, `ipywidgets`, `Pillow`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

No new dependencies needed.
