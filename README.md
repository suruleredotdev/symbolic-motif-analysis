# symbolic-motif-analysis

A computer-vision pipeline for extracting, vectorizing, and clustering
individual carved motifs from photographs of African panel art — Yoruba door
panels, Ifa divination boards (opon Ifa), and Benin relief plaques — so that
recurring symbolic elements can be found, compared, and labeled across a
scattered archival record.

<img width="500" alt="Screenshot 2026-07-19 at 4 53 31 PM" src="https://github.com/user-attachments/assets/c98874f7-35c5-4c87-a880-1eea64030217" />
<!-- <img width="400" alt="Screenshot 2026-07-19 at 4 45 43 PM" src="https://github.com/user-attachments/assets/d29d8812-ae54-46d3-9dc3-655a382f1f95" /> -->
<img height="400" alt="25bc53f9-d6dc-4ea2-a960-15d963894e62_1188x1712" src="https://github.com/user-attachments/assets/266e8e97-4db9-4ea5-98f9-d4a6aebcc633" />


This is built upon earlier inquiries into [African symbolic systems](https://suruleredotdev.substack.com/p/african-symbolism-inquiry)
and [Yoruba art as historic storytelling](https://suruleredotdev.substack.com/p/on-yoruba-art-as-historic-storytelling),
as well as using some of the data aggregation tooling developed as part of our
earlier project [indexing African artifacts](https://suruleredotdev.substack.com/p/indexing-african-artifacts)


## Why this exists

Carved door panels and divination boards from Yoruba, Ifa, and related West
African traditions encode recurring visual vocabularies — figures, knotwork,
geometric borders, registers of narrative scenes - which we argue are not merely aesthetic but communicative in function.
Other symbolic/writing systems like Egyptian or Mayan hieroglyphs have been the subject of much inquiry and interpretation,
using established computational techniques. Our goal is to use such techniques - as well
as knowledge of history, mythology and cross-cultural semiotics - to establish an interpretive
framework that may be used to expand/enrich the corpus of Yoruba history.

As far as the history of such inqury, the Leo Frobenius-led [German archeaological expeditions of 1910-1912](https://archive.org/details/voiceofafricabei02frobuoft/page/n11/mode/thumb)
as welll as museum collections and archival photo libraries of the
[Frobenius Institut Bildarchiv](http://bildarchiv.frobenius-katalog.de/), provide a rich set of archaic panels with rich symbolic forms, as well as apparent inquiry into specific motifs and their meaning

<img width="625" alt="Screenshot 2026-07-19 at 5 52 42 PM" src="https://github.com/user-attachments/assets/aa62afae-50ec-4d30-a9b8-de678fccb6e3" />
<img height="300" alt="EBA-Div_00303_Ado_Ekiti_q166559_i1_panel_00" src="https://github.com/user-attachments/assets/405e279f-38fa-4b6d-b3b0-9aa77c788196" />
<img height="300" alt="EBA-B_00425_Ibadan_q97912_i1_panel_0_cropped" src="https://github.com/user-attachments/assets/781254ea-64f5-4807-8305-0ae230b585a6" />

This pipeline frames "find and cluster the motifs" as a computer-vision
problem: segment carved regions out of photographs of varying form and
quality (aged B&W photographs, pen-and-ink survey drawings, modern colour
photos), normalize them into a comparable visual space, embed them, and
cluster the embeddings to surface visual families — then let a human (with
LLM assistance) name and interpret what's been found.

## Pipeline architecture
<img width="1057" height="460" alt="Screenshot 2026-07-19 at 5 52 54 PM" src="https://github.com/user-attachments/assets/13aed61d-f276-476f-aa96-d019ae1fdd55" />

```
images/<source>.png                                  (photo or illustration)
    │
    ▼  Phase 1 — Preprocessing            panel_art/preprocess.py
XDoG line-art extraction                   (or binarize+clean if already line art)
    │
    ▼  Phase 2 — Panel detection           panel_art/panel_detect.py
Otsu threshold → connected components → per-panel crops
    │
    ▼  Phase 3 — Motif segmentation        panel_art/motif_segment.py
SAM (Segment Anything) auto-mask generation → scale-classified, NMS-filtered bboxes
    │
    ▼  Phase 4 — Vectorization             panel_art/vectorize.py
potrace bitmap → SVG, normalised to a 100×100 viewBox, stroke-only
    │
    ▼  Phase 5 — Similarity & clustering   panel_art/similarity.py
Hu moments + CLIP/DINO embeddings → HDBSCAN clustering (no preset cluster count)
    │
    ▼  Phase 6 — Interpretation            panel_art/layout.py + interpret.py
register/symmetry recovery + cluster statistics → Claude: cluster briefs →
panel readings → corpus synthesis
    │
    ▼
clusters.json, similarity_graph.json, annotated images, SVG motif library,
interpretation/ (cluster briefs, per-panel readings, corpus essay)
```

**Phase-by-phase rationale:**

- **Preprocessing (XDoG).** Extended Difference-of-Gaussians produces coherent
  closed-contour line art rather than the noisy, disconnected fragments Canny
  edge detection gives you — matching the pen-and-ink illustration style
  already present in part of the archive. Image type (photograph vs.
  illustration) is auto-detected from the standard deviation of the Laplacian,
  since illustrations are already near-binary and don't need the full XDoG
  treatment.
- **Panel detection.** Frobenius archive photos use a consistent plain grey/
  white studio background, so classic Otsu thresholding + morphological
  closing + connected components reliably splits a single photograph
  containing multiple physical door panels into individual per-panel crops —
  no neural detector needed.
- **Motif segmentation (SAM).** Carved wood has no colour contrast between
  motif and background — everything is the same wood tone — so segmentation
  has to work from edge/texture cues alone. SAM's zero-shot mask generation
  handles this without domain-specific training. Detections are classified
  into scale bands (zone / motif / element, by mask-area fraction) and
  deduplicated with IoU-based non-max suppression.
  - The pipeline has a second-generation segmentation mode,
    **IGSM (Image-Guided SAM prompting)**: instead of a uniform point grid,
    Otsu-binarize first, take connected components, then SAM point-prompt
    each component individually. This follows the approach in
    Fuentes-Ferrer et al. (2025), who found that grid-based automatic SAM
    tends to produce masks "mixed between" adjacent motifs on carved stone —
    the same failure mode this pipeline hits on carved wood. IGSM constrains
    SAM to look only where thresholding already found a discrete dark region,
    which better separates interlaced knotwork borders and dense narrative
    registers. Both modes are available side-by-side in `motif_tuning_v2.ipynb`.
- **Vectorization.** Each detected region is cropped from the line-art image,
  binarized, and traced to an SVG with `potrace` (via `svgpathtools`), then
  normalised to a fixed viewBox — giving a compact, scale/position-invariant
  shape representation for the original SVG-based similarity workflow
  (`panel_art/similarity.py`, Phase 5).
- **Similarity & clustering.** Two descriptor families feed HDBSCAN:
  Hu moments (7 values, invariant to translation/scale/rotation, cheap) for
  gross shape class, and CLIP (ViT-L/14) or DINOv2 embeddings for semantic
  visual similarity. HDBSCAN is density-based and needs no preset cluster
  count — motif families emerge from the data rather than being imposed on it.
  **The CLIP/DINO-embedding workflow (`motif_similarity.ipynb`) has superseded
  the original SVG/Hu-moment clustering (`panel_art/similarity.py`) as the
  primary similarity method** — it's more robust to the inconsistent line
  quality across photographs vs. illustrations.
- **Interpretation.** Clustering answers "which motifs recur"; it does not
  answer what a panel says. Phase 6 joins the three things a reading needs —
  the per-motif descriptions, the embedding clusters (evidence about recurrence
  that no single crop carries), and relative position — through three widening
  Claude passes: a brief per motif family across the whole corpus, then a
  register-by-register reading of each panel that uses those briefs, then a
  synthesis over everything. The spatial half is deterministic
  (`panel_art/layout.py` recovers registers, bilateral pairs, nesting, and
  reading order from bounding boxes alone), so it can be inspected before any
  API call. See [`INTERPRETATION.md`](./INTERPRETATION.md).

See [`PIPELINE.md`](./PIPELINE.md) for the exact CLI/notebook sequence, file
naming conventions, and current artifact counts from the last full run.

### Two generations of tooling
<img width="1036" height="388" alt="Screenshot 2026-07-19 at 5 53 10 PM" src="https://github.com/user-attachments/assets/922b2092-2dba-47de-a18d-235725309235" />

The pipeline evolved from a modular, CLI/notebook-per-phase design into a
single unified interactive notebook:

- **v1 — modular pipeline.** `panel_art/pipeline.py` orchestrates Phases 1–5
  end to end from the command line. `motif_tuning.ipynb` tunes SAM parameters
  interactively before a full run; `bbox_review.ipynb` lets you include/
  exclude individual detections per panel before cropping; `extract_crops.py`
  and `normalize_motifs.py` handle crop extraction and medium-agnostic
  normalization (photo / B&W scan / illustration → uniform line art);
  `motif_similarity.ipynb` computes CLIP embeddings, t-SNE layout, and
  HDBSCAN clusters; `motif_labeling.ipynb` is a cluster-gallery labeling UI
  with Claude-generated label suggestions.
- **v2 — unified pipeline (`motif_pipeline.ipynb`).** Consolidates the whole
  human-in-the-loop loop — Setup → Segment (review/manual-draw/SAM-Refine
  all in one UI) → Cluster → Gallery → Label (with LLM suggestions) →
  Interpret (Layout / Cluster Brief / Panel Reading / Corpus Synthesis) →
  Export — into one notebook with shared, cell-independent state.
  Every bbox, label, and cluster assignment records its own provenance
  (`manual` / `sam_prompted` / `llm` / `human`) and timestamp. This is the
  current recommended entry point for day-to-day use;
  see [`HITL_PLAN.md`](./HITL_PLAN.md), [`LABELING_PLAN.md`](./LABELING_PLAN.md),
  and [`INTERPRETATION.md`](./INTERPRETATION.md)
  for the design rationale behind the human-review and interpretation
  extensions layered onto the original automated pipeline.

## Repository layout

```
panel_art/                  Core package — the 5-phase pipeline
  preprocess.py                Phase 1: XDoG line-art extraction
  panel_detect.py               Phase 2: multi-panel ROI splitting
  motif_segment.py              Phase 3: SAM auto-mask generation + NMS
  vectorize.py                  Phase 4: bbox → normalised SVG
  similarity.py                 Phase 5: Hu moments/embeddings + HDBSCAN
  layout.py                     Phase 6: registers, symmetry, nesting from bboxes
  interpret.py                  Phase 6: cluster briefs → panel readings → synthesis
  pipeline.py                   CLI orchestration of Phases 1-5
  pipeline_state.py              Run tracking / staleness checks (HITL Phase 2)

scripts/                    Standalone data-prep & pipeline utility scripts
  filter_frobenius_panel_art.py  Metadata filter → curated panel-art allowlist
  scrape_frobenius.py            Frobenius Institut Bildarchiv scraper
  extract_motif_patches.py       SAM 2 automatic mask generation → patch crops
  embed_motif_patches.py         CLIP (ViT-L/14) + DINOv2 (ViT-L/14) embeddings
  describe_motif_patches.py      Claude-generated structured visual descriptions
  interpret_motifs.py            Phase 6 driver (cluster/panel/corpus stages)
  dino_perceptual_hash_demo.py   DINO-based perceptual hashing experiment

tests/                      pytest suite for the deterministic layers
  test_layout.py                 Register/symmetry/containment geometry
  test_interpret.py              Corpus loading, cluster stats, prompt assembly
  test_interpret_cli.py          CLI stages, resume, failure handling

extract_crops.py            PNG crop extraction from SAM detections (geometric
                             containment filtering to drop sub-crop artefacts)
normalize_motifs.py         Medium-agnostic normalisation (photo/B&W/illustration
                             → uniform line-art space, for CLIP embeddings)
export_html.sh              nbconvert notebooks → self-contained static HTML

motif_tuning.ipynb          v1: interactive SAM parameter tuning
motif_tuning_v2.ipynb       v2: automatic-grid vs. IGSM segmentation, side by side
bbox_review.ipynb           Per-panel detection review UI (include/exclude/draw)
motif_similarity.ipynb      CLIP embeddings + t-SNE + HDBSCAN scatter view
motif_labeling.ipynb        Cluster-gallery labeling UI with LLM suggestions
motif_pipeline.ipynb        v2 unified pipeline: segment→cluster→label→interpret→export
_gen_pipeline_notebook.py   Generates motif_pipeline.ipynb programmatically
_gen_labeling_notebook.py   Generates motif_labeling.ipynb programmatically

motif_embeddings_*.npy      Example embeddings from a prior full run
motif_paths_*.txt           Corresponding crop paths (edges / grayscale preprocessing)
motif_scatter_*.png         t-SNE scatter visualisations from that run
motif_params_edges.json     Parameters used to produce the above

PIPELINE.md                 End-to-end CLI/notebook reference, current run stats
HITL_PLAN.md                 Human-in-the-loop pipeline extension design
LABELING_PLAN.md             motif_labeling.ipynb design spec
INTERPRETATION.md            Phase 6 design: the three-pass interpretation join

panel_art_dataset/           Sample curated metadata + example queries
  README.md                    Dataset sources, categories, directory layout
  db_queries.sql               Reference SQL for a downstream searchable DB
  a1.json … a6.json             Example curated metadata records

plans/                       Dated planning documents (architecture, HITL extension)
todos/                       Dated TODO checklists tracked alongside the plans

pyproject.toml, requirements.txt, uv.lock   Python 3.12 project (uv-managed)
```

## Setup

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

External dependencies not installable via pip:

- **`potrace`** (system binary, used by `panel_art/vectorize.py`):
  `brew install potrace` (macOS) or `apt install potrace` (Linux).
- **SAM checkpoint** (Phase 3, v1 pipeline): download a ViT-B/L/H checkpoint
  from [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
  and pass its path via `--checkpoint`.
- **SAM 2** (used by `scripts/extract_motif_patches.py` and the v2 notebook):
  not on PyPI —
  `uv pip install git+https://github.com/facebookresearch/sam2.git`, then
  download a checkpoint, e.g. the ~40MB tiny model:
  ```bash
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
  wget https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
  ```
- **`ANTHROPIC_API_KEY`** (optional): enables LLM-generated motif label/
  description suggestions in `motif_labeling.ipynb` and
  `scripts/describe_motif_patches.py`, and the Phase 6 interpretation passes.
  All degrade gracefully without it — Phase 6's spatial analysis and cluster
  statistics run offline (`--dry-run`).

## Usage

**Metadata & images.** `panel_art/pipeline.py` consumes a JSON allowlist of
curated records (a `{"records": [...]}` object keyed by registration number —
see `load_allowed_images()` in `panel_art/pipeline.py`) plus a directory of
downloaded images and a `manifest.json` mapping registration numbers to
filenames. In the source project this allowlist was produced by
`scripts/filter_frobenius_panel_art.py` from a full Frobenius-archive dump
gathered by `scripts/scrape_frobenius.py` (and cross-referenced against
museum records gathered from the Art Institute of Chicago, SMB, and the
British Museum by a sibling TypeScript project). Neither the full archive
dump nor the downloaded images are included here — only the pipeline that
processes them. `panel_art_dataset/a1–a6.json` are example *museum records*
(id/title/culture/museum/image URL) illustrating the record schema those
upstream sources produce, not a ready-to-run metadata allowlist; see
`panel_art_dataset/README.md` for the full sourcing/format details.

**v1 — modular pipeline, end to end:**

```bash
# Step 0: filter raw archive metadata to panel-type records
python3 scripts/filter_frobenius_panel_art.py

# Steps 2-5: panel detection → SAM segmentation → vectorize → cluster
uv run python -m panel_art.pipeline \
  --image-dir  images/ \
  --metadata   panel_art_dataset/frobenius_panel_art.json \
  --img-manifest images/manifest.json \
  --checkpoint sam_vit_b_01ec64.pth \
  --out-dir    analysis/
```

Tune SAM parameters interactively first with `motif_tuning.ipynb` (or
`motif_tuning_v2.ipynb` to compare automatic-grid vs. IGSM), review/curate
detections with `bbox_review.ipynb`, extract crops with `extract_crops.py`,
normalize with `normalize_motifs.py`, then run `motif_similarity.ipynb` for
embeddings/clustering and `motif_labeling.ipynb` to label. Full step-by-step
instructions, flags, and output paths are in [`PIPELINE.md`](./PIPELINE.md).

**v2 — unified pipeline (recommended):**

```bash
uv run jupyter notebook motif_pipeline.ipynb
```

Run Setup once, then use the Segment / Cluster / Gallery / Label / Interpret
/ Export cells independently — each can be re-run without affecting the
others. See [`HITL_PLAN.md`](./HITL_PLAN.md) for the review-workflow design
(candidate pools, manual bbox drawing, dirty-state autosave, run versioning).

**Phase 6 — interpretation.** The final join runs either from the notebook's
Stage 5 or headless. Both write to the same `analysis/interpretation/`
directory and read each other's output, so a run can start in one and finish
in the other:

```bash
# Inspect the joined prompt and recovered geometry — no API calls
uv run python scripts/interpret_motifs.py --analysis-dir analysis/ --dry-run

# The default: ONE call joining the whole analysis into an interpretation
uv run python scripts/interpret_motifs.py \
  --analysis-dir analysis/ \
  --embeddings   motif_embeddings_edges.npy \
  --paths        motif_paths_edges.txt

# The three-pass flow — a call per cluster and per panel, so the model sees
# the actual crops and annotated panels. Use it where the depth is worth it.
uv run python scripts/interpret_motifs.py \
  --analysis-dir analysis/ \
  --embeddings   motif_embeddings_edges.npy \
  --paths        motif_paths_edges.txt \
  --stage all --resume
```

Output: `interpretation/clusters.{json,md}`, `interpretation/panels/<stem>.{json,md}`,
`interpretation/layouts/<stem>.json`, and `interpretation/corpus.md`.
See [`INTERPRETATION.md`](./INTERPRETATION.md) for the design.

**The shippable artifact.** `export_interpretation_site.py` turns the
interpretation into one self-contained HTML page — every panel image and motif
crop embedded, no server and no asset directory. Each panel is presented as an
annotated plate: numbered detection boxes tinted by motif family, the recovered
registers drawn as bands, and an apparatus column carrying the panel's reading
and, when a box is selected, that motif's annotation and its family.

```bash
uv run python scripts/export_interpretation_site.py \
  --analysis-dir analysis/ \
  --embeddings   motif_embeddings_edges.npy \
  --paths        motif_paths_edges.txt
# → analysis/interpretation/site.html
```

It needs no API key and degrades on partial data: panels without a reading
still render with their motifs and geometry, families without a brief still
show their members and statistics. Arrow keys walk a plate in the reading order
the layout pass computed. Use `--max-dim` / `--quality` to trade file size
against image detail.

**Bootstrapping labels.** Cluster briefs work on a corpus with zero labels,
so the fastest route to a complete v1 is to let them seed the labels:

```bash
# Briefs first (they need no labels), then propagate them — no API calls
uv run python scripts/interpret_motifs.py --analysis-dir analysis/ --stage clusters
uv run python scripts/label_motifs.py     --analysis-dir analysis/ --from-briefs

# Refine only where the family label is too coarse (one call per motif)
uv run python scripts/label_motifs.py --analysis-dir analysis/ --per-motif --clusters 3
```

Every label records its provenance, human annotation is never overwritten
without `--overwrite`, and the interpretation prompts present human labels as
the strongest evidence while marking generated ones as provisional. Briefs
store a fingerprint of the labels they were written from, so after correcting
annotations the CLI names just the families that need regenerating. See
[`INTERPRETATION.md`](./INTERPRETATION.md) → The annotation loop.

**Tests.** The deterministic layers (geometry, corpus loading, cluster
statistics, prompt assembly, CLI staging) are covered by a pytest suite that
needs no API key:

```bash
uv run pytest
```

**Static export.** Convert an executed notebook to a self-contained,
shareable HTML file (widgets render as their last saved state):

```bash
bash export_html.sh motif_similarity
```

## Roadmap

Tracked in more detail in `HITL_PLAN.md`, `LABELING_PLAN.md`, and `todos/`:

- Pipeline-run versioning (`pipeline_state.json` staleness checks) so a
  fresh notebook kernel can detect and reload the parameters of the last run.
- `cluster_review.ipynb` — dedicated drag-and-drop cluster-reassignment UI
  (partially landed inside `motif_pipeline.ipynb`'s Gallery/Cluster cells).
- Feeding manually-drawn bounding boxes back into SAM fine-tuning
  (SAM-2 LoRA or similar) — deferred, not yet started.
- Moving containment filtering (currently a post-hoc pass in
  `extract_crops.py`) upstream into Phase 3 segmentation.
- Vertical-column detection in `panel_art/layout.py` — registers are currently
  horizontal-only, so a column-organised panel collapses into one band.
- Near-duplicate detection across panels (the same carving photographed twice),
  so Phase 6 can relate motifs directly and not only through their cluster.

## Sharing data with collaborators

The image corpus is gitignored and large (~900 MB), most of it source
photographs that carry the tightest licensing restrictions. `make_subset.sh`
stages a small, shareable slice instead — the panels listed in
[`subset_panels.txt`](./subset_panels.txt) plus everything derived from them:

```bash
./make_subset.sh --tar
```

The default shortlist of 10 panels produces ~24 MB (204 files) and resolves to
just **5 source photographs**, since several panels are cut from the same plate.
Edit `subset_panels.txt` to change the selection, or pass `--list`. Use
`--no-source` to ship derived crops only, with no archive photographs at all.

Each bundle gets a `MANIFEST.md` recording its panels and source catalogue
numbers. See [`TERMS.md`](./TERMS.md) before sharing one onward.

Upload the staging directory to the shared Drive folder:

```bash
rclone sync subset_share gdrive:motif-subset --progress
```

### Running `motif_pipeline.ipynb` in Colab

Stage 0 locates its data automatically: local paths first, then an
already-mounted Drive, and only mounting Drive if nothing local matched. It
raises with every path it tried when none do.

**Preferred — download the bundle, no Drive mount.** `drive.mount()` cannot be
scoped to one folder; it exposes the collaborator's entire Drive to the
notebook. At 23 MB the bundle doesn't need a mount at all. Share
`subset_share.tar.gz` and pull it by file id:

```python
!git clone https://github.com/suruleredotdev/symbolic-motif-analysis.git
%cd symbolic-motif-analysis
!gdown <file-id> -O /content/bundle.tar.gz
!mkdir -p /content/subset_share && tar xf /content/bundle.tar.gz -C /content/subset_share --strip-components=1
```

Stage 0 finds `/content/subset_share` and never touches Drive. Set
`MOTIF_NO_DRIVE=1` to make that guarantee explicit — Stage 0 will then refuse to
mount and fail loudly instead. The trade-off: `gdown` needs the file to be
link-shared, which is weaker than named-account sharing. Rotate the link after
each labeling round, and keep it pointed at the derived-crops bundle
(`--no-source`) so a leaked link doesn't expose archive photographs.

**Alternative — mount Drive.** Stronger access control (named accounts,
revocable), at the cost of a full-Drive mount:

```python
!git clone https://github.com/suruleredotdev/symbolic-motif-analysis.git
%cd symbolic-motif-analysis
from google.colab import drive; drive.mount('/content/drive')
!ls /content/drive/MyDrive/motif-subset
```

Either way the clone matters — `panel_art/` and the notebook's sibling data
files are imported from the checkout, so without the `%cd` Stage 0 fails on
`import panel_art`. The `ls` should print `MANIFEST.md  analysis  images`; if it
prints `subset_share`, the upload is nested one level too deep.

Two things to tell collaborators:

- A folder shared with them will **not** appear under `MyDrive` until they add a
  shortcut to it (right-click the folder → *Organize* → *Add shortcut to
  Drive*). Set `MOTIF_DRIVE_FOLDER` if it is named something else.
- Set `MOTIF_LABELER` to their name before labeling. Stage 6 rewrites the whole
  label JSON on save, so a shared path means whoever saves last erases everyone
  else's work; with `MOTIF_LABELER=ade` they write `motif_labels.ade.json`
  instead, and the files are merged afterwards.

**What works in Colab, and what doesn't.** Not every stage survives the move:

| Stage | Colab |
|---|---|
| 0 — load state | works |
| 1 — Segment / manual draw | **degraded** — needs the `ipympl` widget backend for drag-to-draw; falls back to inline, so panels render but boxes can't be drawn |
| 2 — Embeddings / clustering | needs `torch` + `open_clip` (~2 GB of installs) |
| 3 — Gallery | works; degrades gracefully with no embeddings |
| 4 — Label | works — this is the informant-facing surface |
| 5 — Interpret | works; LLM suggestions need `ANTHROPIC_API_KEY` |
| 6 — Export | works, but writes into the data dir, so the Drive folder must be shared read-write |

Stage 4 references gallery state built in Stage 3, so informants run 0 → 3 → 4
and skip 1 and 2. Colab preinstalls everything those stages need.

## Provenance

Extracted from
[`suruleredotdev/african-artifacts#14`](https://github.com/suruleredotdev/african-artifacts/pull/14)
with `git filter-repo`, preserving original authorship and commit history for
every commit that touched this code. The sibling repository retains the
museum-collection web app and database that this pipeline's outputs were
originally built to feed into.

## License

MIT — see [`LICENSE`](./LICENSE).
