# Phase 6 — Interpretation

Design notes for `panel_art/layout.py`, `panel_art/interpret.py`, and
`scripts/interpret_motifs.py`: the stage that joins the cluster analysis into a
single reading of the collection.

---

## The problem this solves

Phases 1–5 produce fragments. After a full run you have, per motif, a crop, a
CLIP embedding, an HDBSCAN cluster id, a bounding box, and — if someone has
been through `motif_labeling.ipynb` — a short label and description. What you
do *not* have is any statement about what a panel says, or what the collection
shares.

The gap is not "run an LLM over the crops". A per-crop description is what
`scripts/describe_motif_patches.py` already does, and reading those descriptions
in sequence does not produce an interpretation: it produces a list. Three things
have to be joined before a reading is possible:

1. **What each motif is** — the descriptions and labels.
2. **What family it belongs to** — the embedding clusters, which are evidence
   about recurrence that no single crop carries.
3. **Where it sits** — relative position, register membership, symmetry,
   nesting. A figure flanked by two mirrored attendants over a band of
   interlace is a composition; the same three crops in a list are not.

Phase 6 is the join.

---

## Two paths: start cheap

There are two ways to run the join, and **the cheap one is the default**.

| | `--stage direct` (default) | `--stage all` |
|---|---|---|
| API calls | **1** | 1 per cluster + 1 per panel + 1 |
| Sees the images | no | yes |
| Input | the analysis already on disk | same, plus crops and annotated panels |
| Good for | the whole-collection interpretation | depth on a panel that deserves it |

At the corpus sizes this pipeline produces, the entire join — every motif
label and description, every cluster's statistics, every panel's recovered
layout — is only tens of thousands of tokens. That fits in one request with
room to spare, so **one call is the right default** and the three-pass flow
has to earn its cost.

What the extra calls buy is images. A per-panel call can attach the annotated
panel so the model actually looks at the carving; a per-cluster call can
attach exemplar crops. Everything else the three passes produce, the single
call can produce too — from the pipeline's *description* of the images rather
than the images themselves, which the direct prompt says explicitly so the
model can qualify readings that depend on it.

Recommended order: run `--stage direct` first, read it, and escalate to
`--stage panels --panels <stem>` for the specific panels where the description
clearly isn't enough.

## Three widening passes

Each pass consumes the one before it. That ordering is what makes the output
cohere rather than repeat itself — a panel reading can refer to "the interlace
family that also frames seven other panels" because the cluster brief
established that family first, across the whole corpus, before any single panel
was read.

```
   embeddings ─┐
   labels ─────┼──▶  ① cluster brief   (one call per cluster)
   crops ──────┘            │
                            ▼
   layout ────────────▶ ② panel reading  (one call per panel)
   panel image ───────▶     │
                            ▼
                       ③ corpus synthesis (one call, over everything)
```

**① Cluster brief** — what is this visual family, seen across every panel it
appears on? Receives the exemplar crops closest to the cluster centroid, the
family's spread across panels and scales, its embedding cohesion, its nearest
sibling clusters, and whatever labels its members already carry. Returns a
name, a visual definition, an account of internal variation, a note on what the
distribution implies about function, an iconographic reading, and a confidence.

**② Panel reading** — what does *this* panel say? Receives the panel image with
every detection outlined and numbered, the deterministic spatial description
(below), the per-motif notes, and the pass-① briefs for every family present on
the panel. Returns a register-by-register reading, an account of the
composition, a narrative reading, links to the wider corpus, and explicit
uncertainties.

**③ Corpus synthesis** — what does the collection say? Receives every brief and
every panel reading, no images. Returns a Markdown essay.

---

## The deterministic half

`panel_art/layout.py` computes the spatial structure without a model. This
matters for three reasons: it is cheap, it is testable, and when an
interpretation looks wrong you can check whether the geometry it was given was
wrong first (`--dry-run`, or the notebook's **Layout** button).

| What it recovers | How |
|---|---|
| Relative geometry | Centre, size, and area fraction normalised to the panel |
| Zone | 3×3 naming — "upper left", "centre", "lower right" |
| **Registers** | Single-link clustering on vertical spans; a new band starts when the next motif's span doesn't overlap the band so far. No register count is imposed. |
| Field / ground | Detections covering ≥55% of the panel are treated as the ground the rest sits on, not as members of a register |
| Reading order | Register-major, then left to right; field detections last |
| Adjacency | *k* nearest neighbours per motif, with direction named in picture-plane terms ("upper-left of", not "northwest") |
| Bilateral pairs | Motifs that reflect across the panel's vertical mid-axis and match in size — the flanking-attendant device |
| Nesting | Which detections enclose which, at ≥80% area containment |

Register detection is the load-bearing piece. Carved door panels are organised
into stacked horizontal bands, and reading a panel is largely reading those
bands in order; recovering them from bboxes gives the interpretation its
skeleton. The algorithm degrades gracefully: a panel with no banded structure
comes back as one register per motif rather than failing.

`render_layout_text()` turns all of this into the compact block sent to the
model. It is deliberately plain — the model sees the annotated image alongside
it, so the text's job is to index and measure what the image shows, not to
re-describe it.

## The embedding half

`compute_cluster_stats()` is the other non-LLM input:

- **Cohesion** — mean pairwise cosine similarity within a cluster. Tells the
  model whether it is looking at one tight family or several merged things,
  which is a question it cannot answer from four exemplar crops alone.
- **Centroid exemplars** — the members nearest the cluster centroid, so what
  gets sent is the family's centre of gravity rather than an arbitrary
  first-four sample.
- **Cluster adjacency** — nearest *other* clusters by centroid similarity. This
  is what lets an interpretation say two families are variants of one another
  rather than treating every cluster as an island.

All three degrade to `None`/empty when no embeddings are supplied; the passes
still run on labels and counts alone, just with less to go on.

---

## Prompting

One system prompt (`interpret.SYSTEM_PROMPT`) across all three passes. Its
substantive job is to keep three kinds of claim separable in the output:

> (a) what the image shows, (b) what the pipeline's grouping implies, and
> (c) what is being inferred from cultural knowledge.

It also states plainly that the clusters are data-derived rather than
established iconographic categories, that the existing labels are a mix of
human and model guesses, and that segmentation errors are common — a detection
may be a fragment, a duplicate, or an artefact of the photograph. Without that,
readings tend to treat a bad bbox as a carved element and build on it.

Passes ① and ② use structured outputs (`output_config.format`) rather than
asking for JSON in prose; pass ③ is free-form Markdown. Every call uses
adaptive thinking and streams, because a corpus synthesis at high effort runs
well past the non-streaming timeout.

Prompt construction is separated from the API call (`build_cluster_prompt`,
`build_panel_prompt`, `build_corpus_prompt`) so that `--dry-run` prints exactly
what would be sent, and so the assembly is unit-testable without a key.

---

## Output

```
analysis/interpretation/
  clusters.json         cluster id → brief (+ the statistics it was given)
  clusters.md           the same, as a reference sheet
  layouts/<stem>.json   deterministic geometry — written even on --dry-run
  panels/<stem>.json    structured panel reading (+ the layout it used)
  panels/<stem>.md      the same reading, rendered for reading
  corpus.md             the synthesis essay
```

Every artefact records the model and timestamp that produced it, matching the
provenance convention the rest of the pipeline uses (`HITL_PLAN.md`). Cluster
briefs are checkpointed after each call, so an interrupted run resumes with
`--resume` rather than starting over.

---

## Usage

```bash
# Inspect the joined prompt — no API key, no calls
python3 scripts/interpret_motifs.py \
  --analysis-dir frobenius_artifacts/analysis --dry-run

# The default: ONE call over the whole corpus
python3 scripts/interpret_motifs.py \
  --analysis-dir frobenius_artifacts/analysis \
  --embeddings   motif_embeddings_edges.npy \
  --paths        motif_paths_edges.txt

# The three-pass flow, when a panel deserves the model's eyes on it
python3 scripts/interpret_motifs.py \
  --analysis-dir frobenius_artifacts/analysis \
  --embeddings   motif_embeddings_edges.npy \
  --paths        motif_paths_edges.txt \
  --stage all --resume
```

Or interactively, in `motif_pipeline.ipynb` Stage 5 — **Layout** (no API),
**Cluster Brief**, **Panel Reading**, **Corpus Synthesis**. The notebook and
the CLI write to the same directory and read each other's output, so a run can
start in one and finish in the other.

---

## The site

`scripts/export_interpretation_site.py` is the shippable end of the pipeline: it
reads the analysis directory and whatever `interpretation/` contains, and writes
a single self-contained HTML file.

The page is built as **a plate and its apparatus**, borrowing the archive's own
idiom — a numbered figure with letterpress notes beside it — rather than a
dashboard. Two things it draws are things the pipeline *discovered* and that are
invisible in the JSON:

- **Registers as bands.** The structure `layout.py` recovered from bounding
  boxes alone, laid over the image where it can be checked against the carving.
- **Boxes tinted by family.** The same tint recurring across different plates is
  the corpus-level argument, made visible in one glance.

Everything is embedded as a data URI, so the file works from disk, over email,
or on any static host. It needs no API key. It also degrades: a panel with no
reading still renders with its motifs and geometry, and a family with no brief
still shows its members and statistics — a partial interpretation gives a
partial but honest page rather than an error.

`panel_art/site_template.py` holds the page (HTML, CSS, JS) so the exporter
stays readable as data assembly; the exporter passes it one JSON payload.

## The annotation loop

Labels sharpen the interpretation, and the interpretation bootstraps the
labels. The mechanics that make that a loop rather than a one-way trip:

**Clusters do not require labels.** `analysis/clusters.json` is the
authoritative record of assignments, written by the notebook's *Save Clusters*
and read by `load_corpus()`. Cluster briefs work on a corpus with zero labels —
they are written from centroid exemplars and the embedding statistics.

**Bootstrap the corpus for free.** `scripts/label_motifs.py --from-briefs`
propagates each brief's name and visual definition onto its unlabelled members.
No API calls, and the labels agree with one another by construction because
they come from the same characterisation. `--per-motif` spends one call each
where a family label is genuinely too coarse.

**Provenance decides authority.** Every label records its source —
`cluster-brief`, `llm`, `llm-edited`, or `human`. A missing source counts as
human, which is the safe default: it protects existing annotation. Generated
labels are freely replaced by later passes; a human label is never overwritten
without `--overwrite`. In the prompts, human annotation is presented first and
explicitly as *"the strongest evidence available"*, while generated labels are
marked *"provisional, may be wrong, and safe to contradict"*.

**Two writers, one file.** `motif_labels.json` is written by both the
notebook and `label_motifs.py`, often with a notebook kernel holding a stale
copy while the script runs. Three rules keep them from fighting:

- *One motif, one key.* Both writers use `motif_label_key()`. They previously
  did not — the notebook baked the motif's IoU into the key and the script
  hardcoded `1.000` — so the same motif got two entries and whichever sorted
  later silently won on load. Legacy duplicates are collapsed on write.
- *The script only fills gaps.* It never edits an existing label. Replacing
  generated ones is `--refresh-generated`; replacing human ones is
  `--overwrite`.
- *The notebook writes only what you edited.* Save All State merges the
  session's edits (`MotifRecord.dirty`) into whatever is on disk, leaving
  everything else alone. Note the flag, not the label source, is what marks a
  change: a motif loaded from disk already carries `source="human"` without
  anyone having touched it this session, so source alone would rewrite the
  whole file and regress anything improved on disk meanwhile.

**Briefs know when they are out of date.** Each brief stores a fingerprint of
the member labels it was written from (`label_fingerprint`). After annotation
work, `interpret_motifs.py` names the families whose labels have moved:

```
  3 motifs annotated by a person (weighted above model-generated labels)
  1 cluster brief(s) predate the current labels: 7
```

So a correction costs one brief to regenerate, not the whole corpus.

The intended cycle: cluster → brief → `--from-briefs` → read the interpretation
→ correct the labels that are wrong → regenerate the briefs flagged stale →
re-synthesise. Each turn, more of the corpus rests on human judgement and less
on the model's first guess, and the prompts know which is which.

## Known limits

- **Registers are horizontal only.** A panel organised into vertical columns
  comes back as one register containing everything. Column detection would be
  the same algorithm on the other axis, gated on aspect ratio.
- **Cohesion is scale-free but not calibrated.** The "below ~0.5 suggests a
  merged cluster" guidance in the prompt is a rule of thumb from CLIP
  embeddings on this corpus, not a measured threshold.
- **Pass ③ sees no images.** The synthesis works from the two earlier passes'
  text, so an error in a panel reading propagates rather than being caught.
- **No cross-panel motif matching below the cluster level.** Two motifs in the
  same family are related through the family, not directly; near-duplicate
  detection across panels (the same carving photographed twice) is not done.
