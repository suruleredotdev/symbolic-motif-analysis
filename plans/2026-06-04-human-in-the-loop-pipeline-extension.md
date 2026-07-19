Plan: Human-in-the-loop pipeline extension

---
Phase 0 — Pipeline integrity baseline (prerequisite, do first)

Before extending anything, lock down the current end-to-end flow so each step is trustworthy.

TODOs:

- 0a. Verify full run completes cleanly — confirm motif_similarity finishes and motif_labeling loads 145 motifs / 56 panels with no missing bboxes
- 0b. Add stale-input warnings to motif_labeling cell 2 — check mtime of .npy vs motifs_norm/ directory; warn if motifs are newer than embeddings (tells you to re-run motif_similarity before labeling)
- 0c. Parameter sidecar files — after each notebook writes .npy/.txt, also write a <name>_params.json next to it capturing every tuning variable used (PREPROCESS_MODE, clustering params, etc.). On next open, reload these and pre-fill the
widgets so you can reproduce an exact run
- 0d. Notebook header staleness checks — Cell 1 of each notebook checks expected input files exist and are non-empty; prints a clear "re-run step X first" message rather than crashing downstream

---
Phase 1 — Extended bbox_review: manual drawing + more SAM candidates

Problem: SAM at current thresholds misses some motifs; bbox_review can only approve/reject what SAM found, not add new regions.

Architecture:

Keep the existing file schema; add two things:

1. _detections_raw.json — SAM output at a lower threshold (store all masks, unfiltered), written alongside _detections.json. This is the "candidate pool" for bbox_review to pull from.
2. "source" field in _approved.json entries: "sam_approved" | "sam_candidate" | "manual". Existing extract_crops.py works unchanged (reads bbox dict regardless of source).

TODOs:

- 1a. motif_segment.py: write _detections_raw.json — run mask generation once, write two outputs: the current filtered _detections.json and an unfiltered _detections_raw.json (just area bounds, no NMS). One extra file per panel, no pipeline
change.
- 1b. bbox_review: "Show candidates" toggle — add a button to load _detections_raw.json for the current panel and display the rejected masks as faded/dashed boxes. User can click one to promote it to approved.
- 1c. bbox_review: draw-mode — add a "Draw bbox" button. When active, click+drag on the panel image creates a new bbox, stored as "source": "manual" in the approved state. Store in the same _approved.json so extract_crops.py picks it up
without changes.
- 1d. bbox_review: dirty-state tracking — show unsaved indicator when panel state changes; autosave on panel switch.

Deferred (follow-up): Manual bboxes feeding back into SAM fine-tuning (requires SAM-2 or LoRA; separate thread).

---
Phase 2 — Pipeline versioning and run tracking

Problem: Hard to know which files are stale after a step re-runs with new parameters. Hard to reproduce a prior run.

Architecture: Lightweight analysis/pipeline_state.json — one record per step per run. No new tooling, just a JSON file each script appends to.

{
  "extract_crops": {
    "last_run": "2026-06-04T14:30:00",
    "params": {"filter_containment": true, "containment_threshold": 0.8},
    "outputs": {"n_crops": 145, "n_panels": 56},
    "input_mtime": {"annotated/": "2026-06-04T12:00:00"}
  },
  "normalize_motifs": { ... },
  "motif_similarity": {
    "last_run": "...",
    "params": {"PREPROCESS_MODE": "grayscale", "USE_NORMALIZED": true, "perplexity": 30},
    "outputs": {"n_embeddings": 145}
  }
}

TODOs:

- 2a. pipeline_state.py helper — tiny module: record_run(step, params, outputs) writes/updates the JSON; check_staleness(step, input_paths) compares last-run mtime against input files and returns a warning string if stale. Used by scripts and
notebooks.
- 2b. Wire into scripts — extract_crops.py and normalize_motifs.py call record_run on completion.
- 2c. Wire into notebooks — Cell 1 calls check_staleness and prints a banner if inputs are newer than last run. Cell that writes .npy calls record_run.
- 2d. Parameter reload — Notebooks load pipeline_state.json at startup and set widget defaults from the last recorded params, so a fresh kernel reproduces the prior run automatically.

---
Phase 3 — Clustering review (follow-up, not EOD)

New cluster_review.ipynb or an added tab in motif_labeling:

- Drag-and-drop grid: move thumbnail cards between cluster columns
- Saves motif_clusters_override.json — maps path_norm → cluster_id
- motif_similarity.ipynb loads this at cell 3 and applies overrides after HDBSCAN, before writing embeddings

---
EOD scope

┌───────────────────────────────────────┬─────────┬───────────┐
│                 Phase                 │  Today  │ Follow-up │
├───────────────────────────────────────┼─────────┼───────────┤
│ 0 — baseline integrity                │ ✓ all   │ —         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 1a — _detections_raw.json             │ ✓       │ —         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 1b — candidates toggle in bbox_review │ ✓       │ —         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 1c — draw-mode in bbox_review         │ ✓       │ —         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 1d — dirty state / autosave           │ ✓       │ —         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 2a–2d — pipeline_state versioning     │ stretch │ ✓         │
├───────────────────────────────────────┼─────────┼───────────┤
│ 3 — cluster_review                    │ —       │ ✓         │
└───────────────────────────────────────┴─────────┴───────────┘

Phase 0 unblocks labeling immediately. Phase 1 is the most valuable feature addition and is self-contained (no re-architecture). Phase 2 is incremental hardening. Phase 3 is a follow-up notebook.

---
Does this match your intent, or do you want to adjust scope/priority before I start implementing?