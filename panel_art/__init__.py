"""
panel_art — Frobenius panel art symbolic motif analysis pipeline.

Phases:
    1. preprocess   — XDoG line-art extraction (photo → boundary-ified image)
    2. panel_detect — split multi-panel images into individual ROI crops
    3. motif_segment — SAM-2 auto-mask generation + NMS bounding boxes
    4. vectorize    — motif region → normalised SVG via contour tracing
    5. similarity   — DINO/CLIP embeddings + HDBSCAN clustering
    6. layout       — registers, symmetry, and nesting recovered from bboxes
       interpret    — cluster briefs → panel readings → corpus synthesis
    7. pipeline     — end-to-end orchestration + CLI

See plans/2026-04-15-frobenius-motif-analysis.md for architecture overview.
"""
