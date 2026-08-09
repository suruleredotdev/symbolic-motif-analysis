"""Shared fixtures — a synthetic analysis directory shaped like a real pipeline run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PANEL_W, PANEL_H = 400, 900

# A two-register panel: a mirrored pair of figures over a centred figure,
# with a full-height interlace border behind everything.
PANEL_A_BOXES = [
    # index, x, y, w, h, scale
    (0, 20, 20, 360, 860, "register"),      # field / border, ~86% of panel
    (1, 40, 80, 90, 120, "motif"),          # upper left  ─┐ mirrored pair
    (2, 270, 80, 90, 120, "motif"),         # upper right ─┘
    (3, 150, 520, 100, 140, "motif"),       # lower centre
]

PANEL_B_BOXES = [
    (0, 30, 60, 100, 130, "motif"),
    (1, 260, 65, 100, 130, "motif"),
    (2, 140, 600, 120, 150, "motif"),
]


def _write_panel_png(path: Path, width: int = PANEL_W, height: int = PANEL_H) -> None:
    Image.new("RGB", (width, height), (210, 200, 185)).save(path)


def _detections(boxes) -> list[dict]:
    return [
        {
            "index": idx,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "scale": scale,
            "area_ratio": round((w * h) / (PANEL_W * PANEL_H), 5),
            "predicted_iou": 0.9,
            "stability_score": 0.9,
            "source": "sam_auto",
        }
        for idx, x, y, w, h, scale in boxes
    ]


@pytest.fixture
def analysis_dir(tmp_path: Path) -> Path:
    """A minimal but complete analysis/ tree: panels, approved bboxes, labels."""
    root = tmp_path / "analysis"
    panels = root / "panels"
    annotated = root / "annotated"
    panels.mkdir(parents=True)
    annotated.mkdir(parents=True)

    for stem, boxes in (("panel_a", PANEL_A_BOXES), ("panel_b", PANEL_B_BOXES)):
        _write_panel_png(panels / f"{stem}.png")
        dets = _detections(boxes)
        (annotated / f"{stem}_detections.json").write_text(json.dumps(dets))
        (annotated / f"{stem}_approved.json").write_text(json.dumps(dets))

    # Labels keyed on crop paths, exactly as motif_labeling.ipynb writes them.
    labels = {
        "../../analysis/motifs_norm/panel_a/001_motif_iou0.900.png": {
            "label": "standing_figure",
            "description": "Upright figure with raised arms",
            "iconography": "possible attendant",
            "notes": "",
            "cluster": 0,
            "source": "human",
            "timestamp": "2026-06-01T10:00:00",
        },
        "../../analysis/motifs_norm/panel_a/002_motif_iou0.900.png": {
            "label": "standing_figure",
            "description": "Mirror of #1",
            "iconography": "possible attendant",
            "notes": "",
            "cluster": 0,
            "source": "llm",
            "timestamp": "2026-06-01T10:01:00",
        },
        "../../analysis/motifs_norm/panel_a/003_motif_iou0.900.png": {
            "label": "seated_figure",
            "description": "Frontal seated figure",
            "iconography": "unclear",
            "notes": "",
            "cluster": 1,
            "source": "human",
            "timestamp": "2026-06-01T10:02:00",
        },
        "../../analysis/motifs_norm/panel_a/000_motif_iou0.900.png": {
            "label": "interlace_border",
            "description": "Full-height knotwork",
            "iconography": "unclear",
            "notes": "",
            "cluster": 2,
            "source": "human",
            "timestamp": "2026-06-01T10:03:00",
        },
        "../../analysis/motifs_norm/panel_b/000_motif_iou0.900.png": {
            "label": "standing_figure",
            "description": "Upright figure",
            "iconography": "unclear",
            "notes": "",
            "cluster": 0,
            "source": "llm",
            "timestamp": "2026-06-01T10:04:00",
        },
        "../../analysis/motifs_norm/panel_b/001_motif_iou0.900.png": {
            "label": "standing_figure",
            "description": "Upright figure",
            "iconography": "unclear",
            "notes": "",
            "cluster": 0,
            "source": "llm",
            "timestamp": "2026-06-01T10:05:00",
        },
        "../../analysis/motifs_norm/panel_b/002_motif_iou0.900.png": {
            "label": "seated_figure",
            "description": "Frontal seated figure",
            "iconography": "unclear",
            "notes": "",
            "cluster": 1,
            "source": "human",
            "timestamp": "2026-06-01T10:06:00",
        },
    }
    (root / "motif_labels.json").write_text(json.dumps(labels, indent=2))
    return root


@pytest.fixture
def embeddings(tmp_path: Path) -> tuple[Path, Path]:
    """Embeddings whose geometry matches the fixture's cluster assignments.

    Cluster 0 members sit near one axis, cluster 1 near a second, cluster 2 near
    a third, so cohesion and centroid exemplars have something real to measure.
    """
    keys = [
        ("panel_a", 0, 2), ("panel_a", 1, 0), ("panel_a", 2, 0), ("panel_a", 3, 1),
        ("panel_b", 0, 0), ("panel_b", 1, 0), ("panel_b", 2, 1),
    ]
    rng = np.random.default_rng(0)
    vectors = []
    paths = []
    for stem, idx, cluster in keys:
        base = np.zeros(8)
        base[cluster] = 1.0
        vectors.append(base + rng.normal(scale=0.05, size=8))
        paths.append(f"../../analysis/motifs_norm/{stem}/{idx:03d}_motif_iou0.900.png")

    npy = tmp_path / "emb.npy"
    txt = tmp_path / "paths.txt"
    np.save(npy, np.asarray(vectors))
    txt.write_text("\n".join(paths))
    return npy, txt
