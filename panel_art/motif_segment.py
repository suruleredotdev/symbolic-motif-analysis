"""
motif_segment.py — Phase 3: SAM automatic motif segmentation

Uses Meta's Segment Anything Model (SAM-1 / ViT-B) to generate zero-shot
segmentation masks for each panel crop, then filters and classifies them
into two scale levels:

  register — large compositional band     (> 25% of panel area)
             Typically one of the horizontal thirds of a vertical panel:
             a full knotwork band, a row of figures, a border register.

  motif    — individual carved unit       (3–25%)
             A complete figure (humanoid, animal), a complete geometric cell,
             or a self-contained symbolic unit within a register.

Sub-motif fragments (body parts, partial pattern sections) are suppressed
by a 3% minimum area floor and area-sorted NMS (IoU 0.35) that prefers
the largest mask when two candidates overlap substantially.

Why SAM over traditional CV: carved wood panels have no colour contrast
between motif and background (same wood tone throughout). SAM's edge and
texture cues provide robust region proposals without domain-specific training.

Model: SAM ViT-B (~375 MB) — good balance of quality and speed. The ViT-H
checkpoint can be swapped in for higher recall at the cost of more memory.

Install:
  uv pip install segment-anything
  # Checkpoint (already handled by pipeline setup):
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

CLI usage:
  uv run python -m panel_art.motif_segment <panel_image> [<panel_image> ...]
  uv run python -m panel_art.motif_segment --checkpoint /path/to/sam.pth <img>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# SAM is optional — pipeline degrades gracefully if unavailable
try:
    import torch
    from segment_anything import (
        SamAutomaticMaskGenerator,
        SamPredictor,
        sam_model_registry,
    )
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT",
    str(Path(__file__).parent.parent / "sam_vit_b_01ec64.pth"),
)
DEFAULT_MODEL_TYPE = "vit_b"

# Mask quality thresholds.
# SAM quality scores are calibrated for clean, distinct objects. Large carved
# texture regions (whole register bands, full knotwork panels) score lower
# because they have ambiguous boundaries by SAM's metric. Lowering these lets
# whole-figure and whole-band masks through; NMS + area filter cleans up the rest.
DEFAULT_IOU_THRESH       = 0.70
DEFAULT_STABILITY_THRESH = 0.75
# NMS: when two boxes overlap > 40%, drop the smaller one.
# Sort by area so large whole-figure/whole-band masks win over body parts.
# 0.40 (up from 0.35) avoids over-suppressing adjacent small motifs at the
# lower 1% min_area floor.
DEFAULT_NMS_IOU  = 0.40
# 1% floor: admits individual carved symbols (1–3% of panel area) while
# suppressing sub-pixel noise. Was 3% which over-suppressed on dense panels
# such as Ado Ekiti (153/157 raw masks dropped).
DEFAULT_MIN_AREA = 0.01
# 85% ceiling: allows large register bands (e.g. a full knotwork body spanning
# 70-80% of a narrow vertical panel) while still blocking the degenerate
# "entire panel" catch-all mask that SAM sometimes generates.
DEFAULT_MAX_AREA = 0.85
# 7.0 aspect ratio cap: allows tall standing figures (≈2:7 proportions).
# Was 5.0 which clipped elongated humanoid forms.
DEFAULT_MAX_ASPECT = 7.0

# Annotation colours (R, G, B) for each scale level
SCALE_COLOURS = {
    "register": (255, 80,  80),   # red  — full horizontal band (~1/3 of panel)
    "motif":    (80,  200, 80),   # green — individual carved unit
}

PALETTE = list(SCALE_COLOURS.values()) + [
    (200, 80, 255), (255, 165, 50), (50, 220, 200), (255, 220, 50),
]

# SAM grid density: 32 points gives better coverage of small carved symbols
# (1–3% of panel area) without the full cost of 64.
DEFAULT_POINTS_PER_SIDE = 32

# Raw candidate pool thresholds — much lower than the quality gate so
# _detections_raw.json contains masks the main filter rejects.
# These are passed to SamAutomaticMaskGenerator internally; our own
# filter_and_nms() then applies DEFAULT_IOU_THRESH / DEFAULT_STABILITY_THRESH
# for the curated _detections.json output.
RAW_IOU_THRESH       = 0.40
RAW_STABILITY_THRESH = 0.50


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One detected region within a panel."""
    index: int
    bbox: dict          # {x, y, w, h} in pixels
    scale: str          # "zone" | "motif" | "element"
    area_ratio: float
    predicted_iou: float
    stability_score: float
    segmentation: object = field(repr=False, default=None)  # H×W bool array
    # Set by prompted_segment(): what the review queue is ordered by.
    edge_density: float = 0.0
    novelty: float = 1.0
    rank_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "bbox": self.bbox,
            "scale": self.scale,
            "area_ratio": round(self.area_ratio, 5),
            "predicted_iou": round(self.predicted_iou, 4),
            "stability_score": round(self.stability_score, 4),
        }


# ── Scale classification ──────────────────────────────────────────────────────

def classify_scale(area_ratio: float) -> str:
    """
    Map mask area (as fraction of panel) to a semantic scale level.

      register : > 25%  — a full horizontal band, roughly one third of a
                          vertical panel (knotwork band, row of figures,
                          border register).
      motif    : 3–25% — a complete carved unit: a whole humanoid figure,
                          a complete geometric cell, a single Ifa symbol.
                          Sub-motif fragments (arms, legs, partial knots)
                          are excluded by the 3% min-area floor before
                          this function is reached.

    Post-inference step for annotation of the detected segmentation masks,
    for our usecase.
    """
    if area_ratio > 0.25:
        return "register"
    return "motif"


# ── Filtering and NMS ─────────────────────────────────────────────────────────

def _iou(a: list[int], b: list[int]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def filter_and_nms(
    masks: list[dict],
    img_area: int,
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float = DEFAULT_MAX_AREA,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    stability_thresh: float = DEFAULT_STABILITY_THRESH,
    nms_iou: float = DEFAULT_NMS_IOU,
    max_aspect: float = DEFAULT_MAX_ASPECT,
) -> list[dict]:
    """
    Quality + geometry filter followed by greedy NMS.

    Sorted by predicted_iou descending before NMS so higher-quality masks
    win when two masks overlap significantly.
    """
    kept = []
    for m in masks:
        x, y, w, h = m["bbox"]
        area_ratio = m["area"] / img_area

        if area_ratio < min_area or area_ratio > max_area:
            continue
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue
        if m["predicted_iou"] < iou_thresh:
            continue
        if m["stability_score"] < stability_thresh:
            continue
        kept.append(m)

    # Sort by area descending so larger masks (whole figures, whole bands) win
    # NMS over smaller body-part masks that overlap them.
    kept.sort(key=lambda m: m["area"], reverse=True)

    final, suppressed = [], set()
    for i, m in enumerate(kept):
        if i in suppressed:
            continue
        final.append(m)
        for j in range(i + 1, len(kept)):
            if j not in suppressed and _iou(m["bbox"], kept[j]["bbox"]) > nms_iou:
                suppressed.add(j)

    return final


# ── Model loading ─────────────────────────────────────────────────────────────

_sam_model_cache: dict[str, object] = {}
_generator_cache: dict[str, "SamAutomaticMaskGenerator"] = {}


def _resolve_device() -> str:
    if not SAM_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # SAM-1 uses float64 operations internally; MPS (Apple Metal) does not
    # support float64, so we fall back to CPU even when MPS is present.
    return "cpu"


def _load_sam_model(
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
) -> object:
    """Load and cache the SAM model — shared between generator and predictor."""
    if not SAM_AVAILABLE:
        raise ImportError(
            "SAM not available. Install with:\n"
            "  uv pip install segment-anything\n"
            "Then download a checkpoint:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        )

    cache_key = f"{checkpoint}:{model_type}"
    if cache_key in _sam_model_cache:
        return _sam_model_cache[cache_key]

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            f"Set SAM_CHECKPOINT env var or pass --checkpoint."
        )

    device = _resolve_device()
    print(f"  Loading SAM {model_type} on {device} …", flush=True)
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    _sam_model_cache[cache_key] = sam
    return sam


def load_generator(
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    points_per_side: int = DEFAULT_POINTS_PER_SIDE,
) -> "SamAutomaticMaskGenerator":
    """
    Load SAM and return a configured automatic mask generator.
    Results are cached by checkpoint path so the model is only loaded once.
    """
    cache_key = f"{checkpoint}:{model_type}:{points_per_side}"
    if cache_key in _generator_cache:
        return _generator_cache[cache_key]

    sam = _load_sam_model(checkpoint, model_type)

    # Use low internal thresholds so generator.generate() returns a wide
    # candidate pool.  Our filter_and_nms() applies the stricter quality
    # gate (DEFAULT_IOU_THRESH / DEFAULT_STABILITY_THRESH) for curated output.
    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=RAW_IOU_THRESH,
        stability_score_thresh=RAW_STABILITY_THRESH,
        min_mask_region_area=200,   # absolute px² — drops sub-pixel noise
    )
    _generator_cache[cache_key] = generator
    return generator


# ── Per-panel pipeline ────────────────────────────────────────────────────────

def segment_panel(
    panel_img: np.ndarray,
    generator: "SamAutomaticMaskGenerator",
    min_area: float = DEFAULT_MIN_AREA,
    max_area: float = DEFAULT_MAX_AREA,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[Detection]:
    """
    Run SAM on one panel image and return filtered, classified detections.

    Parameters
    ----------
    panel_img : H×W×3 uint8 RGB array (a panel crop from Phase 2)
    generator : pre-loaded SAM mask generator (reused across calls)

    Returns
    -------
    List of Detection objects sorted by area descending (largest first).
    """
    img_h, img_w = panel_img.shape[:2]
    img_area = img_h * img_w

    # SAM expects RGB uint8
    raw_masks = generator.generate(panel_img)

    kept = filter_and_nms(
        raw_masks, img_area,
        min_area=min_area,
        max_area=max_area,
        nms_iou=nms_iou,
    )

    detections = []
    for idx, m in enumerate(kept):
        x, y, w, h = [int(v) for v in m["bbox"]]
        area_ratio = m["area"] / img_area
        detections.append(Detection(
            index=idx,
            bbox={"x": x, "y": y, "w": w, "h": h},
            scale=classify_scale(area_ratio),
            area_ratio=area_ratio,
            predicted_iou=float(m["predicted_iou"]),
            stability_score=float(m["stability_score"]),
            segmentation=m["segmentation"],
        ))

    # Sort largest → smallest so register-level context comes first
    detections.sort(key=lambda d: d.area_ratio, reverse=True)
    for i, d in enumerate(detections):
        d.index = i

    return detections


# ── Prompted re-segmentation (HITL feedback) ─────────────────────────────────
#
# Defaults below were set from a hand review of every SAM Generate candidate on
# FoA_04-5947_q48640_i1_panel_00 (25 candidates, 7 kept), then checked with
# scripts/eval_prompted_segment.py against the boxes that were approved:
#
#   - Blank surface (tray recesses, the mirror, background) had edge density
#     0.03–0.05; every kept motif had 0.21–0.34. A 12% floor drops the blanks.
#   - SAM's predicted IoU was *highest* on blank surface (~0.95), so it is used
#     as a gate only, never to order the queue.
#   - The remaining misses were clumps of two or three neighbouring figures.
#     They match a motif on every measure except size, and one board's figures
#     share a scale — so the size window comes from the panel's own approved
#     boxes once there are enough of them, and a candidate that swallows two
#     already-kept candidates is dropped.
#   - Grid probes landed between figures and a median-sized box prompt made SAM
#     fill the box. Probes now sit on local peaks of carving density (roughly
#     one per figure), with negative points on the neighbouring peaks so SAM
#     separates adjacent figures cut from the same wood.

PROMPT_MIN_SCORE = 0.80
PROMPT_MIN_EDGE_DENSITY = 0.12
# Small carved figures on a dense board are ~0.6% of the panel as a mask
# (the bbox is larger than the mask); a 1% floor rejected most of them.
PROMPT_MIN_AREA = 0.005
# Long thin masks were strips along the board's frame, not motifs.
PROMPT_MAX_ASPECT = 4.0
PANEL_TEMPLATE_MIN = 3          # approved boxes on this panel before its own scale is used
PANEL_ENVELOPE_MARGIN = 0.35
GLOBAL_ENVELOPE_MARGIN = 0.5
MAX_PROBES = 160

# The behaviour before the defaults above: a sparse grid, a 3% edge floor and
# every passing mask kept in probe order. Kept so the two can be compared.
LEGACY_PROMPT_PARAMS = dict(
    min_edge_density=0.03, min_area=0.01, max_aspect=100.0,
    probe="grid", negative_points=False, pick="all",
)


def _trim_outsized(templates: list[dict], factor: float = 2.5) -> list[dict]:
    """Drop boxes more than *factor* × the median area.

    A panel's approved boxes mix figures with the odd large form (a whole
    border band, the curved horn bundle on an opon); one of those drags the
    median prompt size up until every prompt spans two figures.
    """
    if len(templates) < 3:
        return templates
    areas = sorted(t["w"] * t["h"] for t in templates)
    med = areas[len(areas) // 2]
    return [t for t in templates if t["w"] * t["h"] <= factor * med]


def _size_envelope(templates: list[dict], margin: float = 0.5, min_n: int = 3):
    """Compute (w_lo, w_hi, h_lo, h_hi, med_w, med_h) from bbox templates.

    Uses the 10th–90th percentile of widths/heights (min/max below ten boxes),
    expanded by *margin* (0.5 = 50%).  Returns None if fewer than *min_n*.
    """
    if len(templates) < min_n:
        return None
    widths  = sorted(t["w"] for t in templates)
    heights = sorted(t["h"] for t in templates)
    n = len(widths)
    p10, p90 = max(n // 10, 0), min(n - 1 - n // 10, n - 1)
    w_lo = int(widths[p10]  * (1 - margin))
    w_hi = int(widths[p90]  * (1 + margin))
    h_lo = int(heights[p10] * (1 - margin))
    h_hi = int(heights[p90] * (1 + margin))
    med_w = widths[n // 2]
    med_h = heights[n // 2]
    return w_lo, w_hi, h_lo, h_hi, med_w, med_h


def _edge_density(img_gray: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of mask pixels that are Canny edges.

    Carved motifs have grooves and relief boundaries → high edge density
    (0.2–0.35 on the Frobenius photographs).  Flat wood, the smooth mirror of
    an opon and the photographic backdrop sit at 0.03–0.05.
    """
    edges = cv2.Canny(img_gray, 50, 150)
    mask_px = int(mask.sum())
    if mask_px == 0:
        return 0.0
    return float((edges[mask] > 0).sum()) / mask_px


def _containment_ratio(inner: list[int], outer: dict) -> float:
    """Fraction of inner bbox area that falls inside outer bbox."""
    ix1, iy1 = inner[0], inner[1]
    ix2, iy2 = ix1 + inner[2], iy1 + inner[3]
    ox1, oy1 = outer["x"], outer["y"]
    ox2, oy2 = ox1 + outer["w"], oy1 + outer["h"]
    inter_w = max(0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = inner[2] * inner[3]
    if inner_area == 0:
        return 0.0
    return (inter_w * inter_h) / inner_area


def carving_peaks(
    img_gray: np.ndarray,
    spacing: int,
    min_density: float,
    occupied: np.ndarray | None = None,
    limit: int = MAX_PROBES,
) -> list[tuple[int, int, float]]:
    """Local maxima of carving density — roughly one per carved figure.

    Density is the fraction of Canny edge pixels in a *spacing*-sized window.
    Peaks closer than 0.75·spacing to a stronger one are merged, peaks below
    *min_density* (flat wood) and inside *occupied* (existing boxes) are
    dropped. Returns (x, y, density), densest first.
    """
    edges = (cv2.Canny(img_gray, 50, 150) > 0).astype(np.float32)
    k = max(3, int(spacing) | 1)
    dens = cv2.blur(edges, (k, k))
    local_max = dens >= cv2.dilate(dens, np.ones((k, k), np.uint8)) - 1e-6
    ys, xs = np.nonzero(local_max & (dens >= min_density))
    order = np.argsort(-dens[ys, xs])
    min_d2 = (0.75 * spacing) ** 2
    kept: list[tuple[int, int, float]] = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if occupied is not None and occupied[y, x]:
            continue
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_d2 for kx, ky, _ in kept):
            kept.append((x, y, float(dens[y, x])))
            if len(kept) >= limit:
                break
    return kept


def rank_score(edge_density: float, box_area: float, median_area: float,
               novelty: float) -> float:
    """Queue order for a candidate: carved, motif-sized, and not already boxed.

    0.4 · edge density (saturating at 0.30, the top of the motif range)
    0.4 · scale fit — 1.0 at the median template area, 0.5 at 2× or ½×
    0.2 · novelty — 1 − max IoU with boxes already on the panel
    """
    edge_term = min(edge_density / 0.30, 1.0)
    scale_fit = float(np.exp(-abs(np.log(max(box_area, 1.0) / max(median_area, 1.0)))))
    return 0.4 * edge_term + 0.4 * scale_fit + 0.2 * novelty


def prompted_segment(
    panel_img: np.ndarray,
    existing_bboxes: list[dict],
    approved_templates: list[dict] | None = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    min_score: float = PROMPT_MIN_SCORE,
    min_area: float = PROMPT_MIN_AREA,
    max_area: float = 0.50,
    max_aspect: float = PROMPT_MAX_ASPECT,
    min_edge_density: float = PROMPT_MIN_EDGE_DENSITY,
    containment_thresh: float = 0.70,
    grid_spacing: int = 80,
    panel_templates: list[dict] | None = None,
    probe: str = "peaks",
    negative_points: bool = True,
    pick: str = "best",
    verbose: bool = False,
) -> list[Detection]:
    """
    Find motifs shaped like approved templates in uncovered areas.

    Strategy
    --------
    1. Compute a **size envelope** (width/height range). The panel's own
       approved boxes (*panel_templates*) set it once there are at least
       PANEL_TEMPLATE_MIN of them — one board's figures share a scale.
       Otherwise *approved_templates* from every panel set it, more loosely.
    2. Probe at **carving-density peaks** (``probe="peaks"``) — about one per
       figure, none on flat wood — or on a sparse grid plus the edges of
       existing boxes (``probe="grid"``, the older behaviour).
    3. Each probe is a positive point plus a box prompt at the median template
       size, tried upright and transposed when the templates are not square.
       With *negative_points*, neighbouring peaks and the centres of nearby
       existing boxes are passed as background so SAM does not merge adjacent
       figures.
    4. Filter pipeline (each mask must pass ALL):
       a. SAM score >= min_score
       b. Mask area between min_area and max_area (fraction of panel)
       c. Size within the envelope, aspect ratio <= max_aspect
       d. Edge density >= min_edge_density (rejects featureless surface)
       e. Not >containment_thresh contained in any existing bbox
       f. IoU <= 0.3 with existing bboxes
    5. ``pick="best"`` keeps the best-ranked mask per probe, then walks all of
       them best-first, dropping any that overlaps (IoU > 0.3) or swallows two
       or more already-kept candidates. ``pick="all"`` keeps every passing
       mask in probe order (the older behaviour).

    Every returned Detection carries ``edge_density``, ``novelty`` and
    ``rank_score`` (see rank_score()); the list is sorted by rank_score.

    Parameters
    ----------
    panel_img           : H×W×3 uint8 RGB array
    existing_bboxes     : bboxes already on this panel (skipped in output)
    approved_templates  : bbox dicts from approved panels — defines what a
                          "good motif" looks like (size, aspect ratio)
    panel_templates     : bboxes approved on *this* panel; preferred over
                          approved_templates when there are enough of them
    min_score           : minimum predicted_iou to keep a mask
    min_area            : minimum mask area as fraction of panel (0.005 = 0.5%)
    max_area            : maximum mask area as fraction of panel (0.50 = 50%)
    max_aspect          : longest side over shortest side of the mask's bbox
    min_edge_density    : minimum Canny edge fraction inside the mask;
                          rejects flat/featureless regions (0.12 = 12%)
    containment_thresh  : if this fraction of the new mask's bbox falls
                          inside an existing bbox, reject it as a sub-motif
    grid_spacing        : pixel spacing for the sparse grid (probe="grid")
    """
    sam = _load_sam_model(checkpoint, model_type)
    predictor = SamPredictor(sam)
    predictor.set_image(panel_img)

    img_h, img_w = panel_img.shape[:2]
    img_area = img_h * img_w
    img_gray = cv2.cvtColor(panel_img, cv2.COLOR_RGB2GRAY)

    # ── Size envelope — this panel's scale first, the corpus second ───────
    panel_templates = _trim_outsized(panel_templates or [])
    envelope = _size_envelope(panel_templates, PANEL_ENVELOPE_MARGIN,
                              min_n=PANEL_TEMPLATE_MIN)
    envelope_from = "panel"
    if envelope is None:
        envelope = _size_envelope(approved_templates or [], GLOBAL_ENVELOPE_MARGIN)
        envelope_from = "corpus"
    if envelope:
        w_lo, w_hi, h_lo, h_hi, med_w, med_h = envelope
    else:
        envelope_from = "none"
        w_lo, h_lo = 20, 20
        w_hi, h_hi = img_w // 2, img_h // 2
        med_w, med_h = img_w // 4, img_h // 4
    median_area = float(med_w * med_h)

    if verbose:
        n_t = len(panel_templates or []) if envelope_from == "panel" else len(approved_templates or [])
        print(f"  Size envelope ({envelope_from}, {n_t} boxes): w=[{w_lo}–{w_hi}], "
              f"h=[{h_lo}–{h_hi}], median={med_w}x{med_h}")
        print(f"  Filters: score>={min_score}, area=[{min_area:.1%}–{max_area:.0%}], "
              f"edge>={min_edge_density:.1%}, containment<{containment_thresh:.0%}")
        print(f"  Probes: {probe}, negative points: {negative_points}, pick: {pick}")

    _rej = {"score": 0, "area": 0, "size": 0, "edge": 0,
            "containment": 0, "iou": 0, "dedup": 0, "empty": 0, "clump": 0,
            "aspect": 0}

    # ── Occupancy map ─────────────────────────────────────────────────────
    occupied = np.zeros((img_h, img_w), dtype=bool)
    for bb in existing_bboxes:
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        occupied[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)] = True

    # ── Probe points ──────────────────────────────────────────────────────
    peaks: list[tuple[int, int, float]] = []
    points: list[tuple[int, int]] = []
    if probe == "peaks":
        spacing = max(24, int(min(med_w, med_h) * 0.6))
        peaks = carving_peaks(img_gray, spacing, min_edge_density, occupied)
        points = [(x, y) for x, y, _ in peaks]
    else:
        for bb in existing_bboxes:
            x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
            margin = max(med_w, med_h) // 2
            for ex, ey in [
                (x - margin, y + h // 2),
                (x + w + margin, y + h // 2),
                (x + w // 2, y - margin),
                (x + w // 2, y + h + margin),
                (x - margin, y),
                (x + w + margin, y),
                (x - margin, y + h),
                (x + w + margin, y + h),
            ]:
                ex, ey = int(ex), int(ey)
                if 0 <= ex < img_w and 0 <= ey < img_h and not occupied[ey, ex]:
                    points.append((ex, ey))

        half = grid_spacing // 2
        for py in range(half, img_h, grid_spacing):
            for px in range(half, img_w, grid_spacing):
                r = grid_spacing // 4
                y1c, y2c = max(0, py - r), min(img_h, py + r)
                x1c, x2c = max(0, px - r), min(img_w, px + r)
                if occupied[y1c:y2c, x1c:x2c].mean() < 0.5:
                    points.append((px, py))

    if verbose:
        print(f"  Probe points: {len(points)} "
              f"({img_h}x{img_w} panel, {int(occupied.mean()*100)}% occupied)")

    if not points:
        if verbose:
            print("  No probe points — panel fully occupied")
        return []

    # Box prompt shapes: the median, plus its transpose when the templates
    # are clearly not square (tall figures on side strips, wide on top/bottom).
    shapes = [(med_w, med_h)]
    if probe == "peaks" and max(med_w, med_h) > 1.3 * min(med_w, med_h):
        shapes.append((med_h, med_w))
    existing_centres = [(b["x"] + b["w"] // 2, b["y"] + b["h"] // 2) for b in existing_bboxes]
    neighbour_pool = [(x, y) for x, y, _ in peaks] + existing_centres

    def _negatives(px: int, py: int, bw: int, bh: int) -> list[tuple[int, int]]:
        # A neighbour sits within 1.5× the prompt box but outside its central
        # 70% — close enough for SAM to bleed into, far enough to be another figure.
        out = []
        for nx, ny in neighbour_pool:
            dx, dy = abs(nx - px), abs(ny - py)
            if dx <= 0.75 * bw and dy <= 0.75 * bh and (dx > 0.35 * bw or dy > 0.35 * bh):
                out.append((nx, ny))
        return out[:8]

    def _evaluate(mask: np.ndarray, score: float):
        """Apply the filter chain; return (bbox_list, area_ratio, ed, novelty) or None."""
        if score < min_score:
            _rej["score"] += 1; return None
        ys, xs = np.where(mask)
        if len(xs) == 0:
            _rej["empty"] += 1; return None
        mx, my_c = int(xs.min()), int(ys.min())
        mw, mh = int(xs.max()) - mx, int(ys.max()) - my_c
        if mw <= 0 or mh <= 0:
            _rej["empty"] += 1; return None
        area_ratio = int(mask.sum()) / img_area
        if area_ratio < min_area or area_ratio > max_area:
            _rej["area"] += 1; return None
        if mw < w_lo or mw > w_hi or mh < h_lo or mh > h_hi:
            _rej["size"] += 1; return None
        if max(mw, mh) > max_aspect * min(mw, mh):
            _rej["aspect"] += 1; return None
        ed = _edge_density(img_gray, mask)
        if ed < min_edge_density:
            _rej["edge"] += 1; return None
        bb_list = [mx, my_c, mw, mh]
        if any(_containment_ratio(bb_list, e) > containment_thresh
               for e in existing_bboxes):
            _rej["containment"] += 1; return None
        ious = [_iou(bb_list, [e["x"], e["y"], e["w"], e["h"]]) for e in existing_bboxes]
        if any(v > 0.3 for v in ious):
            _rej["iou"] += 1; return None
        novelty = 1.0 - max(ious, default=0.0)
        return bb_list, area_ratio, ed, novelty

    # ── Probe each point ──────────────────────────────────────────────────
    candidates: list[Detection] = []
    _total_masks = 0
    for px, py in points:
        probe_cands: list[Detection] = []
        for bw, bh in shapes:
            coords, labels = [[px, py]], [1]
            if negative_points:
                for nx, ny in _negatives(px, py, bw, bh):
                    coords.append([nx, ny]); labels.append(0)
            box = np.array([max(0, px - bw // 2), max(0, py - bh // 2),
                            min(img_w, px + bw // 2), min(img_h, py + bh // 2)])
            masks, scores, _ = predictor.predict(
                point_coords=np.array(coords),
                point_labels=np.array(labels),
                box=box,
                multimask_output=True,
            )
            for mi in range(len(scores)):
                _total_masks += 1
                score = float(scores[mi])
                res = _evaluate(masks[mi], score)
                if res is None:
                    continue
                bb_list, area_ratio, ed, novelty = res
                probe_cands.append(Detection(
                    index=0,
                    bbox={"x": bb_list[0], "y": bb_list[1], "w": bb_list[2], "h": bb_list[3]},
                    scale=classify_scale(area_ratio),
                    area_ratio=area_ratio,
                    predicted_iou=score,
                    stability_score=score,
                    segmentation=masks[mi],
                    edge_density=ed,
                    novelty=novelty,
                    rank_score=rank_score(ed, bb_list[2] * bb_list[3], median_area, novelty),
                ))
        if pick == "best" and probe_cands:
            candidates.append(max(probe_cands, key=lambda d: d.rank_score))
        else:
            candidates.extend(probe_cands)

    # ── Dedup ─────────────────────────────────────────────────────────────
    if pick == "best":
        candidates.sort(key=lambda d: d.rank_score, reverse=True)
    detections: list[Detection] = []
    seen: list[list[int]] = []
    for d in candidates:
        bb_list = [d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"]]
        if any(_iou(bb_list, s) > 0.3 for s in seen):
            _rej["dedup"] += 1; continue
        if pick == "best":
            as_dict = d.bbox
            swallowed = sum(
                _containment_ratio(s, as_dict) > 0.7 for s in seen)
            if swallowed >= 2:
                _rej["clump"] += 1; continue
            if any(_containment_ratio(bb_list, {"x": s[0], "y": s[1], "w": s[2], "h": s[3]}) > 0.7
                   for s in seen):
                _rej["dedup"] += 1; continue
        d.index = len(detections)
        detections.append(d)
        seen.append(bb_list)

    detections.sort(key=lambda d: d.rank_score, reverse=True)
    for i, d in enumerate(detections):
        d.index = i

    if verbose:
        print(f"  SAM returned {_total_masks} masks from {len(points)} probes")
        print("  Rejected: " + ", ".join(f"{k}={v}" for k, v in _rej.items() if v))
        print(f"  Kept: {len(detections)}")

    return detections


def snap_boxes(
    panel_img: np.ndarray,
    boxes: list[dict],
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    min_score: float = PROMPT_MIN_SCORE,
    min_iou: float = 0.6,
) -> list[dict | None]:
    """Tighten boxes to carved edges with a SAM box prompt.

    For each box, SAM segments inside it and the mask whose bbox best matches
    the original (IoU >= *min_iou*, score >= *min_score*) replaces it. Returns
    one entry per input box: the snapped bbox, or None where no mask agreed —
    keep the original there. Used to clean up boxes placed by eye or by an LLM.
    """
    if not boxes:
        return []
    sam = _load_sam_model(checkpoint, model_type)
    predictor = SamPredictor(sam)
    predictor.set_image(panel_img)
    out: list[dict | None] = []
    for b in boxes:
        masks, scores, _ = predictor.predict(
            box=np.array([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]),
            multimask_output=True,
        )
        best, best_iou = None, min_iou
        for mask, score in zip(masks, scores):
            if float(score) < min_score:
                continue
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            cand = [int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min()), int(ys.max() - ys.min())]
            v = _iou(cand, [b["x"], b["y"], b["w"], b["h"]])
            if v >= best_iou:
                best, best_iou = cand, v
        out.append(None if best is None else
                   {"x": best[0], "y": best[1], "w": best[2], "h": best[3]})
    return out


# ── Annotation ────────────────────────────────────────────────────────────────

def annotate_detections(
    panel_img: np.ndarray,
    detections: list[Detection],
    out_path: str | Path,
) -> None:
    """
    Draw coloured bounding boxes and scale labels on the panel image.
    Colour by scale: red=register, green=motif.
    """
    img = cv2.cvtColor(panel_img, cv2.COLOR_RGB2BGR)
    img_h, img_w = img.shape[:2]
    font_scale = max(0.3, min(img_w, img_h) / 800)
    thickness = max(1, int(min(img_w, img_h) / 400))

    for d in detections:
        colour = SCALE_COLOURS.get(d.scale, (200, 200, 200))
        bgr = (colour[2], colour[1], colour[0])
        x, y, w, h = d.bbox["x"], d.bbox["y"], d.bbox["w"], d.bbox["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, thickness + 1)

        label = f"#{d.index} {d.scale} {d.area_ratio*100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ly = max(y - 2, th + 4)
        cv2.rectangle(img, (x, ly - th - baseline - 2), (x + tw + 4, ly + 2), bgr, -1)
        cv2.putText(img, label, (x + 2, ly - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


# ── Batch processing ──────────────────────────────────────────────────────────

def segment_to_files(
    panel_image_path: str | Path,
    out_dir: str | Path,
    generator: "SamAutomaticMaskGenerator",
    **kwargs,
) -> list[dict]:
    """
    Segment a panel image, save annotated JPEG and patches JSON.
    Returns list of detection dicts (without segmentation masks).

    Writes two JSON files:
      <stem>_detections_raw.json — all SAM masks before any filtering
                                   (candidate pool for bbox_review Phase 1b)
      <stem>_detections.json     — filtered + NMS detections (pipeline default)
    """
    path = Path(panel_image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_pil = Image.open(path).convert("RGB")
    img_np = np.array(img_pil)

    img_h, img_w = img_np.shape[:2]
    img_area = img_h * img_w

    # Run SAM once — reuse raw masks for both raw JSON and filtered detections
    raw_masks = generator.generate(img_np)

    # ── _detections_raw.json: all masks before filtering (Phase 1a) ────────────
    raw_meta = []
    for idx, m in enumerate(raw_masks):
        x, y, w, h = [int(v) for v in m["bbox"]]
        raw_meta.append({
            "index": idx,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area_ratio": round(m["area"] / img_area, 5),
            "predicted_iou": round(float(m["predicted_iou"]), 4),
            "stability_score": round(float(m["stability_score"]), 4),
        })
    raw_json_path = out_dir / f"{path.stem}_detections_raw.json"
    with open(raw_json_path, "w") as f:
        json.dump(raw_meta, f, indent=2)

    # ── Filter + NMS → Detection objects ───────────────────────────────────────
    min_area = kwargs.get("min_area", DEFAULT_MIN_AREA)
    max_area = kwargs.get("max_area", DEFAULT_MAX_AREA)
    nms_iou  = kwargs.get("nms_iou",  DEFAULT_NMS_IOU)

    kept = filter_and_nms(raw_masks, img_area,
                          min_area=min_area, max_area=max_area, nms_iou=nms_iou)

    detections = []
    for idx, m in enumerate(kept):
        x, y, w, h = [int(v) for v in m["bbox"]]
        area_ratio = m["area"] / img_area
        detections.append(Detection(
            index=idx,
            bbox={"x": x, "y": y, "w": w, "h": h},
            scale=classify_scale(area_ratio),
            area_ratio=area_ratio,
            predicted_iou=float(m["predicted_iou"]),
            stability_score=float(m["stability_score"]),
            segmentation=m["segmentation"],
        ))

    # Sort largest → smallest so register-level context comes first
    detections.sort(key=lambda d: d.area_ratio, reverse=True)
    for i, d in enumerate(detections):
        d.index = i

    # Annotated image
    ann_path = out_dir / f"{path.stem}_annotated.jpg"
    annotate_detections(img_np, detections, ann_path)

    # JSON metadata (no raw segmentation arrays — too large)
    meta = [d.to_dict() for d in detections]
    json_path = out_dir / f"{path.stem}_detections.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 3: SAM automatic motif segmentation on panel crops",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("images", nargs="+", metavar="PANEL_IMAGE")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-type", default=DEFAULT_MODEL_TYPE,
                   choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--out-dir",
                   default="frobenius_artifacts/analysis/annotated")
    p.add_argument("--points-per-side", type=int, default=DEFAULT_POINTS_PER_SIDE)
    p.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    p.add_argument("--max-area", type=float, default=DEFAULT_MAX_AREA)
    p.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    p.add_argument(
        "--resegment", action="store_true",
        help="HITL mode: read manual bboxes from _approved.json and use them "
             "as SamPredictor box prompts to get refined masks. Merges results "
             "back into _approved.json with source='sam_prompted'.",
    )
    return p


def _resegment_from_approved(
    panel_path: Path, out_dir: Path, checkpoint: str, model_type: str,
) -> None:
    """Probe uncovered areas of a panel using existing bboxes + cross-panel sizes."""
    stem = panel_path.stem
    out_dir = Path(out_dir)
    approved_path = out_dir / f"{stem}_approved.json"
    if not approved_path.exists():
        print(f"  skip — no _approved.json")
        return

    approved = json.loads(approved_path.read_text())
    existing = [d["bbox"] for d in approved]
    if not existing:
        print(f"  skip — no bboxes in approved")
        return

    # Collect approved bbox templates from all other panels
    templates: list[dict] = []
    for ap in out_dir.glob("*_approved.json"):
        if stem in ap.stem:
            continue
        try:
            for r in json.loads(ap.read_text()):
                b = r.get("bbox", {})
                if b.get("w", 0) > 0 and b.get("h", 0) > 0:
                    templates.append(b)
        except Exception:
            pass

    print(f"  {len(existing)} existing bbox(es), "
          f"{len(templates)} approved templates → probing uncovered areas")

    img_np = np.array(Image.open(panel_path).convert("RGB"))

    prompted = prompted_segment(
        img_np, existing,
        approved_templates=templates or None,
        checkpoint=checkpoint, model_type=model_type,
    )

    next_idx = max((d.get("index", 0) for d in approved), default=-1) + 1
    added = 0
    for det in prompted:
        d = det.to_dict()
        d["source"] = "sam_prompted"
        d["index"] = next_idx
        approved.append(d)
        next_idx += 1
        added += 1

    approved_path.write_text(json.dumps(approved, indent=2))
    print(f"  +{added} prompted masks → {approved_path.name} "
          f"(total {len(approved)})")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.resegment:
        for img_path in args.images:
            path = Path(img_path)
            print(f"\n{path.name}")
            _resegment_from_approved(
                path, args.out_dir,
                checkpoint=args.checkpoint,
                model_type=args.model_type,
            )
        return

    generator = load_generator(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        points_per_side=args.points_per_side,
    )

    for img_path in args.images:
        path = Path(img_path)
        print(f"\n{path.name}")
        meta = segment_to_files(
            path, args.out_dir, generator,
            min_area=args.min_area,
            max_area=args.max_area,
            nms_iou=args.nms_iou,
        )
        by_scale = {}
        for d in meta:
            by_scale.setdefault(d["scale"], 0)
            by_scale[d["scale"]] += 1
        total = len(meta)
        print(f"  {total} detections: " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_scale.items())))
        print(f"  → {args.out_dir}/{path.stem}_annotated.jpg")


if __name__ == "__main__":
    main(sys.argv[1:])
