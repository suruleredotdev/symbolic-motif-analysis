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

def _size_envelope(templates: list[dict], margin: float = 0.5):
    """Compute (w_lo, w_hi, h_lo, h_hi) from approved bbox templates.

    Uses the 10th–90th percentile of widths/heights, expanded by *margin*
    (0.5 = 50%).  Returns None if fewer than 3 templates.
    """
    if len(templates) < 3:
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


def prompted_segment(
    panel_img: np.ndarray,
    existing_bboxes: list[dict],
    approved_templates: list[dict] | None = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    model_type: str = DEFAULT_MODEL_TYPE,
    min_score: float = 0.80,
    grid_spacing: int = 80,
) -> list[Detection]:
    """
    Find motifs shaped like approved templates in uncovered areas.

    Strategy
    --------
    1. Compute a **size envelope** (width/height range) from
       approved_templates — only masks matching those dimensions are kept.
    2. Prioritise probing **edges of existing bboxes** (adjacent motifs)
       plus a sparse grid over uncovered space.
    3. For each probe point, pass a box prompt at the median template size
       so SAM knows what scale to segment at.
    4. Keep masks that score >= min_score, fall within the size envelope,
       and don't overlap existing bboxes.

    Parameters
    ----------
    panel_img           : H×W×3 uint8 RGB array
    existing_bboxes     : bboxes already on this panel (skipped in output)
    approved_templates  : bbox dicts from approved panels — defines what a
                          "good motif" looks like (size, aspect ratio)
    min_score           : minimum predicted_iou to keep a mask
    grid_spacing        : pixel spacing for the sparse exploration grid
    """
    sam = _load_sam_model(checkpoint, model_type)
    predictor = SamPredictor(sam)
    predictor.set_image(panel_img)

    img_h, img_w = panel_img.shape[:2]
    img_area = img_h * img_w

    # ── Size envelope from approved templates ─────────────────────────────
    envelope = _size_envelope(approved_templates or [])
    if envelope:
        w_lo, w_hi, h_lo, h_hi, med_w, med_h = envelope
    else:
        # No template data — permissive defaults
        w_lo, h_lo = 20, 20
        w_hi, h_hi = img_w // 2, img_h // 2
        med_w, med_h = img_w // 4, img_h // 4

    # ── Occupancy map ─────────────────────────────────────────────────────
    occupied = np.zeros((img_h, img_w), dtype=bool)
    for bb in existing_bboxes:
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        occupied[max(0, y):min(img_h, y + h), max(0, x):min(img_w, x + w)] = True

    # ── Probe points — edges first, then sparse grid ──────────────────────
    points: list[tuple[int, int]] = []

    # 1) Edge points just outside each existing bbox (highest value)
    for bb in existing_bboxes:
        x, y, w, h = bb["x"], bb["y"], bb["w"], bb["h"]
        margin = max(med_w, med_h) // 2   # probe at ~half a motif away
        for ex, ey in [
            (x - margin, y + h // 2),       # left
            (x + w + margin, y + h // 2),    # right
            (x + w // 2, y - margin),        # top
            (x + w // 2, y + h + margin),    # bottom
            (x - margin, y),                 # top-left
            (x + w + margin, y),             # top-right
            (x - margin, y + h),             # bottom-left
            (x + w + margin, y + h),         # bottom-right
        ]:
            ex, ey = int(ex), int(ey)
            if 0 <= ex < img_w and 0 <= ey < img_h and not occupied[ey, ex]:
                points.append((ex, ey))

    # 2) Sparse grid over remaining uncovered areas
    half = grid_spacing // 2
    for py in range(half, img_h, grid_spacing):
        for px in range(half, img_w, grid_spacing):
            r = grid_spacing // 4
            y1c, y2c = max(0, py - r), min(img_h, py + r)
            x1c, x2c = max(0, px - r), min(img_w, px + r)
            if occupied[y1c:y2c, x1c:x2c].mean() < 0.5:
                points.append((px, py))

    if not points:
        return []

    # ── Probe each point with a template-sized box prompt ─────────────────
    detections: list[Detection] = []
    seen: list[list[int]] = []

    for px, py in points:
        coords = np.array([[px, py]])
        labels = np.array([1])

        # Box prompt centred on the probe point, sized to median template
        bx1 = max(0, px - med_w // 2)
        by1 = max(0, py - med_h // 2)
        bx2 = min(img_w, px + med_w // 2)
        by2 = min(img_h, py + med_h // 2)

        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=np.array([bx1, by1, bx2, by2]),
            multimask_output=True,
        )

        for mi in range(len(scores)):
            if float(scores[mi]) < min_score:
                continue
            mask = masks[mi]
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            mx, my_c = int(xs.min()), int(ys.min())
            mw, mh = int(xs.max()) - mx, int(ys.max()) - my_c
            if mw <= 0 or mh <= 0:
                continue

            # ── Size envelope filter — the key precision gate ─────────
            if mw < w_lo or mw > w_hi or mh < h_lo or mh > h_hi:
                continue

            area_ratio = int(mask.sum()) / img_area
            bb_list = [mx, my_c, mw, mh]

            # Skip if overlapping existing bboxes
            if any(_iou(bb_list, [e["x"], e["y"], e["w"], e["h"]]) > 0.3
                   for e in existing_bboxes):
                continue
            # Internal dedup
            if any(_iou(bb_list, s) > 0.3 for s in seen):
                continue

            detections.append(Detection(
                index=len(detections),
                bbox={"x": mx, "y": my_c, "w": mw, "h": mh},
                scale=classify_scale(area_ratio),
                area_ratio=area_ratio,
                predicted_iou=float(scores[mi]),
                stability_score=float(scores[mi]),
                segmentation=mask,
            ))
            seen.append(bb_list)

    return detections


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
