"""
llm_segment.py — Claude as a reviewer and proposer of motif bounding boxes.

SAM finds carved regions well but cannot tell one figure from three touching
figures cut from the same wood, and it happily segments blank surface. A
vision model reading the whole panel can: it knows what a figure, a mask face
or an emblem looks like, and that border registers rotate their figures.

One call per panel. Claude sees

  1. the panel with a labelled pixel grid, so it can read coordinates off the
     image instead of estimating them, and
  2. the same panel with the boxes already approved (grey, E0…) and the SAM
     candidates still in the review queue (numbered 0…) drawn on it,

and returns a verdict for every SAM candidate (accept / adjust / reject) plus
any motifs neither SAM nor the reviewer has boxed yet. The result becomes the
notebook's draft queue — the same Add / Skip flow as SAM Refine — so a person
(or Claude driving the browser) still approves every box.

Optionally each box is snapped to carved edges with a SAM box prompt
(motif_segment.snap_boxes): Claude decides *what* is a motif, SAM decides
exactly where its edges are.

Usage:
    from panel_art.llm_segment import suggest_boxes
    suggestions = suggest_boxes(panel_rgb, existing, sam_candidates)
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    ANTHROPIC_AVAILABLE = False


DEFAULT_MODEL = os.environ.get("MOTIF_LLM_MODEL", "claude-opus-5-5")
# Claude Opus 5.5 defaults to medium; placing boxes on a dense board is worth more.
DEFAULT_EFFORT = os.environ.get("MOTIF_LLM_EFFORT", "high")
MODEL_OPTIONS = ["claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5"]
if DEFAULT_MODEL not in MODEL_OPTIONS:           # an env override the list doesn't know
    MODEL_OPTIONS.insert(0, DEFAULT_MODEL)
EFFORT_OPTIONS = ["low", "medium", "high", "xhigh", "max"]

# Long edge sent to the API. Larger images are downscaled by the API anyway;
# doing it here keeps the grid labels legible and the coordinate maths exact.
MAX_IMAGE_EDGE = 1568
MIN_BOX_PX = 8


class LLMSegmentError(RuntimeError):
    """The model could not produce usable boxes (refusal, truncation, bad JSON)."""


SYSTEM_PROMPT = """\
You are helping annotate images of Yoruba carved wood — door panels, Ifá \
divination trays (opon Ifá), bowls and lidded vessels, house posts — for a \
study of recurring symbolic motifs. The images are photographs or ink \
drawings of the carvings. Your job is to place bounding boxes, one per motif.

A motif is one self-contained carved unit: a single human or animal figure \
(including whatever it holds, its headdress and limbs), a mask face, a single \
emblem or object, or one unit of a repeated ornament such as a leaf column. \
Figures in border registers or on curved surfaces are often rotated, \
foreshortened or upside down; they are still motifs.

Not motifs: blank or smooth surface (tray recesses, the central mirror of an \
opon, plain background), the photographic backdrop or paper, the plain frame \
of the board, all-over texture (hatching, chip-carved triangles) and clumps — \
a box covering two or more figures that could each be boxed separately. When \
you reject a candidate as a clump, add the individual motifs inside it.

Box every motif you can identify. A reviewer approves each box, so mark a \
motif you are unsure of as low confidence rather than leaving it out — but do \
not box texture or empty surface to fill the list. Boxes should be tight: the \
whole motif, as little of its neighbours as possible. Read coordinates off the \
grid in the first image. All coordinates are in pixels of that image, origin \
at the top-left, x to the right, y down. Do not re-box anything already \
approved.\
"""

_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "w": {"type": "integer"},
        "h": {"type": "integer"},
    },
    "required": ["x", "y", "w", "h"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "description": "One entry per numbered SAM candidate.",
            "items": {
                "type": "object",
                "properties": {
                    "candidate": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["accept", "adjust", "reject"]},
                    "bbox": _BBOX_SCHEMA,
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["candidate", "verdict", "bbox", "label", "reason"],
                "additionalProperties": False,
            },
        },
        "additions": {
            "type": "array",
            "description": "Motifs not covered by an approved box or an accepted candidate.",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": _BBOX_SCHEMA,
                    "label": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
                "required": ["bbox", "label", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reviews", "additions"],
    "additionalProperties": False,
}


@dataclass
class Suggestion:
    """One box for the review queue."""
    bbox: dict                      # {x, y, w, h} in panel pixels
    label: str
    reason: str
    origin: str                     # "sam_accept" | "sam_adjust" | "llm_add"
    confidence: str = "high"
    sam_index: int | None = None    # position in the SAM queue it came from
    snapped: bool = False

    @property
    def source(self) -> str:
        """MotifRecord.source for an accepted suggestion."""
        return "llm" if self.sam_index is None else "sam_llm"


@dataclass
class LLMResult:
    suggestions: list[Suggestion]
    rejected: list[dict] = field(default_factory=list)   # {candidate, reason}
    usage: dict = field(default_factory=dict)
    model: str = ""


# ── Rendering ─────────────────────────────────────────────────────────────────

def _font(size: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def grid_step(width: int, height: int) -> int:
    """Grid spacing in image pixels: ~10–20 lines along the long edge."""
    long_edge = max(width, height)
    for step in (25, 50, 100, 200):
        if long_edge / step <= 20:
            return step
    return 250


def render_grid(img: Image.Image) -> Image.Image:
    """The panel with a labelled coordinate grid — the model reads positions off it."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    step = grid_step(*out.size)
    major = step * 2 if step < 100 else step
    font = _font(max(11, out.width // 70))
    for x in range(step, out.width, step):
        strong = x % major == 0
        d.line([(x, 0), (x, out.height)], fill=(255, 40, 40, 150 if strong else 70), width=1)
        if strong:
            d.text((x + 2, 2), str(x), fill=(255, 60, 60, 255), font=font)
    for y in range(step, out.height, step):
        strong = y % major == 0
        d.line([(0, y), (out.width, y)], fill=(255, 40, 40, 150 if strong else 70), width=1)
        if strong:
            d.text((2, y + 2), str(y), fill=(255, 60, 60, 255), font=font)
    return out


def render_boxes(img: Image.Image, existing: list[dict], candidates: list[dict]) -> Image.Image:
    """Approved boxes in grey (E0…), SAM candidates in colour (0…)."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    font = _font(max(12, out.width // 60))
    palette = [(255, 70, 70), (60, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255)]
    for i, b in enumerate(existing):
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=(170, 170, 170), width=2)
        d.text((b["x"] + 3, b["y"] + 2), f"E{i}", fill=(200, 200, 200), font=font)
    for i, b in enumerate(candidates):
        col = palette[i % len(palette)]
        d.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=col, width=2)
        d.rectangle([b["x"] + 1, b["y"] + 1, b["x"] + 24, b["y"] + 18], fill=(0, 0, 0))
        d.text((b["x"] + 4, b["y"] + 2), str(i), fill=col, font=font)
    return out


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _scale_box(b: dict, s: float) -> dict:
    return {k: int(round(b[k] * s)) for k in ("x", "y", "w", "h")}


# ── Request / response ────────────────────────────────────────────────────────

def build_content(
    panel: Image.Image,
    existing: list[dict],
    candidates: list[dict],
    context: str = "",
) -> list[dict]:
    """User-turn content blocks. Boxes must already be in *panel*'s pixel space."""
    has_cands = bool(candidates)
    lines = [
        f"Panel size: {panel.width}×{panel.height} px.",
        "Already approved (grey, E0…): "
        + (json.dumps(existing) if existing else "none") + ".",
    ]
    if has_cands:
        lines.append("SAM candidates awaiting review (numbered in image 2): "
                     + json.dumps([{"candidate": i, **b} for i, b in enumerate(candidates)]) + ".")
        lines.append("Give a verdict for every candidate: accept it as drawn, adjust it "
                     "(return the corrected box), or reject it. For accept, return the "
                     "candidate's box unchanged.")
    else:
        lines.append("There are no SAM candidates — return an empty reviews list.")
    lines.append("Then add every remaining motif that no approved box or accepted/adjusted "
                 "candidate covers.")
    if context:
        lines.append(f"Context: {context}")

    content: list[dict] = [
        {"type": "text", "text": "Image 1 — the panel with a pixel grid:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": _b64_png(render_grid(panel))}},
        {"type": "text", "text": "Image 2 — approved boxes and SAM candidates:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": _b64_png(render_boxes(panel, existing, candidates))}},
        {"type": "text", "text": "\n".join(lines)},
    ]
    return content


def _clamp(b: dict, width: int, height: int) -> dict | None:
    x = max(0, min(int(b["x"]), width - 1))
    y = max(0, min(int(b["y"]), height - 1))
    w = max(0, min(int(b["w"]), width - x))
    h = max(0, min(int(b["h"]), height - y))
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def parse_response(
    data: dict,
    candidates: list[dict],
    width: int,
    height: int,
    scale: float = 1.0,
) -> tuple[list[Suggestion], list[dict]]:
    """Turn the model's JSON into queue order: reviewed SAM boxes, then additions.

    *scale* maps the model's coordinates back to panel pixels (1 / the
    downscale applied before sending). Boxes are clamped to the panel; any
    smaller than MIN_BOX_PX a side are dropped.
    """
    suggestions: list[Suggestion] = []
    rejected: list[dict] = []
    seen: set[int] = set()
    for r in data.get("reviews", []):
        i = r.get("candidate")
        if not isinstance(i, int) or not 0 <= i < len(candidates) or i in seen:
            continue
        seen.add(i)
        if r.get("verdict") == "reject":
            rejected.append({"candidate": i, "reason": r.get("reason", "")})
            continue
        if r.get("verdict") == "accept":
            box = dict(candidates[i])          # exact SAM box — no rounding drift
        else:
            box = _clamp(_scale_box(r["bbox"], scale), width, height)
            if box is None:
                continue
        suggestions.append(Suggestion(
            bbox=box, label=r.get("label", ""), reason=r.get("reason", ""),
            origin="sam_accept" if r.get("verdict") == "accept" else "sam_adjust",
            sam_index=i))
    suggestions.sort(key=lambda s: s.sam_index)

    adds = []
    for a in data.get("additions", []):
        box = _clamp(_scale_box(a["bbox"], scale), width, height)
        if box is None:
            continue
        adds.append(Suggestion(
            bbox=box, label=a.get("label", ""), reason=a.get("reason", ""),
            origin="llm_add", confidence=a.get("confidence", "medium")))
    adds.sort(key=lambda s: _CONF_ORDER.get(s.confidence, 1))
    return suggestions + adds, rejected


def _text_of(message: Any) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")


def suggest_boxes(
    panel_rgb: np.ndarray | Image.Image,
    existing: list[dict],
    candidates: list[dict] | None = None,
    *,
    context: str = "",
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    client: Any = None,
) -> LLMResult:
    """Ask Claude to review *candidates* and add missing motifs on one panel.

    *existing* and *candidates* are {x, y, w, h} dicts in panel pixels.
    Returns suggestions in review order and the candidates Claude rejected.
    Raises LLMSegmentError on a refusal, a truncated reply or unusable JSON.
    """
    candidates = candidates or []
    panel = panel_rgb if isinstance(panel_rgb, Image.Image) else Image.fromarray(panel_rgb)
    panel = panel.convert("RGB")
    width, height = panel.size

    down = min(1.0, MAX_IMAGE_EDGE / max(width, height))
    sent = panel if down == 1.0 else panel.resize(
        (round(width * down), round(height * down)), Image.LANCZOS)
    content = build_content(
        sent,
        [_scale_box(b, down) for b in existing],
        [_scale_box(b, down) for b in candidates],
        context,
    )

    if client is None:
        if not ANTHROPIC_AVAILABLE:
            raise LLMSegmentError("anthropic is not installed: pip install anthropic")
        client = anthropic.Anthropic()

    # Streaming: at high effort the model can think for a while before the JSON
    # starts, longer than a non-streaming request should be left open.
    with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
        messages=[{"role": "user", "content": content}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        category = getattr(getattr(message, "stop_details", None), "category", None)
        raise LLMSegmentError(f"{model} declined this panel (category: {category})")
    if message.stop_reason == "max_tokens":
        raise LLMSegmentError("reply was cut off at max_tokens — try a lower effort")
    try:
        data = json.loads(_text_of(message))
    except json.JSONDecodeError as e:
        raise LLMSegmentError(f"could not parse the reply as JSON: {e}") from e

    suggestions, rejected = parse_response(data, candidates, width, height, scale=1.0 / down)
    usage = getattr(message, "usage", None)
    return LLMResult(
        suggestions=suggestions,
        rejected=rejected,
        usage={
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        },
        model=getattr(message, "model", model),
    )
