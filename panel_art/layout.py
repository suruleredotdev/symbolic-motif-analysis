"""
layout.py — Deterministic spatial analysis of a panel's motifs.

Phase 6 (interpretation) needs to talk about *where* motifs sit, not just what
they look like.  Carved Yoruba door panels are organised into stacked
horizontal registers, and a reading of the panel is usually a reading of those
registers in order — so this module recovers that structure from bounding
boxes alone, before any LLM is involved.

Nothing here calls an API or loads a model: it is pure geometry, which makes
it cheap to run, easy to test, and inspectable when an interpretation looks
wrong.  `render_layout_text()` produces the compact description that
`panel_art/interpret.py` feeds to Claude.

Typical use:

    from panel_art.layout import Placement, analyze_layout, render_layout_text

    placements = [Placement(key="p/0", index=0, bbox={...}, scale="motif"), ...]
    layout = analyze_layout(placements, panel_w=980, panel_h=1460)
    print(render_layout_text(layout))
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

# ── Tuning constants ─────────────────────────────────────────────────────────
# A detection covering more than this fraction of the panel is treated as
# ground/frame rather than as a motif sitting *in* a register.  The v1 pipeline
# calls these "register" scale; a full knotwork body can reach ~0.82 of a
# narrow panel (see todos/2026-04-15, Phase 3 validation).
FIELD_AREA_FRACTION = 0.55

# Two motifs belong to the same register if their vertical spans overlap by at
# least this fraction of the shorter span.
REGISTER_OVERLAP = 0.35

# Mirror-pair detection: centres must reflect across the panel's vertical axis
# to within this fraction of panel width, and match in size to within
# MIRROR_SIZE_TOLERANCE relative difference.
MIRROR_AXIS_TOLERANCE = 0.06
MIRROR_SIZE_TOLERANCE = 0.35

# A bbox is "inside" another when this much of its area is contained.
CONTAINMENT = 0.80


# ── Records ──────────────────────────────────────────────────────────────────

@dataclass
class Placement:
    """One motif's geometry, decoupled from the rest of the pipeline.

    `key` is any stable identifier (the pipeline uses ``panel_stem/index``);
    `label` is optional and only used to make the rendered text readable.
    """

    key: str
    index: int
    bbox: dict                      # {x, y, w, h} in panel pixels
    scale: str = "motif"            # "motif" | "register" | "element"
    cluster: int = -1
    label: str | None = None

    # ── Derived (filled in by analyze_layout) ─────────────────────────────
    cx: float = 0.0                 # centre, 0–1 relative to panel width
    cy: float = 0.0                 # centre, 0–1 relative to panel height
    rw: float = 0.0                 # width,  0–1
    rh: float = 0.0                 # height, 0–1
    area_fraction: float = 0.0
    zone: str = ""                  # "upper left", "centre", …
    register: int = -1              # index into PanelLayout.registers, -1 = field
    is_field: bool = False          # ground/frame rather than a placed motif

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "index": self.index,
            "bbox": self.bbox,
            "scale": self.scale,
            "cluster": self.cluster,
            "label": self.label,
            "centre": [round(self.cx, 4), round(self.cy, 4)],
            "size": [round(self.rw, 4), round(self.rh, 4)],
            "area_fraction": round(self.area_fraction, 5),
            "zone": self.zone,
            "register": self.register,
            "is_field": self.is_field,
        }


@dataclass
class Register:
    """A horizontal band of motifs — the panel's basic narrative unit."""

    index: int                      # 0 = topmost
    y_top: float                    # 0–1 relative to panel height
    y_bottom: float
    members: list[str] = field(default_factory=list)   # Placement keys, left→right

    @property
    def height(self) -> float:
        return self.y_bottom - self.y_top

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "y_top": round(self.y_top, 4),
            "y_bottom": round(self.y_bottom, 4),
            "members": list(self.members),
        }


@dataclass
class Relation:
    """A directed spatial relation between two motifs."""

    source: str
    target: str
    direction: str                  # "above", "below", "left of", "upper-right of", …
    distance: float                 # centre-to-centre, in relative units

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "direction": self.direction,
            "distance": round(self.distance, 4),
        }


@dataclass
class PanelLayout:
    """Everything the interpretation stage needs to know about panel geometry."""

    panel_stem: str
    panel_width: int
    panel_height: int
    placements: list[Placement]
    registers: list[Register]
    reading_order: list[str]
    adjacency: list[Relation]
    mirror_pairs: list[tuple[str, str]]
    containment: dict[str, list[str]]           # container key → contained keys
    aspect_ratio: float = 0.0

    def by_key(self, key: str) -> Placement | None:
        return next((p for p in self.placements if p.key == key), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_stem": self.panel_stem,
            "panel_width": self.panel_width,
            "panel_height": self.panel_height,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "placements": [p.as_dict() for p in self.placements],
            "registers": [r.as_dict() for r in self.registers],
            "reading_order": list(self.reading_order),
            "adjacency": [r.as_dict() for r in self.adjacency],
            "mirror_pairs": [list(pair) for pair in self.mirror_pairs],
            "containment": {k: list(v) for k, v in self.containment.items()},
        }


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _span_overlap(a_top: float, a_bot: float, b_top: float, b_bot: float) -> float:
    """Fraction of the *shorter* vertical span that the two spans share."""
    lo, hi = max(a_top, b_top), min(a_bot, b_bot)
    if hi <= lo:
        return 0.0
    shorter = min(a_bot - a_top, b_bot - b_top)
    return (hi - lo) / shorter if shorter > 0 else 0.0


def _containment_fraction(inner: dict, outer: dict) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    ix1, iy1 = inner["x"], inner["y"]
    ix2, iy2 = ix1 + inner["w"], iy1 + inner["h"]
    ox1, oy1 = outer["x"], outer["y"]
    ox2, oy2 = ox1 + outer["w"], oy1 + outer["h"]
    x1, y1 = max(ix1, ox1), max(iy1, oy1)
    x2, y2 = min(ix2, ox2), min(iy2, oy2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inner_area = inner["w"] * inner["h"]
    return ((x2 - x1) * (y2 - y1)) / inner_area if inner_area > 0 else 0.0


def zone_name(cx: float, cy: float) -> str:
    """Name a 3×3 zone of the panel from a relative centre point."""
    col = "left" if cx < 1 / 3 else ("right" if cx > 2 / 3 else "centre")
    row = "upper" if cy < 1 / 3 else ("lower" if cy > 2 / 3 else "middle")
    if row == "middle" and col == "centre":
        return "centre"
    if col == "centre":
        return f"{row} centre"
    if row == "middle":
        return f"middle {col}"
    return f"{row} {col}"


def direction_name(source: Placement, target: Placement) -> str:
    """Describe where `target` sits relative to `source`.

    Uses the bearing between centres, snapped to eight compass-style terms in
    picture-plane language (no cardinal directions — a panel has no north).
    """
    dx = target.cx - source.cx
    dy = target.cy - source.cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "coincident with"

    angle = math.degrees(math.atan2(-dy, dx))     # y grows downward in image space
    if angle < 0:
        angle += 360.0

    octant = int((angle + 22.5) // 45) % 8
    return [
        "to the right of",
        "upper-right of",
        "above",
        "upper-left of",
        "to the left of",
        "lower-left of",
        "below",
        "lower-right of",
    ][octant]


# ── Register detection ───────────────────────────────────────────────────────

def detect_registers(placements: Sequence[Placement]) -> list[Register]:
    """Group placements into stacked horizontal bands.

    Single-link clustering on vertical spans: walk the motifs top to bottom and
    start a new register whenever the next motif's span does not overlap the
    band accumulated so far.  This handles the common case (clean stacked rows)
    without imposing a register count, and degrades to one register per motif
    when a panel has no banded structure at all.

    Field-scale detections (`is_field`) are excluded — they *contain* registers
    rather than sitting in one.
    """
    banded = sorted(
        (p for p in placements if not p.is_field),
        key=lambda p: (p.cy - p.rh / 2, p.cx),
    )
    registers: list[Register] = []
    cur_members: list[Placement] = []
    cur_top = cur_bottom = 0.0

    for p in banded:
        top, bottom = p.cy - p.rh / 2, p.cy + p.rh / 2
        if cur_members and _span_overlap(cur_top, cur_bottom, top, bottom) < REGISTER_OVERLAP:
            registers.append(_finish_register(len(registers), cur_members, cur_top, cur_bottom))
            cur_members, cur_top, cur_bottom = [], top, bottom
        elif not cur_members:
            cur_top, cur_bottom = top, bottom
        else:
            cur_top, cur_bottom = min(cur_top, top), max(cur_bottom, bottom)
        cur_members.append(p)

    if cur_members:
        registers.append(_finish_register(len(registers), cur_members, cur_top, cur_bottom))

    for reg in registers:
        for key in reg.members:
            match = next(p for p in banded if p.key == key)
            match.register = reg.index

    return registers


def _finish_register(idx: int, members: list[Placement], top: float, bottom: float) -> Register:
    ordered = sorted(members, key=lambda p: p.cx)
    return Register(index=idx, y_top=top, y_bottom=bottom,
                    members=[p.key for p in ordered])


# ── Relations ────────────────────────────────────────────────────────────────

def nearest_neighbours(
    placements: Sequence[Placement],
    k: int = 3,
) -> list[Relation]:
    """For each motif, the `k` nearest others by centre distance, with direction."""
    relations: list[Relation] = []
    for p in placements:
        others = [
            (math.dist((p.cx, p.cy), (q.cx, q.cy)), q)
            for q in placements if q.key != p.key
        ]
        others.sort(key=lambda t: t[0])
        for dist, q in others[:k]:
            relations.append(Relation(
                source=p.key, target=q.key,
                direction=direction_name(p, q), distance=dist,
            ))
    return relations


def find_mirror_pairs(placements: Sequence[Placement]) -> list[tuple[str, str]]:
    """Motifs that reflect one another across the panel's vertical mid-axis.

    Bilateral symmetry about a central figure is a recurring compositional
    device on carved door panels; flagging it lets the interpretation talk
    about flanking attendants rather than two unrelated motifs.
    """
    candidates = [p for p in placements if not p.is_field]
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()

    for i, a in enumerate(candidates):
        if a.key in used:
            continue
        best: tuple[float, Placement] | None = None
        for b in candidates[i + 1:]:
            if b.key in used:
                continue
            # Reflected centre of `a` should land on `b`.
            axis_error = abs((1.0 - a.cx) - b.cx)
            row_error = abs(a.cy - b.cy)
            if axis_error > MIRROR_AXIS_TOLERANCE or row_error > MIRROR_AXIS_TOLERANCE:
                continue
            size_error = _relative_difference(a.rw * a.rh, b.rw * b.rh)
            if size_error > MIRROR_SIZE_TOLERANCE:
                continue
            score = axis_error + row_error + 0.25 * size_error
            if best is None or score < best[0]:
                best = (score, b)
        if best is not None:
            pairs.append((a.key, best[1].key))
            used.update({a.key, best[1].key})

    return pairs


def _relative_difference(a: float, b: float) -> float:
    larger = max(a, b)
    return abs(a - b) / larger if larger > 0 else 0.0


def find_containment(placements: Sequence[Placement]) -> dict[str, list[str]]:
    """Map each container motif to the motifs nested inside it.

    The segmentation stage deliberately keeps both whole-figure and
    sub-element detections, so a register box often contains several motifs.
    Knowing which is which stops the interpretation from reading a border and
    the elements woven into it as separate, unrelated things.
    """
    by_area = sorted(placements, key=lambda p: -(p.rw * p.rh))
    nested: dict[str, list[str]] = {}
    for i, outer in enumerate(by_area):
        for inner in by_area[i + 1:]:
            if _containment_fraction(inner.bbox, outer.bbox) >= CONTAINMENT:
                nested.setdefault(outer.key, []).append(inner.key)
    return nested


# ── Top-level analysis ───────────────────────────────────────────────────────

def analyze_layout(
    placements: Iterable[Placement],
    panel_width: int,
    panel_height: int,
    panel_stem: str = "",
    neighbours: int = 3,
    field_area_fraction: float = FIELD_AREA_FRACTION,
) -> PanelLayout:
    """Compute the full spatial description of one panel.

    Mutates the passed placements in place (filling their derived fields) and
    returns them inside a `PanelLayout`.
    """
    placements = list(placements)
    if panel_width <= 0 or panel_height <= 0:
        raise ValueError(f"panel dimensions must be positive, got {panel_width}×{panel_height}")

    for p in placements:
        b = p.bbox
        p.cx = (b["x"] + b["w"] / 2) / panel_width
        p.cy = (b["y"] + b["h"] / 2) / panel_height
        p.rw = b["w"] / panel_width
        p.rh = b["h"] / panel_height
        p.area_fraction = p.rw * p.rh
        p.zone = zone_name(p.cx, p.cy)
        p.is_field = p.area_fraction >= field_area_fraction
        p.register = -1

    registers = detect_registers(placements)

    reading_order = [key for reg in registers for key in reg.members]
    # Field detections read last — they are the ground the rest sits on.
    reading_order += [p.key for p in placements if p.is_field]

    return PanelLayout(
        panel_stem=panel_stem,
        panel_width=panel_width,
        panel_height=panel_height,
        placements=placements,
        registers=registers,
        reading_order=reading_order,
        adjacency=nearest_neighbours(placements, k=neighbours),
        mirror_pairs=find_mirror_pairs(placements),
        containment=find_containment(placements),
        aspect_ratio=panel_height / panel_width,
    )


# ── Prompt rendering ─────────────────────────────────────────────────────────

def render_layout_text(layout: PanelLayout, max_relations: int = 40) -> str:
    """Render a layout as the compact text block sent to the model.

    Kept deliberately plain: the model reads this alongside the panel image, so
    its job is to name and index what the image shows, not to re-describe it.
    """
    lines: list[str] = []
    shape = "tall/narrow" if layout.aspect_ratio >= 2.0 else (
        "wide" if layout.aspect_ratio <= 0.75 else "roughly rectangular")
    lines.append(
        f"Panel {layout.panel_stem or '(unnamed)'}: "
        f"{layout.panel_width}×{layout.panel_height}px, {shape} "
        f"(height/width {layout.aspect_ratio:.2f})."
    )
    lines.append(
        f"{len(layout.placements)} detected motifs in "
        f"{len(layout.registers)} horizontal register(s), numbered top to bottom."
    )

    lines.append("")
    lines.append("REGISTERS (top to bottom; motifs listed left to right):")
    if not layout.registers:
        lines.append("  (no banded structure detected)")
    for reg in layout.registers:
        band = f"  Register {reg.index} — spans {reg.y_top:.2f}–{reg.y_bottom:.2f} of panel height:"
        lines.append(band)
        for key in reg.members:
            p = layout.by_key(key)
            if p is None:
                continue
            lines.append(f"    {_placement_line(p)}")

    fields = [p for p in layout.placements if p.is_field]
    if fields:
        lines.append("")
        lines.append("FIELD / GROUND (detections covering most of the panel):")
        for p in fields:
            lines.append(f"  {_placement_line(p)}")

    if layout.containment:
        lines.append("")
        lines.append("NESTING (motifs enclosed by a larger detection):")
        for outer, inners in layout.containment.items():
            lines.append(f"  #{_short(outer)} encloses {', '.join('#' + _short(i) for i in inners)}")

    if layout.mirror_pairs:
        lines.append("")
        lines.append("BILATERAL PAIRS (mirrored across the panel's vertical axis):")
        for a, b in layout.mirror_pairs:
            lines.append(f"  #{_short(a)} ↔ #{_short(b)}")

    if layout.adjacency:
        lines.append("")
        lines.append("ADJACENCY (nearest neighbours):")
        for rel in layout.adjacency[:max_relations]:
            lines.append(
                f"  #{_short(rel.source)} — #{_short(rel.target)} lies "
                f"{rel.direction} it (distance {rel.distance:.2f})"
            )
        if len(layout.adjacency) > max_relations:
            lines.append(f"  … {len(layout.adjacency) - max_relations} further relations omitted")

    lines.append("")
    lines.append("READING ORDER (register-major, then left to right): "
                 + " → ".join("#" + _short(k) for k in layout.reading_order))
    return "\n".join(lines)


def _placement_line(p: Placement) -> str:
    bits = [f"#{_short(p.key)}", f"{p.zone}", f"{p.scale}"]
    bits.append(f"covers {p.area_fraction * 100:.1f}% of panel")
    if p.cluster >= 0:
        bits.append(f"cluster {p.cluster}")
    if p.label:
        bits.append(f'labelled "{p.label}"')
    return " | ".join(bits)


def _short(key: str) -> str:
    """Trailing index of a ``panel_stem/index`` key, for readable prompt text."""
    return key.rsplit("/", 1)[-1]
