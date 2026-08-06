"""Geometry tests for panel_art.layout — no API, no model, no images."""

from __future__ import annotations

import pytest

from panel_art.layout import (
    Placement,
    analyze_layout,
    detect_registers,
    direction_name,
    find_containment,
    find_mirror_pairs,
    render_layout_text,
    zone_name,
)


def make(key: str, x: int, y: int, w: int, h: int, **kwargs) -> Placement:
    return Placement(key=key, index=int(key.rsplit("/", 1)[-1]),
                     bbox={"x": x, "y": y, "w": w, "h": h}, **kwargs)


# ── zone_name ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cx,cy,expected", [
    (0.5, 0.5, "centre"),
    (0.1, 0.1, "upper left"),
    (0.9, 0.9, "lower right"),
    (0.5, 0.1, "upper centre"),
    (0.1, 0.5, "middle left"),
    (0.9, 0.5, "middle right"),
    (0.5, 0.9, "lower centre"),
])
def test_zone_name(cx, cy, expected):
    assert zone_name(cx, cy) == expected


# ── direction_name ───────────────────────────────────────────────────────────

def test_direction_name_uses_picture_plane_terms():
    origin = make("p/0", 100, 100, 20, 20)
    analyze_layout([origin], 400, 400)

    def rel(x, y):
        other = make("p/1", x, y, 20, 20)
        analyze_layout([origin, other], 400, 400)
        return direction_name(origin, other)

    assert rel(300, 100) == "to the right of"
    assert rel(100, 300) == "below"
    assert rel(100, 10) == "above"
    assert rel(10, 100) == "to the left of"


def test_direction_name_handles_coincident_centres():
    a = make("p/0", 100, 100, 20, 20)
    b = make("p/1", 100, 100, 20, 20)
    analyze_layout([a, b], 400, 400)
    assert direction_name(a, b) == "coincident with"


# ── register detection ───────────────────────────────────────────────────────

def test_detect_registers_groups_a_row_and_stacks_bands():
    # Two motifs side by side at the top, one below.
    placements = [
        make("p/0", 20, 40, 80, 100),
        make("p/1", 250, 45, 80, 100),
        make("p/2", 130, 500, 90, 120),
    ]
    layout = analyze_layout(placements, 400, 900)

    assert len(layout.registers) == 2
    assert layout.registers[0].members == ["p/0", "p/1"]      # left to right
    assert layout.registers[1].members == ["p/2"]
    assert layout.by_key("p/0").register == 0
    assert layout.by_key("p/2").register == 1


def test_detect_registers_orders_members_left_to_right():
    placements = [
        make("p/0", 300, 40, 60, 60),
        make("p/1", 20, 45, 60, 60),
        make("p/2", 160, 42, 60, 60),
    ]
    layout = analyze_layout(placements, 400, 400)
    assert layout.registers[0].members == ["p/1", "p/2", "p/0"]


def test_field_detections_are_excluded_from_registers():
    field = make("p/0", 10, 10, 380, 880)          # ~93% of the panel
    motif = make("p/1", 100, 100, 80, 80)
    layout = analyze_layout([field, motif], 400, 900)

    assert layout.by_key("p/0").is_field is True
    assert layout.by_key("p/0").register == -1
    assert layout.by_key("p/1").is_field is False
    assert all("p/0" not in reg.members for reg in layout.registers)
    # Field reads last — it is the ground the rest sits on.
    assert layout.reading_order[-1] == "p/0"


def test_detect_registers_on_empty_input():
    assert detect_registers([]) == []


# ── mirror pairs ─────────────────────────────────────────────────────────────

def test_find_mirror_pairs_detects_flanking_motifs():
    left = make("p/0", 40, 80, 90, 120)
    right = make("p/1", 270, 80, 90, 120)          # reflects across x=200
    layout = analyze_layout([left, right], 400, 900)
    assert layout.mirror_pairs == [("p/0", "p/1")]


def test_mirror_pairs_reject_different_rows():
    left = make("p/0", 40, 80, 90, 120)
    right = make("p/1", 270, 600, 90, 120)
    layout = analyze_layout([left, right], 400, 900)
    assert layout.mirror_pairs == []


def test_mirror_pairs_reject_mismatched_sizes():
    left = make("p/0", 40, 80, 90, 120)
    right = make("p/1", 290, 100, 40, 50)
    layout = analyze_layout([left, right], 400, 900)
    assert layout.mirror_pairs == []


def test_each_motif_pairs_at_most_once():
    placements = [
        make("p/0", 40, 80, 90, 120),
        make("p/1", 270, 80, 90, 120),
        make("p/2", 275, 82, 90, 120),             # near-duplicate of p/1
    ]
    layout = analyze_layout(placements, 400, 900)
    paired = [k for pair in layout.mirror_pairs for k in pair]
    assert len(paired) == len(set(paired))


# ── containment ──────────────────────────────────────────────────────────────

def test_find_containment_maps_outer_to_inner():
    outer = make("p/0", 0, 0, 400, 400)
    inner = make("p/1", 50, 50, 60, 60)
    outside = make("p/2", 500, 500, 40, 40)
    layout = analyze_layout([outer, inner, outside], 1000, 1000)

    assert layout.containment == {"p/0": ["p/1"]}


def test_partial_overlap_is_not_containment():
    a = make("p/0", 0, 0, 100, 100)
    b = make("p/1", 60, 60, 100, 100)             # 40% overlap at most
    assert find_containment(_prepared([a, b])) == {}


# ── analyze_layout ───────────────────────────────────────────────────────────

def test_analyze_layout_computes_relative_geometry():
    layout = analyze_layout([make("p/0", 100, 200, 200, 400)], 400, 800)
    p = layout.by_key("p/0")
    assert p.cx == pytest.approx(0.5)
    assert p.cy == pytest.approx(0.5)
    assert p.rw == pytest.approx(0.5)
    assert p.rh == pytest.approx(0.5)
    assert p.area_fraction == pytest.approx(0.25)
    assert p.zone == "centre"


def test_analyze_layout_rejects_bad_panel_dimensions():
    with pytest.raises(ValueError, match="positive"):
        analyze_layout([make("p/0", 0, 0, 10, 10)], 0, 100)


def test_reading_order_is_register_major_then_left_to_right():
    placements = [
        make("p/0", 250, 500, 60, 60),
        make("p/1", 30, 40, 60, 60),
        make("p/2", 250, 45, 60, 60),
        make("p/3", 30, 505, 60, 60),
    ]
    layout = analyze_layout(placements, 400, 900)
    assert layout.reading_order == ["p/1", "p/2", "p/3", "p/0"]


def test_adjacency_reports_k_neighbours_per_motif():
    placements = [make(f"p/{i}", 20 + 60 * i, 100, 40, 40) for i in range(4)]
    layout = analyze_layout(placements, 400, 400, neighbours=2)
    assert len(layout.adjacency) == 4 * 2
    assert {r.source for r in layout.adjacency} == {"p/0", "p/1", "p/2", "p/3"}


def test_layout_round_trips_to_json_safe_dict():
    layout = analyze_layout([make("p/0", 10, 10, 50, 50, cluster=3, label="x")], 400, 400)
    data = layout.as_dict()
    import json
    assert json.loads(json.dumps(data))["placements"][0]["cluster"] == 3


# ── prompt rendering ─────────────────────────────────────────────────────────

def test_render_layout_text_covers_every_section():
    placements = [
        make("p/0", 10, 10, 380, 880),                       # field
        make("p/1", 40, 80, 90, 120, cluster=0, label="figure"),
        make("p/2", 270, 80, 90, 120, cluster=0),
        make("p/3", 150, 520, 100, 140, cluster=1),
    ]
    layout = analyze_layout(placements, 400, 900, panel_stem="panel_a")
    text = render_layout_text(layout)

    assert "Panel panel_a" in text
    assert "REGISTERS" in text
    assert "FIELD / GROUND" in text
    assert "BILATERAL PAIRS" in text
    assert "NESTING" in text
    assert "READING ORDER" in text
    assert 'labelled "figure"' in text
    assert "cluster 0" in text


def test_render_layout_text_truncates_long_relation_lists():
    placements = [make(f"p/{i}", 20 * i, 100, 15, 15) for i in range(15)]
    layout = analyze_layout(placements, 400, 400)
    text = render_layout_text(layout, max_relations=5)
    assert "further relations omitted" in text


def _prepared(placements):
    analyze_layout(placements, 1000, 1000)
    return placements
