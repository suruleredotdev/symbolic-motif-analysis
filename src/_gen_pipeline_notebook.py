#!/usr/bin/env python3
"""Generate motif_pipeline.ipynb — run with: uv run python _gen_pipeline_notebook.py"""
import json
from pathlib import Path


def code(cid, src):
    lines = src.strip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": source}


def md(cid, src):
    lines = src.strip("\n").split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": source}


cells = []

# ══════════════════════════════════════════════════════════════════════════════
# Cell 0: Header
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md("mp-0", """\
# Frobenius Motif Pipeline

Unified pipeline for motif segmentation, clustering, labeling, and interpretation.

**Cells are independent** — re-run any cell after the setup cell without affecting others.

| Cell | Stage | What it does |
|------|-------|-------------|
| 1 | Setup | Load panels, approved bboxes, labels into shared state |
| 2 | Segment | Review detections, manual draw, include/exclude |
| 3 | SAM Draft | One-at-a-time SAM suggestions — Accept or Skip each |
| 4 | Cluster | CLIP embeddings + HDBSCAN clustering |
| 5 | Gallery | Browse motifs grouped by cluster / panel / label |
| 6 | Label | Edit labels + LLM Suggest via Claude |
| 7 | Interpret | LLM-powered motif and panel interpretation |
| 8 | Export | Export crops, save session, progress charts |

**Data provenance**: every bbox, label, and cluster assignment records its source
(`manual`, `sam_prompted`, `llm`, `human`) and timestamp.\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1: Setup — imports, paths, shared state
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-1", """\
import json
import io
import re
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from PIL import Image

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_NB   = Path(".").resolve()                           # src/python/
_REPO = _NB.parent.parent
_ANA  = _REPO / "frobenius_artifacts/analysis"

PANELS_DIR    = _ANA / "panels"
ANNOTATED_DIR = _ANA / "annotated"
MOTIFS_DIR    = _ANA / "motifs"
LABELS_PATH   = _ANA / "motif_labels.json"

# ── Shared state ──────────────────────────────────────────────────────────────
from panel_art.pipeline_state import PipelineState

PS = PipelineState()
PS.load_from_disk(
    annotated_dir=ANNOTATED_DIR,
    panels_dir=PANELS_DIR,
    labels_path=LABELS_PATH,
)

# Quick summary
_n_manual = len(PS.manual_templates())
_n_labeled = sum(1 for m in PS.motifs if m.label)
print(f"Manual templates: {_n_manual}  |  Labeled: {_n_labeled}")
print("Re-run this cell to reload from disk (revert all in-memory changes)")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 2: Segment — Panel Review + Manual Draw
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-2", """\
%matplotlib widget
## ── Segment: Panel Review + Manual Draw ─────────────────────────────────────

import io as _io

# ── Helpers ───────────────────────────────────────────────────────────────────
def _containment(a, b):
    ax1, ay1 = a["x"], a["y"];  ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"];  bx2, by2 = bx1 + b["w"], by1 + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    min_area = min(a["w"] * a["h"], b["w"] * b["h"])
    return inter / min_area if min_area > 0 else 0.0

def _auto_exclude(dets, threshold):
    n = len(dets)
    areas = [d.bbox["w"] * d.bbox["h"] for d in dets]
    by_size = sorted(range(n), key=lambda i: -areas[i])
    suppress = set()
    for pos, i in enumerate(by_size):
        if i in suppress: continue
        for j in by_size[pos + 1:]:
            if j not in suppress and _containment(dets[i].bbox, dets[j].bbox) >= threshold:
                suppress.add(j)
    return suppress

def _thumb(panel_img, bbox, size=96, bg=(30, 30, 30)):
    iw, ih = panel_img.size
    b = bbox
    crop = panel_img.crop((max(0, b["x"]), max(0, b["y"]),
                           min(iw, b["x"] + b["w"]), min(ih, b["y"] + b["h"])))
    crop.thumbnail((size, size))
    sq = Image.new("RGB", (size, size), bg)
    sq.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    buf = _io.BytesIO();  sq.save(buf, format="PNG")
    return widgets.Image(value=buf.getvalue(), format="png", width=size, height=size)


# ── Persistent figure ─────────────────────────────────────────────────────────
MAX_DISPLAY_W = 900
_seg_fig, _seg_ax = plt.subplots(1, 1, dpi=96,
    gridspec_kw={"left": 0, "right": 1, "top": 1, "bottom": 0})
_seg_fig.patch.set_facecolor("#111")
_seg_ax.set_facecolor("#111")
_seg_ax.axis("off")

_seg_state = {"stem": None, "dirty": False}
_seg_img_cache = {"stem": None, "arr": None, "scale": 1.0}
_seg_drag = {"active": False, "x0": 0, "y0": 0}
_seg_draw = {"mode_on": False}
_seg_no_obs = [False]


def _seg_scale():
    if _seg_state["stem"] is None: return 1.0
    info = PS.panels[_seg_state["stem"]]
    return min(1.0, MAX_DISPLAY_W / info.width)


def _seg_motifs():
    # Current panel motifs from PipelineState.
    stem = _seg_state["stem"]
    return PS.motifs_for_panel(stem) if stem else []


def _seg_redraw(with_pending=False):
    if _seg_state["stem"] is None: return
    stem = _seg_state["stem"]
    panel_img = PS.panel_image(stem)
    iw, ih = panel_img.size
    scale = min(1.0, MAX_DISPLAY_W / iw)
    dw, dh = int(iw * scale), int(ih * scale)

    _seg_ax.clear()
    _seg_ax.axis("off")

    if _seg_img_cache["stem"] != stem:
        _seg_img_cache["arr"] = np.array(panel_img.resize((dw, dh), Image.LANCZOS))
        _seg_img_cache["stem"] = stem
        _seg_img_cache["scale"] = scale
    _seg_ax.imshow(_seg_img_cache["arr"])

    motifs = _seg_motifs()
    for i, m in enumerate(motifs):
        b = m.bbox
        x, y, w, h = b["x"]*scale, b["y"]*scale, b["w"]*scale, b["h"]*scale
        inc = _seg_cbs[i].value if i < len(_seg_cbs) else m.included
        color = "#44dd44" if inc else "#ff4444"
        _seg_ax.add_patch(mpatches.Rectangle(
            (x, y), w, h,
            linewidth=2.5 if inc else 1.5,
            edgecolor=color, facecolor="none", alpha=0.9 if inc else 0.55))
        _seg_ax.text(x+3, y+3, str(m.index),
            fontsize=max(6, int(10*scale)), color=color, va="top", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.4, pad=1, linewidth=0))

    if with_pending:
        pb = {"x": w_mx.value, "y": w_my.value, "w": w_mw.value, "h": w_mh.value}
        if pb["w"] > 0 and pb["h"] > 0:
            x, y = pb["x"]*scale, pb["y"]*scale
            w2, h2 = pb["w"]*scale, pb["h"]*scale
            _seg_ax.add_patch(mpatches.Rectangle(
                (x, y), w2, h2, linewidth=0, facecolor="#00ffff", alpha=0.12))
            _seg_ax.add_patch(mpatches.Rectangle(
                (x, y), w2, h2, linewidth=2.5, edgecolor="#00ffff",
                facecolor="none", alpha=0.95, linestyle=(0, (6, 3))))
            _seg_ax.text(x+4, y+4, "new",
                fontsize=max(7, int(11*scale)), color="#00ffff", va="top",
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.45, pad=1, linewidth=0))

    if _seg_draw["mode_on"]:
        _seg_ax.set_title("Draw mode ON — drag to select a motif region",
                          color="#00cccc", fontsize=10, pad=3)
    _seg_fig.canvas.draw_idle()


# ── Mouse handlers ────────────────────────────────────────────────────────────
def _seg_on_press(event):
    if not _seg_draw["mode_on"]: return
    if event.inaxes != _seg_ax or event.button != 1: return
    scale = _seg_img_cache["scale"] or _seg_scale()
    _seg_drag["active"] = True
    _seg_drag["x0"] = max(0, int(event.xdata / scale)) if event.xdata is not None else 0
    _seg_drag["y0"] = max(0, int(event.ydata / scale)) if event.ydata is not None else 0

def _seg_on_move(event):
    if not _seg_drag["active"] or not _seg_draw["mode_on"]: return
    if event.inaxes != _seg_ax or event.xdata is None: return
    scale = _seg_img_cache["scale"] or _seg_scale()
    x0, y0 = _seg_drag["x0"], _seg_drag["y0"]
    x1 = max(0, int(event.xdata / scale))
    y1 = max(0, int(event.ydata / scale))
    if _seg_state["stem"]:
        info = PS.panels[_seg_state["stem"]]
        x1 = min(x1, info.width - 1); y1 = min(y1, info.height - 1)
    x = min(x0, x1); y = min(y0, y1)
    w = max(1, abs(x1 - x0)); h = max(1, abs(y1 - y0))
    _seg_no_obs[0] = True
    w_mx.value = x; w_my.value = y; w_mw.value = w; w_mh.value = h
    _seg_no_obs[0] = False
    _seg_redraw(with_pending=True)

def _seg_on_release(event):
    if not _seg_drag["active"]: return
    _seg_drag["active"] = False
    _seg_redraw(with_pending=True)

_seg_fig.canvas.mpl_connect("button_press_event", _seg_on_press)
_seg_fig.canvas.mpl_connect("motion_notify_event", _seg_on_move)
_seg_fig.canvas.mpl_connect("button_release_event", _seg_on_release)


# ── Widgets ───────────────────────────────────────────────────────────────────
_panel_opts = [(s, s) for s in sorted(PS.panels.keys())]
w_seg_panel = widgets.Dropdown(options=_panel_opts, description="Panel:",
    style={"description_width": "60px"}, layout=widgets.Layout(width="90%"))

_sl_kw = dict(continuous_update=False, style={"description_width": "140px"},
              layout=widgets.Layout(width="55%"))
w_subcrop = widgets.FloatSlider(min=0, max=1, step=0.05, value=0.75,
    description="sub-crop filter", readout_format=".2f", **_sl_kw)
w_min_qual = widgets.FloatSlider(min=0, max=1, step=0.05, value=0.0,
    description="min SAM quality", readout_format=".2f", **_sl_kw)

btn_save    = widgets.Button(description="Save approved", button_style="success",
                             layout=widgets.Layout(width="150px"))
btn_revert  = widgets.Button(description="Revert to saved", button_style="danger",
                             layout=widgets.Layout(width="135px"))
btn_draw    = widgets.Button(description="[D] Draw bbox", button_style="",
                             layout=widgets.Layout(width="130px"))

_draw_sl = dict(continuous_update=True, style={"description_width": "18px"},
                layout=widgets.Layout(width="88%"))
w_mx = widgets.IntSlider(value=0, min=0, max=2000, step=1, description="x", **_draw_sl)
w_my = widgets.IntSlider(value=0, min=0, max=2000, step=1, description="y", **_draw_sl)
w_mw = widgets.IntSlider(value=100, min=1, max=2000, step=1, description="w", **_draw_sl)
w_mh = widgets.IntSlider(value=100, min=1, max=2000, step=1, description="h", **_draw_sl)
btn_add = widgets.Button(description="Add bbox ✓", button_style="info",
                         layout=widgets.Layout(width="110px"))

out_seg_cards  = widgets.Output()
out_seg_status = widgets.Output()

_seg_cbs: list[widgets.Checkbox] = []  # detection card checkboxes


# ── Handlers ──────────────────────────────────────────────────────────────────
def _seg_mark_dirty():
    _seg_state["dirty"] = True
    btn_save.description = "Save approved ●"
    btn_save.button_style = "warning"

def _seg_mark_clean():
    _seg_state["dirty"] = False
    btn_save.description = "Save approved"
    btn_save.button_style = "success"

def _seg_build_cards():
    global _seg_cbs
    stem = _seg_state["stem"]
    if not stem: return
    motifs = _seg_motifs()
    panel_img = PS.panel_image(stem)
    auto_excl = _auto_exclude(motifs, w_subcrop.value)
    min_q = w_min_qual.value

    cbs, cards = [], []
    for i, m in enumerate(motifs):
        tw = _thumb(panel_img, m.bbox)
        below_q = m.predicted_iou < min_q
        auto_tag = " ⚠ sub-crop" if i in auto_excl else (" ⚠ low quality" if below_q else "")
        src_tag = f" [{m.source}]"
        cb = widgets.Checkbox(value=(i not in auto_excl and not below_q and m.included),
                              description="Include", indent=False,
                              style={"description_width": "initial"},
                              layout=widgets.Layout(width="100px"))
        cbs.append(cb)
        cb.observe(lambda c: (_seg_mark_dirty(), _seg_redraw()), names="value")
        iou_s = f"{m.predicted_iou:.3f}"
        area_s = f"{m.area_ratio:.3f}"
        b = m.bbox
        meta = widgets.HTML(
            f"<div style='font-size:11px;line-height:1.5;color:#ccc'>"
            f"<b>#{m.index}</b> {m.scale}{src_tag}<br>"
            f"area: {area_s}<br>iou: {iou_s}<br>"
            f"{b['w']}×{b['h']} px"
            f"<span style='color:#ff8888'>{auto_tag}</span></div>")
        cards.append(widgets.VBox([tw, meta, cb],
            layout=widgets.Layout(border="1px solid #333", padding="4px",
                                  margin="3px", width="120px", background="#1a1a1a")))

    _seg_cbs = cbs
    rows = [widgets.HBox(cards[r:r+6]) for r in range(0, len(cards), 6)]
    out_seg_cards.clear_output(wait=True)
    with out_seg_cards:
        display(widgets.VBox(rows) if rows else widgets.HTML("<i>No detections</i>"))


def _seg_load_panel(_=None):
    stem = w_seg_panel.value
    old_stem = _seg_state["stem"]
    if _seg_state["dirty"] and old_stem and old_stem != stem:
        _seg_save()  # autosave on panel switch
    _seg_state["stem"] = stem
    _seg_img_cache["stem"] = None  # invalidate
    info = PS.panels[stem]
    w_mx.value = 0; w_my.value = 0
    w_mw.value = min(100, info.width); w_mh.value = min(100, info.height)
    w_mx.max = info.width - 1; w_my.max = info.height - 1
    w_mw.max = info.width; w_mh.max = info.height
    _seg_mark_clean()
    _seg_build_cards()
    _seg_redraw()
    out_seg_status.clear_output()
    with out_seg_status:
        motifs = _seg_motifs()
        n_inc = sum(1 for m in motifs if m.included)
        print(f"{stem} — {len(motifs)} detections ({n_inc} included)")
        print(f"Panel: {info.width}×{info.height} px")


def _seg_save(_=None):
    stem = _seg_state["stem"]
    if not stem: return
    # Sync checkbox state → MotifRecord.included
    motifs = _seg_motifs()
    for i, m in enumerate(motifs):
        m.included = _seg_cbs[i].value if i < len(_seg_cbs) else m.included
    path = PS.save_approved(stem)
    _seg_mark_clean()
    n = sum(1 for m in motifs if m.included)
    out_seg_status.clear_output()
    with out_seg_status:
        print(f"Saved: {n}/{len(motifs)} → {path.name}")


def _seg_revert(_=None):
    _seg_state["dirty"] = False
    # Reload this panel from disk
    stem = _seg_state["stem"]
    if not stem: return
    PS.motifs = [m for m in PS.motifs if m.panel_stem != stem]
    # Re-read from disk
    approved_path = ANNOTATED_DIR / f"{stem}_approved.json"
    det_path = ANNOTATED_DIR / f"{stem}_detections.json"
    src_path = approved_path if approved_path.exists() else det_path
    if src_path.exists():
        raw = json.loads(src_path.read_text())
        for d in raw:
            from panel_art.pipeline_state import MotifRecord
            PS.motifs.append(MotifRecord(
                panel_stem=stem, index=d.get("index", 0), bbox=d["bbox"],
                scale=d.get("scale", "motif"),
                area_ratio=d.get("area_ratio", 0.0),
                predicted_iou=d.get("predicted_iou", d.get("pred_iou", 0.0)),
                stability_score=d.get("stability_score", 0.0),
                source=d.get("source", "sam_auto"),
                created_at=d.get("created_at", ""), included=True))
    _seg_load_panel()
    out_seg_status.clear_output()
    with out_seg_status:
        print(f"Reverted to saved state — {len(_seg_motifs())} detections")


def _seg_on_draw_toggle(_=None):
    _seg_draw["mode_on"] = not _seg_draw["mode_on"]
    if _seg_draw["mode_on"]:
        btn_draw.description = "[D] Draw ● ON"
        btn_draw.button_style = "info"
    else:
        btn_draw.description = "[D] Draw bbox"
        btn_draw.button_style = ""
        _seg_drag["active"] = False
    _seg_redraw(with_pending=_seg_draw["mode_on"])


def _seg_on_add(_=None):
    stem = _seg_state["stem"]
    if not stem: return
    bbox = {"x": w_mx.value, "y": w_my.value, "w": w_mw.value, "h": w_mh.value}
    if bbox["w"] <= 0 or bbox["h"] <= 0: return
    m = PS.add_motif(stem, bbox, source="manual")
    _seg_mark_dirty()
    _seg_build_cards()
    _seg_redraw()
    out_seg_status.clear_output()
    with out_seg_status:
        print(f"Added manual bbox #{m.index}: ({bbox['x']},{bbox['y']}) {bbox['w']}×{bbox['h']}")


def _seg_on_slider(_=None):
    if _seg_no_obs[0]: return
    _seg_redraw(with_pending=True)

def _seg_on_filter(_=None):
    _seg_build_cards()
    _seg_redraw()

# ── Wire up ───────────────────────────────────────────────────────────────────
w_seg_panel.observe(_seg_load_panel, names="value")
w_subcrop.observe(_seg_on_filter, names="value")
w_min_qual.observe(_seg_on_filter, names="value")
btn_save.on_click(_seg_save)
btn_revert.on_click(_seg_revert)
btn_draw.on_click(_seg_on_draw_toggle)
btn_add.on_click(_seg_on_add)
for _w in (w_mx, w_my, w_mw, w_mh):
    _w.observe(_seg_on_slider, names="value")


# ── Layout ────────────────────────────────────────────────────────────────────
_filter_help = widgets.HTML(
    "<div style='font-size:11px;color:#999;margin:2px 0 4px 0'>"
    "<b>sub-crop filter</b>: higher = stricter, auto-excludes nested bboxes "
    "&nbsp;|&nbsp; "
    "<b>min SAM quality</b>: raise to hide low-confidence detections</div>")

_draw_section = widgets.VBox([
    widgets.HTML("<b style='font-size:12px;color:#00cccc'>"
                 "Pending bbox — drag on image or use sliders:</b>"),
    widgets.HBox([
        widgets.VBox([widgets.HBox([w_mx]), widgets.HBox([w_my])],
                     layout=widgets.Layout(width="50%")),
        widgets.VBox([widgets.HBox([w_mw]), widgets.HBox([w_mh])],
                     layout=widgets.Layout(width="50%")),
    ]),
    btn_add,
], layout=widgets.Layout(border="1px solid #006666", padding="6px 10px",
                         margin="4px 0", background="#001a1a"))

display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 1: Segment</h3>"),
    w_seg_panel, _filter_help,
    widgets.HBox([w_subcrop, w_min_qual]),
    widgets.HBox([btn_draw, btn_save, btn_revert]),
    out_seg_status,
    _seg_fig.canvas,
    _draw_section,
    widgets.HTML("<b style='font-size:13px'>Detection cards</b>"),
    out_seg_cards,
)
_seg_load_panel()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 3: SAM Draft Bbox Review
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-3", """\
## ── Stage 1b: SAM Draft Review ──────────────────────────────────────────────
#
# One candidate at a time.  Click "Generate" to run SAM, then Accept/Skip
# through candidates.  Each acceptance feeds back as a template.

import panel_art.motif_segment as _ms
import importlib; importlib.reload(_ms)

# ── Widgets ───────────────────────────────────────────────────────────────────
_ref_sl = dict(continuous_update=False, style={"description_width": "130px"},
               layout=widgets.Layout(width="48%"))

w_dr_score = widgets.FloatSlider(min=0.50, max=0.99, step=0.01, value=0.85,
    description="min SAM score", readout_format=".2f", **_ref_sl)
w_dr_min_area = widgets.FloatSlider(min=0.005, max=0.20, step=0.005, value=0.01,
    description="min area %", readout_format=".1%", **_ref_sl)
w_dr_max_area = widgets.FloatSlider(min=0.10, max=0.80, step=0.05, value=0.50,
    description="max area %", readout_format=".0%", **_ref_sl)
w_dr_edge = widgets.FloatSlider(min=0.0, max=0.15, step=0.005, value=0.03,
    description="min edge density", readout_format=".1%", **_ref_sl)

btn_dr_generate = widgets.Button(description="Generate SAM Candidates",
    button_style="", layout=widgets.Layout(width="220px"))
btn_dr_accept = widgets.Button(description="Accept ✓", button_style="success",
    layout=widgets.Layout(width="110px"))
btn_dr_skip = widgets.Button(description="Skip →", button_style="warning",
    layout=widgets.Layout(width="100px"))
btn_dr_reset = widgets.Button(description="Reset Queue", button_style="",
    layout=widgets.Layout(width="120px"))

# Draft bbox adjustment sliders (synced from Cell 2's draw sliders concept)
_dsl = dict(continuous_update=True, style={"description_width": "18px"},
            layout=widgets.Layout(width="44%"))
w_dx = widgets.IntSlider(value=0, min=0, max=2000, step=1, description="x", **_dsl)
w_dy = widgets.IntSlider(value=0, min=0, max=2000, step=1, description="y", **_dsl)
w_dw = widgets.IntSlider(value=100, min=1, max=2000, step=1, description="w", **_dsl)
w_dh = widgets.IntSlider(value=100, min=1, max=2000, step=1, description="h", **_dsl)

out_draft = widgets.Output()
out_draft_status = widgets.Output()

# ── Draft figure (separate from Cell 2) ───────────────────────────────────────
_dr_fig, _dr_ax = plt.subplots(1, 1, dpi=96,
    gridspec_kw={"left": 0, "right": 1, "top": 1, "bottom": 0})
_dr_fig.patch.set_facecolor("#111")
_dr_ax.set_facecolor("#111")
_dr_ax.axis("off")
_dr_img_cache = {"stem": None, "arr": None, "scale": 1.0}


def _dr_redraw():
    # Draw panel with existing bboxes (green) + current draft candidate (cyan).
    stem = _seg_state.get("stem") or (w_seg_panel.value if 'w_seg_panel' in dir() else None)
    if not stem or stem not in PS.panels: return

    panel_img = PS.panel_image(stem)
    iw, ih = panel_img.size
    scale = min(1.0, MAX_DISPLAY_W / iw)
    dw, dh = int(iw * scale), int(ih * scale)

    _dr_ax.clear(); _dr_ax.axis("off")
    if _dr_img_cache["stem"] != stem:
        _dr_img_cache["arr"] = np.array(panel_img.resize((dw, dh), Image.LANCZOS))
        _dr_img_cache["stem"] = stem
        _dr_img_cache["scale"] = scale
    _dr_ax.imshow(_dr_img_cache["arr"])

    # Existing included bboxes (green, thin)
    for m in PS.motifs_for_panel(stem):
        if not m.included: continue
        b = m.bbox
        x, y, w, h = b["x"]*scale, b["y"]*scale, b["w"]*scale, b["h"]*scale
        _dr_ax.add_patch(mpatches.Rectangle((x, y), w, h,
            linewidth=1.5, edgecolor="#44dd44", facecolor="none", alpha=0.6))

    # Current draft candidate (cyan, bold)
    pb = {"x": w_dx.value, "y": w_dy.value, "w": w_dw.value, "h": w_dh.value}
    if pb["w"] > 0 and pb["h"] > 0:
        x, y = pb["x"]*scale, pb["y"]*scale
        w2, h2 = pb["w"]*scale, pb["h"]*scale
        _dr_ax.add_patch(mpatches.Rectangle((x, y), w2, h2,
            linewidth=0, facecolor="#00ffff", alpha=0.12))
        _dr_ax.add_patch(mpatches.Rectangle((x, y), w2, h2,
            linewidth=2.5, edgecolor="#00ffff", facecolor="none",
            alpha=0.95, linestyle=(0, (6, 3))))
        _dr_ax.text(x+4, y+4, "draft",
            fontsize=10, color="#00ffff", va="top", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, pad=1, linewidth=0))

    _dr_fig.canvas.draw_idle()


def _dr_show_candidate():
    # Display the current draft candidate info and update sliders.
    stem = _seg_state.get("stem")
    if not stem: return

    cand = PS.next_draft(stem)
    total, cursor, accepted, skipped = PS.draft_count(stem)

    out_draft_status.clear_output()
    with out_draft_status:
        if cand is None:
            print(f"No more candidates for this panel. "
                  f"(Accepted: {accepted}, Skipped: {skipped}/{total})")
            print("Click 'Reset Queue' to review again, or 'Generate' with new settings.")
            btn_dr_accept.disabled = True
            btn_dr_skip.disabled = True
            return

        btn_dr_accept.disabled = False
        btn_dr_skip.disabled = False
        rec = cand["record"]
        print(f"Candidate {cursor+1}/{total} — "
              f"SAM score={rec.predicted_iou:.3f}, "
              f"edge={cand['edge_density']:.3f}, "
              f"novelty={cand['novelty']:.2f}, "
              f"composite={cand['score']:.3f}")
        print(f"Accepted: {accepted} | Skipped: {skipped} | "
              f"Remaining: {total - cursor}")

        # Set sliders to candidate bbox
        b = rec.bbox
        info = PS.panels[stem]
        w_dx.max = info.width - 1; w_dy.max = info.height - 1
        w_dw.max = info.width; w_dh.max = info.height
        w_dx.value = b["x"]; w_dy.value = b["y"]
        w_dw.value = b["w"]; w_dh.value = b["h"]

    _dr_redraw()


# ── Handlers ──────────────────────────────────────────────────────────────────
def _dr_generate(_=None):
    stem = _seg_state.get("stem")
    if not stem:
        with out_draft_status: print("Select a panel in Cell 2 first")
        return

    btn_dr_generate.description = "Running SAM…"
    btn_dr_generate.disabled = True

    out_draft_status.clear_output()
    with out_draft_status:
        existing = [m.bbox for m in PS.motifs_for_panel(stem) if m.included]
        templates = PS.manual_templates()
        print(f"SamPredictor: {len(existing)} existing bboxes, "
              f"{len(templates)} manual templates…")

    try:
        img_np = np.array(PS.panel_image(stem))
        candidates = _ms.prompted_segment(
            img_np, existing,
            approved_templates=templates or None,
            min_score=w_dr_score.value,
            min_area=w_dr_min_area.value,
            max_area=w_dr_max_area.value,
            min_edge_density=w_dr_edge.value,
            verbose=True,
        )
        n = PS.cache_sam_candidates(stem, candidates)
        out_draft_status.clear_output()
        with out_draft_status:
            print(f"Generated {n} candidates, sorted by composite score")
        _dr_show_candidate()
    except Exception:
        out_draft_status.clear_output()
        with out_draft_status:
            import traceback; traceback.print_exc()
    finally:
        btn_dr_generate.description = "Generate SAM Candidates"
        btn_dr_generate.disabled = False


def _dr_accept(_=None):
    stem = _seg_state.get("stem")
    if not stem: return
    adjusted = {"x": w_dx.value, "y": w_dy.value, "w": w_dw.value, "h": w_dh.value}
    rec = PS.accept_draft(stem, adjusted_bbox=adjusted)
    # Rebuild Cell 2 cards to show the new detection
    _seg_build_cards()
    _seg_redraw()
    _seg_mark_dirty()
    _dr_show_candidate()


def _dr_skip(_=None):
    stem = _seg_state.get("stem")
    if not stem: return
    PS.skip_draft(stem)
    _dr_show_candidate()


def _dr_reset(_=None):
    stem = _seg_state.get("stem")
    if not stem: return
    PS.reset_draft_queue(stem)
    _dr_show_candidate()


def _dr_on_slider(_=None):
    _dr_redraw()


# ── Wire up ───────────────────────────────────────────────────────────────────
btn_dr_generate.on_click(_dr_generate)
btn_dr_accept.on_click(_dr_accept)
btn_dr_skip.on_click(_dr_skip)
btn_dr_reset.on_click(_dr_reset)
for _w in (w_dx, w_dy, w_dw, w_dh):
    _w.observe(_dr_on_slider, names="value")


# ── Layout ────────────────────────────────────────────────────────────────────
display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 1b: SAM Draft Review</h3>"),
    widgets.HTML(
        "<div style='font-size:12px;color:#dd8800;margin-bottom:6px'>"
        "Adjust SAM settings, click Generate, then Accept/Skip each candidate. "
        "Each acceptance tightens the template for future suggestions.</div>"),
    widgets.HBox([w_dr_score, w_dr_min_area]),
    widgets.HBox([w_dr_max_area, w_dr_edge]),
    widgets.HBox([btn_dr_generate, btn_dr_accept, btn_dr_skip, btn_dr_reset]),
    out_draft_status,
    _dr_fig.canvas,
    widgets.HTML("<b style='font-size:12px;color:#00cccc'>Adjust draft bbox:</b>"),
    widgets.HBox([w_dx, w_dy]),
    widgets.HBox([w_dw, w_dh]),
)\
"""))


# ══════════════════════════════════════════════════════════════════════════════
# Write notebook
# ══════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "motif_pipeline.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")
