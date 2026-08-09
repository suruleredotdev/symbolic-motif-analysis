#!/usr/bin/env python3
"""Generate motif_pipeline.ipynb — run with: uv run python _gen_pipeline_notebook.py"""
import json
import sys
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
| 2 | Segment | Review detections, manual draw, SAM Refine — all in one UI |
| 3 | Cluster | CLIP embeddings + HDBSCAN clustering |
| 4 | Gallery | Browse motifs grouped by cluster / panel / label |
| 5 | Label | Edit labels + LLM Suggest via Claude |
| 6 | Interpret | LLM-powered motif and panel interpretation |
| 7 | Export | Save state, export crops, progress charts |

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
from IPython.display import display, clear_output, HTML, Markdown
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
CLUSTERS_PATH = _ANA / "clusters.json"
EMBED_CACHE   = _ANA / "embeddings_cache.npy"
EMBED_KEYS    = _ANA / "embeddings_cache_keys.txt"

# ── Shared state ──────────────────────────────────────────────────────────────
from panel_art.pipeline_state import PipelineState

PS = PipelineState()
PS.load_from_disk(
    annotated_dir=ANNOTATED_DIR,
    panels_dir=PANELS_DIR,
    labels_path=LABELS_PATH,
    clusters_path=CLUSTERS_PATH,
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
    # Add pending bbox — works for both manual draws and SAM candidates
    stem = _seg_state["stem"]
    if not stem: return
    bbox = {"x": w_mx.value, "y": w_my.value, "w": w_mw.value, "h": w_mh.value}
    if bbox["w"] <= 0 or bbox["h"] <= 0: return
    cand = PS.next_draft(stem)
    if cand is not None:
        rec = PS.accept_draft(stem, adjusted_bbox=bbox)
        tag = "sam_prompted"
        _seg_load_next_candidate()
        _seg_show_sam_status()
    else:
        rec = PS.add_motif(stem, bbox, source="manual")
        tag = "manual"
    _seg_mark_dirty()
    _seg_build_cards()
    has_next = PS.next_draft(stem) is not None
    _seg_redraw(with_pending=has_next)
    out_seg_status.clear_output()
    with out_seg_status:
        print(f"Added {tag} bbox #{rec.index}: ({bbox['x']},{bbox['y']}) {bbox['w']}×{bbox['h']}")


def _seg_on_slider(_=None):
    if _seg_no_obs[0]: return
    _seg_redraw(with_pending=True)

def _seg_on_filter(_=None):
    _seg_build_cards()
    _seg_redraw()


# ── SAM Refine (integrated) ──────────────────────────────────────────────────
import panel_art.motif_segment as _ms
import importlib; importlib.reload(_ms)

_ref_sl = dict(continuous_update=False, style={"description_width": "130px"},
               layout=widgets.Layout(width="48%"))
w_sam_score = widgets.FloatSlider(min=0.50, max=0.99, step=0.01, value=0.85,
    description="min SAM score", readout_format=".2f", **_ref_sl)
w_sam_min_area = widgets.FloatSlider(min=0.005, max=0.20, step=0.005, value=0.01,
    description="min area %", readout_format=".1%", **_ref_sl)
w_sam_max_area = widgets.FloatSlider(min=0.10, max=0.80, step=0.05, value=0.50,
    description="max area %", readout_format=".0%", **_ref_sl)
w_sam_edge = widgets.FloatSlider(min=0.0, max=0.15, step=0.005, value=0.03,
    description="min edge density", readout_format=".1%", **_ref_sl)

btn_sam_gen  = widgets.Button(description="SAM Generate", button_style="",
    layout=widgets.Layout(width="140px"))
btn_sam_skip = widgets.Button(description="Skip →", button_style="warning",
    layout=widgets.Layout(width="90px"), disabled=True)
btn_sam_reset = widgets.Button(description="Reset Queue", button_style="",
    layout=widgets.Layout(width="110px"))
out_sam_status = widgets.Output()


def _seg_load_next_candidate():
    stem = _seg_state["stem"]
    if not stem: return
    cand = PS.next_draft(stem)
    if cand is None:
        btn_sam_skip.disabled = True
        return
    btn_sam_skip.disabled = False
    rec = cand["record"]
    b = rec.bbox
    info = PS.panels[stem]
    _seg_no_obs[0] = True
    w_mx.max = info.width - 1; w_my.max = info.height - 1
    w_mw.max = info.width; w_mh.max = info.height
    w_mx.value = b["x"]; w_my.value = b["y"]
    w_mw.value = b["w"]; w_mh.value = b["h"]
    _seg_no_obs[0] = False
    _seg_redraw(with_pending=True)


def _seg_show_sam_status():
    stem = _seg_state["stem"]
    if not stem: return
    total, cursor, accepted, skipped = PS.draft_count(stem)
    cand = PS.next_draft(stem)
    out_sam_status.clear_output()
    with out_sam_status:
        if cand is None:
            if total > 0:
                print(f"Queue done. Accepted: {accepted}, Skipped: {skipped}/{total}")
            btn_sam_skip.disabled = True
        else:
            rec = cand["record"]
            print(f"SAM {cursor+1}/{total} — score={rec.predicted_iou:.3f}, "
                  f"edge={cand['edge_density']:.2f}, novelty={cand['novelty']:.2f}")
            print(f"Accepted: {accepted} | Skipped: {skipped} | Remaining: {total - cursor}")
            print("Adjust with drag/sliders, then Add bbox or Skip")
            btn_sam_skip.disabled = False


def _seg_on_sam_gen(_=None):
    stem = _seg_state["stem"]
    if not stem: return
    btn_sam_gen.description = "Running SAM..."
    btn_sam_gen.disabled = True
    out_sam_status.clear_output()
    with out_sam_status: print("Loading SAM...")
    try:
        existing = [m.bbox for m in PS.motifs_for_panel(stem) if m.included]
        templates = PS.manual_templates()
        img_np = np.array(PS.panel_image(stem))
        candidates = _ms.prompted_segment(
            img_np, existing,
            approved_templates=templates or None,
            min_score=w_sam_score.value,
            min_area=w_sam_min_area.value,
            max_area=w_sam_max_area.value,
            min_edge_density=w_sam_edge.value,
            verbose=True,
        )
        n = PS.cache_sam_candidates(stem, candidates)
        out_sam_status.clear_output()
        with out_sam_status: print(f"Generated {n} candidates")
        _seg_load_next_candidate()
        _seg_show_sam_status()
    except Exception:
        out_sam_status.clear_output()
        with out_sam_status: import traceback; traceback.print_exc()
    finally:
        btn_sam_gen.description = "SAM Generate"
        btn_sam_gen.disabled = False


def _seg_on_sam_skip(_=None):
    stem = _seg_state["stem"]
    if not stem: return
    PS.skip_draft(stem)
    _seg_load_next_candidate()
    _seg_show_sam_status()


def _seg_on_sam_reset(_=None):
    stem = _seg_state["stem"]
    if not stem: return
    PS.reset_draft_queue(stem)
    _seg_load_next_candidate()
    _seg_show_sam_status()


# ── Wire up ───────────────────────────────────────────────────────────────────
w_seg_panel.observe(_seg_load_panel, names="value")
w_subcrop.observe(_seg_on_filter, names="value")
w_min_qual.observe(_seg_on_filter, names="value")
btn_save.on_click(_seg_save)
btn_revert.on_click(_seg_revert)
btn_draw.on_click(_seg_on_draw_toggle)
btn_add.on_click(_seg_on_add)
btn_sam_gen.on_click(_seg_on_sam_gen)
btn_sam_skip.on_click(_seg_on_sam_skip)
btn_sam_reset.on_click(_seg_on_sam_reset)
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
                 "Pending bbox — drag on image, use sliders, or SAM Generate:</b>"),
    widgets.HBox([
        widgets.VBox([widgets.HBox([w_mx]), widgets.HBox([w_my])],
                     layout=widgets.Layout(width="50%")),
        widgets.VBox([widgets.HBox([w_mw]), widgets.HBox([w_mh])],
                     layout=widgets.Layout(width="50%")),
    ]),
    widgets.HBox([btn_add, btn_sam_skip]),
], layout=widgets.Layout(border="1px solid #006666", padding="6px 10px",
                         margin="4px 0", background="#001a1a"))

_sam_section = widgets.VBox([
    widgets.HTML("<b style='font-size:12px;color:#dd8800'>"
                 "SAM Refine — candidates appear one at a time as pending bboxes:</b>"),
    widgets.HBox([w_sam_score, w_sam_min_area]),
    widgets.HBox([w_sam_max_area, w_sam_edge]),
    widgets.HBox([btn_sam_gen, btn_sam_reset]),
    out_sam_status,
], layout=widgets.Layout(border="1px solid #664400", padding="6px 10px",
                         margin="4px 0", background="#1a0d00"))

display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 1: Segment</h3>"),
    w_seg_panel, _filter_help,
    widgets.HBox([w_subcrop, w_min_qual]),
    widgets.HBox([btn_draw, btn_save, btn_revert]),
    out_seg_status,
    _seg_fig.canvas,
    _draw_section,
    _sam_section,
    widgets.HTML("<b style='font-size:13px'>Detection cards</b>"),
    out_seg_cards,
)
_seg_load_panel()\
"""))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 3: Cluster — Embeddings + t-SNE + HDBSCAN
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-3", """\
## ── Stage 2: Cluster — CLIP Embeddings + HDBSCAN ────────────────────────────
#
# Computes CLIP embeddings from in-memory crops (no disk files needed),
# runs t-SNE for 2D layout, HDBSCAN for clustering.
# Cluster assignments are stored back on PS.motifs[].cluster.

import cv2 as _cv2
import torch as _torch
import open_clip as _oc
from sklearn.manifold import TSNE as _TSNE
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
from sklearn.cluster import AgglomerativeClustering as _Agglom
import hdbscan as _hdbscan

# ── Preprocessing mode ────────────────────────────────────────────────────────
CLIP_MODEL  = "ViT-B-32"
CLIP_WEIGHTS = "openai"
BATCH = 32

def _preprocess_crop(img, mode):
    # Apply mode transform before CLIP
    img = img.convert("RGB")
    arr = np.array(img)
    if mode == "color":
        return img
    elif mode == "grayscale":
        gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)
        return Image.fromarray(np.stack([gray]*3, axis=-1))
    elif mode == "edges":
        gray = _cv2.cvtColor(arr, _cv2.COLOR_RGB2GRAY)
        med = float(np.median(gray))
        edges = _cv2.Canny(gray, max(0, 0.5*med), min(255, 1.5*med))
        edges = 255 - edges
        return Image.fromarray(np.stack([edges]*3, axis=-1))
    elif mode == "clahe":
        lab = _cv2.cvtColor(arr, _cv2.COLOR_RGB2LAB).astype(np.uint8)
        cl = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        lab[:,:,0] = cl.apply(lab[:,:,0])
        return Image.fromarray(_cv2.cvtColor(lab, _cv2.COLOR_LAB2RGB))
    return img

_clip_cache = {}
def _get_clip():
    if not _clip_cache:
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP {CLIP_MODEL} on {device}...")
        model, _, prep = _oc.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_WEIGHTS)
        model = model.to(device).eval()
        _clip_cache.update(model=model, prep=prep, device=device)
    return _clip_cache["model"], _clip_cache["prep"], _clip_cache["device"]


# ── Widgets ───────────────────────────────────────────────────────────────────
_cl_sl = dict(continuous_update=False, style={"description_width": "160px"},
              layout=widgets.Layout(width="55%"))

w_prep_mode = widgets.ToggleButtons(
    options=["edges", "grayscale", "color", "clahe"], value="edges",
    description="Preprocessing:", style={"description_width": "120px", "button_width": "80px"})

btn_embed = widgets.Button(description="Compute Embeddings", button_style="info",
    layout=widgets.Layout(width="200px"))

w_mcs    = widgets.IntSlider(min=2, max=20, step=1, value=3,
    description="min_cluster_size", **_cl_sl)
w_ms     = widgets.IntSlider(min=1, max=10, step=1, value=1,
    description="min_samples", **_cl_sl)
w_method = widgets.ToggleButtons(options=["leaf", "eom"], value="leaf",
    description="selection method",
    style={"description_width": "160px", "button_width": "70px"})
w_pass2  = widgets.Checkbox(value=True,
    description="Sub-cluster noise points (pass 2)",
    style={"description_width": "initial"})
w_nsub   = widgets.IntSlider(min=2, max=30, step=1, value=8,
    description="pass-2 sub-clusters", **_cl_sl)

out_embed  = widgets.Output()
out_clust  = widgets.Output()

# ── State ─────────────────────────────────────────────────────────────────────
_cl_state = {
    "motif_keys": [],     # ordered keys matching embeddings rows
    "embeddings": None,
    "coords": None,       # t-SNE 2D
    "sim_matrix": None,
    "labels": None,
    "n_clusters": 0,
}


def _compute_embeddings(_=None):
    mode = w_prep_mode.value
    included = PS.included_motifs()
    if not included:
        with out_embed: print("No included motifs — approve some in Stage 1 first")
        return

    btn_embed.description = "Computing..."
    btn_embed.disabled = True
    out_embed.clear_output()

    try:
        model, prep, device = _get_clip()
        keys, all_feats = [], []
        with out_embed:
            print(f"Embedding {len(included)} motifs (mode={mode})...")

        for i in range(0, len(included), BATCH):
            batch = included[i:i+BATCH]
            imgs = _torch.stack([
                prep(_preprocess_crop(PS.crop(m), mode)) for m in batch
            ]).to(device)
            with _torch.no_grad():
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().float().numpy())
            keys.extend(m.motif_key for m in batch)

        embs = np.vstack(all_feats)
        N = len(keys)

        # t-SNE
        with out_embed:
            print(f"Running t-SNE (n={N})...")
        perp = min(30, max(5, N // 8))
        import sklearn as _sk
        _ver = tuple(int(x) for x in _sk.__version__.split(".")[:2])
        _ik = "max_iter" if _ver >= (1, 5) else "n_iter"
        coords = _TSNE(n_components=2, perplexity=perp, metric="euclidean",
                        random_state=42, init="pca", **{_ik: 1500}).fit_transform(embs)

        sim = _cos_sim(embs)

        _cl_state["motif_keys"] = keys
        _cl_state["embeddings"] = embs
        _cl_state["coords"] = coords
        _cl_state["sim_matrix"] = sim
        PS.embeddings = embs
        PS.tsne_xy = coords
        PS.sim_matrix = sim

        out_embed.clear_output()
        with out_embed:
            print(f"Done: {embs.shape} embeddings, t-SNE coords ready")

        # Run initial clustering
        _run_clustering()
    except Exception:
        out_embed.clear_output()
        with out_embed:
            import traceback; traceback.print_exc()
    finally:
        btn_embed.description = "Compute Embeddings"
        btn_embed.disabled = False


def _run_clustering(_=None):
    embs = _cl_state["embeddings"]
    if embs is None:
        with out_clust: print("Compute embeddings first")
        return

    mcs, ms = w_mcs.value, w_ms.value
    method = w_method.value
    do_pass2, n_sub = w_pass2.value, w_nsub.value

    cl = _hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                           metric="euclidean", cluster_selection_method=method)
    lbl = cl.fit_predict(embs).copy()
    nc1 = len(set(lbl)) - (1 if -1 in lbl else 0)
    nn = int((lbl == -1).sum())

    nc2 = 0
    if do_pass2 and nn > 0:
        noise_idx = np.where(lbl == -1)[0]
        n_sub_real = min(n_sub, len(noise_idx))
        if n_sub_real >= 2:
            sub_lbl = _Agglom(n_clusters=n_sub_real).fit_predict(embs[noise_idx])
            for i, ni in enumerate(noise_idx):
                lbl[ni] = nc1 + sub_lbl[i]
            nc2 = n_sub_real

    _cl_state["labels"] = lbl
    _cl_state["n_clusters"] = nc1 + nc2

    # Write back to PS.motifs
    keys = _cl_state["motif_keys"]
    key_to_lbl = dict(zip(keys, lbl))
    for m in PS.motifs:
        if m.motif_key in key_to_lbl:
            m.cluster = int(key_to_lbl[m.motif_key])

    remaining = int((lbl == -1).sum())
    out_clust.clear_output()
    with out_clust:
        print(f"Pass 1: {nc1} clusters, {nn} noise  [mcs={mcs}, ms={ms}, {method}]")
        if do_pass2 and nc2:
            print(f"Pass 2: {nc2} sub-clusters from {nn} noise points")
        print(f"Total: {nc1+nc2} clusters, {remaining} unclustered")
        for c in sorted(set(lbl)):
            tag = "noise" if c == -1 else f"C{c:2d}"
            n = int((lbl == c).sum())
            bar = "█" * min(40, n)
            print(f"  {tag}: {n:3d}  {bar}")





# ── Persistence ───────────────────────────────────────────────────────────────
# Cluster ids are saved on their own, not inside label records: a motif does
# not need a label to keep its cluster, and most never get one.

btn_save_clusters = widgets.Button(description="Save Clusters", button_style="success",
    layout=widgets.Layout(width="150px"),
    tooltip="Write clusters.json — survives a kernel restart, unlike in-memory state")
btn_load_embed = widgets.Button(description="Load Cached", button_style="",
    layout=widgets.Layout(width="140px"),
    tooltip="Reuse the cached embedding matrix instead of recomputing CLIP")
out_cl_save = widgets.Output()


def _cluster_params():
    return {"preprocess": w_prep_mode.value, "min_cluster_size": w_mcs.value,
            "min_samples": w_ms.value, "selection_method": w_method.value,
            "pass2": w_pass2.value, "n_subclusters": w_nsub.value,
            "clip_model": CLIP_MODEL}


def _save_clusters(_=None):
    out_cl_save.clear_output()
    with out_cl_save:
        n = sum(1 for m in PS.motifs if m.included and m.cluster >= 0)
        if not n:
            print("Nothing to save — compute embeddings and cluster first")
            return
        PS.save_clusters(CLUSTERS_PATH, params=_cluster_params())
        print(f"Saved {n} cluster assignments -> {CLUSTERS_PATH.name}")
        if _cl_state["embeddings"] is not None:
            PS.save_embeddings(EMBED_CACHE, EMBED_KEYS, _cl_state["motif_keys"],
                              embeddings=_cl_state["embeddings"])
            print(f"Cached {len(_cl_state['motif_keys'])} embeddings "
                  f"-> {EMBED_CACHE.name} (skip recompute next session)")


def _load_cached_embeddings(_=None):
    out_embed.clear_output()
    with out_embed:
        keys = PS.load_embeddings(EMBED_CACHE, EMBED_KEYS)
        if not keys:
            print("No usable cache — click Compute Embeddings")
            return
        _cl_state["motif_keys"] = keys
        _cl_state["embeddings"] = PS.embeddings
        print(f"Loaded {len(keys)} cached embeddings from {EMBED_CACHE.name}")
        print("Adjust the sliders below to re-cluster, or Save Clusters to keep "
              "the assignments already restored from disk.")


# ── Wire up ───────────────────────────────────────────────────────────────────
btn_embed.on_click(_compute_embeddings)
btn_save_clusters.on_click(_save_clusters)
btn_load_embed.on_click(_load_cached_embeddings)
for _w in (w_mcs, w_ms, w_method, w_pass2, w_nsub):
    _w.observe(_run_clustering, names="value")


# ── Layout ────────────────────────────────────────────────────────────────────
display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 2: Cluster</h3>"),
    widgets.HTML("<div style='font-size:12px;color:#999;margin-bottom:4px'>"
        "Select preprocessing mode, compute embeddings, then tune clustering. "
        "<b>Save Clusters</b> writes them to clusters.json and caches the "
        "embeddings — without it the clustering is lost when the kernel stops. "
        "View scatter map in Stage 3 (Gallery).</div>"),
    w_prep_mode, widgets.HBox([btn_embed, btn_load_embed]), out_embed,
    widgets.HTML("<b style='font-size:13px;margin-top:8px'>HDBSCAN clustering</b>"),
    w_mcs, w_ms, w_method,
    widgets.HTML("<b>Pass 2 — sub-cluster noise</b>"),
    w_pass2, w_nsub,
    out_clust,
    widgets.HTML("<hr style='border-color:#444;margin:8px 0 4px'>"),
    btn_save_clusters, out_cl_save,
)\
"""))


# ══════════════════════════════════════════════════════════════════════════════
# Cell 4: Gallery — Browse + Group
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-4", """\
## ── Stage 3: Gallery ────────────────────────────────────────────────────────
#
# Browse motifs grouped by cluster, panel, or label.
# Click a thumbnail to select it as active for the Label stage.

import io as _gio

THUMB_PX = 72

_gal_state = {"cursor": 0, "selected": set()}
_gal_cards: dict = {}
_gal_dots: dict = {}
_gal_chks: dict = {}
# Callbacks that fire when gallery selection changes — label cell registers into this
_gal_on_select_cbs: list = []

out_gallery = widgets.Output()
out_context = widgets.Output()
out_nn_strip = widgets.Output()
out_gal_scatter = widgets.Output()

w_group_by = widgets.ToggleButtons(
    options=["cluster", "panel", "label", "all clusters", "all panels"], value="cluster",
    description="Group by:", style={"description_width": "80px", "button_width": "90px"})


def _gal_png(img):
    buf = _gio.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()


def _gal_thumb(motif, size=THUMB_PX):
    # In-memory crop thumbnail
    try:
        crop = PS.crop(motif)
        crop.thumbnail((size, size), Image.LANCZOS)
        sq = Image.new("RGB", (size, size), (50, 50, 50))
        sq.paste(crop, ((size - crop.width)//2, (size - crop.height)//2))
        return _gal_png(sq)
    except Exception:
        return _gal_png(Image.new("RGB", (size, size), (80, 80, 80)))


def _gal_dot(motif):
    c = "#44cc44" if motif.label else "#666"
    return f'<div style="width:8px;height:8px;border-radius:50%;background:{c};margin:2px auto"></div>'


def _gal_refresh_context():
    out_context.clear_output(wait=True)
    included = PS.included_motifs()
    if not included: return
    idx = _gal_state["cursor"]
    if idx >= len(included): idx = 0
    m = included[idx]
    with out_context:
        # Show panel with highlighted bbox + zoomed crop
        panel_img = PS.panel_image(m.panel_stem)
        from PIL import ImageDraw
        pdraw = panel_img.copy()
        drw = ImageDraw.Draw(pdraw)
        b = m.bbox
        drw.rectangle([b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]],
                      outline=(0, 255, 64), width=3)
        # Zoomed crop
        pad = 30
        iw, ih = panel_img.size
        zoom = panel_img.crop((max(0, b["x"]-pad), max(0, b["y"]-pad),
                               min(iw, b["x"]+b["w"]+pad), min(ih, b["y"]+b["h"]+pad)))
        zoom.thumbnail((220, 420), Image.LANCZOS)
        pdraw.thumbnail((560, 700), Image.LANCZOS)
        row = []
        row.append(widgets.VBox([
            widgets.HTML("<b style='color:#ccc'>Crop</b>"),
            widgets.Image(value=_gal_png(zoom), format="png",
                          layout=widgets.Layout(max_width="220px")),
        ]))
        row.append(widgets.VBox([
            widgets.HTML("<b style='color:#ccc'>Panel context</b>"),
            widgets.Image(value=_gal_png(pdraw), format="png",
                          layout=widgets.Layout(max_width="560px")),
        ]))
        display(widgets.HBox(row, layout=widgets.Layout(gap="16px")))
        # Info line
        info = (f"<b>{m.panel_stem}</b> idx={m.index} {m.scale} "
                f"cluster={m.cluster} source={m.source}")
        if m.label:
            info += f" label=<b>{m.label}</b>"
        display(widgets.HTML(f"<div style='color:#aaa;font-size:12px;margin-top:4px'>{info}</div>"))


def _gal_refresh_nn():
    out_nn_strip.clear_output(wait=True)
    included = PS.included_motifs()
    if not included or PS.sim_matrix is None: return
    idx = _gal_state["cursor"]
    if idx >= len(included): return
    sim_row = PS.sim_matrix[idx].copy()
    sim_row[idx] = -1
    top_k = np.argsort(-sim_row)[:6]
    with out_nn_strip:
        ws = []
        for ni in top_k:
            if ni >= len(included): continue
            nm = included[ni]
            tb = _gal_thumb(nm)
            sim_v = float(sim_row[ni])
            lbl_txt = nm.label or ""
            ws.append(widgets.VBox([
                widgets.Image(value=tb, format="png",
                              layout=widgets.Layout(width=f"{THUMB_PX}px")),
                widgets.HTML(f"<div style='font-size:9px;color:#888;text-align:center'>"
                             f"{sim_v:.2f}<br>{lbl_txt}</div>"),
            ], layout=widgets.Layout(width=f"{THUMB_PX+8}px", margin="3px")))
        if ws:
            display(widgets.HTML("<b style='color:#888;font-size:11px'>Nearest neighbours:</b>"))
            display(widgets.HBox(ws))


def _update_move_count():
    n = len(_gal_state["selected"])
    btn_move.description = f"Move {n} to cluster" if n else "Move to cluster"
    btn_move.disabled = (n == 0)

def _build_gallery(_=None):
    _gal_cards.clear(); _gal_dots.clear(); _gal_chks.clear()
    _gal_state["selected"].clear()
    _update_move_count()
    w_move_target.options = _cluster_options()
    included = PS.included_motifs()
    if not included:
        out_gallery.clear_output(wait=True)
        with out_gallery: display(widgets.HTML("<i>No included motifs</i>"))
        return

    mode = w_group_by.value
    groups = {}
    for i, m in enumerate(included):
        if mode in ("cluster", "all clusters"):
            key = f"C{m.cluster}" if m.cluster >= 0 else "Noise"
        elif mode in ("panel", "all panels"):
            key = m.panel_stem
        elif mode == "label":
            key = m.label or "(unlabeled)"
        else:
            key = m.label or "(unlabeled)"
        groups.setdefault(key, []).append((i, m))

    tab_kids, tab_titles = [], []
    for gname in sorted(groups.keys()):
        items = groups[gname]
        cards = []
        for gi, m in items:
            tb = _gal_thumb(m)
            imgw = widgets.Image(value=tb, format="png",
                layout=widgets.Layout(width=f"{THUMB_PX}px", height=f"{THUMB_PX}px"))
            chk = widgets.Checkbox(value=False, indent=False,
                layout=widgets.Layout(width="18px", height="18px"))
            def _on_chk(change, idx=gi):
                if change["new"]: _gal_state["selected"].add(idx)
                else: _gal_state["selected"].discard(idx)
                _update_move_count()
            chk.observe(_on_chk, names="value")
            dot = widgets.HTML(_gal_dot(m), layout=widgets.Layout(height="12px"))
            sel_btn = widgets.Button(description=str(gi),
                layout=widgets.Layout(width=f"{THUMB_PX}px", height="18px", padding="0"))
            def _on_sel(b, idx=gi):
                _gal_state["cursor"] = idx
                for ci, card in _gal_cards.items():
                    card.layout.border = "2px solid #44aaff" if ci == idx else "2px solid transparent"
                _gal_refresh_context()
                _gal_refresh_nn()
                for cb in _gal_on_select_cbs:
                    try: cb()
                    except Exception: pass
            sel_btn.on_click(_on_sel)
            _gal_chks[gi] = chk
            card = widgets.VBox([imgw, chk, dot, sel_btn],
                layout=widgets.Layout(width=f"{THUMB_PX+6}px", margin="3px",
                    padding="1px", border="2px solid transparent"))
            _gal_cards[gi] = card
            _gal_dots[gi] = dot
            cards.append(card)
        grid = widgets.HBox(cards,
            layout=widgets.Layout(flex_flow="row wrap", max_height="280px",
                overflow_y="auto", overflow_x="auto"))
        tab_kids.append(grid)
        tab_titles.append(f"{gname} ({len(items)})")

    if mode in ("all clusters", "all panels"):
        # Vertical stack: one row per group, all visible at once
        rows_vbox = []
        for gname in sorted(groups.keys()):
            items = groups[gname]
            cards_in_row = []
            for gi, m in items:
                tb = _gal_thumb(m)
                imgw = widgets.Image(value=tb, format="png",
                    layout=widgets.Layout(width=f"{THUMB_PX}px", height=f"{THUMB_PX}px"))
                chk = widgets.Checkbox(value=False, indent=False,
                    layout=widgets.Layout(width="18px", height="18px"))
                def _on_chk_r(change, idx=gi):
                    if change["new"]: _gal_state["selected"].add(idx)
                    else: _gal_state["selected"].discard(idx)
                    _update_move_count()
                chk.observe(_on_chk_r, names="value")
                dot = widgets.HTML(_gal_dot(m), layout=widgets.Layout(height="12px"))
                sel_btn = widgets.Button(description=str(gi),
                    layout=widgets.Layout(width=f"{THUMB_PX}px", height="18px", padding="0"))
                def _on_sel_r(b, idx=gi):
                    _gal_state["cursor"] = idx
                    for ci, card in _gal_cards.items():
                        card.layout.border = "2px solid #44aaff" if ci == idx else "2px solid transparent"
                    _gal_refresh_context()
                    _gal_refresh_nn()
                    for cb in _gal_on_select_cbs:
                        try: cb()
                        except Exception: pass
                sel_btn.on_click(_on_sel_r)
                _gal_chks[gi] = chk
                card = widgets.VBox([imgw, chk, dot, sel_btn],
                    layout=widgets.Layout(width=f"{THUMB_PX+6}px", margin="2px",
                        padding="1px", border="2px solid transparent"))
                _gal_cards[gi] = card
                _gal_dots[gi] = dot
                cards_in_row.append(card)
            row_grid = widgets.HBox(cards_in_row,
                layout=widgets.Layout(flex_flow="row wrap", overflow_x="auto"))
            row_label = widgets.HTML(
                f"<div style='font-weight:600;color:#ccc;font-size:13px;"
                f"margin:8px 0 2px 0;border-bottom:1px solid #444;"
                f"padding-bottom:2px'>{gname} ({len(items)})</div>")
            rows_vbox.append(widgets.VBox([row_label, row_grid]))

        out_gallery.clear_output(wait=True)
        with out_gallery:
            display(widgets.VBox(rows_vbox,
                layout=widgets.Layout(max_height="600px", overflow_y="auto")))
    else:
        # Tab view (existing)
        tab = widgets.Tab(children=tab_kids)
        for i, t in enumerate(tab_titles):
            tab.set_title(i, t)
        out_gallery.clear_output(wait=True)
        with out_gallery: display(tab)

    _gal_refresh_context()
    _gal_refresh_nn()


w_group_by.observe(_build_gallery, names="value")

btn_rebuild_gal = widgets.Button(description="Refresh Gallery", button_style="info",
    layout=widgets.Layout(width="150px"))
btn_rebuild_gal.on_click(_build_gallery)

# ── Move to cluster ──────────────────────────────────────────────────────────
def _cluster_options():
    included = PS.included_motifs()
    ids = sorted(set(m.cluster for m in included))
    opts = [(f"C{c}" if c >= 0 else "Noise", c) for c in ids]
    # Add "New cluster" option
    next_id = max(ids, default=-1) + 1
    if next_id < 0: next_id = 0
    opts.append((f"New (C{next_id})", next_id))
    return opts

w_move_target = widgets.Dropdown(
    options=_cluster_options(),
    description="Target:",
    style={"description_width": "55px"},
    layout=widgets.Layout(width="200px"))

btn_move = widgets.Button(description="Move to cluster", button_style="warning",
    layout=widgets.Layout(width="170px"), disabled=True)
out_move = widgets.Output()


def _on_move(_=None):
    sel = _gal_state["selected"]
    if not sel: return
    target = w_move_target.value
    included = PS.included_motifs()
    moved = 0
    for idx in sel:
        if idx < len(included):
            included[idx].cluster = target
            moved += 1
    _gal_state["selected"].clear()
    # Update dropdown options (new cluster may have been created)
    w_move_target.options = _cluster_options()
    _build_gallery()
    _gal_draw_scatter()
    out_move.clear_output()
    with out_move:
        print(f"Moved {moved} motif(s) to cluster {target}")

btn_move.on_click(_on_move)


# ── Interactive scatter map (click to select, drag to reassign cluster) ───────
_scat_fig, _scat_ax = plt.subplots(1, 1, figsize=(14, 10), dpi=100)
_scat_fig.patch.set_facecolor("#111")
_scat_ax.set_facecolor("#111")
_scat_ax.axis("off")
_scat_drag = {"active": False, "idx": -1, "start_xy": None}

_SPAL = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
_SNOISE = (0.45, 0.45, 0.45)
def _scc(l):
    return _SNOISE if l == -1 else _SPAL[l % len(_SPAL)]


def _gal_draw_scatter(_=None):
    # Draw scatter from PS state (coords + clusters)
    included = PS.included_motifs()
    coords = PS.tsne_xy
    if coords is None or len(included) == 0:
        _scat_ax.clear(); _scat_ax.axis("off")
        _scat_ax.text(0.5, 0.5, "Run Compute Embeddings in Stage 2 first",
            transform=_scat_ax.transAxes, ha="center", va="center",
            color="#888", fontsize=12)
        _scat_fig.canvas.draw_idle()
        return

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    THUMB = 32; ZOOM = 0.45; BD = 2
    half = ZOOM * THUMB / 2

    _scat_ax.clear(); _scat_ax.axis("off")

    # Load thumbnails
    for i, m in enumerate(included):
        if i >= len(coords): break
        x, y = coords[i]
        col = _scc(m.cluster)
        _scat_ax.add_patch(plt.Rectangle(
            (x - half - BD, y - half - BD), 2*half + 2*BD, 2*half + 2*BD,
            color=col, zorder=1, linewidth=0))

        crop = PS.crop(m)
        crop.thumbnail((THUMB, THUMB))
        sq = Image.new("RGB", (THUMB, THUMB), (25, 25, 25))
        sq.paste(crop, ((THUMB - crop.width)//2, (THUMB - crop.height)//2))
        ab = AnnotationBbox(OffsetImage(np.array(sq), zoom=ZOOM), (x, y),
                            frameon=False, zorder=2)
        _scat_ax.add_artist(ab)

        _scat_ax.text(x, y - half - BD - 1, f"#{m.index}",
            fontsize=3.5, color="white", ha="center", va="top", zorder=3, alpha=0.85)

    # Highlight selected
    cursor = _gal_state.get("cursor", 0)
    if cursor < len(coords):
        cx, cy = coords[cursor]
        _scat_ax.add_patch(plt.Rectangle(
            (cx - half - BD - 2, cy - half - BD - 2),
            2*half + 2*BD + 4, 2*half + 2*BD + 4,
            edgecolor="#44aaff", facecolor="none", linewidth=2.5, zorder=4))

    _scat_ax.autoscale_view()
    margin = THUMB * 1.2
    _scat_ax.set_xlim(coords[:,0].min()-margin, coords[:,0].max()+margin)
    _scat_ax.set_ylim(coords[:,1].min()-margin, coords[:,1].max()+margin)

    n_cl = len(set(m.cluster for m in included)) - (1 if any(m.cluster == -1 for m in included) else 0)
    _scat_ax.set_title(
        f"Cluster scatter — {len(included)} motifs, {n_cl} clusters  "
        f"(click to select, drag onto another cluster to reassign)",
        color="white", fontsize=9, pad=6)
    _scat_fig.canvas.draw_idle()


def _scat_find_nearest(event):
    # Find the motif nearest to the click position
    coords = PS.tsne_xy
    if coords is None or event.xdata is None: return -1
    dists = (coords[:,0] - event.xdata)**2 + (coords[:,1] - event.ydata)**2
    idx = int(np.argmin(dists))
    # Only match if reasonably close
    if dists[idx] > (50**2): return -1
    return idx


def _scat_on_press(event):
    if event.inaxes != _scat_ax or event.button != 1: return
    idx = _scat_find_nearest(event)
    if idx < 0: return
    _scat_drag["active"] = True
    _scat_drag["idx"] = idx
    _scat_drag["start_xy"] = (event.xdata, event.ydata)
    # Select this motif
    _gal_state["cursor"] = idx
    for ci, card in _gal_cards.items():
        card.layout.border = "2px solid #44aaff" if ci == idx else "2px solid transparent"
    _gal_refresh_context()
    _gal_refresh_nn()
    for cb in _gal_on_select_cbs:
        try: cb()
        except: pass
    _gal_draw_scatter()


def _scat_on_release(event):
    if not _scat_drag["active"]: return
    _scat_drag["active"] = False
    if event.inaxes != _scat_ax or event.xdata is None: return

    src_idx = _scat_drag["idx"]
    sx, sy = _scat_drag["start_xy"]
    dx = abs(event.xdata - sx)
    dy = abs(event.ydata - sy)

    # Only reassign if dragged a meaningful distance
    if dx + dy < 5: return

    # Find which motif we dropped near
    target_idx = _scat_find_nearest(event)
    if target_idx < 0 or target_idx == src_idx: return

    included = PS.included_motifs()
    if src_idx >= len(included) or target_idx >= len(included): return

    # Reassign source motif to the target motif cluster
    old_cluster = included[src_idx].cluster
    new_cluster = included[target_idx].cluster
    if old_cluster == new_cluster: return

    included[src_idx].cluster = new_cluster
    out_context.clear_output()
    with out_context:
        display(widgets.HTML(
            f"<div style='color:#ffaa22'>Moved motif #{included[src_idx].index} "
            f"from cluster {old_cluster} to {new_cluster}</div>"))
    _gal_draw_scatter()

_scat_fig.canvas.mpl_connect("button_press_event", _scat_on_press)
_scat_fig.canvas.mpl_connect("button_release_event", _scat_on_release)


btn_refresh_scatter = widgets.Button(description="Refresh Scatter", button_style="",
    layout=widgets.Layout(width="150px"))
btn_refresh_scatter.on_click(_gal_draw_scatter)


# ── Spotlight launcher (web-based scatter UI) ─────────────────────────────────
btn_spotlight = widgets.Button(description="Launch Spotlight", button_style="info",
    layout=widgets.Layout(width="170px"),
    tooltip="Open interactive web scatter in a new browser tab")
out_spotlight = widgets.Output()


def _launch_spotlight(_=None):
    included = PS.included_motifs()
    coords = PS.tsne_xy
    if not included or coords is None:
        with out_spotlight: print("Compute embeddings in Stage 2 first")
        return

    btn_spotlight.description = "Starting..."
    btn_spotlight.disabled = True
    out_spotlight.clear_output()

    try:
        import pandas as pd
        from renumics import spotlight

        # Save crops to a temp dir so Spotlight can load them as images
        import tempfile, shutil
        _spot_dir = Path(tempfile.mkdtemp(prefix="spotlight_crops_"))

        rows = []
        for i, m in enumerate(included):
            if i >= len(coords): break
            # Write crop to temp file for Spotlight
            crop_path = _spot_dir / f"{m.panel_stem}_{m.index:03d}.png"
            PS.crop(m).save(crop_path)
            rows.append({
                "image": str(crop_path),
                "panel": m.panel_stem,
                "index": m.index,
                "scale": m.scale,
                "cluster": m.cluster,
                "label": m.label or "",
                "source": m.source,
                "predicted_iou": m.predicted_iou,
                "area_ratio": m.area_ratio,
                "tsne_x": float(coords[i, 0]),
                "tsne_y": float(coords[i, 1]),
            })
            # Add embedding dimensions
            if PS.embeddings is not None and i < len(PS.embeddings):
                rows[-1]["embedding"] = PS.embeddings[i].tolist()

        df = pd.DataFrame(rows)
        with out_spotlight:
            print(f"Spotlight: {len(df)} motifs, temp crops in {_spot_dir}")
            print("Opening in browser... (close the Spotlight tab to stop the server)")

        spotlight.show(
            df,
            dtype={"image": spotlight.Image, "embedding": spotlight.Embedding},
            port=8765,
            no_browser=False,
        )
    except Exception:
        out_spotlight.clear_output()
        with out_spotlight:
            import traceback; traceback.print_exc()
    finally:
        btn_spotlight.description = "Launch Spotlight"
        btn_spotlight.disabled = False


btn_spotlight.on_click(_launch_spotlight)


display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 3: Gallery</h3>"),
    widgets.HBox([w_group_by, btn_rebuild_gal]),
    widgets.HBox([w_move_target, btn_move],
        layout=widgets.Layout(margin="4px 0")),
    out_move,
    out_gallery,
    widgets.HTML("<b style='font-size:13px;margin-top:8px'>Cluster scatter map</b>"
        "<div style='font-size:11px;color:#888'>Click a motif to select it. "
        "Drag a motif onto another cluster to reassign it.</div>"),
    widgets.HBox([btn_refresh_scatter, btn_spotlight]),
    out_spotlight,
    _scat_fig.canvas,
    widgets.HTML("<b style='font-size:13px;margin-top:8px'>Context</b>"),
    out_context,
    out_nn_strip,
)
_build_gallery()
_gal_draw_scatter()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 5: Label — Editor + LLM Suggest
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-5", """\
## ── Stage 4: Label ──────────────────────────────────────────────────────────
#
# Edit labels for the currently selected motif from the Gallery.
# LLM Suggest sends the crop + context to Claude for a pre-fill.

import os as _os
import base64 as _b64mod

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

_WL = widgets.Layout
_si = {"description_width": "110px"}
_fw = _WL(width="480px")

w_lbl_info  = widgets.HTML("<i>Select a motif in the Gallery</i>")
w_lbl_label = widgets.Text(placeholder="e.g. interlaced_knotwork",
    description="Label", layout=_fw, style=_si)
w_lbl_desc  = widgets.Textarea(placeholder="One sentence on what is visually present",
    description="Description", rows=2, layout=_fw, style=_si)
w_lbl_icon  = widgets.Textarea(placeholder="Iconographic significance",
    description="Iconography", rows=2, layout=_fw, style=_si)
w_lbl_notes = widgets.Text(placeholder="Cross-references, observations",
    description="Notes", layout=_fw, style=_si)

btn_lbl_llm  = widgets.Button(description="LLM Suggest", button_style="info",
    layout=_WL(width="140px"))
btn_lbl_save = widgets.Button(description="Save", button_style="success",
    layout=_WL(width="100px"))
btn_lbl_prev = widgets.Button(description="Prev", layout=_WL(width="80px"))
btn_lbl_next = widgets.Button(description="Next", layout=_WL(width="80px"))
btn_lbl_nxtu = widgets.Button(description="Next Unlabeled", layout=_WL(width="140px"))

out_lbl_status = widgets.Output()
out_lbl_llm    = widgets.Output()

_lbl_llm_data = [None]  # mutable ref for LLM suggestion


def _lbl_load_fields():
    included = PS.included_motifs()
    idx = _gal_state.get("cursor", 0)
    if idx >= len(included):
        w_lbl_info.value = "<i>No motif selected</i>"
        return
    m = included[idx]
    info = f"<b>{m.panel_stem}</b> idx={m.index} {m.scale} cluster={m.cluster}"
    w_lbl_info.value = info
    w_lbl_label.value = m.label or ""
    w_lbl_desc.value = m.description or ""
    w_lbl_icon.value = m.iconography or ""
    w_lbl_notes.value = m.notes or ""
    _lbl_llm_data[0] = None


def _lbl_save(_=None):
    included = PS.included_motifs()
    idx = _gal_state.get("cursor", 0)
    if idx >= len(included): return
    m = included[idx]
    ld = _lbl_llm_data[0]
    src = "human"
    if ld:
        src = "llm" if w_lbl_label.value.strip() == ld.get("label", "").strip() else "llm-edited"
    PS.save_label(m, w_lbl_label.value, w_lbl_desc.value,
                  w_lbl_icon.value, w_lbl_notes.value, source=src)
    _lbl_llm_data[0] = None
    # Update gallery dot
    if idx in _gal_dots:
        _gal_dots[idx].value = _gal_dot(m)
    out_lbl_status.clear_output()
    with out_lbl_status:
        n_labeled = sum(1 for mm in PS.motifs if mm.label)
        total = len(included)
        print(f"Saved. {n_labeled}/{total} labeled ({n_labeled/total*100:.0f}%)")


def _lbl_navigate(delta):
    included = PS.included_motifs()
    if not included: return
    _gal_state["cursor"] = (_gal_state["cursor"] + delta) % len(included)
    _lbl_load_fields()
    _gal_refresh_context()
    _gal_refresh_nn()
    # Update gallery card borders
    for ci, card in _gal_cards.items():
        card.layout.border = ("2px solid #44aaff" if ci == _gal_state["cursor"]
                              else "2px solid transparent")


def _lbl_next_unlabeled(_=None):
    included = PS.included_motifs()
    start = _gal_state.get("cursor", 0)
    for off in range(1, len(included) + 1):
        idx = (start + off) % len(included)
        if not included[idx].label:
            _gal_state["cursor"] = idx
            _lbl_load_fields()
            _gal_refresh_context()
            _gal_refresh_nn()
            for ci, card in _gal_cards.items():
                card.layout.border = ("2px solid #44aaff" if ci == idx
                                      else "2px solid transparent")
            return
    out_lbl_status.clear_output()
    with out_lbl_status: print("All motifs labeled!")


def _lbl_llm_suggest(_=None):
    included = PS.included_motifs()
    idx = _gal_state.get("cursor", 0)
    if idx >= len(included): return
    m = included[idx]

    key = _os.environ.get("ANTHROPIC_API_KEY")
    if not key or not _ANTHROPIC_OK:
        with out_lbl_llm: print("ANTHROPIC_API_KEY not set or anthropic not installed")
        return

    btn_lbl_llm.description = "Thinking..."
    btn_lbl_llm.disabled = True
    out_lbl_llm.clear_output()

    try:
        client = _anthropic.Anthropic(api_key=key)
        content = []

        # Send crop image
        crop = PS.crop(m)
        crop_resized = crop.copy()
        crop_resized.thumbnail((512, 512), Image.LANCZOS)
        buf = _gio.BytesIO(); crop_resized.save(buf, "PNG")
        b64_crop = _b64mod.standard_b64encode(buf.getvalue()).decode()

        content.append({"type": "text", "text": "Image 1 — motif crop:"})
        content.append({"type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": b64_crop}})

        # Context
        info = (f"Yoruba carved door panel. Panel: {m.panel_stem}. "
                f"Scale: {m.scale}. Cluster: {m.cluster}.")
        content.append({"type": "text", "text":
            f"{info}\\n\\nDescribe this carved motif. "
            "Respond with JSON only:\\n"
            '{\\n  "label": "2-4 words snake_case",\\n'
            '  "description": "one sentence on what is visually present",\\n'
            '  "iconography": "cultural significance in Yoruba art"\\n}'
        })

        resp = client.messages.create(
            model="claude-opus-5",
            # Thinking is on by default on this model and shares the max_tokens
            # budget with the reply, so 400 would truncate the JSON mid-object.
            # A label suggestion is a small task — low effort keeps it quick.
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}],
        )
        # content[0] is a thinking block, not text — collect the text blocks.
        raw = "".join(b.text for b in resp.content if b.type == "text")
        # Parse JSON from response
        import re as _re_lbl
        jm = _re_lbl.search(r'\\{[^}]+\\}', raw, _re_lbl.DOTALL)
        if jm:
            data = json.loads(jm.group())
            _lbl_llm_data[0] = data
            w_lbl_label.value = data.get("label", "")
            w_lbl_desc.value = data.get("description", "")
            w_lbl_icon.value = data.get("iconography", "")
            out_lbl_llm.clear_output()
            with out_lbl_llm:
                print(f"LLM suggested: {data.get('label', '?')}")
                print(f"Edit and Save, or Next to skip.")
        else:
            out_lbl_llm.clear_output()
            with out_lbl_llm:
                print(f"Could not parse LLM response:\\n{raw[:300]}")
    except Exception:
        out_lbl_llm.clear_output()
        with out_lbl_llm:
            import traceback; traceback.print_exc()
    finally:
        btn_lbl_llm.description = "LLM Suggest"
        btn_lbl_llm.disabled = False


btn_lbl_save.on_click(_lbl_save)
btn_lbl_prev.on_click(lambda b: _lbl_navigate(-1))
btn_lbl_next.on_click(lambda b: _lbl_navigate(+1))
btn_lbl_nxtu.on_click(_lbl_next_unlabeled)
btn_lbl_llm.on_click(_lbl_llm_suggest)

display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 4: Label</h3>"),
    w_lbl_info,
    w_lbl_label, w_lbl_desc, w_lbl_icon, w_lbl_notes,
    widgets.HBox([btn_lbl_llm, btn_lbl_save, btn_lbl_prev,
                  btn_lbl_next, btn_lbl_nxtu]),
    out_lbl_llm, out_lbl_status,
)
# Register with gallery so clicking a card also updates the label editor
_gal_on_select_cbs.append(_lbl_load_fields)
_lbl_load_fields()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 6: Interpret — join clusters, descriptions, and positions
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-6", """\
## ── Stage 5: Interpret ──────────────────────────────────────────────────────
#
# The join: motif descriptions + embedding-derived cluster families + relative
# positions, run through Claude in three widening passes.
#
#   Layout          — deterministic geometry only, no API call. Run it first to
#                     check the registers and symmetry the model will be told about.
#   Cluster Brief   — what the selected motif's family is, across every panel.
#   Panel Reading   — this panel read register by register, using the briefs.
#   Corpus Synthesis— all briefs + all readings joined into one essay.
#
# Each pass feeds the next, so run them in order. Results persist to
# analysis/interpretation/ and reload on re-run.

import os as _os7

from IPython.display import Markdown
from panel_art.interpret import (
    Corpus, InterpretationStore, Interpreter,
    build_cluster_prompt, build_panel_prompt,
    cluster_context_lines, compute_cluster_stats, corpus_scale,
    render_clusters_markdown, render_panel_markdown,
)
from panel_art.layout import render_layout_text

INTERPRET_DIR = _ANA / "interpretation"
_store = InterpretationStore(INTERPRET_DIR)
_store.ensure_dirs()

_interp_state = {
    "corpus": None,
    "stats": {},
    "briefs": _store.load_clusters(),
    "readings": _store.load_panels(),
}


def _interp_corpus(rebuild=False):
    \"\"\"Build a Corpus over the approved motifs, reusing Stage 2's embeddings.\"\"\"
    if _interp_state["corpus"] is None or rebuild:
        embeddings = _cl_state.get("embeddings") if "_cl_state" in globals() else None
        keys = _cl_state.get("motif_keys") if "_cl_state" in globals() else None
        corpus = Corpus.from_pipeline_state(PS, embeddings, keys)
        _interp_state["corpus"] = corpus
        _interp_state["stats"] = compute_cluster_stats(corpus)
    return _interp_state["corpus"]


def _interp_client():
    if not _os7.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set — Layout still works without it")
    return Interpreter(model=w_interp_model.value, effort=w_interp_effort.value)


def _selected_motif(corpus):
    included = PS.included_motifs()
    idx = _gal_state.get("cursor", 0)
    if not included or idx >= len(included):
        return None
    return corpus.by_key(included[idx].motif_key)


def _current_stem(corpus):
    stem = _seg_state.get("stem")
    if stem:
        return stem
    motif = _selected_motif(corpus)
    return motif.panel_stem if motif else None


# ── Widgets ───────────────────────────────────────────────────────────────────
w_interp_model = widgets.Dropdown(
    options=["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
    value="claude-opus-5", description="Model:",
    layout=widgets.Layout(width="260px"))
w_interp_effort = widgets.Dropdown(
    options=["low", "medium", "high", "xhigh", "max"], value="high",
    description="Effort:", layout=widgets.Layout(width="200px"))

btn_interp_layout = widgets.Button(description="Layout", button_style="",
    layout=widgets.Layout(width="130px"),
    tooltip="Registers, symmetry, nesting — computed, no API call")
btn_interp_cluster = widgets.Button(description="Cluster Brief", button_style="info",
    layout=widgets.Layout(width="150px"),
    tooltip="Characterise the selected motif's family across the whole corpus")
btn_interp_panel = widgets.Button(description="Panel Reading", button_style="info",
    layout=widgets.Layout(width="150px"),
    tooltip="Read the current panel register by register, using the cluster briefs")
btn_interp_corpus = widgets.Button(description="Corpus Synthesis", button_style="success",
    layout=widgets.Layout(width="170px"),
    tooltip="Join every brief and reading into a single interpretation")

out_interp = widgets.Output()
out_interp_status = widgets.Output()


def _interp_status():
    out_interp_status.clear_output()
    with out_interp_status:
        scale = corpus_scale(_interp_corpus())
        print(f"{scale['panels']} panels | {scale['motifs']} motifs | "
              f"{scale['clusters']} families | {scale['labelled']} labelled")
        print(f"Briefs: {len(_interp_state['briefs'])}/{scale['clusters']}  |  "
              f"Readings: {len(_interp_state['readings'])}/{scale['panels']}  |  "
              f"Corpus essay: {'yes' if _store.corpus_path.exists() else 'no'}")
        print(f"Saved under {INTERPRET_DIR}")


def _interp_busy(button, label):
    button._label = button.description
    button.description = label
    button.disabled = True


def _interp_done(button):
    button.description = getattr(button, "_label", button.description)
    button.disabled = False


# ── Layout (no API) ───────────────────────────────────────────────────────────
def _on_layout(_=None):
    out_interp.clear_output()
    corpus = _interp_corpus(rebuild=True)
    stem = _current_stem(corpus)
    with out_interp:
        if not stem:
            print("Select a panel in Stage 1, or a motif in the gallery, first")
            return
        layout = corpus.layout_for(stem)
        _store.save_layout(stem, layout)
        print(render_layout_text(layout))
        print(f"\\nSaved to {_store.layouts_dir / (stem + '.json')}")
    _interp_status()


# ── Pass 1: cluster brief ─────────────────────────────────────────────────────
def _on_cluster(_=None):
    out_interp.clear_output()
    corpus = _interp_corpus(rebuild=True)
    motif = _selected_motif(corpus)
    if motif is None or motif.cluster < 0:
        with out_interp:
            print("Select a clustered motif in the gallery first "
                  "(unclustered motifs have no family to brief)")
        return

    stats = _interp_state["stats"].get(motif.cluster)
    _interp_busy(btn_interp_cluster, "Thinking...")
    try:
        with out_interp:
            print(f"Briefing cluster {motif.cluster} ({stats.size} motifs across "
                  f"{stats.panel_spread} panel(s))...")
        brief = _interp_client().cluster_brief(corpus, stats)
        _interp_state["briefs"][str(motif.cluster)] = brief
        _store.save_clusters(_interp_state["briefs"])
        (INTERPRET_DIR / "clusters.md").write_text(
            render_clusters_markdown(_interp_state["briefs"]), encoding="utf-8")

        out_interp.clear_output()
        with out_interp:
            display(HTML(f"<h4>Cluster {motif.cluster}: {brief.get('name', '?')}</h4>"))
            for key in ("visual_definition", "variation", "distribution_note",
                        "iconographic_reading", "relation_to_neighbours"):
                if brief.get(key):
                    display(HTML(f"<b>{key.replace('_', ' ').title()}.</b> {brief[key]}"))
            display(HTML(f"<i>Confidence: {brief.get('confidence', '?')}</i>"))
            for q in brief.get("open_questions", []):
                print(f"  ? {q}")
    except Exception:
        out_interp.clear_output()
        with out_interp:
            import traceback; traceback.print_exc()
    finally:
        _interp_done(btn_interp_cluster)
        _interp_status()


# ── Pass 2: panel reading ─────────────────────────────────────────────────────
def _on_panel(_=None):
    out_interp.clear_output()
    corpus = _interp_corpus(rebuild=True)
    stem = _current_stem(corpus)
    if not stem:
        with out_interp:
            print("Select a panel in Stage 1 first")
        return

    _interp_busy(btn_interp_panel, "Thinking...")
    try:
        with out_interp:
            n_briefs = len(_interp_state["briefs"])
            print(f"Reading {stem} with {n_briefs} cluster brief(s) as context...")
            if not n_briefs:
                print("  (no briefs yet — run Cluster Brief first for corpus-aware readings)")
        reading = _interp_client().panel_reading(
            corpus, stem, _interp_state["briefs"], _interp_state["stats"])
        _interp_state["readings"][stem] = reading
        _store.save_panel(stem, reading)

        out_interp.clear_output()
        with out_interp:
            display(Markdown(render_panel_markdown(reading)))
    except Exception:
        out_interp.clear_output()
        with out_interp:
            import traceback; traceback.print_exc()
    finally:
        _interp_done(btn_interp_panel)
        _interp_status()


# ── Pass 3: corpus synthesis ──────────────────────────────────────────────────
def _on_corpus(_=None):
    out_interp.clear_output()
    corpus = _interp_corpus(rebuild=True)
    briefs, readings = _interp_state["briefs"], _interp_state["readings"]
    if not briefs and not readings:
        with out_interp:
            print("Nothing to synthesise yet — run Cluster Brief and Panel Reading first")
        return

    _interp_busy(btn_interp_corpus, "Thinking...")
    try:
        with out_interp:
            print(f"Synthesising {len(briefs)} families and {len(readings)} panel "
                  "readings into one interpretation... (this is the slowest pass)")
        markdown = _interp_client().corpus_synthesis(
            briefs, readings, corpus_scale(corpus))
        _store.save_corpus(markdown)

        out_interp.clear_output()
        with out_interp:
            display(Markdown(markdown))
    except Exception:
        out_interp.clear_output()
        with out_interp:
            import traceback; traceback.print_exc()
    finally:
        _interp_done(btn_interp_corpus)
        _interp_status()


btn_interp_layout.on_click(_on_layout)
btn_interp_cluster.on_click(_on_cluster)
btn_interp_panel.on_click(_on_panel)
btn_interp_corpus.on_click(_on_corpus)

display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 5: Interpret</h3>"),
    widgets.HTML("<div style='font-size:12px;color:#999;margin-bottom:6px'>"
        "Three widening passes, each feeding the next: "
        "<b>Cluster Brief</b> characterises the selected motif's family across every "
        "panel; <b>Panel Reading</b> reads the current panel register by register "
        "using those briefs; <b>Corpus Synthesis</b> joins everything into one "
        "interpretation. <b>Layout</b> shows the geometry the model is given — "
        "no API call.</div>"),
    widgets.HBox([w_interp_model, w_interp_effort]),
    widgets.HBox([btn_interp_layout, btn_interp_cluster,
                  btn_interp_panel, btn_interp_corpus]),
    out_interp_status, out_interp,
)
_interp_status()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Cell 7: Export + Progress
# ══════════════════════════════════════════════════════════════════════════════
cells.append(code("mp-7", """\
## ── Stage 6: Export + Progress ──────────────────────────────────────────────

btn_save_state = widgets.Button(description="Save All State",
    button_style="success", layout=widgets.Layout(width="160px"),
    tooltip="Save approved bboxes + labels + cluster assignments to disk")
btn_export_crops = widgets.Button(description="Export Crop Images",
    button_style="", layout=widgets.Layout(width="170px"),
    tooltip="Write cropped PNG images to disk (optional — bbox metadata is saved separately)")
out_export = widgets.Output()
out_progress = widgets.Output()


def _on_save_state(_=None):
    # Save everything: approved bboxes + labels
    out_export.clear_output()
    with out_export:
        saved_panels = 0
        for stem in PS.panels:
            motifs = PS.motifs_for_panel(stem)
            if any(m.included for m in motifs):
                PS.save_approved(stem)
                saved_panels += 1
        # Only what was edited in this session, merged into whatever is on
        # disk. The label file is shared with scripts/label_motifs.py and with
        # other labellers, so writing every in-memory label would regress
        # anything improved on disk while this kernel held a stale copy.
        path, n_lbl = PS.save_all_labels(LABELS_PATH, only_changed=True)
        print(f"Saved: {saved_panels} panel _approved.json files "
              f"(bboxes + source + timestamps)")
        print(f"Saved: {n_lbl} label(s) edited this session, merged into {path.name}"
              if n_lbl else f"Labels: nothing edited this session ({path.name} untouched)")

        # Clusters go to their own file. Previously they rode along inside
        # label records, so every unlabelled motif lost its assignment here.
        n_cl = sum(1 for m in PS.motifs if m.included and m.cluster >= 0)
        if n_cl:
            cpath = PS.save_clusters(CLUSTERS_PATH)
            print(f"Saved: {n_cl} cluster assignments to {cpath.name}")
        else:
            print("No cluster assignments to save (run Stage 2 first)")


def _on_export_crops(_=None):
    out_dir = _ANA / "motifs"
    out_export.clear_output()
    with out_export:
        n = PS.export_crops(out_dir)
        print(f"Exported {n} crop PNGs to {out_dir}")
        print(f"(bbox metadata is already saved via Save All State)")


def _show_progress(_=None):
    included = PS.included_motifs()
    n_total = len(included)
    n_labeled = sum(1 for m in included if m.label)
    n_clustered = sum(1 for m in included if m.cluster >= 0)
    n_panels = len(set(m.panel_stem for m in included))
    n_manual = sum(1 for m in included if m.source == "manual")
    n_sam = sum(1 for m in included if m.source in ("sam_auto", "sam_prompted"))

    out_progress.clear_output(wait=True)
    with out_progress:
        print(f"Pipeline Progress")
        print(f"{'='*40}")
        print(f"Panels:      {n_panels}")
        print(f"Motifs:      {n_total}")
        print(f"  manual:    {n_manual}")
        print(f"  SAM:       {n_sam}")
        print(f"Clustered:   {n_clustered}/{n_total} ({n_clustered/n_total*100:.0f}%)" if n_total else "")
        print(f"Labeled:     {n_labeled}/{n_total} ({n_labeled/n_total*100:.0f}%)" if n_total else "")

        if n_labeled > 0:
            # Label frequency
            from collections import Counter
            lbl_counts = Counter(m.label for m in included if m.label)
            print(f"\\nTop labels:")
            for lbl, cnt in lbl_counts.most_common(15):
                bar = "█" * min(30, cnt)
                print(f"  {lbl:30s} {cnt:3d}  {bar}")


btn_save_state.on_click(_on_save_state)
btn_export_crops.on_click(_on_export_crops)

display(
    widgets.HTML("<h3 style='margin:4px 0'>Stage 6: Export + Progress</h3>"),
    widgets.HTML("<div style='font-size:12px;color:#999;margin-bottom:4px'>"
        "<b>Save All State</b>: writes _approved.json (bbox metadata) + "
        "motif_labels.json (labels + clusters) to disk.<br>"
        "<b>Export Crop Images</b>: optional — writes actual PNG sub-images "
        "(bbox metadata is saved separately).</div>"),
    widgets.HBox([btn_save_state, btn_export_crops]),
    out_export,
    widgets.HTML("<hr style='border-color:#444'>"),
    out_progress,
)
_show_progress()\
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


def merge_with_existing(nb: dict, existing_path: Path) -> dict:
    """Fold the generated notebook into whatever is already on disk.

    The checked-in notebook is not purely generated. It is also an executed
    record of a run, it carries the collapse state its author set in
    JupyterLab, and it has been hand-extended in Colab with setup cells this
    script knows nothing about (Drive mount, clone, pip install). Regenerating
    must not silently discard any of that, so this merges rather than
    overwrites:

    - **Foreign cells** — anything on disk this script does not generate — are
      kept verbatim, in place. They are somebody's hand-authored work.
    - **Cell metadata** (``jupyter.source_hidden``, Colab's ``@title`` state)
      is a display preference about the cell, not about the code in it, so it
      survives an edit to the source.
    - **Outputs** are only carried over when the source is unchanged; stale
      output shown under new code is worse than no output.
    - **Notebook-level metadata** (``colab``, ``widgets``) is preserved, with
      the generated kernelspec filling in anything absent.

    A generated cell whose on-disk source has *diverged* from what this script
    produces is also left alone, and reported. The notebook has been adapted
    for Colab in place (environment resolution in Stage 1, a matplotlib backend
    fallback in Stage 2), so blindly overwriting would undo that. Pass
    ``--force`` once you have folded those edits back into this file and want
    the generated source to win.
    """
    generated = nb["cells"]
    if not existing_path.exists():
        return nb
    try:
        previous = json.loads(existing_path.read_text())
    except (json.JSONDecodeError, OSError):
        return nb

    force = "--force" in sys.argv
    gen_by_id = {c["id"]: c for c in generated}
    merged: list[dict] = []
    placed: set[str] = set()
    diverged: list[str] = []
    kept_outputs = kept_metadata = foreign = 0

    for old in previous.get("cells", []):
        cid = old.get("id")
        cell = gen_by_id.get(cid)
        if cell is None:                      # hand-authored — not ours to touch
            merged.append(old)
            foreign += 1
            continue
        if old.get("source") != cell["source"] and not force:
            # Diverged: either this script changed, or somebody edited the
            # notebook directly. Keeping the on-disk version is the safe
            # default — a lost hand-edit is unrecoverable, a skipped
            # regeneration is not.
            diverged.append(cid)
            merged.append(old)
            placed.add(cid)
            continue
        if old.get("metadata"):
            cell["metadata"] = old["metadata"]
            kept_metadata += 1
        if old.get("outputs") and old.get("source") == cell["source"]:
            cell["outputs"] = old["outputs"]
            cell["execution_count"] = old.get("execution_count")
            kept_outputs += 1
        merged.append(cell)
        placed.add(cid)

    # Generated cells the previous notebook did not have (a newly added stage).
    merged.extend(c for c in generated if c["id"] not in placed)

    # Preserve the existing key order; only add what is missing. Reordering
    # metadata that Colab wrote produces a diff carrying no content change.
    metadata = dict(previous.get("metadata", {}))
    for key, value in nb["metadata"].items():
        metadata.setdefault(key, value)
    nb = {**nb, "cells": merged, "metadata": metadata}

    print(f"  merged with existing: kept {foreign} hand-authored cell(s), "
          f"outputs for {kept_outputs}, metadata for {kept_metadata}")
    if diverged:
        print(f"  SKIPPED {len(diverged)} generated cell(s) whose on-disk source has "
              f"diverged: {', '.join(diverged)}")
        print("    The notebook was adapted in place (Colab setup, backend fallback).")
        print("    Fold those edits into this script, or re-run with --force to")
        print("    overwrite them with the generated source.")
    return nb


nb = merge_with_existing(nb, out)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Written: {out}  ({out.stat().st_size // 1024} KB, {len(nb['cells'])} cells)")
