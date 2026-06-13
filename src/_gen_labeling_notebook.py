#!/usr/bin/env python3
"""Generate motif_labeling.ipynb — run once with: uv run python _gen_labeling_notebook.py"""
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

# ── Cell 0: header ─────────────────────────────────────────────────────────────
cells.append(md("ml-0", """\
# Frobenius Motif Labeling

Interactive labeling, clustering review, and LLM-assisted description of extracted motifs.

**Workflow:**
1. Run all cells (Kernel → Restart & Run All)
2. Browse clusters in the gallery tabs — click a thumbnail number to select it as active
3. Check the checkbox on multiple thumbnails to view them in context together
4. Use **✨ LLM Suggest** to pre-fill a label suggestion from Claude, then edit and **Save**
5. Navigate with **← Prev / Next → / Next Unlabeled →**

**Requirements:** `motif_embeddings_*.npy` and `motif_paths_*.txt` must exist (run `motif_similarity.ipynb` first).
Set `ANTHROPIC_API_KEY` in your environment for LLM suggestions.\
"""))

# ── Cell 1: imports + paths ────────────────────────────────────────────────────
cells.append(code("ml-1", """\
import base64
import io
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import ipywidgets as widgets
import numpy as np
from IPython.display import clear_output, display
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

try:
    import hdbscan as _hdbscan_mod
    _HDBSCAN = _hdbscan_mod.HDBSCAN
except ImportError:
    from sklearn.cluster import DBSCAN as _HDBSCAN  # fallback

try:
    import anthropic as _anthropic
    _ANTHROPIC_OK = True
except ImportError:
    _ANTHROPIC_OK = False

# ── Paths ──────────────────────────────────────────────────────────────────────
_NB    = Path(".").resolve()                             # src/python/
_REPO  = _NB.parent.parent
_ANA   = _REPO / "frobenius_artifacts/analysis"

PANELS_DIR    = _ANA / "panels"
ANNOTATED_DIR = _ANA / "annotated"
MOTIFS_DIR    = _ANA / "motifs"
MOTIFS_NORM   = _ANA / "motifs_norm"
LABELS_PATH   = _ANA / "motif_labels.json"
META_JSON     = _REPO / "src/typescript/backend/lib/data/frobenius_panel_art.json"

print(f"Repo  : {_REPO}")
print(f"Labels: {LABELS_PATH}")\
"""))

# ── Cell 2: data load ──────────────────────────────────────────────────────────
cells.append(code("ml-2", """\
# ── Auto-detect mode from most recent params sidecar, fallback to "edges" ─────
_params_files = sorted(_NB.glob("motif_params_*.json"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
PREPROCESS_MODE = "edges"  # default
if _params_files:
    try:
        PREPROCESS_MODE = json.loads(_params_files[0].read_text())["PREPROCESS_MODE"]
        print(f"Auto-detected PREPROCESS_MODE={PREPROCESS_MODE!r} from {_params_files[0].name}")
    except Exception:
        pass

_emb_file   = _NB / f"motif_embeddings_{PREPROCESS_MODE}.npy"
_paths_file = _NB / f"motif_paths_{PREPROCESS_MODE}.txt"

if not _emb_file.exists():
    raise FileNotFoundError(
        f"{_emb_file.name} not found — run motif_similarity.ipynb first."
    )

embeddings = np.load(_emb_file).astype(np.float32)
_raw_paths = _paths_file.read_text().splitlines()

# ── Stale-input check (Phase 0b) ──────────────────────────────────────────────
# Warn if motifs_norm/ has files newer than the saved embeddings.
# This means motifs changed after the last motif_similarity run — re-run it first.
_npy_mtime   = _emb_file.stat().st_mtime
_norm_files  = list(MOTIFS_NORM.rglob("*.png")) if MOTIFS_NORM.exists() else []
if _norm_files:
    _newest_norm = max(f.stat().st_mtime for f in _norm_files)
    if _newest_norm > _npy_mtime:
        import datetime as _dt
        _delta = _dt.datetime.fromtimestamp(_newest_norm) - _dt.datetime.fromtimestamp(_npy_mtime)
        print(f"  ⚠ motifs_norm/ has files {_delta} newer than embeddings.")
        print(f"    Re-run motif_similarity.ipynb before labeling for accurate results.")

# ── Artifact metadata ──────────────────────────────────────────────────────────
_meta_by_reg: dict = {}
try:
    _meta_recs = json.loads(META_JSON.read_text()).get("records", [])
    for _r in _meta_recs:
        _reg = _r.get("registration_number", "")
        if _reg:
            _meta_by_reg[_reg] = _r
except Exception as _e:
    print(f"  Note: artifact metadata not loaded ({_e})")

# ── BBox cache ─────────────────────────────────────────────────────────────────
_bbox_cache: dict = {}

def _panel_bboxes(stem: str) -> dict:
    if stem not in _bbox_cache:
        for sfx in ("_approved.json", "_detections.json"):
            p = ANNOTATED_DIR / f"{stem}{sfx}"
            if p.exists():
                data = json.loads(p.read_text())
                _bbox_cache[stem] = {d["index"]: d["bbox"] for d in data}
                break
        else:
            _bbox_cache[stem] = {}
    return _bbox_cache[stem]

# ── Parse paths ────────────────────────────────────────────────────────────────
_PATH_RE = re.compile(r"motifs(?:_norm)?/([^/]+)/(\\d+)_([^_]+)_iou([\\d.]+)\\.png")

def _abs(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (_NB / raw).resolve()

def _find_meta(stem: str) -> dict | None:
    for reg, m in _meta_by_reg.items():
        if reg.replace(" ", "_").replace("/", "-") in stem or reg in stem:
            return m
    return None

records: list[dict] = []
for _raw in _raw_paths:
    _m = _PATH_RE.search(_raw)
    if not _m:
        continue
    _stem, _idx_s, _scale, _iou_s = _m.groups()
    _na  = _abs(_raw)
    _oa  = Path(str(_na).replace("motifs_norm", "motifs"))
    _ppng = PANELS_DIR / f"{_stem}.png"
    if not _ppng.exists():
        _ppng = PANELS_DIR / f"{_stem}_cropped.png"
    records.append({
        "path_norm"    : _raw,
        "path_norm_abs": _na,
        "path_orig_abs": _oa,
        "path_display" : _na if _na.exists() else _oa,
        "panel_stem"   : _stem,
        "index"        : int(_idx_s),
        "scale"        : _scale,
        "iou"          : float(_iou_s),
        "bbox"         : _panel_bboxes(_stem).get(int(_idx_s)),
        "panel_png"    : _ppng,
        "artifact_meta": _find_meta(_stem),
        "cluster"      : -1,
        "xy"           : (0.0, 0.0),
        "nn_indices"   : [],
    })

N = len(records)
_n_panels = len(set(r["panel_stem"] for r in records))
_no_bbox  = sum(1 for r in records if not r["bbox"])
_no_panel = sum(1 for r in records if not r["panel_png"].exists())
print(f"Loaded {N} motifs from {_n_panels} panels")
if _no_bbox or _no_panel:
    print(f"  ⚠ {_no_bbox} missing bbox  |  {_no_panel} missing panel PNG")\
"""))

# ── Cell 3: clustering + similarity ───────────────────────────────────────────
cells.append(code("ml-3", """\
print("t-SNE …", end=" ", flush=True)
_perp = min(30, max(5, N // 5))
_xy   = TSNE(n_components=2, perplexity=_perp, max_iter=1000,
             random_state=42).fit_transform(embeddings)
print("done.")

print("HDBSCAN …", end=" ", flush=True)
try:
    _lbl = _HDBSCAN(
        min_cluster_size=max(3, N // 20),
        min_samples=2, metric="euclidean",
    ).fit_predict(_xy)
except TypeError:
    # sklearn DBSCAN fallback
    _lbl = _HDBSCAN(eps=3.0, min_samples=2).fit_predict(_xy)
print("done.")

for i, r in enumerate(records):
    r["cluster"] = int(_lbl[i])
    r["xy"]      = (float(_xy[i, 0]), float(_xy[i, 1]))

_n_clusters = int(_lbl.max()) + 1 if _lbl.max() >= 0 else 0
_n_noise    = int((_lbl == -1).sum())
print(f"{_n_clusters} clusters  |  {_n_noise} noise ({_n_noise / N * 100:.0f}%)")

print("Similarity matrix …", end=" ", flush=True)
_sim     = cosine_similarity(embeddings)                  # (N, N)
_nn_all  = np.argsort(-_sim, axis=1)[:, 1:7]             # top-6 NN excl. self
for i, r in enumerate(records):
    r["nn_indices"] = _nn_all[i].tolist()
print("done.")\
"""))

# ── Cell 4: label store ────────────────────────────────────────────────────────
cells.append(code("ml-4", """\
_db: dict = {}
if LABELS_PATH.exists():
    _db = json.loads(LABELS_PATH.read_text())
    _matched = sum(1 for r in records if r["path_norm"] in _db)
    print(f"Loaded {len(_db)} labels  ({_matched} match current records)")

def get_label(rec: dict) -> dict | None:
    return _db.get(rec["path_norm"])

def save_label(rec: dict, label: str, description: str, notes: str,
               iconography: str = "", source: str = "human", llm_data: dict = None):
    entry = {
        "label"      : label.strip(),
        "description": description.strip(),
        "notes"      : notes.strip(),
        "iconography": iconography.strip(),
        "cluster"    : rec["cluster"],
        "source"     : source,
        "timestamp"  : datetime.now().isoformat(timespec="seconds"),
    }
    if llm_data:
        entry["llm_suggestion"] = llm_data.get("label", "")
        entry["llm_reasoning"]  = llm_data.get("raw", "")
    _db[rec["path_norm"]] = entry
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(_db, indent=2, ensure_ascii=False))

def label_progress() -> tuple[int, int]:
    return sum(1 for r in records if r["path_norm"] in _db), N\
"""))

# ── Cell 5: display utilities ──────────────────────────────────────────────────
cells.append(code("ml-5", """\
THUMB_PX  = 72
PANEL_MAX = 560   # max px for panel display

_tcache: dict[str, bytes] = {}

def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO(); img.save(buf, "PNG"); return buf.getvalue()

def _thumb(path: Path, size: int = THUMB_PX) -> bytes:
    key = f"{path}:{size}"
    if key not in _tcache:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((size, size), Image.LANCZOS)
            sq  = Image.new("RGB", (size, size), (50, 50, 50))
            sq.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
            _tcache[key] = _png(sq)
        except Exception:
            sq = Image.new("RGB", (size, size), (80, 80, 80))
            _tcache[key] = _png(sq)
    return _tcache[key]

def _resized(path: Path, max_d: int) -> bytes | None:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_d, max_d), Image.LANCZOS)
        return _png(img)
    except Exception:
        return None

def _b64(path: Path, max_d: int = 512) -> str | None:
    b = _resized(path, max_d)
    return base64.standard_b64encode(b).decode() if b else None

def _get_font(size: int = 16):
    for path in ("/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()

_LIME = (0, 255, 64)

def _draw_single(rec: dict) -> tuple[bytes | None, bytes | None]:
    '''Return (zoom_bytes, panel_bytes) for single-select context.'''
    if not rec["panel_png"].exists():
        return None, None
    try:
        panel = Image.open(rec["panel_png"]).convert("RGB")
    except Exception:
        return None, None
    pdraw = panel.copy()
    zoom_b = None
    if rec["bbox"]:
        b = rec["bbox"]
        drw = ImageDraw.Draw(pdraw)
        drw.rectangle([b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]],
                      outline=_LIME, width=3)
        pad = 30
        iw, ih = panel.size
        zoom = panel.crop((max(0, b["x"]-pad), max(0, b["y"]-pad),
                           min(iw, b["x"]+b["w"]+pad), min(ih, b["y"]+b["h"]+pad)))
        zoom.thumbnail((220, 420), Image.LANCZOS)
        zoom_b = _png(zoom)
    pdraw.thumbnail((PANEL_MAX, PANEL_MAX * 2), Image.LANCZOS)
    return zoom_b, _png(pdraw)

def _draw_multi_panel(panel_stem: str, sel_recs: list, global_nums: list) -> tuple:
    '''Draw panel with all selected motifs numbered.
    Returns (panel_bytes, [crop_bytes | None]).
    '''
    ppng = next((r["panel_png"] for r in sel_recs if r["panel_png"].exists()), None)
    if not ppng:
        return None, [None] * len(sel_recs)
    try:
        panel = Image.open(ppng).convert("RGB")
    except Exception:
        return None, [None] * len(sel_recs)
    pdraw = panel.copy()
    drw   = ImageDraw.Draw(pdraw)
    font  = _get_font(18)
    crops = []
    for rec, n in zip(sel_recs, global_nums):
        if not rec["bbox"]:
            crops.append(None); continue
        b = rec["bbox"]
        drw.rectangle([b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]],
                      outline=_LIME, width=3)
        lbl = str(n)
        tx, ty = b["x"] + 4, b["y"] + 4
        try:
            bb = drw.textbbox((tx, ty), lbl, font=font)
            drw.rectangle([bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2], fill=(0, 0, 0))
        except AttributeError:
            drw.rectangle([tx-2, ty-2, tx+20, ty+20], fill=(0, 0, 0))
        drw.text((tx, ty), lbl, fill=_LIME, font=font)
        pad = 20
        iw, ih = panel.size
        crop = panel.crop((max(0, b["x"]-pad), max(0, b["y"]-pad),
                           min(iw, b["x"]+b["w"]+pad), min(ih, b["y"]+b["h"]+pad)))
        crop.thumbnail((100, 200), Image.LANCZOS)
        crops.append(_png(crop))
    pdraw.thumbnail((PANEL_MAX, PANEL_MAX * 2), Image.LANCZOS)
    return _png(pdraw), crops\
"""))

# ── Cell 6: state + gallery ────────────────────────────────────────────────────
cells.append(code("ml-6", """\
# ── Shared state ──────────────────────────────────────────────────────────────
_S = {"cursor": 0, "selected": set(), "llm_data": None}

# ── Widget registries ─────────────────────────────────────────────────────────
_cards: dict[int, widgets.VBox]     = {}
_chks:  dict[int, widgets.Checkbox] = {}
_dots:  dict[int, widgets.HTML]     = {}

# ── Output areas ──────────────────────────────────────────────────────────────
_ctx_out = widgets.Output(layout=widgets.Layout(min_height="200px"))
_nn_out  = widgets.Output()
_sts_out = widgets.Output()
_llm_out = widgets.Output()

# ── Editor widgets ─────────────────────────────────────────────────────────────
_WL = widgets.Layout
_si = {"description_width": "110px"}
_fw = _WL(width="480px")

_w_info  = widgets.HTML("<i>no selection</i>")
_w_lbl   = widgets.Text(placeholder="e.g. interlaced_knotwork",
                        description="Label", layout=_fw, style=_si)
_w_desc  = widgets.Textarea(placeholder="One sentence on what is visually present",
                             description="Description", rows=2, layout=_fw, style=_si)
_w_icon  = widgets.Textarea(placeholder="Iconographic significance, or 'unclear'",
                             description="Iconography", rows=2, layout=_fw, style=_si)
_w_notes = widgets.Text(placeholder="Cross-references, observations",
                        description="Notes", layout=_fw, style=_si)

_WB = lambda desc, style="", w="140px": widgets.Button(
    description=desc, button_style=style, layout=_WL(width=w))
_btn_llm  = _WB("✨ LLM Suggest", "info", "150px")
_btn_save = _WB("✓ Save", "success", "100px")
_btn_prev = _WB("← Prev", "", "100px")
_btn_next = _WB("Next →", "", "100px")
_btn_nxt_u = _WB("Next Unlabeled →", "", "160px")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _dot_html(labeled):
    c = "#44cc44" if labeled else "#666"
    return (f'<div style="width:8px;height:8px;border-radius:50%;'
            f'background:{c};margin:2px auto"></div>')

def _card_border(idx):
    c = _cards.get(idx)
    if not c: return
    if idx == _S["cursor"]:
        c.layout.border = "2px solid #44aaff"
    elif idx in _S["selected"]:
        c.layout.border = "2px solid #ffaa22"
    else:
        c.layout.border = "2px solid transparent"

def _refresh_gallery():
    for idx in _cards:
        _card_border(idx)
        if idx in _dots:
            _dots[idx].value = _dot_html(records[idx]["path_norm"] in _db)

def _load_fields():
    r   = records[_S["cursor"]]
    lbl = get_label(r)
    m   = r.get("artifact_meta") or {}
    reg = m.get("registration_number", r["panel_stem"])
    loc = m.get("historical_location") or m.get("location", "")
    cat = ", ".join(m.get("categories", []))
    extra = f"<br><span style='color:#888;font-size:11px'>{reg}  {loc}  {cat}</span>" if (loc or cat) else ""
    _w_info.value = (f"<b>{r['panel_stem']}</b> · idx {r['index']} "
                     f"· {r['scale']} · cluster {r['cluster']}{extra}")
    if lbl:
        _w_lbl.value  = lbl.get("label", "")
        _w_desc.value = lbl.get("description", "")
        _w_icon.value = lbl.get("iconography", "")
        _w_notes.value= lbl.get("notes", "")
    else:
        _w_lbl.value = _w_desc.value = _w_icon.value = _w_notes.value = ""

def _refresh_ctx():
    with _ctx_out:
        clear_output(wait=True)
        sel = _S["selected"]
        if len(sel) <= 1:
            r = records[_S["cursor"]]
            zoom_b, panel_b = _draw_single(r)
            row = []
            if zoom_b:
                row.append(widgets.VBox([
                    widgets.HTML("<b style='color:#ccc'>Crop (2×)</b>"),
                    widgets.Image(value=zoom_b, format="png",
                                  layout=_WL(max_width="220px", max_height="420px")),
                ]))
            if panel_b:
                row.append(widgets.VBox([
                    widgets.HTML("<b style='color:#ccc'>Panel context</b>"),
                    widgets.Image(value=panel_b, format="png",
                                  layout=_WL(max_width="560px", max_height="700px")),
                ]))
            if row:
                display(widgets.HBox(row, layout=_WL(
                    flex_flow="row wrap", gap="16px", align_items="flex-start")))
            else:
                display(widgets.HTML("<i style='color:#888'>No panel image found</i>"))
        else:
            sorted_sel = sorted(sel)
            by_panel: dict[str, list] = {}
            for gn, idx in enumerate(sorted_sel, 1):
                by_panel.setdefault(records[idx]["panel_stem"], []).append((idx, gn))
            for stem, pairs in by_panel.items():
                recs_sel = [records[idx] for idx, _ in pairs]
                gnums    = [gn for _, gn in pairs]
                panel_b, crops = _draw_multi_panel(stem, recs_sel, gnums)
                display(widgets.HTML(
                    f"<div style='font-weight:600;color:#bbb;margin-top:12px'>"
                    f"{stem}  ({len(recs_sel)} selected)</div>"))
                if panel_b:
                    display(widgets.Image(value=panel_b, format="png",
                                          layout=_WL(max_width="580px")))
                crop_ws = []
                for cb, (_, gn) in zip(crops, pairs):
                    if cb:
                        crop_ws.append(widgets.VBox([
                            widgets.Image(value=cb, format="png",
                                          layout=_WL(width="100px")),
                            widgets.HTML(f"<center style='color:#888;font-size:11px'>#{gn}</center>"),
                        ]))
                if crop_ws:
                    display(widgets.HBox(crop_ws, layout=_WL(flex_flow="row wrap", gap="8px")))

def _refresh_nn():
    with _nn_out:
        clear_output(wait=True)
        r  = records[_S["cursor"]]
        ws = []
        for nn_i in r["nn_indices"]:
            nn_r  = records[nn_i]
            tb    = _thumb(nn_r["path_display"])
            sim_v = float(_sim[_S["cursor"], nn_i])
            nl    = get_label(nn_r)
            nl_tx = f"<div style='font-size:9px;color:#888;word-break:break-all'>{nl['label'] if nl else ''}</div>"
            sb    = widgets.Button(description="→", layout=_WL(width=f"{THUMB_PX}px", height="16px", padding="0"))
            def _goto(_, t=nn_i):
                _S["cursor"] = t; _refresh_all()
            sb.on_click(_goto)
            ws.append(widgets.VBox([
                widgets.Image(value=tb, format="png",
                              layout=_WL(width=f"{THUMB_PX}px", height=f"{THUMB_PX}px")),
                widgets.HTML(f"<div style='font-size:9px;color:#666;text-align:center'>{sim_v:.2f}</div>" + nl_tx),
                sb,
            ], layout=_WL(width=f"{THUMB_PX+8}px", margin="3px")))
        display(widgets.HBox(ws))

def _refresh_sts():
    with _sts_out:
        clear_output(wait=True)
        ln, tot = label_progress()
        pct = ln / tot * 100 if tot else 0
        display(widgets.HTML(f"<span style='color:#888'>Labels: {ln}/{tot} ({pct:.0f}%)</span>"))

def _refresh_all():
    _refresh_gallery()
    _refresh_ctx()
    _refresh_nn()
    _load_fields()
    _refresh_sts()

# ── Build gallery ──────────────────────────────────────────────────────────────
def _make_card(idx: int) -> widgets.VBox:
    r    = records[idx]
    tb   = _thumb(r["path_display"])
    imgw = widgets.Image(value=tb, format="png",
                         layout=_WL(width=f"{THUMB_PX}px", height=f"{THUMB_PX}px"))
    chk  = widgets.Checkbox(value=False, indent=False,
                             layout=_WL(width="18px", height="18px"))
    dot  = widgets.HTML(_dot_html(r["path_norm"] in _db),
                        layout=_WL(height="12px"))
    sel_btn = widgets.Button(description=str(idx), tooltip=r["panel_stem"],
                              layout=_WL(width=f"{THUMB_PX}px", height="18px", padding="0"))

    def _on_sel(b, i=idx):
        _S["cursor"] = i; _refresh_all()
    def _on_chk(change, i=idx):
        if change["new"]: _S["selected"].add(i)
        else: _S["selected"].discard(i)
        _card_border(i); _refresh_ctx()

    sel_btn.on_click(_on_sel)
    chk.observe(_on_chk, names="value")

    card = widgets.VBox([imgw, chk, dot, sel_btn],
                        layout=_WL(width=f"{THUMB_PX+6}px", margin="3px",
                                   padding="1px", border="2px solid transparent"))
    _cards[idx] = card; _chks[idx] = chk; _dots[idx] = dot
    return card

print("Building gallery …", end=" ", flush=True)
_cids = sorted({r["cluster"] for r in records if r["cluster"] >= 0}) + ([-1] if any(r["cluster"] == -1 for r in records) else [])
_tab_kids, _tab_titles = [], []
for _cid in _cids:
    _recs_c = [(i, r) for i, r in enumerate(records) if r["cluster"] == _cid]
    _grid   = widgets.HBox(
        [_make_card(i) for i, _ in _recs_c],
        layout=_WL(flex_flow="row wrap", align_items="flex-start",
                   overflow_x="auto", max_height="260px", overflow_y="auto"),
    )
    _tab_kids.append(_grid)
    _tab_titles.append(f"{'Noise' if _cid == -1 else f'C{_cid}'} ({len(_recs_c)})")

_gallery_tab = widgets.Tab(children=_tab_kids)
for _i, _t in enumerate(_tab_titles):
    _gallery_tab.set_title(_i, _t)
print(f"done — {len(_cards)} thumbnails across {len(_tab_kids)} tabs.")\
"""))

# ── Cell 7: button handlers + LLM ─────────────────────────────────────────────
cells.append(code("ml-7", """\
# ── Navigation & save ─────────────────────────────────────────────────────────
def _on_save(b):
    r   = records[_S["cursor"]]
    ld  = _S.get("llm_data")
    src = "human"
    if ld:
        src = ("llm" if (_w_lbl.value.strip() == ld.get("label","").strip()
                         and _w_desc.value.strip() == ld.get("description","").strip())
               else "llm-edited")
    save_label(r, _w_lbl.value, _w_desc.value, _w_notes.value,
               _w_icon.value, src, ld)
    _S["llm_data"] = None
    _refresh_all()
    with _sts_out:
        clear_output(wait=True)
        ln, tot = label_progress()
        display(widgets.HTML(f"<span style='color:#44cc44'>Saved. {ln}/{tot} labeled.</span>"))

def _navigate(delta: int):
    _S["cursor"] = (_S["cursor"] + delta) % N
    _refresh_all()

def _nav_unlabeled():
    start = _S["cursor"]
    for off in range(1, N + 1):
        idx = (start + off) % N
        if records[idx]["path_norm"] not in _db:
            _S["cursor"] = idx; _refresh_all(); return
    with _sts_out:
        clear_output(wait=True)
        display(widgets.HTML("<span style='color:#44cc44'>All motifs labeled!</span>"))

_btn_save.on_click(_on_save)
_btn_prev.on_click(lambda b: _navigate(-1))
_btn_next.on_click(lambda b: _navigate(+1))
_btn_nxt_u.on_click(lambda b: _nav_unlabeled())

# ── LLM suggestion ─────────────────────────────────────────────────────────────
def _call_llm(rec: dict) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"error": "ANTHROPIC_API_KEY not set"}
    if not _ANTHROPIC_OK:
        return {"error": "anthropic package not available"}

    client  = _anthropic.Anthropic(api_key=key)
    content = []

    def _add_img(path: Path, caption: str, max_d: int = 512):
        b64 = _b64(path, max_d)
        if not b64: return
        content.append({"type": "text", "text": caption})
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64}})

    _add_img(rec["path_orig_abs"], "Image 1 — original crop (photo/illustration):")
    if rec["path_norm_abs"].exists():
        _add_img(rec["path_norm_abs"], "Image 2 — normalised line-art:")
    _img_offset = 3 if rec["path_norm_abs"].exists() else 2
    for rank, nn_i in enumerate(rec["nn_indices"][:3], 1):
        nn_r = records[nn_i]; nl = get_label(nn_r)
        cap  = f"Image {_img_offset + rank - 1} — nearest neighbour #{rank}"
        if nl: cap += f" (labeled: {nl['label']})"
        _add_img(nn_r["path_orig_abs"], cap + ":", max_d=256)

    m    = rec.get("artifact_meta") or {}
    reg  = m.get("registration_number", rec["panel_stem"])
    loc  = m.get("historical_location") or m.get("location", "")
    cats = ", ".join(m.get("categories", [])) or "panel art"
    nc   = sum(1 for r in records if r["cluster"] == rec["cluster"])
    cstr = (f"Cluster {rec['cluster']} ({nc} similar)" if rec["cluster"] >= 0 else "Unclustered")
    nns  = [get_label(records[ni]) for ni in rec["nn_indices"][:3]]
    nn_names = [l["label"] for l in nns if l and l.get("label")]
    nn_str   = f"\\nNearest labelled: {', '.join(nn_names)}." if nn_names else ""

    content.append({"type": "text", "text":
        f"You are describing carved motifs from Yoruba door panels in the Frobenius Institute archive.\\n\\n"
        f"Artifact: {reg}  |  Location: {loc}  |  Type: {cats}  |  {cstr}{nn_str}\\n\\n"
        "Use Image 1 (original) for cultural/medium context.\\n"
        "Use Image 2 (line-art, if present) for structure.\\n"
        "Use nearest-neighbour images for visual comparison.\\n\\n"
        "Respond with JSON only — no markdown, no explanation outside JSON:\\n"
        '{\\n  "label": "2-4 words snake_case",\\n'
        '  "description": "one sentence on what is visually present",\\n'
        '  "iconography": "one sentence on symbolic significance, or \\'unclear\\'"\\n}'
    })

    try:
        resp = client.messages.create(
            model="claude-opus-4-6", max_tokens=300,
            messages=[{"role": "user", "content": content}],
        )
        raw   = resp.content[0].text.strip()
        clean = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", raw, flags=re.MULTILINE).strip()
        result = json.loads(clean)
        result["raw"] = raw
        return result
    except json.JSONDecodeError:
        return {"label": "", "description": raw, "iconography": "", "raw": raw}
    except Exception as exc:
        return {"error": str(exc)}

def _on_llm(b):
    _btn_llm.disabled = True
    _btn_llm.description = "Thinking …"
    with _llm_out:
        clear_output(wait=True)
        display(widgets.HTML("<span style='color:#aaa'>Calling Claude…</span>"))
    try:
        result = _call_llm(records[_S["cursor"]])
    except Exception as e:
        result = {"error": str(e)}
    _btn_llm.disabled = False
    _btn_llm.description = "✨ LLM Suggest"
    with _llm_out:
        clear_output(wait=True)
        if "error" in result:
            display(widgets.HTML(f"<span style='color:#f66'>{result['error']}</span>"))
            return
    _S["llm_data"] = result
    _w_lbl.value   = result.get("label", "")
    _w_desc.value  = result.get("description", "")
    _w_icon.value  = result.get("iconography", "")
    with _llm_out:
        clear_output(wait=True)
        display(widgets.HTML("<span style='color:#44cc44'>Suggestion loaded — edit and Save.</span>"))

_btn_llm.on_click(_on_llm)\
"""))

# ── Cell 8: launch UI ──────────────────────────────────────────────────────────
cells.append(code("ml-8", """\
_refresh_all()

_editor = widgets.VBox([
    _w_info,
    widgets.HTML("<hr style='border-color:#333;margin:6px 0'>"),
    _w_lbl, _w_desc, _w_icon, _w_notes,
    widgets.HBox([_btn_llm, _btn_save]),
    widgets.HBox([_btn_prev, _btn_next, _btn_nxt_u]),
    _llm_out,
    _sts_out,
], layout=widgets.Layout(padding="10px", max_width="520px"))

_right = widgets.VBox([
    widgets.HTML("<span style='color:#aaa;font-weight:600'>Context</span>"),
    _ctx_out,
    widgets.HTML("<hr style='border-color:#333;margin:8px 0'>"
                 "<span style='color:#aaa;font-weight:600'>Nearest neighbours</span>"),
    _nn_out,
    widgets.HTML("<hr style='border-color:#333;margin:8px 0'>"
                 "<span style='color:#aaa;font-weight:600'>Label editor</span>"),
    _editor,
])

display(widgets.VBox([
    widgets.HTML("<h2 style='color:#eee;margin:0 0 4px'>Motif Labeling</h2>"),
    widgets.HTML(
        "<div style='color:#888;font-size:12px;margin-bottom:10px'>"
        "Click index button to set active · Checkbox to multi-select · "
        "Active = blue border · Multi-select = orange border</div>"
    ),
    _gallery_tab,
    widgets.HTML("<hr style='border-color:#333;margin:12px 0'>"),
    _right,
], layout=widgets.Layout(padding="16px")))\
"""))

# ── Cell 9: progress + export ──────────────────────────────────────────────────
cells.append(code("ml-9", """\
# Run this cell any time to refresh progress stats (safe to re-run, no side effects).
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

_lb_recs = [r for r in records if get_label(r)]
_ln, _tot = len(_lb_recs), N
_lbl_vals = [get_label(r)["label"] for r in _lb_recs if get_label(r).get("label")]

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
fig.patch.set_facecolor("#1a1a1a")
for ax in axes:
    ax.set_facecolor("#252525")
    for spine in ax.spines.values(): spine.set_color("#444")
    ax.tick_params(colors="#aaa"); ax.xaxis.label.set_color("#aaa"); ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#eee")

# Coverage
axes[0].barh(["Labeled", "Unlabeled"], [_ln, _tot - _ln], color=["#44cc44", "#444"])
axes[0].set_xlim(0, _tot)
axes[0].set_title(f"Coverage: {_ln}/{_tot} ({_ln/_tot*100:.0f}%)" if _tot else "Coverage")

# Label frequency
if _lbl_vals:
    _top = Counter(_lbl_vals).most_common(15)
    _ns, _cs = zip(*_top)
    axes[1].barh([n[:30] for n in reversed(_ns)], list(reversed(_cs)), color="#5599ff")
    axes[1].set_title(f"Top {len(_top)} Labels")
else:
    axes[1].text(0.5, 0.5, "No labels yet", ha="center", va="center",
                 transform=axes[1].transAxes, color="#666")

# Cluster × label heatmap
if len(_lb_recs) >= 2 and _lbl_vals:
    _pairs = [(r["cluster"], get_label(r)["label"]) for r in _lb_recs if get_label(r).get("label")]
    _cls = sorted({c for c, _ in _pairs}); _lbs = sorted({l for _, l in _pairs})
    if _cls and _lbs:
        _mat = np.zeros((len(_cls), len(_lbs)), dtype=int)
        for c, l in _pairs: _mat[_cls.index(c), _lbs.index(l)] += 1
        sns.heatmap(_mat, ax=axes[2], xticklabels=[l[:18] for l in _lbs],
                    yticklabels=[str(c) for c in _cls],
                    cmap="Blues", annot=True, fmt="d",
                    cbar_kws={"label": "count"})
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", fontsize=7)
        axes[2].set_title("Cluster × Label")
else:
    axes[2].text(0.5, 0.5, "Need labels to show heatmap", ha="center", va="center",
                 transform=axes[2].transAxes, color="#666")

plt.suptitle("Motif Labeling Progress", color="#eee", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()
if _lbl_vals:
    print("\\nTop labels:")
    for lbl, cnt in Counter(_lbl_vals).most_common(10):
        print(f"  {lbl:<40s} {cnt}")\
"""))

# ── Write notebook ─────────────────────────────────────────────────────────────
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

out = Path(__file__).parent / "motif_labeling.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Written: {out}  ({out.stat().st_size // 1024} KB)")
