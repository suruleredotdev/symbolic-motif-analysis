"""
pipeline_state.py — Shared in-memory state for the unified motif pipeline.

All notebook cells read/write a single PipelineState instance.  Persistence
is backward-compatible: _approved.json and motif_labels.json on disk use the
same schemas as the existing separate notebooks.

Motif crops are computed on the fly from panel_img + bbox — no intermediate
files.  export_crops() writes PNGs as an explicit action.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


# ── Data records ─────────────────────────────────────────────────────────────

# Label-file keys are crop paths for backward compatibility with the standalone
# notebooks.  The canonical form deliberately omits the IoU: it used to be
# baked in, which meant two writers holding slightly different values for the
# same motif produced two entries for it, and whichever landed later in the
# file silently won on load.  Readers parse only the stem and index, so
# dropping the IoU is compatible in both directions.
_LEGACY_KEY_RE = re.compile(r"motifs(?:_norm)?/([^/]+)/(\d+)_")


def motif_label_key(panel_stem: str, index: int) -> str:
    """The one key under which a motif's label is stored.  Use this everywhere."""
    return (f"../../frobenius_artifacts/analysis/motifs_norm/"
            f"{panel_stem}/{index:03d}_motif.png")


def key_to_motif(key: str) -> str | None:
    """``<panel_stem>/<index>`` for a label key, old form or new.  None if unparseable."""
    match = _LEGACY_KEY_RE.search(key)
    return f"{match.group(1)}/{int(match.group(2))}" if match else None


@dataclass
class MotifRecord:
    """One motif detection across the full pipeline lifecycle."""

    # ── Segment stage (set at creation) ───────────────────────────────────
    panel_stem: str
    index: int
    bbox: dict                          # {x, y, w, h}
    scale: str                          # "register" | "motif"
    area_ratio: float
    predicted_iou: float
    stability_score: float
    source: str                         # "sam_auto" | "sam_prompted" | "manual"
    created_at: str = ""                # ISO 8601
    included: bool = True               # user approved this detection?

    # ── Cluster stage (set later) ─────────────────────────────────────────
    cluster: int = -1                   # -1 = unassigned

    # ── Label stage (set later) ───────────────────────────────────────────
    label: str | None = None
    description: str | None = None
    iconography: str | None = None
    notes: str | None = None
    label_source: str | None = None     # "human" | "llm" | "llm-edited"
    label_timestamp: str | None = None

    # Set by save_label(); cleared on load and after a successful write. This is
    # what "changed in this session" means — the label *source* cannot answer
    # that, since a motif loaded from disk already carries source="human"
    # without anyone having touched it this time round.
    dirty: bool = False

    def to_approved_dict(self) -> dict:
        """Serialize to the _approved.json schema (backward compatible)."""
        d: dict[str, Any] = {
            "index": self.index,
            "bbox": self.bbox,
            "scale": self.scale,
            "area_ratio": round(self.area_ratio, 5),
            "predicted_iou": round(self.predicted_iou, 4),
            "stability_score": round(self.stability_score, 4),
            "source": self.source,
        }
        if self.created_at:
            d["created_at"] = self.created_at
        return d

    def to_label_dict(self) -> dict | None:
        """Serialize to the motif_labels.json schema.  Returns None if unlabeled."""
        if not self.label:
            return None
        d: dict[str, Any] = {
            "label": self.label,
            "description": self.description or "",
            "notes": self.notes or "",
            "iconography": self.iconography or "",
            "cluster": self.cluster,
            "source": self.label_source or "human",
            "timestamp": self.label_timestamp or self.created_at,
        }
        return d

    @property
    def motif_key(self) -> str:
        """Unique key for this motif across all panels."""
        return f"{self.panel_stem}/{self.index}"


@dataclass
class PanelInfo:
    """Metadata for one panel image."""
    stem: str
    png_path: Path
    width: int = 0
    height: int = 0


# ── Pipeline state ───────────────────────────────────────────────────────────

class PipelineState:
    """Shared in-memory state for the unified pipeline notebook.

    All cells read/write this object.  Disk I/O is explicit:
    load_from_disk() at startup, save_approved() / save_all_labels() on demand.
    """

    def __init__(self) -> None:
        self.panels: dict[str, PanelInfo] = {}
        self.motifs: list[MotifRecord] = []
        self.embeddings: np.ndarray | None = None
        self.sim_matrix: np.ndarray | None = None
        self.tsne_xy: np.ndarray | None = None

        # Lazy-loaded panel images
        self._panel_img_cache: dict[str, Image.Image] = {}

        # SAM draft-bbox workflow
        self._sam_candidates: dict[str, list[MotifRecord]] = {}
        self._draft_cursor: dict[str, int] = {}
        self._draft_accepted: dict[str, int] = {}
        self._draft_skipped: dict[str, int] = {}

        # Paths (set by load_from_disk)
        self._annotated_dir: Path | None = None
        self._panels_dir: Path | None = None
        self._labels_path: Path | None = None
        self._clusters_path: Path | None = None

    # ── Load from disk ────────────────────────────────────────────────────

    def load_from_disk(
        self,
        annotated_dir: Path,
        panels_dir: Path,
        labels_path: Path | None = None,
        clusters_path: Path | None = None,
    ) -> None:
        """Read existing _approved.json / _detections.json files + labels + clusters."""
        self._annotated_dir = Path(annotated_dir)
        self._panels_dir = Path(panels_dir)
        self._labels_path = Path(labels_path) if labels_path else None
        self._clusters_path = Path(clusters_path) if clusters_path else None

        self.panels.clear()
        self.motifs.clear()
        self._panel_img_cache.clear()

        # Discover panels
        det_files = sorted(self._annotated_dir.glob("*_detections.json"))
        for det_path in det_files:
            stem = det_path.stem.removesuffix("_detections")
            png = self._panels_dir / f"{stem}.png"
            if not png.exists():
                png = self._panels_dir / f"{stem}_cropped.png"
            if not png.exists():
                continue

            # Prefer _approved.json over _detections.json
            approved_path = self._annotated_dir / f"{stem}_approved.json"
            src_path = approved_path if approved_path.exists() else det_path
            src_type = "approved" if approved_path.exists() else "detections"

            raw = json.loads(src_path.read_text())
            img = Image.open(png)
            pw, ph = img.size
            img.close()

            self.panels[stem] = PanelInfo(stem=stem, png_path=png,
                                          width=pw, height=ph)

            for d in raw:
                source = d.get("source", "sam_auto")
                if src_type == "detections" and "source" not in d:
                    source = "sam_auto"
                self.motifs.append(MotifRecord(
                    panel_stem=stem,
                    index=d.get("index", 0),
                    bbox=d["bbox"],
                    scale=d.get("scale", "motif"),
                    area_ratio=d.get("area_ratio", 0.0),
                    predicted_iou=d.get("predicted_iou",
                                        d.get("pred_iou", 0.0)),
                    stability_score=d.get("stability_score", 0.0),
                    source=source,
                    created_at=d.get("created_at", ""),
                    included=True if src_type == "approved" else True,
                ))

        # Merge existing labels
        if self._labels_path and self._labels_path.exists():
            self._merge_labels(self._labels_path)

        # Clusters load after labels so a standalone clusters.json wins — it is
        # the newer, label-independent record of the same assignment.
        if self._clusters_path and self._clusters_path.exists():
            applied = self.load_clusters(self._clusters_path)
            if applied:
                print(f"  Clusters restored: {applied} motifs")

        n_panels = len(self.panels)
        n_motifs = len(self.motifs)
        n_approved = sum(1 for p in self.panels
                         if (self._annotated_dir / f"{p}_approved.json").exists())
        print(f"PipelineState: {n_panels} panels, {n_motifs} motifs, "
              f"{n_approved} with approved files")

    def _merge_labels(self, labels_path: Path) -> None:
        """Merge motif_labels.json into MotifRecord.label fields."""
        raw = json.loads(labels_path.read_text())
        # Build lookup: "panel_stem/index" -> label dict
        # Label keys are paths like "../../.../motifs_norm/<stem>/<idx>_<scale>_iou<X>.png"
        import re
        label_lookup: dict[str, dict] = {}
        pat = re.compile(r"motifs(?:_norm)?/([^/]+)/(\d+)_")
        for key, val in raw.items():
            m = pat.search(key)
            if m:
                stem, idx = m.group(1), int(m.group(2))
                label_lookup[f"{stem}/{idx}"] = val

        matched = 0
        for mr in self.motifs:
            lbl = label_lookup.get(mr.motif_key)
            if lbl:
                mr.label = lbl.get("label")
                mr.description = lbl.get("description")
                mr.iconography = lbl.get("iconography")
                mr.notes = lbl.get("notes")
                mr.cluster = lbl.get("cluster", -1)
                mr.label_source = lbl.get("source")
                mr.label_timestamp = lbl.get("timestamp")
                mr.dirty = False                 # freshly loaded == unmodified
                matched += 1

        if matched:
            print(f"  Labels merged: {matched}/{len(self.motifs)}")

    # ── Save to disk ──────────────────────────────────────────────────────

    def save_approved(self, stem: str) -> Path:
        """Write _approved.json for one panel (included motifs only)."""
        assert self._annotated_dir is not None
        included = [m for m in self.motifs
                    if m.panel_stem == stem and m.included]
        # Re-index
        for i, m in enumerate(included):
            m.index = i
        data = [m.to_approved_dict() for m in included]
        path = self._annotated_dir / f"{stem}_approved.json"
        path.write_text(json.dumps(data, indent=2))
        return path

    # ── Cluster persistence ───────────────────────────────────────────────
    #
    # Cluster ids used to survive only inside label records, so a motif that
    # was clustered but never labelled lost its assignment on save — which is
    # most of them on a fresh run.  Clusters now persist on their own.

    def save_clusters(self, path: Path, params: dict | None = None) -> Path:
        """Write clusters.json — every included motif's cluster assignment.

        Independent of labels: a motif does not need a label to keep its
        cluster.  `params` records the settings that produced the run so a
        later session can tell whether an assignment is still current.
        """
        assignments = {
            m.motif_key: m.cluster
            for m in self.motifs
            if m.included and m.cluster is not None and m.cluster >= 0
        }
        data = {
            "generated_at": datetime.now().isoformat(),
            "params": params or {},
            "n_clusters": len(set(assignments.values())),
            "assignments": dict(sorted(assignments.items())),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        return path

    def load_clusters(self, path: Path) -> int:
        """Apply clusters.json onto the motif records.  Returns the count applied.

        Accepts the shape `save_clusters` writes, and also a bare
        ``{motif_key: cluster_id}`` mapping, which is what the interpretation
        CLI's ``--clusters`` override takes.
        """
        path = Path(path)
        if not path.exists():
            return 0
        raw = json.loads(path.read_text())
        assignments = raw.get("assignments", raw) if isinstance(raw, dict) else {}

        applied = 0
        by_key = {m.motif_key: m for m in self.motifs}
        for key, value in assignments.items():
            cluster = value.get("cluster") if isinstance(value, dict) else value
            motif = by_key.get(key)
            if motif is not None and cluster is not None:
                motif.cluster = int(cluster)
                applied += 1
        return applied

    # ── Embedding cache ───────────────────────────────────────────────────

    def save_embeddings(self, npy_path: Path, keys_path: Path,
                        keys: list[str] | None = None,
                        embeddings: "np.ndarray | None" = None) -> Path:
        """Cache the embedding matrix and its row keys.

        Recomputing CLIP over every crop costs minutes each session; the
        vectors only change when the crops or the preprocessing mode do.

        Pass `embeddings` explicitly to save a matrix the caller is holding —
        the notebook keeps its working copy in cell state, and relying on that
        having been mirrored onto `self.embeddings` is a coupling worth not
        depending on.  Defaults to `self.embeddings`.
        """
        matrix = embeddings if embeddings is not None else self.embeddings
        if matrix is None:
            raise ValueError("no embeddings to save — compute them first")
        keys = keys or [m.motif_key for m in self.included_motifs()]
        if len(keys) != len(matrix):
            raise ValueError(
                f"{len(keys)} keys but {len(matrix)} embedding rows")

        npy_path, keys_path = Path(npy_path), Path(keys_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, matrix)
        keys_path.write_text("\n".join(keys))
        return npy_path

    def load_embeddings(self, npy_path: Path, keys_path: Path) -> list[str]:
        """Restore a cached embedding matrix.  Returns its row keys ([] if absent)."""
        npy_path, keys_path = Path(npy_path), Path(keys_path)
        if not npy_path.exists() or not keys_path.exists():
            return []
        embeddings = np.load(npy_path)
        keys = [k for k in keys_path.read_text().splitlines() if k.strip()]
        if len(keys) != len(embeddings):
            return []                       # stale pair; recompute rather than guess
        self.embeddings = embeddings
        return keys

    def save_all_labels(self, path: Path | None = None,
                        only_changed: bool = False) -> tuple[Path, int]:
        """Merge this session's labels into motif_labels.json.

        Merges rather than replaces. The file is shared with
        `scripts/label_motifs.py` and with other labellers, so entries this
        session knows nothing about — a panel that was not loaded, a label
        written after this kernel started — are preserved untouched. Rewriting
        the whole file from memory silently deleted them.

        `only_changed` writes just the motifs edited in this session (see
        `MotifRecord.dirty`), which is the safe default for a long-running
        notebook: it cannot regress a label somebody else improved on disk
        while this kernel held a stale copy of it.

        Returns ``(path, number of entries written)``.
        """
        path = path or self._labels_path
        assert path is not None

        existing: dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:
                existing = {}                    # corrupt file: start clean

        candidates = [m for m in self.motifs if m.label
                      and (m.dirty or not only_changed)]

        written = 0
        for m in candidates:
            lbl = m.to_label_dict()
            if not lbl:
                continue
            # Drop any other key that refers to this same motif — older writers
            # embedded the IoU, so one motif could hold several entries.
            for stale in [k for k in existing
                          if key_to_motif(k) == m.motif_key]:
                del existing[stale]
            existing[motif_label_key(m.panel_stem, m.index)] = lbl
            written += 1

        path.write_text(json.dumps(dict(sorted(existing.items())), indent=2))
        for m in candidates:
            m.dirty = False
        return path, written

    # ── Panel image access ────────────────────────────────────────────────

    def panel_image(self, stem: str) -> Image.Image:
        """Get panel image (lazy loaded, cached)."""
        if stem not in self._panel_img_cache:
            info = self.panels[stem]
            self._panel_img_cache[stem] = Image.open(info.png_path).convert("RGB")
        return self._panel_img_cache[stem]

    def crop(self, motif: MotifRecord, padding: int = 4) -> Image.Image:
        """Crop a motif from its panel image. No disk I/O."""
        img = self.panel_image(motif.panel_stem)
        iw, ih = img.size
        b = motif.bbox
        x1 = max(0, b["x"] - padding)
        y1 = max(0, b["y"] - padding)
        x2 = min(iw, b["x"] + b["w"] + padding)
        y2 = min(ih, b["y"] + b["h"] + padding)
        return img.crop((x1, y1, x2, y2))

    def export_crops(self, out_dir: Path, padding: int = 4) -> int:
        """Write all included motif crops as PNGs. Returns count written."""
        out_dir = Path(out_dir)
        written = 0
        for m in self.motifs:
            if not m.included:
                continue
            panel_dir = out_dir / m.panel_stem
            panel_dir.mkdir(parents=True, exist_ok=True)
            crop = self.crop(m, padding=padding)
            fname = f"{m.index:03d}_{m.scale}_iou{m.predicted_iou:.3f}.png"
            crop.save(panel_dir / fname)
            written += 1
        return written

    # ── Motif queries ─────────────────────────────────────────────────────

    def motifs_for_panel(self, stem: str) -> list[MotifRecord]:
        """All motifs for a panel, in index order."""
        return sorted([m for m in self.motifs if m.panel_stem == stem],
                      key=lambda m: m.index)

    def included_motifs(self) -> list[MotifRecord]:
        """All approved motifs across all panels."""
        return [m for m in self.motifs if m.included]

    def motif_by_key(self, key: str) -> MotifRecord | None:
        """Look up one motif by its ``<panel_stem>/<index>`` key."""
        return next((m for m in self.motifs if m.motif_key == key), None)

    def manual_templates(self) -> list[dict]:
        """All source='manual' bboxes — ground-truth for SAM Refine templates."""
        return [m.bbox for m in self.motifs
                if m.included and m.source == "manual"]

    # ── Motif mutations ───────────────────────────────────────────────────

    def add_motif(self, stem: str, bbox: dict, source: str = "manual",
                  predicted_iou: float = 1.0,
                  stability_score: float = 1.0) -> MotifRecord:
        """Add a new motif with provenance timestamp."""
        panel = self.panels[stem]
        area_ratio = (bbox["w"] * bbox["h"]) / (panel.width * panel.height)
        existing = self.motifs_for_panel(stem)
        new_idx = max((m.index for m in existing), default=-1) + 1
        rec = MotifRecord(
            panel_stem=stem,
            index=new_idx,
            bbox=bbox,
            scale="motif" if area_ratio < 0.25 else "register",
            area_ratio=round(area_ratio, 5),
            predicted_iou=predicted_iou,
            stability_score=stability_score,
            source=source,
            created_at=datetime.now().isoformat(),
            included=True,
        )
        self.motifs.append(rec)
        return rec

    def remove_motif(self, motif: MotifRecord) -> None:
        """Mark a motif as excluded (does not delete)."""
        motif.included = False

    def save_label(self, motif: MotifRecord, label: str,
                   description: str = "", iconography: str = "",
                   notes: str = "", source: str = "human") -> None:
        """Set label fields with provenance timestamp."""
        motif.label = label
        motif.description = description
        motif.iconography = iconography
        motif.notes = notes
        motif.label_source = source
        motif.label_timestamp = datetime.now().isoformat()
        motif.dirty = True

    # ── SAM draft-bbox workflow ───────────────────────────────────────────

    def cache_sam_candidates(
        self,
        stem: str,
        candidates: list,
    ) -> int:
        """Store SAM candidates for the draft-bbox workflow.

        candidates: list of Detection objects from prompted_segment().
        Returns the number of candidates queued.
        """
        from panel_art.motif_segment import _edge_density
        import cv2

        panel_img = self.panel_image(stem)
        img_np = np.array(panel_img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        existing = [m.bbox for m in self.motifs_for_panel(stem) if m.included]

        records = []
        for det in candidates:
            # Compute novelty: inverse of max IoU with existing
            from panel_art.motif_segment import _iou
            bb = [det.bbox["x"], det.bbox["y"], det.bbox["w"], det.bbox["h"]]
            max_iou = max(
                (_iou(bb, [e["x"], e["y"], e["w"], e["h"]]) for e in existing),
                default=0.0,
            )
            novelty = 1.0 - max_iou

            # Edge density for this mask
            ed = 0.0
            if det.segmentation is not None:
                ed = _edge_density(img_gray, det.segmentation)

            # Composite score for ranking
            score = (0.5 * det.predicted_iou
                     + 0.3 * ed
                     + 0.2 * novelty)

            rec = MotifRecord(
                panel_stem=stem,
                index=det.index,
                bbox=det.bbox,
                scale=det.scale,
                area_ratio=det.area_ratio,
                predicted_iou=det.predicted_iou,
                stability_score=det.stability_score,
                source="sam_prompted",
                created_at="",
                included=False,  # draft, not yet accepted
            )
            records.append((score, ed, novelty, rec))

        # Sort by composite score descending (best first)
        records.sort(key=lambda t: t[0], reverse=True)

        # Store with metadata for status display
        self._sam_candidates[stem] = [
            {"record": r, "score": s, "edge_density": e, "novelty": n}
            for s, e, n, r in records
        ]
        self._draft_cursor[stem] = 0
        self._draft_accepted[stem] = 0
        self._draft_skipped[stem] = 0

        return len(records)

    def draft_count(self, stem: str) -> tuple[int, int, int, int]:
        """Returns (total, cursor, accepted, skipped) for draft queue."""
        total = len(self._sam_candidates.get(stem, []))
        cursor = self._draft_cursor.get(stem, 0)
        accepted = self._draft_accepted.get(stem, 0)
        skipped = self._draft_skipped.get(stem, 0)
        return total, cursor, accepted, skipped

    def next_draft(self, stem: str) -> dict | None:
        """Get the next SAM candidate for review.

        Returns dict with 'record', 'score', 'edge_density', 'novelty'
        or None if queue is exhausted.
        """
        cands = self._sam_candidates.get(stem, [])
        cursor = self._draft_cursor.get(stem, 0)
        if cursor >= len(cands):
            return None
        return cands[cursor]

    def accept_draft(self, stem: str, adjusted_bbox: dict | None = None) -> MotifRecord:
        """Accept the current draft candidate, optionally with an adjusted bbox."""
        cand = self.next_draft(stem)
        assert cand is not None, "No draft to accept"
        rec = cand["record"]

        if adjusted_bbox:
            rec.bbox = adjusted_bbox
            panel = self.panels[stem]
            rec.area_ratio = round(
                (adjusted_bbox["w"] * adjusted_bbox["h"])
                / (panel.width * panel.height), 5)

        rec.included = True
        rec.created_at = datetime.now().isoformat()
        rec.index = max((m.index for m in self.motifs
                         if m.panel_stem == stem), default=-1) + 1
        self.motifs.append(rec)

        self._draft_cursor[stem] = self._draft_cursor.get(stem, 0) + 1
        self._draft_accepted[stem] = self._draft_accepted.get(stem, 0) + 1
        return rec

    def skip_draft(self, stem: str) -> None:
        """Skip the current draft candidate."""
        self._draft_cursor[stem] = self._draft_cursor.get(stem, 0) + 1
        self._draft_skipped[stem] = self._draft_skipped.get(stem, 0) + 1

    def reset_draft_queue(self, stem: str) -> None:
        """Reset the cursor to re-review all candidates."""
        self._draft_cursor[stem] = 0
        self._draft_accepted.pop(stem, None)
        self._draft_skipped.pop(stem, None)
