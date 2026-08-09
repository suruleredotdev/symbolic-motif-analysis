"""
interpret.py — Phase 6: joining the cluster analysis into one interpretation.

Everything upstream of this module produces *fragments*: a crop, an embedding,
a cluster id, a hand-written label, a bounding box.  None of them say what a
panel means.  This module is the join — it takes those fragments and drives
Claude through three widening passes:

    1. cluster brief   — what is this visual family, seen across every panel
                         it appears on?  (one call per cluster)
    2. panel reading   — what does *this* panel say, register by register,
                         given its motifs, their cluster identities, and their
                         relative positions?  (one call per panel)
    3. corpus synthesis — what does the collection say once the panel readings
                         and cluster briefs are set beside each other?
                         (one call, over everything)

Each pass consumes the pass before it, which is what makes the output cohere:
a panel reading can say "the interlace family that also frames seven other
panels", because the cluster brief established that family first.

There is also a **single-call path** (`direct_synthesis`, `--stage direct`)
that joins the analysis already on disk — motif labels and descriptions,
cluster membership and statistics, recovered layout — and asks for the
interpretation in one request.  Prefer it by default: at the corpus sizes this
pipeline produces, the whole join is tens of thousands of tokens, so the three
passes above buy depth per panel and a look at the actual images, at roughly
one call per cluster plus one per panel.  Start direct; escalate when a
specific panel deserves it.

The deterministic inputs — cluster statistics from the embeddings, spatial
structure from `panel_art/layout.py` — are computed here without the model, so
that what Claude receives is evidence rather than raw pixels alone, and so the
non-LLM half stays inspectable (`--dry-run` in `scripts/interpret_motifs.py`
prints exactly what would be sent).

Requires ANTHROPIC_API_KEY for the LLM passes; the loading, statistics, and
layout passes work without it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

from panel_art.layout import Placement, PanelLayout, analyze_layout, render_layout_text

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    NUMPY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    PIL_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:                                   # pragma: no cover
    ANTHROPIC_AVAILABLE = False


DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"

# Motif crop paths look like  .../motifs_norm/<panel_stem>/<NNN>_<scale>_iou<X>.png
_MOTIF_PATH_RE = re.compile(r"motifs(?:_norm)?/([^/]+)/(\d+)_")


# ══════════════════════════════════════════════════════════════════════════════
# Corpus — the assembled inputs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MotifView:
    """A single motif, flattened across every stage that touched it."""

    key: str                        # "<panel_stem>/<index>"
    panel_stem: str
    index: int
    bbox: dict
    scale: str = "motif"
    cluster: int = -1
    label: str | None = None
    description: str | None = None
    iconography: str | None = None
    notes: str | None = None
    label_source: str | None = None

    # Sources a model wrote. Anything else — including a missing source on an
    # older record — is treated as a person's assertion, which is the safe way
    # round: it protects human work from being overwritten or discounted.
    GENERATED_SOURCES = frozenset({"llm", "cluster-brief", "llm-edited"})

    @property
    def has_text(self) -> bool:
        return bool(self.label or self.description or self.iconography)

    @property
    def is_human_labelled(self) -> bool:
        return bool(self.label) and self.label_source not in self.GENERATED_SOURCES

    def summary_line(self) -> str:
        """One line of text describing this motif for a prompt."""
        bits = [f"#{self.index} ({self.scale})"]
        if self.label:
            bits.append(f'"{self.label}"')
        if self.description:
            bits.append(self.description)
        if self.iconography and self.iconography.lower() not in {"unclear", "unknown", "none"}:
            bits.append(f"iconography: {self.iconography}")
        if self.notes:
            bits.append(f"note: {self.notes}")
        if self.label_source:
            bits.append(f"[label source: {self.label_source}]")
        return " — ".join(bits)


@dataclass
class PanelView:
    stem: str
    png_path: Path
    width: int
    height: int


class Corpus:
    """Panels, motifs, cluster assignments, and (optionally) their embeddings.

    Built either from a live `PipelineState` (notebook use) or straight from
    the analysis directory on disk (CLI use).
    """

    def __init__(
        self,
        panels: dict[str, PanelView],
        motifs: list[MotifView],
        embeddings: "np.ndarray | None" = None,
        embedding_keys: Sequence[str] | None = None,
    ) -> None:
        self.panels = panels
        self.motifs = motifs
        self.embeddings = embeddings
        self.embedding_keys: list[str] = list(embedding_keys or [])
        self._panel_img_cache: dict[str, Any] = {}

        # Index each embedding under both its literal key and its canonical form.
        # Crop paths sometimes carry a "_cropped" stem suffix that PipelineState
        # strips (it derives stems from *_detections.json and only falls back to
        # <stem>_cropped.png for the image), so an exact-match-only index would
        # silently drop those rows — and a missing embedding degrades quietly
        # into "no cohesion, arbitrary exemplars" rather than failing loudly.
        self._embed_row: dict[str, int] = {}
        for row, key in enumerate(self.embedding_keys):
            if not key:
                continue
            self._embed_row.setdefault(key, row)
            self._embed_row.setdefault(canonical_motif_key(key), row)

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_pipeline_state(
        cls,
        state: Any,
        embeddings: "np.ndarray | None" = None,
        embedding_keys: Sequence[str] | None = None,
    ) -> "Corpus":
        """Adapt a `panel_art.pipeline_state.PipelineState` (the notebook's state)."""
        panels = {
            stem: PanelView(stem=stem, png_path=info.png_path,
                            width=info.width, height=info.height)
            for stem, info in state.panels.items()
        }
        motifs = [
            MotifView(
                key=m.motif_key, panel_stem=m.panel_stem, index=m.index,
                bbox=m.bbox, scale=m.scale, cluster=m.cluster,
                label=m.label, description=m.description,
                iconography=m.iconography, notes=m.notes,
                label_source=m.label_source,
            )
            for m in state.motifs if m.included
        ]
        return cls(panels, motifs, embeddings, embedding_keys)

    # ── Queries ───────────────────────────────────────────────────────────

    def motifs_for_panel(self, stem: str) -> list[MotifView]:
        return sorted((m for m in self.motifs if m.panel_stem == stem),
                      key=lambda m: m.index)

    def by_key(self, key: str) -> MotifView | None:
        return next((m for m in self.motifs if m.key == key), None)

    def clusters(self, include_noise: bool = False) -> dict[int, list[MotifView]]:
        """Cluster id → members.  HDBSCAN's noise label (-1) is excluded by default."""
        out: dict[int, list[MotifView]] = {}
        for m in self.motifs:
            if m.cluster < 0 and not include_noise:
                continue
            out.setdefault(m.cluster, []).append(m)
        return dict(sorted(out.items()))

    def panel_stems(self) -> list[str]:
        return sorted(self.panels)

    def embedding_for(self, key: str) -> "np.ndarray | None":
        if self.embeddings is None:
            return None
        row = self._embed_row.get(key)
        if row is None:
            row = self._embed_row.get(canonical_motif_key(key))
        return None if row is None else self.embeddings[row]

    def embedding_coverage(self) -> tuple[int, int]:
        """``(motifs with an embedding, total motifs)`` — surfaces a bad join.

        A low rate here means the embeddings were computed from a different run
        than the approved bboxes on disk, which is worth knowing *before*
        spending a corpus's worth of API calls on degraded statistics.
        """
        if self.embeddings is None:
            return 0, len(self.motifs)
        matched = sum(1 for m in self.motifs if self.embedding_for(m.key) is not None)
        return matched, len(self.motifs)

    # ── Images ────────────────────────────────────────────────────────────

    def panel_image(self, stem: str):
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for image access: pip install pillow")
        if stem not in self._panel_img_cache:
            self._panel_img_cache[stem] = Image.open(self.panels[stem].png_path).convert("RGB")
        return self._panel_img_cache[stem]

    def crop(self, motif: MotifView, padding: int = 4):
        img = self.panel_image(motif.panel_stem)
        iw, ih = img.size
        b = motif.bbox
        return img.crop((
            max(0, b["x"] - padding), max(0, b["y"] - padding),
            min(iw, b["x"] + b["w"] + padding), min(ih, b["y"] + b["h"] + padding),
        ))

    # ── Layout ────────────────────────────────────────────────────────────

    def layout_for(self, stem: str, **kwargs) -> PanelLayout:
        panel = self.panels[stem]
        placements = [
            Placement(key=m.key, index=m.index, bbox=m.bbox,
                      scale=m.scale, cluster=m.cluster, label=m.label)
            for m in self.motifs_for_panel(stem)
        ]
        return analyze_layout(placements, panel.width, panel.height,
                              panel_stem=stem, **kwargs)


# ── Disk loading ─────────────────────────────────────────────────────────────

def parse_motif_key(path_like: str) -> str | None:
    """Recover a ``<panel_stem>/<index>`` key from a motif crop path.

    Label files and embedding manifests both key on crop paths, so this is the
    join between them and the in-memory records.
    """
    match = _MOTIF_PATH_RE.search(path_like)
    if not match:
        return None
    return f"{match.group(1)}/{int(match.group(2))}"


def label_fingerprint(motifs: Sequence["MotifView"]) -> str:
    """A short digest of the labels a cluster brief was written from.

    Stored on the brief so a later run can tell which briefs your annotation
    work has invalidated: relabel a few motifs and only the affected families
    need regenerating, instead of re-running the whole corpus or — worse —
    leaving a brief that quietly contradicts the labels beneath it.
    """
    payload = "|".join(
        f"{m.key}:{m.label or ''}:{m.iconography or ''}:{m.label_source or ''}"
        for m in sorted(motifs, key=lambda m: m.key)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stale_briefs(briefs: dict[str, dict],
                 stats: dict[int, "ClusterStats"]) -> list[int]:
    """Cluster ids whose brief was written against different labels than today's."""
    stale = []
    for cid, st in stats.items():
        brief = briefs.get(str(cid))
        if not brief:
            continue
        recorded = (brief.get("stats") or {}).get("label_fingerprint")
        if recorded and recorded != st.label_fingerprint:
            stale.append(cid)
    return sorted(stale)


def text_of(message: Any) -> str:
    """Concatenate the text blocks of a Claude response.

    Never index `content[0]`. On models where thinking is on by default the
    first block is a thinking block, and `content[0].text` raises
    ``AttributeError: 'ThinkingBlock' object has no attribute 'text'``. Any
    block type that is not text — thinking, tool use, server tool results — is
    skipped here rather than crashing the caller.
    """
    return "".join(
        block.text for block in getattr(message, "content", [])
        if getattr(block, "type", None) == "text"
    ).strip()


def canonical_motif_key(key: str) -> str:
    """Normalise a motif key so crop paths and PipelineState records agree.

    Panel crops are written as either ``<stem>.png`` or ``<stem>_cropped.png``,
    but the detections JSON is always ``<stem>_detections.json`` — so the same
    motif can be keyed either way depending on which side produced the key.
    """
    stem, _, index = key.rpartition("/")
    if not stem:
        return key
    return f"{stem.removesuffix('_cropped')}/{index}"


def load_corpus(
    analysis_dir: Path,
    embeddings_path: Path | None = None,
    paths_path: Path | None = None,
    clusters_path: Path | None = None,
) -> Corpus:
    """Load panels, approved bboxes, labels, clusters, and embeddings from disk.

    Mirrors the layout `PIPELINE.md` documents:
        analysis/panels/<stem>.png
        analysis/annotated/<stem>_{approved,detections}.json
        analysis/motif_labels.json
    """
    from panel_art.pipeline_state import PipelineState

    analysis_dir = Path(analysis_dir)
    state = PipelineState()
    # clusters.json is the notebook's Save Clusters output and the authoritative
    # record of the assignments — cluster ids inside label records only cover
    # motifs that happen to have been labelled, which is a small minority.
    default_clusters = analysis_dir / "clusters.json"
    state.load_from_disk(
        annotated_dir=analysis_dir / "annotated",
        panels_dir=analysis_dir / "panels",
        labels_path=analysis_dir / "motif_labels.json",
        clusters_path=default_clusters if default_clusters.exists() else None,
    )

    embeddings, embedding_keys = _load_embeddings(embeddings_path, paths_path)
    corpus = Corpus.from_pipeline_state(state, embeddings, embedding_keys)

    # An explicit --clusters file overrides whatever was just loaded.
    if clusters_path:
        _apply_cluster_overrides(corpus, Path(clusters_path))

    return corpus


def _load_embeddings(
    embeddings_path: Path | None,
    paths_path: Path | None,
) -> tuple["np.ndarray | None", list[str]]:
    if not embeddings_path or not paths_path:
        return None, []
    if not NUMPY_AVAILABLE:
        raise RuntimeError("numpy is required to load embeddings")

    embeddings = np.load(Path(embeddings_path))
    raw_paths = Path(paths_path).read_text(encoding="utf-8").splitlines()
    keys = [parse_motif_key(p) or "" for p in raw_paths if p.strip()]

    if len(keys) != len(embeddings):
        raise ValueError(
            f"embeddings/paths mismatch: {len(embeddings)} vectors but {len(keys)} paths "
            f"({embeddings_path.name} vs {paths_path.name})"
        )
    return embeddings, keys


def _apply_cluster_overrides(corpus: Corpus, clusters_path: Path) -> int:
    """Apply a ``{motif_key_or_crop_path: cluster_id}`` mapping over the corpus."""
    raw = json.loads(clusters_path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for key, value in raw.items():
        cluster = value.get("cluster") if isinstance(value, dict) else value
        if cluster is None:
            continue
        mapping[parse_motif_key(key) or key] = int(cluster)

    applied = 0
    for motif in corpus.motifs:
        if motif.key in mapping:
            motif.cluster = mapping[motif.key]
            applied += 1
    return applied


# ══════════════════════════════════════════════════════════════════════════════
# Cluster statistics — the embedding half of the join, no LLM involved
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterStats:
    cluster_id: int
    size: int
    panel_counts: dict[str, int] = field(default_factory=dict)
    scale_counts: dict[str, int] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)
    described: int = 0
    cohesion: float | None = None                  # mean pairwise cosine similarity
    exemplar_keys: list[str] = field(default_factory=list)
    neighbour_clusters: list[tuple[int, float]] = field(default_factory=list)
    human_labelled: int = 0
    label_fingerprint: str = ""                    # see label_fingerprint()

    @property
    def panel_spread(self) -> int:
        return len(self.panel_counts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "panel_spread": self.panel_spread,
            "panel_counts": self.panel_counts,
            "scale_counts": self.scale_counts,
            "label_counts": self.label_counts,
            "described": self.described,
            "cohesion": round(self.cohesion, 4) if self.cohesion is not None else None,
            "human_labelled": self.human_labelled,
            "label_fingerprint": self.label_fingerprint,
            "exemplar_keys": self.exemplar_keys,
            "neighbour_clusters": [[c, round(s, 4)] for c, s in self.neighbour_clusters],
        }


def compute_cluster_stats(
    corpus: Corpus,
    exemplars: int = 4,
    neighbours: int = 3,
) -> dict[int, ClusterStats]:
    """Summarise each cluster: spread, composition, cohesion, exemplars, kin.

    Cohesion is the mean pairwise cosine similarity within the cluster, and
    exemplars are the members closest to the cluster centroid — so what goes
    to the model is the family's centre of gravity rather than an arbitrary
    first-four sample.  `neighbour_clusters` names the nearest *other*
    clusters by centroid similarity, which is what lets an interpretation say
    two families are variants of one another.

    Falls back to label/count statistics alone when embeddings are absent.
    """
    grouped = corpus.clusters()
    stats: dict[int, ClusterStats] = {}
    centroids: dict[int, Any] = {}

    for cid, members in grouped.items():
        st = ClusterStats(
            cluster_id=cid,
            size=len(members),
            panel_counts=dict(Counter(m.panel_stem for m in members).most_common()),
            scale_counts=dict(Counter(m.scale for m in members).most_common()),
            label_counts=dict(Counter(m.label for m in members if m.label).most_common()),
            described=sum(1 for m in members if m.has_text),
            human_labelled=sum(1 for m in members if m.is_human_labelled),
            label_fingerprint=label_fingerprint(members),
        )

        vectors, keys = _cluster_vectors(corpus, members)
        if vectors is not None and len(vectors) >= 1:
            centroid = vectors.mean(axis=0)
            centroid_n = _normalise(centroid)
            centroids[cid] = centroid_n

            sims_to_centroid = _normalise_rows(vectors) @ centroid_n
            order = list(np.argsort(-sims_to_centroid))
            st.exemplar_keys = [keys[i] for i in order[:exemplars]]

            if len(vectors) >= 2:
                st.cohesion = float(_mean_pairwise_cosine(vectors))
        else:
            st.exemplar_keys = [m.key for m in members[:exemplars]]

        stats[cid] = st

    _attach_cluster_neighbours(stats, centroids, neighbours)
    return stats


def _cluster_vectors(corpus: Corpus, members: Sequence[MotifView]):
    if corpus.embeddings is None or not NUMPY_AVAILABLE:
        return None, []
    rows, keys = [], []
    for m in members:
        vec = corpus.embedding_for(m.key)
        if vec is not None:
            rows.append(vec)
            keys.append(m.key)
    if not rows:
        return None, []
    return np.asarray(rows, dtype=float), keys


def _normalise(vec):
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _normalise_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _mean_pairwise_cosine(vectors) -> float:
    unit = _normalise_rows(vectors)
    sims = unit @ unit.T
    n = len(unit)
    # Mean of the strict upper triangle — excludes the self-similarity diagonal.
    return float((sims.sum() - np.trace(sims)) / (n * (n - 1)))


def _attach_cluster_neighbours(
    stats: dict[int, ClusterStats],
    centroids: dict[int, Any],
    neighbours: int,
) -> None:
    if len(centroids) < 2:
        return
    ids = sorted(centroids)
    matrix = np.asarray([centroids[i] for i in ids])
    sims = matrix @ matrix.T
    for row, cid in enumerate(ids):
        ranked = sorted(
            ((ids[col], float(sims[row][col])) for col in range(len(ids)) if col != row),
            key=lambda t: -t[1],
        )
        stats[cid].neighbour_clusters = ranked[:neighbours]


# ══════════════════════════════════════════════════════════════════════════════
# Prompt construction — separated from the API call so it can be inspected
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an art historian specialising in West African visual culture, with deep \
expertise in Yoruba carved wooden objects — door panels (ilekun), Ifa divination \
boards (opon Ifa), house posts, and Benin relief plaques — and in the early \
20th-century record of them, especially the Frobenius expeditions of 1910–1912.

You are reading the output of a computer-vision pipeline: motifs segmented from \
archival photographs and pen-and-ink survey drawings, embedded with CLIP, and \
clustered with HDBSCAN. The clusters are visual families discovered from the \
data, not established iconographic categories, and the labels attached to them \
are a mixture of human and model-generated guesses.

Work from what is visually present and from the evidence you are given. Where \
you draw on knowledge of Yoruba iconography, mythology, or the Frobenius record, \
say so explicitly and mark how confident you are. Distinguish clearly between \
(a) what the image shows, (b) what the pipeline's grouping implies, and (c) what \
you are inferring from cultural knowledge. Segmentation errors are common — a \
detection may be a fragment, a duplicate, or an artefact of the photograph \
rather than a carved element — so say when a reading depends on a detection you \
doubt. Do not invent provenance, dates, or attributions that were not given to you.\
"""

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "2-4 words, snake_case, naming the visual family "
                           "(e.g. interlaced_knotwork, standing_attendant_figure)",
        },
        "visual_definition": {
            "type": "string",
            "description": "What every member of this family has in common, visually.",
        },
        "variation": {
            "type": "string",
            "description": "How members differ from one another, and whether the "
                           "cluster looks like one family or several merged.",
        },
        "distribution_note": {
            "type": "string",
            "description": "What the spread across panels and scales suggests — "
                           "a border element, a narrative subject, an artefact.",
        },
        "iconographic_reading": {
            "type": "string",
            "description": "Probable symbolic or iconographic significance, with "
                           "the basis stated. Use 'unclear' rather than guessing.",
        },
        "relation_to_neighbours": {
            "type": "string",
            "description": "How this family relates to the embedding-adjacent "
                           "clusters listed, if the relation is meaningful.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What would have to be checked to firm this up.",
        },
    },
    "required": [
        "name", "visual_definition", "variation", "distribution_note",
        "iconographic_reading", "relation_to_neighbours", "confidence",
        "open_questions",
    ],
    "additionalProperties": False,
}

PANEL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short descriptive title for the panel."},
        "summary": {"type": "string", "description": "Two or three sentences: what this panel is."},
        "register_readings": {
            "type": "array",
            "description": "One entry per register, top to bottom.",
            "items": {
                "type": "object",
                "properties": {
                    "register": {"type": "integer"},
                    "motifs": {"type": "array", "items": {"type": "integer"},
                               "description": "Motif indices in this register, left to right."},
                    "reading": {"type": "string",
                                "description": "What this band depicts and how it is organised."},
                },
                "required": ["register", "motifs", "reading"],
                "additionalProperties": False,
            },
        },
        "composition": {
            "type": "string",
            "description": "How the registers, symmetry, nesting, and field relate — "
                           "the panel's overall organisation.",
        },
        "narrative": {
            "type": "string",
            "description": "The reading of the panel as a whole: what it may be "
                           "recording, telling, or asserting.",
        },
        "cross_panel_links": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Connections to the wider corpus via shared motif families.",
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Detections or readings you doubt, and why.",
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "title", "summary", "register_readings", "composition", "narrative",
        "cross_panel_links", "uncertainties", "confidence",
    ],
    "additionalProperties": False,
}


def build_cluster_prompt(stats: ClusterStats, members: Sequence[MotifView],
                         exemplar_count: int) -> str:
    """Text half of a cluster-brief request (images are attached alongside)."""
    lines: list[str] = []
    lines.append(f"MOTIF FAMILY: cluster {stats.cluster_id}")
    lines.append(f"Members: {stats.size} motifs across {stats.panel_spread} panel(s).")
    lines.append(f"Scale mix: {_fmt_counts(stats.scale_counts)}")
    lines.append(f"Panel distribution: {_fmt_counts(stats.panel_counts, limit=8)}")

    if stats.cohesion is not None:
        lines.append(
            f"Embedding cohesion: {stats.cohesion:.3f} mean pairwise cosine similarity "
            "(higher means the family is visually tight; below ~0.5 suggests the "
            "cluster may have merged distinct things)."
        )
    if stats.neighbour_clusters:
        kin = ", ".join(f"cluster {c} (cos {s:.3f})" for c, s in stats.neighbour_clusters)
        lines.append(f"Nearest other families in embedding space: {kin}")
    if stats.label_counts:
        lines.append(f"Existing labels on members: {_fmt_counts(stats.label_counts, limit=8)}")

    lines.append("")
    lines.append(f"The {exemplar_count} images attached are the members closest to the "
                 "cluster centroid — the most typical examples of this family.")

    # Human annotations first and marked as such: they are the accumulating
    # signal this pipeline is meant to sharpen against, and a model-generated
    # placeholder should never be weighed the same as a person's assertion.
    described = [m for m in members if m.has_text]
    human = [m for m in described if m.is_human_labelled]
    generated = [m for m in described if not m.is_human_labelled]

    if human:
        lines.append("")
        lines.append("ANNOTATED BY A PERSON — treat these as the strongest evidence "
                     "available, and prefer their vocabulary:")
        for m in human[:20]:
            lines.append(f"  {m.panel_stem} {m.summary_line()}")
        if len(human) > 20:
            lines.append(f"  … and {len(human) - 20} more")

    if generated:
        lines.append("")
        lines.append("MODEL-GENERATED PLACEHOLDERS — provisional, may be wrong, and "
                     "safe to contradict if the images say otherwise:")
        for m in generated[:12]:
            lines.append(f"  {m.panel_stem} {m.summary_line()}")
        if len(generated) > 12:
            lines.append(f"  … and {len(generated) - 12} more")

    if not described:
        lines.append("")
        lines.append("No per-motif notes exist for this family yet — work from the images.")

    lines.append("")
    lines.append(
        "Characterise this family as a recurring element of the carving vocabulary. "
        "Judge whether it is one coherent family or several merged, and say what "
        "its distribution across panels implies about its function."
    )
    return "\n".join(lines)


def build_panel_prompt(
    layout: PanelLayout,
    motifs: Sequence[MotifView],
    cluster_context: Sequence[str],
) -> str:
    """Text half of a panel-reading request (the annotated panel image is attached)."""
    lines: list[str] = []
    lines.append("Read this carved panel from the Frobenius archive as a whole composition.")
    lines.append("")
    lines.append("SPATIAL STRUCTURE (computed from the detection geometry):")
    lines.append(render_layout_text(layout))

    lines.append("")
    lines.append("MOTIFS ON THIS PANEL:")
    for m in motifs:
        cluster = f"cluster {m.cluster}" if m.cluster >= 0 else "unclustered"
        lines.append(f"  {m.summary_line()} [{cluster}]")

    if cluster_context:
        lines.append("")
        lines.append("MOTIF FAMILIES PRESENT, AS CHARACTERISED ACROSS THE WHOLE CORPUS:")
        lines.extend(f"  {line}" for line in cluster_context)

    lines.append("")
    lines.append(
        "The attached image is the panel with every detection outlined and numbered "
        "to match the indices above. Read the panel register by register in the "
        "reading order given, then as a single composition. Use the spatial "
        "structure — which motifs share a register, what is centred, what is "
        "mirrored, what encloses what — as evidence for how the panel is organised, "
        "and use the corpus-wide family descriptions to say what recurs here from "
        "elsewhere in the collection versus what is particular to this panel."
    )
    return "\n".join(lines)


def build_corpus_prompt(
    cluster_briefs: dict[str, dict],
    panel_readings: dict[str, dict],
    corpus_stats: dict[str, Any],
) -> str:
    """Text for the final synthesis pass — no images, purely a join of the passes above."""
    lines: list[str] = []
    lines.append(
        "Below are the results of a motif-level analysis of a collection of carved "
        "West African panels: first every recurring motif family found across the "
        "collection, then a reading of each individual panel. Synthesise them into "
        "one interpretation of the collection."
    )
    lines.append("")
    lines.append("CORPUS SCALE:")
    lines.append(
        f"  {corpus_stats.get('panels', 0)} panels, {corpus_stats.get('motifs', 0)} "
        f"approved motif detections, {corpus_stats.get('clusters', 0)} motif families "
        f"({corpus_stats.get('unclustered', 0)} detections left unclustered by HDBSCAN)."
    )

    lines.append("")
    lines.append("═══ MOTIF FAMILIES ═══")
    for cid, brief in cluster_briefs.items():
        lines.append("")
        lines.append(f"— Cluster {cid}: {brief.get('name', '(unnamed)')} "
                     f"[confidence: {brief.get('confidence', '?')}]")
        lines.append(f"  Visual definition: {brief.get('visual_definition', '')}")
        lines.append(f"  Variation: {brief.get('variation', '')}")
        lines.append(f"  Distribution: {brief.get('distribution_note', '')}")
        lines.append(f"  Iconography: {brief.get('iconographic_reading', '')}")
        if brief.get("relation_to_neighbours"):
            lines.append(f"  Related families: {brief['relation_to_neighbours']}")

    lines.append("")
    lines.append("═══ PANEL READINGS ═══")
    for stem, reading in panel_readings.items():
        lines.append("")
        lines.append(f"— {stem}: {reading.get('title', '')} "
                     f"[confidence: {reading.get('confidence', '?')}]")
        lines.append(f"  {reading.get('summary', '')}")
        lines.append(f"  Composition: {reading.get('composition', '')}")
        lines.append(f"  Narrative: {reading.get('narrative', '')}")
        for link in reading.get("cross_panel_links", []) or []:
            lines.append(f"  Link: {link}")

    lines.append("")
    lines.append(
        "Write the synthesis as a Markdown essay. Cover: what visual vocabulary "
        "this collection shares and how it is deployed; which families are "
        "structural (borders, framing, ground) versus which carry subject matter; "
        "what recurring compositional grammar the panel readings have in common "
        "(register structure, symmetry, centring, scale hierarchy); what the "
        "cross-panel recurrences suggest about workshops, regions, or a shared "
        "repertoire; and what this does and does not support saying about these "
        "carvings as a communicative system.\n\n"
        "Be specific — cite cluster names and panel identifiers where they carry "
        "the argument. Keep the evidential distinction visible throughout: what is "
        "observed, what the clustering implies, what is inferred. End with a "
        "section on the weakest points in the analysis and what evidence would "
        "settle them. Do not pad the essay with restatement; length should follow "
        "from what the evidence supports."
    )
    return "\n".join(lines)


def build_direct_prompt(
    corpus: "Corpus",
    stats: dict[int, "ClusterStats"],
    max_panels: int | None = None,
) -> str:
    """Join everything already on disk into one prompt for a single call.

    The cheap path. Where the three-pass flow spends a call per cluster and per
    panel to *generate* intermediate text — and to put images in front of the
    model — this assembles the analysis that already exists (motif labels and
    descriptions, cluster membership and statistics, recovered layout) and asks
    for the interpretation in one go.

    At the corpus sizes this pipeline produces, the whole join is tens of
    thousands of tokens, so it fits comfortably in one request. What it gives
    up is depth per panel and any actual look at the carvings: the model works
    from the pipeline's description of the images, never the images.
    """
    lines: list[str] = []
    scale = corpus_scale(corpus)

    lines.append(
        "Below is a complete motif-level analysis of a collection of carved West "
        "African panels: the recurring motif families found across the collection, "
        "then every panel with its motifs and the spatial structure recovered from "
        "the detection geometry. Interpret the collection."
    )
    lines.append("")
    lines.append(
        f"CORPUS: {scale['panels']} panels, {scale['motifs']} approved motif "
        f"detections, {scale['clusters']} motif families "
        f"({scale['unclustered']} unclustered), {scale['labelled']} labelled."
    )

    lines.append("")
    lines.append("═══ MOTIF FAMILIES (visual clusters, discovered from CLIP embeddings) ═══")
    grouped = corpus.clusters()
    for cid in sorted(grouped):
        st = stats.get(cid)
        members = grouped[cid]
        lines.append("")
        header = f"— Cluster {cid}: {len(members)} motifs"
        if st:
            header += f" across {st.panel_spread} panel(s)"
            if st.cohesion is not None:
                header += f", cohesion {st.cohesion:.3f}"
        lines.append(header)
        if st and st.label_counts:
            lines.append(f"  Labels: {_fmt_counts(st.label_counts, limit=8)}")
        if st and st.neighbour_clusters:
            kin = ", ".join(f"cluster {c} (cos {s:.3f})" for c, s in st.neighbour_clusters)
            lines.append(f"  Nearest other families: {kin}")
        for m in members:
            if m.has_text:
                lines.append(f"  {m.panel_stem} {m.summary_line()}")

    lines.append("")
    lines.append("═══ PANELS ═══")
    stems = corpus.panel_stems()
    if max_panels is not None:
        stems = stems[:max_panels]
    for stem in stems:
        motifs = corpus.motifs_for_panel(stem)
        if not motifs:
            continue
        lines.append("")
        lines.append(f"── {stem} ──")
        lines.append(render_layout_text(corpus.layout_for(stem), max_relations=12))
        lines.append("Motifs:")
        for m in motifs:
            cluster = f"cluster {m.cluster}" if m.cluster >= 0 else "unclustered"
            lines.append(f"  {m.summary_line()} [{cluster}]")

    lines.append("")
    lines.append(
        "Write the interpretation as a Markdown essay. Cover: what visual "
        "vocabulary this collection shares and how it is deployed; which families "
        "are structural (borders, framing, ground) versus which carry subject "
        "matter; the compositional grammar the panels have in common (register "
        "structure, symmetry, centring, scale hierarchy); what the cross-panel "
        "recurrences suggest about workshops, regions, or a shared repertoire; "
        "and read the most legible individual panels in enough detail to show the "
        "grammar working.\n\n"
        "Be specific — cite cluster numbers and panel identifiers where they carry "
        "the argument. Keep the evidential distinction visible throughout: what the "
        "pipeline observed, what the clustering implies, what you are inferring "
        "from cultural knowledge. Note that you are working from the pipeline's "
        "descriptions and geometry, not from the images themselves, and say where "
        "that limits a reading. End with the weakest points in the analysis and "
        "what evidence would settle them."
    )
    return "\n".join(lines)


def _fmt_counts(counts: dict[Any, int], limit: int = 6) -> str:
    if not counts:
        return "(none)"
    items = list(counts.items())[:limit]
    text = ", ".join(f"{k}×{v}" for k, v in items)
    if len(counts) > limit:
        text += f", … (+{len(counts) - limit} more)"
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ══════════════════════════════════════════════════════════════════════════════

def annotate_panel(image, layout: PanelLayout, width: int = 4):
    """Draw every detection on the panel, numbered to match the prompt's indices."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required: pip install pillow")

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    # Field detections first, so the smaller motifs draw over them.
    ordered = sorted(layout.placements, key=lambda p: -(p.rw * p.rh))
    for p in ordered:
        b = p.bbox
        colour = (255, 160, 0) if p.is_field else (0, 255, 64)
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]],
                       outline=colour, width=width)
        tag = str(p.index)
        tx, ty = b["x"] + 4, b["y"] + 4
        # Dark plate behind the number so it stays readable on light carving.
        draw.rectangle([tx - 2, ty - 2, tx + 8 * len(tag) + 2, ty + 14], fill=(0, 0, 0))
        draw.text((tx, ty), tag, fill=colour)
    return canvas


def encode_image(image, max_dim: int = 1024) -> dict:
    """Encode a PIL image as an Anthropic image content block."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required: pip install pillow")

    img = image.copy().convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8"),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Interpreter — the three Claude passes
# ══════════════════════════════════════════════════════════════════════════════

class Interpreter:
    """Drives the cluster → panel → corpus passes against the Claude API.

    Every call streams, because a corpus synthesis at high effort can run well
    past the SDK's non-streaming timeout; `.get_final_message()` gives back the
    complete message either way.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        api_key: str | None = None,
        heartbeat: float = 15.0,
        on_progress: Any = None,
    ) -> None:
        if client is None:
            if not ANTHROPIC_AVAILABLE:
                raise RuntimeError("anthropic is not installed: pip install anthropic")
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Run with --dry-run to inspect the "
                    "prompts and layouts without calling the API."
                )
            client = anthropic.Anthropic(api_key=key)
        self.client = client
        self.model = model
        self.effort = effort
        self.heartbeat = heartbeat
        self.on_progress = on_progress

    # ── Low-level call ────────────────────────────────────────────────────

    def _message(
        self,
        content: list[dict],
        max_tokens: int,
        schema: dict | None = None,
    ) -> Any:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config=output_config,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            if self.on_progress:
                self._drain_with_heartbeat(stream)
            return stream.get_final_message()

    def _drain_with_heartbeat(self, stream: Any) -> None:
        """Consume the event stream, reporting liveness while the model works.

        Adaptive thinking at high effort can run for minutes before the first
        text arrives, and `thinking.display` defaults to "omitted" on current
        models — so thinking blocks stream with empty text and the connection
        looks dead to anyone watching. Ticking on raw event counts proves the
        socket is alive without depending on the shape of any event.
        """
        started = last = time.monotonic()
        events = 0
        phase = "starting"
        for event in stream:
            events += 1
            kind = getattr(event, "type", "")
            if kind == "content_block_start":
                block_type = getattr(getattr(event, "content_block", None), "type", "")
                if block_type:
                    phase = "thinking" if block_type == "thinking" else f"writing {block_type}"
            now = time.monotonic()
            if now - last >= self.heartbeat:
                self.on_progress(f"{phase}… {now - started:.0f}s, {events} events")
                last = now

    _text_of = staticmethod(text_of)

    def _json_call(self, content: list[dict], max_tokens: int, schema: dict) -> dict:
        message = self._message(content, max_tokens, schema)
        if getattr(message, "stop_reason", None) == "refusal":
            raise RuntimeError(
                "Claude declined this request "
                f"({getattr(getattr(message, 'stop_details', None), 'category', 'unknown')})"
            )
        raw = self._text_of(message)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"could not parse structured response: {raw[:300]}") from exc

    # ── Pass 1: cluster brief ─────────────────────────────────────────────

    def cluster_brief(
        self,
        corpus: Corpus,
        stats: ClusterStats,
        max_tokens: int = 6000,
        max_dim: int = 512,
    ) -> dict:
        members = corpus.clusters().get(stats.cluster_id, [])
        content: list[dict] = []

        shown = 0
        for key in stats.exemplar_keys:
            motif = corpus.by_key(key)
            if motif is None:
                continue
            content.append({"type": "text",
                            "text": f"Exemplar {shown + 1} — {motif.panel_stem} #{motif.index}:"})
            content.append(encode_image(corpus.crop(motif), max_dim=max_dim))
            shown += 1

        content.append({"type": "text",
                        "text": build_cluster_prompt(stats, members, shown)})
        brief = self._json_call(content, max_tokens, CLUSTER_SCHEMA)
        brief["cluster_id"] = stats.cluster_id
        brief["stats"] = stats.as_dict()
        brief["generated_at"] = _now()
        brief["model"] = self.model
        return brief

    # ── Pass 2: panel reading ─────────────────────────────────────────────

    def panel_reading(
        self,
        corpus: Corpus,
        stem: str,
        cluster_briefs: dict[str, dict] | None = None,
        cluster_stats: dict[int, ClusterStats] | None = None,
        max_tokens: int = 16000,
        max_dim: int = 1024,
    ) -> dict:
        layout = corpus.layout_for(stem)
        motifs = corpus.motifs_for_panel(stem)
        context = cluster_context_lines(motifs, cluster_briefs or {}, cluster_stats or {})

        annotated = annotate_panel(corpus.panel_image(stem), layout)
        content: list[dict] = [
            {"type": "text", "text": "Panel with every detection outlined and numbered:"},
            encode_image(annotated, max_dim=max_dim),
            {"type": "text", "text": build_panel_prompt(layout, motifs, context)},
        ]

        reading = self._json_call(content, max_tokens, PANEL_SCHEMA)
        reading["panel_stem"] = stem
        reading["layout"] = layout.as_dict()
        reading["generated_at"] = _now()
        reading["model"] = self.model
        return reading

    # ── Single-call alternative to passes 1-3 ─────────────────────────────

    def direct_synthesis(
        self,
        corpus: Corpus,
        stats: dict[int, ClusterStats],
        max_tokens: int = 32000,
        max_panels: int | None = None,
    ) -> str:
        """One call: join the existing analysis and return the interpretation."""
        prompt = build_direct_prompt(corpus, stats, max_panels=max_panels)
        message = self._message([{"type": "text", "text": prompt}], max_tokens)
        if getattr(message, "stop_reason", None) == "refusal":
            raise RuntimeError("Claude declined the synthesis request")
        return self._text_of(message)

    # ── Pass 3: corpus synthesis ──────────────────────────────────────────

    def corpus_synthesis(
        self,
        cluster_briefs: dict[str, dict],
        panel_readings: dict[str, dict],
        corpus_stats: dict[str, Any],
        max_tokens: int = 32000,
    ) -> str:
        prompt = build_corpus_prompt(cluster_briefs, panel_readings, corpus_stats)
        message = self._message([{"type": "text", "text": prompt}], max_tokens)
        if getattr(message, "stop_reason", None) == "refusal":
            raise RuntimeError("Claude declined the corpus synthesis request")
        return self._text_of(message)


def cluster_context_lines(
    motifs: Sequence[MotifView],
    cluster_briefs: dict[str, dict],
    cluster_stats: dict[int, ClusterStats],
) -> list[str]:
    """One line per motif family present on a panel, drawn from the pass-1 briefs.

    This is the seam that makes a panel reading corpus-aware rather than local.
    """
    lines: list[str] = []
    for cid in sorted({m.cluster for m in motifs if m.cluster >= 0}):
        brief = cluster_briefs.get(str(cid)) or cluster_briefs.get(cid) or {}
        stats = cluster_stats.get(cid)

        name = brief.get("name") or f"cluster_{cid}"
        parts = [f"Cluster {cid} ({name})"]
        if stats:
            others = max(stats.panel_spread - 1, 0)
            parts.append(
                f"{stats.size} members across {stats.panel_spread} panel(s)"
                + (f"; also appears on {others} other panel(s)" if others else
                   "; unique to this panel")
            )
        if brief.get("visual_definition"):
            parts.append(brief["visual_definition"])
        if brief.get("iconographic_reading"):
            parts.append(f"iconography: {brief['iconographic_reading']}")
        lines.append(" — ".join(parts))
    return lines


def corpus_scale(corpus: Corpus) -> dict[str, Any]:
    clustered = corpus.clusters()
    return {
        "panels": len(corpus.panels),
        "motifs": len(corpus.motifs),
        "clusters": len(clustered),
        "unclustered": sum(1 for m in corpus.motifs if m.cluster < 0),
        "labelled": sum(1 for m in corpus.motifs if m.label),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

class InterpretationStore:
    """Reads and writes the `interpretation/` output tree.

    Layout::

        interpretation/
          clusters.json          cluster id → brief
          layouts/<stem>.json    deterministic geometry (no LLM)
          panels/<stem>.json     structured panel reading
          panels/<stem>.md       the same reading, rendered for humans
          corpus.md              final synthesis essay
    """

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = Path(out_dir)
        self.clusters_path = self.out_dir / "clusters.json"
        self.panels_dir = self.out_dir / "panels"
        self.layouts_dir = self.out_dir / "layouts"
        self.corpus_path = self.out_dir / "corpus.md"

    def ensure_dirs(self) -> None:
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.layouts_dir.mkdir(parents=True, exist_ok=True)

    # ── Clusters ──────────────────────────────────────────────────────────

    def load_clusters(self) -> dict[str, dict]:
        if not self.clusters_path.exists():
            return {}
        return json.loads(self.clusters_path.read_text(encoding="utf-8"))

    def save_clusters(self, briefs: dict[str, dict]) -> Path:
        self.ensure_dirs()
        ordered = {k: briefs[k] for k in sorted(briefs, key=lambda s: int(s))}
        self.clusters_path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.clusters_path

    # ── Panels ────────────────────────────────────────────────────────────

    def load_panels(self) -> dict[str, dict]:
        if not self.panels_dir.exists():
            return {}
        out = {}
        for path in sorted(self.panels_dir.glob("*.json")):
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        return out

    def save_panel(self, stem: str, reading: dict) -> Path:
        self.ensure_dirs()
        path = self.panels_dir / f"{stem}.json"
        path.write_text(json.dumps(reading, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.panels_dir / f"{stem}.md").write_text(
            render_panel_markdown(reading), encoding="utf-8")
        return path

    def save_layout(self, stem: str, layout: PanelLayout) -> Path:
        self.ensure_dirs()
        path = self.layouts_dir / f"{stem}.json"
        path.write_text(json.dumps(layout.as_dict(), indent=2), encoding="utf-8")
        return path

    # ── Corpus ────────────────────────────────────────────────────────────

    def save_corpus(self, markdown: str) -> Path:
        self.ensure_dirs()
        self.corpus_path.write_text(markdown, encoding="utf-8")
        return self.corpus_path


# ══════════════════════════════════════════════════════════════════════════════
# Markdown rendering
# ══════════════════════════════════════════════════════════════════════════════

def render_panel_markdown(reading: dict) -> str:
    """Render a structured panel reading as readable Markdown."""
    lines: list[str] = []
    lines.append(f"# {reading.get('title', reading.get('panel_stem', 'Panel'))}")
    lines.append("")
    lines.append(f"*{reading.get('panel_stem', '')}* — confidence: "
                 f"**{reading.get('confidence', 'unknown')}**")
    lines.append("")
    lines.append(reading.get("summary", ""))

    registers = reading.get("register_readings") or []
    if registers:
        lines.append("")
        lines.append("## Registers")
        for reg in registers:
            motifs = ", ".join(f"#{i}" for i in reg.get("motifs", []))
            lines.append("")
            lines.append(f"### Register {reg.get('register')} ({motifs or 'no motifs'})")
            lines.append("")
            lines.append(reg.get("reading", ""))

    if reading.get("composition"):
        lines.append("")
        lines.append("## Composition")
        lines.append("")
        lines.append(reading["composition"])

    if reading.get("narrative"):
        lines.append("")
        lines.append("## Reading")
        lines.append("")
        lines.append(reading["narrative"])

    links = reading.get("cross_panel_links") or []
    if links:
        lines.append("")
        lines.append("## Links to the wider corpus")
        lines.append("")
        lines.extend(f"- {link}" for link in links)

    uncertainties = reading.get("uncertainties") or []
    if uncertainties:
        lines.append("")
        lines.append("## Uncertainties")
        lines.append("")
        lines.extend(f"- {item}" for item in uncertainties)

    lines.append("")
    lines.append("---")
    lines.append(f"Generated {reading.get('generated_at', '')} "
                 f"with `{reading.get('model', '')}`.")
    return "\n".join(lines)


def render_clusters_markdown(briefs: dict[str, dict]) -> str:
    """Render all cluster briefs as a single Markdown reference sheet."""
    lines = ["# Motif families", ""]
    for cid in sorted(briefs, key=lambda s: int(s)):
        brief = briefs[cid]
        stats = brief.get("stats", {})
        lines.append(f"## Cluster {cid} — {brief.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(
            f"{stats.get('size', '?')} motifs across {stats.get('panel_spread', '?')} panel(s)"
            + (f", cohesion {stats['cohesion']:.3f}" if stats.get("cohesion") is not None else "")
            + f" — confidence: **{brief.get('confidence', 'unknown')}**"
        )
        lines.append("")
        for heading, key in (
            ("Visual definition", "visual_definition"),
            ("Variation", "variation"),
            ("Distribution", "distribution_note"),
            ("Iconography", "iconographic_reading"),
            ("Related families", "relation_to_neighbours"),
        ):
            if brief.get(key):
                lines.append(f"**{heading}.** {brief[key]}")
                lines.append("")
        questions = brief.get("open_questions") or []
        if questions:
            lines.append("Open questions:")
            lines.extend(f"- {q}" for q in questions)
            lines.append("")
    return "\n".join(lines)
