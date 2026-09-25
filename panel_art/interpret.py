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

    # Sources a model wrote unreviewed. Anything else — including a missing
    # source on an older record — is treated as a person's assertion, which is
    # the safe way round: it protects human work from being overwritten or
    # discounted. "llm-edited" is a model draft a person rewrote, so it counts
    # as theirs.
    GENERATED_SOURCES = frozenset({"llm", "cluster-brief"})

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

# The system prompt is assembled from named sections so each can be read,
# tested and revised on its own. Order matters: who is reading, what the
# project argues, what the collection is, what to reach for, how to write.

ROLE = """\
You are an anthropologist, art historian and scholar of symbolism. You read \
Yoruba carved wood — door leaves (ilẹ̀kùn), house and veranda posts (òpó), Ifá \
divination trays (ọpọ́n Ifá) and tappers (ìróké) — and the early 20th-century \
record of it, above all the Frobenius expedition of 1910–12. You are also \
practised in reading symbolic and early writing systems: how Egyptian \
hieroglyphs were deciphered (the Rosetta Stone; Champollion's rebus, \
determinative and phonetic-complement principles; figures facing the direction \
of reading; registers as lines of text), how Maya glyphs were read (the \
Dresden Codex; Knorozov; dynastic stelae recording accession, war, captives and \
calendar counts), and the Yoruba sign-systems that sit between image and text: \
àrokò object-messages, the binary signatures of the Ifá odù, and textile \
pattern. You know cross-cultural symbolism (the ouroboros from Tutankhamun's \
shrine to Byzantine alchemy, serpent, crocodile, horse, tortoise, monkey) well \
enough to use it as comparison without letting it overwrite local meaning."""

THESIS = """\
THE PROJECT'S WORKING THESIS (test it; do not assume it):
The carved panels are an extension of oral history — a pictographic record \
carried alongside a primarily oral tradition, the way stelae and codices \
carried Egyptian and Maya history. Two hypotheses guide the reading:
  1. Panels as chronicles. Some panels may be dynastic or civic chronicles: \
sequences of crowned or enthroned figures (a king list such as the ~50 Ọọ̀ni \
of Ifẹ̀), interspersed with scenes of war, tyranny, sacrifice and divination, \
and with widely shared symbols of renewal and succession (the ouroboros / \
coiled serpent).
  2. Pattern as number. Woven, plaited and hatched bands may count something — \
reigns, years, generations, or verses — the way an Ifá tray's marks index the \
16 principal and 256 composite odù.
James C. Scott's suggestion that 'pre-literate' peoples can be post-literate — \
keeping knowledge in orality and material form in times of war and upheaval — \
is why an encoded record in carving is plausible here. For every reading, say \
plainly whether it supports, weakens, or does not bear on these hypotheses."""

CORPUS_CONTEXT = """\
THE COLLECTION AND ITS HISTORY:
- Sources: photographs and pen-and-ink survey drawings from the Frobenius \
Institute archive (EBA, FoA, KBA series), made during and after the German \
Inner-African Expedition (D.I.A.F.E.) of 1910–12. Some KBA items are mount \
sheets reproducing plates from Frobenius, Das unbekannte Afrika (1923).
- Places: Ilé-Ifẹ̀, Modákẹ́kẹ́ (beside Ifẹ̀), Ado-Ekiti. Read place names from \
file names and captions only.
- Background worth knowing, for readings of war and rule: after Ọ̀yọ́ fell \
(early 19th c.) Oyo refugees settled beside Ifẹ̀; Modákẹ́kẹ́ was founded for \
them c. 1845 and fought Ifẹ̀ repeatedly (Ifẹ̀ fell in 1849 and 1882). In the \
Kiriji / Ekitiparapọ war (1877–86) Ibadan, with Modákẹ́kẹ́, fought the \
Ekiti–Ijesa confederacy (Ado-Ekiti among it), which Ifẹ̀ joined after Ibadan \
and Modákẹ́kẹ́ attacked it in 1882; long-range imported rifles gave the war its \
name. The 1886 settlement ordered Modákẹ́kẹ́ evacuated, which was carried out \
in 1909; its people returned only in 1921. Ifẹ̀ had three Ọọ̀ni in 1910 \
(Adélékàn Olúbùṣe I d. 1910; Adékọ́lá, two months; Adémilúyì Ajagun, 1910–30). \
Ado-Ekiti's ruler is the Ewì. So the Modákẹ́kẹ́ objects were recorded in a \
freshly emptied war town, and firearms, horses and captives on a carving are \
period evidence, not only symbols.
- Panel types to test for: chronicle of rulers; war and its structures \
(warriors, rifles, cavalry, captives, walls); divination instruments; \
architectural ornament; and documentation sheets (a drawing or plate about an \
object, not the object)."""

SYMBOL_NOTES = """\
KNOWN AND RECURRING SYMBOLS (starting points, never conclusions):
- Coiled serpent / ouroboros: renewal, continuity, succession, earth and \
threshold power; compare Egyptian and Byzantine ouroboroi; Òṣùmàrè the \
rainbow-serpent.
- Horse and rider: rank, Ọ̀yọ́ cavalry power, war, arrival of a dignitary; \
associated with Ṣàngó.
- Crocodile, lizard, mudfish: water, liminality, Olókun and royal power (compare \
Benin mudfish-legged kings); the agama lizard in origin stories.
- Tortoise (Ìjàpá): cunning, longevity, trickster tales; offering animal.
- Monkey: mimicry, trickery, Ẹ̀ṣù's mischief; masquerade.
- Frontal face at the head of an Ifá tray: ojú ọpọ́n, conventionally Ẹ̀ṣù.
- Kneeling figure: supplication, greeting, offering; figures holding rifles, \
bows or cutlasses: warriors, guards, hunters.
- Interlace / plait / guilloche: framing and binding; read against textile \
(aṣọ olónà) — but, as Lisa Aronson warned of Yoruba cloth, 'the meaning is far \
from watertight': one motif can carry different meanings for different makers, \
media and communities (e.g. the agbérí ẹja 'fish head' tied to Olókun, shared \
with Ijaw trade partners).
- Cross-cultural archives (e.g. ARAS) are for comparison, not decoding. Use \
their three-step order: describe what is seen; give its meaning in its own \
culture; only then offer cross-cultural parallels."""

METHOD = """\
HOW TO READ:
- A detection is one element on a larger object. Read it in place: say where it \
sits (top of the rim, left leaf, third register) and what it does there. Never \
describe a single motif as a group ('members', 'each member', 'this family').
- Keep three kinds of claim distinct and labelled: what is seen; what Yoruba \
context supports (ritual, political, oral-historical); what cross-cultural \
comparison suggests. Mark confidence.
- A label or note written by a person is the strongest evidence you have; build \
on it and use its vocabulary. Model-written labels are drafts you may overrule.
- Orality is the archive: point to where a reading could be checked — oríkì \
(praise poetry), Ifá verses, town histories (Samuel Johnson, The History of \
the Yorubas, 1921), Frobenius's The Voice of Africa (1913), and later \
scholarship (Drewal, Pemberton & Abiodun; Abiodun's Yoruba Art and Language; \
Akintoye; Ajayi & Smith, Yoruba Warfare in the Nineteenth Century). Never \
invent a quotation, a verse, a date, a carver or a provenance.
- Segmentation is imperfect: a box may be a fragment, a duplicate, blank ground \
or a paper mount. Mention it once, briefly, where it changes a reading — not as \
the headline.
- Build upward: motif → panel → story. Each level should say what the level \
below adds up to."""

STYLE = """\
HOW TO WRITE:
Plain, concise, readable English for a curious non-specialist. Short \
paragraphs. No pipeline statistics or bookkeeping unless asked. Do not retitle \
or dismiss the collection; read what is there. Prefer 'may record' to 'is' \
when inferring. Length follows evidence: a thin panel gets a short reading."""

SYSTEM_PROMPT = "\n\n".join([ROLE, THESIS, CORPUS_CONTEXT, SYMBOL_NOTES, METHOD, STYLE])

CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}

MOTIF_READING_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string",
                  "description": "2-4 words, snake_case, naming what this one motif is"},
        "description": {"type": "string",
                        "description": "1-2 sentences: what is visibly carved, and where it "
                                       "sits on its board (e.g. 'at the head of the tray rim')."},
        "iconography": {"type": "string",
                        "description": "1-3 sentences: its likely meaning in Yoruba context, "
                                       "then any cross-cultural parallel, labelled as such. "
                                       "'unclear' if nothing is supportable."},
        "role_on_board": {"type": "string",
                          "description": "One sentence: what this element does in the whole "
                                         "composition (axis, frame, divider, protagonist, "
                                         "attendant, count, ...)."},
        "confidence": CONFIDENCE,
    },
    "required": ["label", "description", "iconography", "role_on_board", "confidence"],
    "additionalProperties": False,
}

CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "2-4 words, snake_case, naming the visual family "
                           "(e.g. interlace_band, kneeling_figure)",
        },
        "motif_gloss": {
            "type": "string",
            "description": "One sentence describing a SINGLE instance of this family as "
                           "it would appear on its own board — written so it can be copied "
                           "onto any one motif. Never say 'members' or 'each'.",
        },
        "visual_definition": {
            "type": "string",
            "description": "1-2 sentences: what instances of this family share, visually.",
        },
        "variation": {
            "type": "string",
            "description": "1-2 sentences: how instances differ, and whether this looks "
                           "like one family or several merged.",
        },
        "distribution_note": {
            "type": "string",
            "description": "1-2 sentences: where on the objects it occurs and what that "
                           "suggests it does (frame, divider, subject, axis).",
        },
        "iconographic_reading": {
            "type": "string",
            "description": "1-3 sentences: Yoruba meaning first, cross-cultural parallel "
                           "second, basis stated. 'unclear' rather than guessing.",
        },
        "relation_to_neighbours": {
            "type": "string",
            "description": "How this family relates to the embedding-adjacent clusters "
                           "listed, if the relation is meaningful; else empty.",
        },
        "confidence": CONFIDENCE,
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "At most 3: what would firm this up.",
        },
    },
    "required": [
        "name", "motif_gloss", "visual_definition", "variation", "distribution_note",
        "iconographic_reading", "relation_to_neighbours", "confidence",
        "open_questions",
    ],
    "additionalProperties": False,
}

PANEL_TYPES = ["chronicle_of_rulers", "war_and_conflict", "divination_instrument",
               "architectural_ornament", "documentation_sheet", "other"]

PANEL_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A short plain title for the object."},
        "object_type": {"type": "string",
                        "description": "What the object is: door leaf, veranda post, "
                                       "opon Ifa (round/rectangular), tapper, vessel, survey "
                                       "drawing, mount sheet, ..."},
        "panel_type": {"type": "string", "enum": PANEL_TYPES,
                       "description": "The closest reading type."},
        "summary": {"type": "string",
                    "description": "Two sentences: what this object is and what it shows."},
        "register_readings": {
            "type": "array",
            "description": "One entry per zone in reading order (registers top to bottom; "
                           "for trays, arcs of the rim from the head). 1-3 sentences each.",
            "items": {
                "type": "object",
                "properties": {
                    "register": {"type": "integer"},
                    "motifs": {"type": "array", "items": {"type": "integer"},
                               "description": "Motif indices in this zone."},
                    "reading": {"type": "string",
                                "description": "What this zone depicts, in 1-3 sentences."},
                },
                "required": ["register", "motifs", "reading"],
                "additionalProperties": False,
            },
        },
        "composition": {
            "type": "string",
            "description": "1-3 sentences: how the parts are organised (axis, frame, "
                           "registers, symmetry, scale).",
        },
        "narrative": {
            "type": "string",
            "description": "One short paragraph: the story or statement this object may "
                           "record, laddered up from its motifs, with the basis stated.",
        },
        "hypotheses": {
            "type": "string",
            "description": "1-2 sentences: does this object support, weaken or not bear on "
                           "the chronicle and pattern-as-number hypotheses, and why.",
        },
        "cross_panel_links": {
            "type": "array",
            "items": {"type": "string"},
            "description": "At most 4 short links to other objects in the collection.",
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "At most 4 short items, including any detection problem that "
                           "changes the reading.",
        },
        "confidence": CONFIDENCE,
    },
    "required": [
        "title", "object_type", "panel_type", "summary", "register_readings",
        "composition", "narrative", "hypotheses", "cross_panel_links", "uncertainties",
        "confidence",
    ],
    "additionalProperties": False,
}

# How the corpus essay is laid out, shared by the three-pass and direct paths.
# Story first, apparatus last: the reader should meet the collection's account
# of itself before they meet the pipeline's caveats.
CORPUS_ESSAY_BRIEF = """\
Write the interpretation as a Markdown essay of roughly 1,200-1,800 words, \
readable by a curious non-specialist. Use this shape:

1. **The story in brief** — one paragraph telling what this collection, read \
as a whole and in a sensible order, may record. No statistics.
2. **Reading the collection in order** — group the objects into a few \
chapters (for example by place, object type or theme: rule, war, divination) \
and walk through them, citing panel identifiers. Show how motifs build into \
each object's statement and how objects build into the larger account.
3. **The vocabulary** — a compact Markdown table of the recurring motifs: \
motif | where it appears | what it may mean (Yoruba context; cross-cultural \
parallel if useful).
4. **What this says about the thesis** — whether the evidence supports, \
weakens or leaves open the chronicle and pattern-as-number hypotheses, and \
what kind of record (chronicle, instrument, emblem, inventory) these carvings \
most look like.
5. **Where to check it** — oral and archival sources that could confirm or \
overturn the main readings (oríkì, Ifá verses, town histories, Frobenius's \
own notes, museum holdings), and at most five short data notes on the pipeline \
where they affect a conclusion.

Cite panel identifiers where they carry the argument. Keep what is seen, what \
Yoruba context supports and what comparison suggests distinguishable, but do \
not let hedging crowd out the reading."""


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
        "Characterise this family as a recurring element of the carving vocabulary, "
        "briefly. Judge whether it is one coherent family or several merged, and "
        "say what its placement on the objects implies about its function. Write "
        "`motif_gloss` for one instance on its own board: it is copied onto "
        "individual motifs, so it must never speak of 'members' or 'each'."
    )
    return "\n".join(lines)


def build_panel_prompt(
    layout: PanelLayout,
    motifs: Sequence[MotifView],
    cluster_context: Sequence[str],
) -> str:
    """Text half of a panel-reading request (the annotated panel image is attached)."""
    lines: list[str] = []
    lines.append(f"Read this object from the Frobenius archive: {layout.panel_stem}.")
    lines.append("Identify what it is first (door leaf, post, Ifa tray, survey drawing, "
                 "mount sheet ...), then read it as one composition.")
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
        "to match the indices above. The detections are parts of one object, not a "
        "set of separate things. Read it zone by zone in its own reading order — "
        "registers top to bottom for doors and posts; for a tray, round the rim from "
        "the face at its head — then as a whole. The computed registers are a guide, "
        "not a fact: override them when the object is round, framed or cropped. Use "
        "the family descriptions to say what recurs elsewhere and what is particular "
        "here. Then ladder up: what story or statement might this object carry, and "
        "does it bear on the chronicle and pattern-as-number hypotheses? Keep every "
        "field short."
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
        "Below are the readings of a collection of carved Yoruba objects from the "
        "Frobenius archive: first the recurring motif families, then a reading of "
        "each object. Read the collection as a whole and tell the story it may "
        "record, building from motifs to objects to the collection."
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
        kind = " / ".join(x for x in (reading.get("object_type"),
                                       reading.get("panel_type")) if x)
        if kind:
            lines.append(f"  Object: {kind}")
        lines.append(f"  {reading.get('summary', '')}")
        lines.append(f"  Composition: {reading.get('composition', '')}")
        lines.append(f"  Narrative: {reading.get('narrative', '')}")
        if reading.get("hypotheses"):
            lines.append(f"  Hypotheses: {reading['hypotheses']}")
        for link in reading.get("cross_panel_links", []) or []:
            lines.append(f"  Link: {link}")

    # Scale goes last and is labelled as apparatus: an essay that opens with
    # detection counts reads as a report on the pipeline, not on the carvings.
    lines.append("")
    lines.append(
        f"DATA NOTE (for section 5 only, not the opening): "
        f"{corpus_stats.get('panels', 0)} panels, {corpus_stats.get('motifs', 0)} "
        f"approved motif detections, {corpus_stats.get('clusters', 0)} motif families "
        f"({corpus_stats.get('unclustered', 0)} detections left unclustered by HDBSCAN)."
    )
    lines.append("")
    lines.append(CORPUS_ESSAY_BRIEF)
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
        "Below is a motif-level analysis of a collection of carved Yoruba objects "
        "from the Frobenius archive: the recurring motif families, then every "
        "object with its motifs and the spatial structure recovered from the "
        "detection geometry. Read the collection and tell the story it may record."
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
        f"DATA NOTE (for section 5 only, not the opening): {scale['panels']} panels, "
        f"{scale['motifs']} approved motif detections, {scale['clusters']} motif "
        f"families ({scale['unclustered']} unclustered), {scale['labelled']} labelled. "
        "You are working from the pipeline's descriptions and geometry, not from the "
        "images themselves; say so where it limits a reading."
    )
    lines.append("")
    lines.append(CORPUS_ESSAY_BRIEF)
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
    kind = " · ".join(x for x in (reading.get("object_type"),
                                   (reading.get("panel_type") or "").replace("_", " "))
                      if x)
    if kind:
        lines.append(f"*{kind}*")
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

    if reading.get("hypotheses"):
        lines.append("")
        lines.append("## Against the thesis")
        lines.append("")
        lines.append(reading["hypotheses"])

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
