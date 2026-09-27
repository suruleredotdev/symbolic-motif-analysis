"""
Cross-panel comparison — the pass that looks at two objects side by side.

The panel pass reads one object at a time, and the corpus pass only has text,
so neither can see that a drawing records a tray that also survives as a
photograph, that two trays come from one workshop, or that a mount sheet
copies the head and foot of another tray. Those identifications change the
story the corpus tells, so this pass puts likely pairs in front of the model
with both images at once.

Two halves, as elsewhere in the pipeline:

  select_pairs()          deterministic, no API. Scores every pair of panels
                          on shared motif families (rarer families count for
                          more), motif embeddings, rare shared words in labels and
                          readings, and catalogue numbers; ranks each signal so
                          none dominates; then keeps each panel's best partners.
  build_comparison_prompt the text half of the call; Interpreter.compare_panels
                          attaches the two annotated panels.

Output lives in interpretation/comparisons.json (+ comparisons.md) and feeds
the corpus synthesis.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMPY_AVAILABLE = False


RELATIONS = [
    "same_object",        # two records (photo, drawing) of one object
    "copy_or_detail",     # one reproduces all or part of the other
    "same_workshop",      # same hand or workshop: shared formula and treatment
    "shared_programme",   # same scheme or motif set, different hands
    "thematic_parallel",  # a shared subject, handled independently
    "unrelated",
]

COMPARISON_SCHEMA = {
    "type": "object",
    "properties": {
        "relation": {"type": "string", "enum": RELATIONS,
                     "description": "The closest relation the images support."},
        "evidence": {"type": "string",
                     "description": "What in the two images supports the relation, "
                                    "in one to three sentences."},
        "motif_matches": {
            "type": "array",
            "description": "Elements that correspond across the two objects.",
            "items": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "integer"},
                          "description": "Detection indices on object A (may be empty "
                                         "if the element is not boxed)."},
                    "b": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string", "description": "What corresponds, briefly."},
                },
                "required": ["a", "b", "note"],
                "additionalProperties": False,
            },
        },
        "differences": {"type": "string",
                        "description": "What differs, briefly: medium, hand, content."},
        "significance": {"type": "string",
                         "description": "What the pair adds to the collection's story or "
                                        "to the chronicle / pattern-as-number hypotheses. "
                                        "Empty if unrelated."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["relation", "evidence", "motif_matches", "differences",
                 "significance", "confidence"],
    "additionalProperties": False,
}


# ── Pair selection ───────────────────────────────────────────────────────────

_STOP = {"a", "an", "the", "of", "and", "or", "with", "in", "on", "at", "to", "by",
         "its", "his", "her", "one", "two", "three", "from", "into", "over", "under",
         "left", "right", "upper", "lower", "top", "bottom", "centre", "center", "side",
         "panel", "motif", "band", "field", "small", "large", "each", "this", "that"}
# Archive identifiers inside a reading ("KBA_17556", "#3") would let a panel
# match whatever its reading happens to cite; strip them so pairs are picked
# on what is described, not on who is named.
_ID_RE = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)?_[\w()-]*\d[\w()-]*|#\d+")
_CATALOGUE_RE = re.compile(r"^([A-Za-z]+(?:-[A-Za-z]+)?(?:_\d{2})?)[_-](\d+)")

WEIGHTS = {"families": 0.35, "embedding": 0.25, "text": 0.25, "catalogue": 0.15}


def _tokens(text: str | None) -> list[str]:
    text = _ID_RE.sub(" ", text or "").lower()
    return [w for w in re.split(r"[^a-zà-ÿ]+", text) if len(w) > 2 and w not in _STOP]


def catalogue_number(stem: str) -> tuple[str, int] | None:
    """("KBA", 19153) for "KBA_19153_Yoruba_q221664_i1_panel_00"."""
    m = _CATALOGUE_RE.match(stem)
    return (m.group(1), int(m.group(2))) if m else None


def source_image(stem: str) -> str:
    return stem.rsplit("_panel_", 1)[0]


def catalogue_affinity(a: str, b: str, window: int = 5) -> float:
    """1 for crops of one photograph, 0.6 for neighbouring catalogue numbers.

    Archive numbering follows accession: neighbours are often the same plate,
    the same album page or the same object photographed twice.
    """
    if source_image(a) == source_image(b):
        return 1.0
    ca, cb = catalogue_number(a), catalogue_number(b)
    if ca and cb and ca[0] == cb[0] and abs(ca[1] - cb[1]) <= window:
        return 0.6
    return 0.0


@dataclass
class PairCandidate:
    a: str
    b: str
    score: float
    reasons: list[str] = field(default_factory=list)
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return pair_key(self.a, self.b)

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "score": round(self.score, 4),
                "reasons": self.reasons}


def pair_key(a: str, b: str) -> str:
    a, b = sorted((a, b))
    return f"{a}|{b}"


def _panel_vectors(corpus, stems: Sequence[str]) -> dict[str, Any]:
    if not NUMPY_AVAILABLE or corpus.embeddings is None:
        return {}
    out = {}
    for stem in stems:
        rows = [corpus.embedding_for(m.key) for m in corpus.motifs_for_panel(stem)]
        rows = [r for r in rows if r is not None]
        if rows:
            mean = np.mean(np.stack(rows), axis=0)
            norm = np.linalg.norm(mean)
            if norm > 0:
                out[stem] = mean / norm
    return out


def _tfidf(docs: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Unit-length TF-IDF vectors: a word every reading uses ("tray", "figure")
    counts for little; a rare one ("pillow", "chameleon") for a lot."""
    df: dict[str, int] = {}
    for toks in docs.values():
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    n = max(len(docs), 1)
    out = {}
    for stem, toks in docs.items():
        tf: dict[str, int] = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        vec = {w: (1 + math.log(c)) * math.log(1 + n / df[w]) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out[stem] = {w: v / norm for w, v in vec.items()}
    return out


def _percentiles(values: list[float | None]) -> list[float | None]:
    """Rank each value among the others (0-1). Raw panel-mean cosines bunch
    between 0.85 and 0.98, so only their order carries information."""
    present = sorted(v for v in values if v is not None)
    if len(present) < 2:
        return [None if v is None else 1.0 for v in values]
    import bisect
    top = len(present) - 1
    return [None if v is None else bisect.bisect_left(present, v) / top for v in values]


def _panel_text(corpus, stem: str, reading: dict) -> list[str]:
    text = " ".join((m.label or "").replace("_", " ") for m in corpus.motifs_for_panel(stem))
    text += " " + " ".join(str(reading.get(k, "")) for k in ("title", "object_type", "summary"))
    return _tokens(text)


def score_pairs(corpus, readings: dict[str, dict] | None = None,
                stems: Sequence[str] | None = None) -> list[PairCandidate]:
    """Score every pair of panels that have motifs, 0-1, highest first."""
    readings = readings or {}
    stems = [s for s in (stems or corpus.panel_stems()) if corpus.motifs_for_panel(s)]

    # A family seen on few panels is strong evidence of a link; one seen
    # everywhere (a generic standing figure) is weak.
    families = {s: {m.cluster for m in corpus.motifs_for_panel(s) if m.cluster >= 0}
                for s in stems}
    spread: dict[int, int] = {}
    for fams in families.values():
        for c in fams:
            spread[c] = spread.get(c, 0) + 1
    n = max(len(stems), 2)
    idf = {c: math.log(n / k) + 0.1 for c, k in spread.items()}

    tfidf = _tfidf({s: _panel_text(corpus, s, readings.get(s, {})) for s in stems})
    vectors = _panel_vectors(corpus, stems)

    pairs, raw = [], {"families": [], "embedding": [], "text": []}
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            pairs.append((a, b))
            shared, union = families[a] & families[b], families[a] | families[b]
            raw["families"].append(
                sum(idf[c] for c in shared) / sum(idf[c] for c in union) if union else None)
            raw["embedding"].append(float(np.dot(vectors[a], vectors[b]))
                                    if a in vectors and b in vectors else None)
            va, vb = tfidf[a], tfidf[b]
            raw["text"].append(sum(v * vb.get(w, 0.0) for w, v in va.items())
                               if va and vb else None)
    ranked = {k: _percentiles(v) for k, v in raw.items()}

    out: list[PairCandidate] = []
    for j, (a, b) in enumerate(pairs):
        parts = {k: ranked[k][j] for k in ranked if ranked[k][j] is not None}
        parts["catalogue"] = catalogue_affinity(a, b)
        score = sum(WEIGHTS[k] * v for k, v in parts.items()) / sum(WEIGHTS[k] for k in parts)

        reasons = []
        shared = families[a] & families[b]
        if shared:
            reasons.append("shared families " + ", ".join(str(c) for c in sorted(shared)))
        if parts.get("embedding", 0) >= 0.9:
            reasons.append("close motif embeddings")
        va, vb = tfidf[a], tfidf[b]
        common = sorted((w for w in va if w in vb), key=lambda w: -(va[w] * vb[w]))[:5]
        if common and parts.get("text", 0) >= 0.5:
            reasons.append("shared terms: " + ", ".join(common))
        if parts["catalogue"] == 1.0:
            reasons.append("crops of one photograph")
        elif parts["catalogue"]:
            reasons.append("neighbouring catalogue numbers")
        ta = readings.get(a, {}).get("panel_type")
        if ta and ta == readings.get(b, {}).get("panel_type"):
            reasons.append(f"both {ta.replace('_', ' ')}")
        out.append(PairCandidate(a, b, score, reasons, parts))

    out.sort(key=lambda p: -p.score)
    return out


SIGNALS = ("families", "embedding", "text")


def select_pairs(corpus, readings: dict[str, dict] | None = None,
                 per_panel: int = 2, max_pairs: int = 100, min_score: float = 0.5,
                 only: Sequence[str] | None = None) -> list[PairCandidate]:
    """The pairs worth a comparison call, strongest first.

    A pair is a candidate when, under any one signal, each panel is among the
    other's `per_panel` nearest (mutual nearest neighbours), or when the two
    sit next to each other in the catalogue. Candidates are chosen on the
    signals separately, not on an average, because the pairs that matter most
    are strong on one signal and blank on another: a survey drawing and a
    photograph of the same tray share rare words but no motif family, since
    drawings and photographs cluster apart.

    Each candidate's `score` is its selection strength: its best supporting
    signal, plus 0.1 for every further signal that supports it. `only` keeps
    pairs that involve at least one of those panels.
    """
    scored = score_pairs(corpus, readings)

    neighbours: dict[str, dict[str, set[str]]] = {}
    for sig in SIGNALS:
        per: dict[str, list[tuple[float, str]]] = {}
        for p in scored:
            v = p.parts.get(sig)
            if v:
                per.setdefault(p.a, []).append((v, p.b))
                per.setdefault(p.b, []).append((v, p.a))
        neighbours[sig] = {stem: {other for _, other in sorted(vals, reverse=True)[:per_panel]}
                           for stem, vals in per.items()}

    wanted = set(only) if only else None
    chosen: list[PairCandidate] = []
    for p in scored:
        if wanted and p.a not in wanted and p.b not in wanted:
            continue
        support = [p.parts[sig] for sig in SIGNALS
                   if p.b in neighbours[sig].get(p.a, ()) and p.a in neighbours[sig].get(p.b, ())]
        catalogue = p.parts.get("catalogue", 0.0)
        if catalogue:
            # A catalogue neighbour on its own is a weak hint; how much it is
            # worth depends on whether anything else about the pair agrees.
            best_other = max((p.parts.get(sig, 0.0) for sig in SIGNALS), default=0.0)
            support.append(catalogue * (0.6 + 0.4 * best_other) / 0.6 if catalogue < 1
                           else 0.6 + 0.4 * best_other)
        if not support:
            continue
        strength = max(support) + 0.1 * (len(support) - 1)
        if strength < min_score:
            continue
        chosen.append(PairCandidate(p.a, p.b, strength, p.reasons, p.parts))

    chosen.sort(key=lambda p: -p.score)
    return chosen[:max_pairs]


# ── Prompt ───────────────────────────────────────────────────────────────────

def _panel_block(tag: str, stem: str, motifs, reading: dict | None) -> list[str]:
    lines = [f"OBJECT {tag}: {stem}"]
    if reading:
        kind = " / ".join(x for x in (reading.get("object_type"), reading.get("panel_type")) if x)
        lines.append(f"  Reading so far: {reading.get('title', '')}" + (f" ({kind})" if kind else ""))
        if reading.get("summary"):
            lines.append(f"  {reading['summary']}")
    lines.append("  Detections:")
    lines.extend(f"    {m.summary_line()}" for m in motifs)
    return lines


def build_comparison_prompt(a: str, b: str, motifs_a, motifs_b,
                            reading_a: dict | None, reading_b: dict | None,
                            reasons: Sequence[str] = ()) -> str:
    lines = [
        "Compare these two objects from the Frobenius archive. The first image is "
        "object A, the second object B, each with its detections outlined and "
        "numbered to match the lists below.",
        "",
    ]
    lines += _panel_block("A", a, motifs_a, reading_a)
    lines.append("")
    lines += _panel_block("B", b, motifs_b, reading_b)
    if reasons:
        lines += ["", "WHY THIS PAIR WAS PICKED (automatic, may be wrong): " + "; ".join(reasons)]
    lines += [
        "",
        "Decide what the images support, from closest to loosest: the same object "
        "recorded twice (a photograph and a survey drawing, or two photographs); one "
        "copying all or part of the other (a mount sheet that excerpts a tray); the "
        "same workshop or hand (the same formula, proportions and carving habits); "
        "a shared programme by different hands; a thematic parallel; or unrelated. "
        "Check it element by element: position on the object, pose, attributes, "
        "count, orientation. Photographs and drawings differ in medium, cropping "
        "and scale, so match on content and arrangement, not on surface. Say "
        "'unrelated' when that is what you see; a false link does more harm than a "
        "missed one. Keep every field short.",
    ]
    return "\n".join(lines)


# ── Rendering and corpus feed ────────────────────────────────────────────────

def comparison_lines(comparisons: dict[str, dict]) -> list[str]:
    """One or two lines per meaningful comparison, for the corpus prompt."""
    lines = []
    order = {r: i for i, r in enumerate(RELATIONS)}
    for c in sorted(comparisons.values(),
                    key=lambda c: (order.get(c.get("relation"), 99), c.get("a", ""))):
        if c.get("relation") in (None, "unrelated"):
            continue
        lines.append(f"— {c.get('a')} ↔ {c.get('b')}: {c.get('relation', '').replace('_', ' ')} "
                     f"[confidence: {c.get('confidence', '?')}]")
        lines.append(f"  {c.get('evidence', '')}")
        if c.get("significance"):
            lines.append(f"  Significance: {c['significance']}")
    return lines


def render_comparisons_markdown(comparisons: dict[str, dict]) -> str:
    lines = ["# Cross-panel comparisons", ""]
    order = {r: i for i, r in enumerate(RELATIONS)}
    for c in sorted(comparisons.values(),
                    key=lambda c: (order.get(c.get("relation"), 99), -c.get("score", 0))):
        lines.append(f"## {c.get('a')} ↔ {c.get('b')}")
        lines.append("")
        lines.append(f"**{c.get('relation', '?').replace('_', ' ')}** — confidence: "
                     f"{c.get('confidence', '?')}")
        lines.append("")
        lines.append(c.get("evidence", ""))
        for m in c.get("motif_matches") or []:
            a = ", ".join(f"#{i}" for i in m.get("a", [])) or "unboxed"
            b = ", ".join(f"#{i}" for i in m.get("b", [])) or "unboxed"
            lines.append(f"- A {a} ↔ B {b}: {m.get('note', '')}")
        if c.get("differences"):
            lines += ["", f"*Differences:* {c['differences']}"]
        if c.get("significance"):
            lines += ["", f"*Significance:* {c['significance']}"]
        lines.append("")
    return "\n".join(lines)
