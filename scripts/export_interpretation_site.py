#!/usr/bin/env python3
"""
export_interpretation_site.py — build the shippable interpretation site.

Reads an analysis directory plus whatever `interpretation/` contains, and
writes ONE self-contained HTML file: every panel image and motif crop is
embedded as a data URI, so the result can be emailed, dropped on a static
host, or opened straight off disk with no server and no asset directory.

The page is four views over one corpus — plates, motifs, families, synthesis —
in the surulere.dev tool chrome. Each is a gallery that opens into a detail,
with the centre holding the object, the right column holding what has been
*said* about it, and the bottom row holding what is *known* about it. Panel
and object identifiers named inside a reading become links back to the plate
they name, so a claim about the corpus can be checked rather than taken on
faith.

Degrades on partial data by design. Panels with no reading still render with
their motifs and geometry; families with no brief still show their members and
statistics. You do not need a finished interpretation to get a useful page.

Usage:
  # Whole corpus, default output next to the interpretation
  python3 scripts/export_interpretation_site.py \\
      --analysis-dir frobenius_artifacts/analysis \\
      --embeddings motif_embeddings_edges.npy \\
      --paths      motif_paths_edges.txt

  # A subset, smaller images, custom destination
  python3 scripts/export_interpretation_site.py \\
      --analysis-dir frobenius_artifacts/analysis \\
      --panels EBA-Div_00311_Ife_q166566_i1_panel_00 \\
      --max-dim 1100 --out site/interpretation.html

Dependencies:
  pip install pillow numpy       (no API key, no network)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from panel_art.interpret import (  # noqa: E402
    Corpus,
    InterpretationStore,
    compute_cluster_stats,
    corpus_scale,
    load_corpus,
)
from panel_art.site_template import render  # noqa: E402

try:
    from PIL import Image
except ImportError:                                        # pragma: no cover
    print("ERROR: Pillow is required — pip install pillow")
    raise SystemExit(1)


# ── Image embedding ──────────────────────────────────────────────────────────

def data_uri(image: "Image.Image", max_dim: int, quality: int) -> str:
    """Downscale and encode as a JPEG data URI.

    JPEG rather than PNG because these are photographs and pen-and-ink scans:
    at the sizes the page displays, JPEG is several times smaller with no
    visible cost, and the whole corpus has to fit in one file.
    """
    img = image.convert("RGB")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()


# ── Markdown → HTML ──────────────────────────────────────────────────────────

def markdown_to_html(text: str) -> str:
    """Render the synthesis essay.

    A deliberately small subset — headings, emphasis, lists, rules, code spans,
    paragraphs — because the essay is Claude-authored Markdown of a known
    shape, and a full parser would be a dependency for no gain. Everything is
    escaped before any tag is introduced, so essay text can never inject HTML.
    """
    if not text.strip():
        return ""

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
        return s

    out: list[str] = []
    list_tag: str | None = None
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            close_list()
            continue

        if re.fullmatch(r"(-\s*){3,}|(\*\s*){3,}|_{3,}", stripped):
            flush_paragraph()
            close_list()
            out.append("<hr>")
            continue

        heading = re.match(r"(#{1,6})\s+(.*)", stripped)
        if heading:
            flush_paragraph()
            close_list()
            level = min(len(heading.group(1)), 4)
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue

        bullet = re.match(r"[-*+]\s+(.*)", stripped)
        numbered = re.match(r"\d+[.)]\s+(.*)", stripped)
        if bullet or numbered:
            flush_paragraph()
            want = "ul" if bullet else "ol"
            if list_tag != want:
                close_list()
                out.append(f"<{want}>")
                list_tag = want
            out.append(f"<li>{inline((bullet or numbered).group(1))}</li>")
            continue

        if list_tag:
            close_list()
        paragraph.append(stripped)

    flush_paragraph()
    close_list()
    return "\n".join(out)


# ── Identifiers ──────────────────────────────────────────────────────────────

_PANEL_SUFFIX = re.compile(r"_panel_\d+$")


def object_id(stem: str) -> str:
    """The source object a panel stem was cut from.

    Panels are named `<object>_panel_NN`, so dropping the suffix recovers the
    object — which is the unit the archive catalogues and the unit the readings
    talk about ("six of nine members come from a single object").
    """
    return _PANEL_SUFFIX.sub("", stem)


def reference_aliases(stems: list[str]) -> dict[str, list[str]]:
    """Identifier → the panels it names, for turning prose into navigation.

    Three spellings of the same thing all get registered, because a reading
    may use any of them: the full stem, the object it came from, and the bare
    catalogue number that opens the object id (`EBA-Div_00302` out of
    `EBA-Div_00302_Ife_q166566_i1_panel_00`).  An alias has to carry a digit
    and at least five characters to qualify, which is what keeps ordinary
    prose words from being linkified.
    """
    refs: dict[str, list[str]] = {}

    def register(alias: str, stem: str) -> None:
        if len(alias) < 5 or not any(c.isdigit() for c in alias):
            return
        bucket = refs.setdefault(alias, [])
        if stem not in bucket:
            bucket.append(stem)

    for stem in stems:
        obj = object_id(stem)
        register(stem, stem)
        register(obj, stem)
        parts = obj.split("_")
        if len(parts) > 2:
            register("_".join(parts[:2]), stem)
    return refs


# ── Payload assembly ─────────────────────────────────────────────────────────

def build_payload(
    corpus: Corpus,
    store: InterpretationStore,
    stems: list[str],
    max_dim: int,
    crop_dim: int,
    quality: int,
    crops_per_family: int,
    motif_crops: bool = True,
    verbose: bool = True,
) -> dict:
    briefs = store.load_clusters()
    readings = store.load_panels()
    stats = compute_cluster_stats(corpus, exemplars=crops_per_family)

    # One crop per motif, keyed and shared: the motif gallery, the family
    # exemplar strips, and the motif card all point at the same bytes.
    crops: dict[str, str] = {}

    panels: list[dict] = []
    for n, stem in enumerate(stems, start=1):
        motifs = corpus.motifs_for_panel(stem)
        if not motifs:
            continue
        layout = corpus.layout_for(stem)
        placement = {p.key: p for p in layout.placements}
        reading = readings.get(stem)

        if verbose:
            print(f"  [{n}/{len(stems)}] {stem} — {len(motifs)} motifs, "
                  f"{len(layout.registers)} register(s)"
                  f"{'' if reading else ', no reading'}")

        if motif_crops:
            for motif in motifs:
                _embed_crop(corpus, motif, crops, crop_dim, quality)

        panels.append({
            "stem": stem,
            "object": object_id(stem),
            "title": (reading or {}).get("title") or stem,
            "width": corpus.panels[stem].width,
            "height": corpus.panels[stem].height,
            "image": data_uri(corpus.panel_image(stem), max_dim, quality),
            "registers": [r.as_dict() for r in layout.registers],
            # Reading order as motif indices — the page walks it with arrow keys.
            "reading_order": [placement[k].index for k in layout.reading_order
                              if k in placement],
            "motifs": [_motif_payload(m, placement.get(m.key)) for m in motifs],
            "reading": _reading_payload(reading),
        })

    families = {}
    for cid, st in stats.items():
        brief = briefs.get(str(cid), {})
        exemplars = []
        for key in st.exemplar_keys[:crops_per_family]:
            motif = corpus.by_key(key)
            if motif is None:
                continue
            if key not in crops and not _embed_crop(corpus, motif, crops,
                                                    crop_dim, quality):
                continue
            exemplars.append(key)
        families[str(cid)] = {
            "name": brief.get("name"),
            "visual_definition": brief.get("visual_definition"),
            "variation": brief.get("variation"),
            "distribution_note": brief.get("distribution_note"),
            "iconographic_reading": brief.get("iconographic_reading"),
            "confidence": brief.get("confidence"),
            "size": st.size,
            "panel_spread": st.panel_spread,
            "cohesion": st.cohesion,
            "exemplars": exemplars,
        }

    scale = corpus_scale(corpus)
    scale["readings"] = sum(1 for p in panels if p["reading"])
    synthesis = markdown_to_html(
        store.corpus_path.read_text(encoding="utf-8")
        if store.corpus_path.exists() else "")

    return {
        "title": "Symbolic Motif Analysis",
        "generated": date.today().isoformat(),
        "panels": panels,
        "families": families,
        "crops": crops,
        "refs": reference_aliases([p["stem"] for p in panels]),
        "scale": scale,
        "synthesis": synthesis,
        "synthesis_lead": _lead_paragraph(synthesis),
    }


def _embed_crop(corpus, motif, crops: dict[str, str], crop_dim: int, quality: int) -> bool:
    """Add `motif`'s crop to the shared index. False if the image is missing."""
    if motif.key in crops:
        return True
    try:
        crops[motif.key] = data_uri(corpus.crop(motif, padding=6), crop_dim, quality)
    except (OSError, KeyError):                  # missing or unreadable panel PNG
        return False
    return True


def _lead_paragraph(synthesis_html: str) -> str:
    """The first paragraph of the synthesis, for the corpus card in the sidebar."""
    match = re.search(r"<p>.*?</p>", synthesis_html, re.DOTALL)
    return match.group(0) if match else ""


def _motif_payload(motif, placement) -> dict:
    b = motif.bbox
    payload = {
        "key": motif.key,
        "panel_stem": motif.panel_stem,
        "index": motif.index,
        "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
        "cluster": motif.cluster,
        "scale": motif.scale,
        "label": motif.label,
        "description": motif.description,
        "iconography": motif.iconography,
        "notes": motif.notes,
        "label_source": motif.label_source,
    }
    if placement is not None:
        payload.update({
            "zone": placement.zone,
            "register": placement.register,
            "is_field": placement.is_field,
            "area_fraction": placement.area_fraction,
        })
    else:                                        # shouldn't happen; stay safe
        payload.update({"zone": "", "register": -1,
                        "is_field": False, "area_fraction": 0.0})
    return payload


def _reading_payload(reading: dict | None) -> dict | None:
    if not reading:
        return None
    return {
        key: reading.get(key) for key in (
            "title", "summary", "composition", "narrative", "confidence",
            "register_readings", "cross_panel_links", "uncertainties",
        )
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export the interpretation as one self-contained HTML page",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--analysis-dir", type=Path, default=Path("frobenius_artifacts/analysis"))
    p.add_argument("--interpretation-dir", type=Path, default=None,
                   help="Default: <analysis-dir>/interpretation")
    p.add_argument("--out", type=Path, default=None,
                   help="Default: <interpretation-dir>/site.html")
    p.add_argument("--embeddings", type=Path, default=None,
                   help="Enables cohesion and centroid-nearest exemplar crops")
    p.add_argument("--paths", type=Path, default=None)
    p.add_argument("--clusters", type=Path, default=None,
                   help="Optional cluster-assignment override JSON")
    p.add_argument("--panels", nargs="*", metavar="STEM", default=None,
                   help="Limit to these panel stems")
    p.add_argument("--max-dim", type=int, default=1400,
                   help="Longest edge for embedded panel images")
    p.add_argument("--crop-dim", type=int, default=160,
                   help="Longest edge for embedded motif crops")
    p.add_argument("--quality", type=int, default=82, help="JPEG quality")
    p.add_argument("--crops-per-family", type=int, default=8,
                   help="Exemplar crops shown per family")
    p.add_argument("--no-motif-crops", action="store_true",
                   help="Embed only family exemplars, not every motif — smaller "
                        "file, but the motif gallery loses its thumbnails")
    p.add_argument("--fragment", action="store_true",
                   help="Emit the page without its <html>/<head> wrapper, for "
                        "hosts that supply their own")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.analysis_dir.exists():
        print(f"ERROR: analysis directory not found: {args.analysis_dir}")
        return 1

    interpretation_dir = args.interpretation_dir or (args.analysis_dir / "interpretation")
    out_path = args.out or (interpretation_dir / "site.html")
    store = InterpretationStore(interpretation_dir)

    print(f"Loading corpus from {args.analysis_dir}")
    corpus = load_corpus(
        args.analysis_dir,
        embeddings_path=args.embeddings,
        paths_path=args.paths,
        clusters_path=args.clusters,
    )
    if not corpus.motifs:
        print("ERROR: no approved motifs found — nothing to show.")
        return 1

    stems = [s for s in (args.panels or corpus.panel_stems())
             if s in corpus.panels and corpus.motifs_for_panel(s)]
    if not stems:
        print("ERROR: no panels with motifs matched.")
        return 1

    scale = corpus_scale(corpus)
    print(f"  {scale['panels']} panels, {scale['motifs']} motifs, "
          f"{scale['clusters']} families, {scale['labelled']} labelled")
    if not interpretation_dir.exists():
        print(f"  NOTE: {interpretation_dir} does not exist — the page will show "
              "geometry and labels, but no readings, briefs, or synthesis.")

    print(f"\nEmbedding {len(stems)} plate(s) at max {args.max_dim}px:")
    payload = build_payload(
        corpus, store, stems,
        max_dim=args.max_dim, crop_dim=args.crop_dim, quality=args.quality,
        crops_per_family=args.crops_per_family,
        motif_crops=not args.no_motif_crops,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(payload, standalone=not args.fragment), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1_048_576

    print(f"\n{'═' * 68}")
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")
    print(f"  {len(payload['panels'])} plates, {payload['scale']['readings']} with readings, "
          f"{len(payload['families'])} families, {len(payload['crops'])} crops"
          f"{', synthesis included' if payload['synthesis'] else ', no synthesis yet'}")
    if size_mb > 16:
        print("  WARNING: over 16 MB — too large to publish as an artifact. "
              "Lower --max-dim or --quality, drop --no-motif-crops in, "
              "or split with --panels.")
    print(f"\nOpen it directly:  open {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
