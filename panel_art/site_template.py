"""
site_template.py — the HTML/CSS/JS shell for the interpretation site.

Kept apart from `export_interpretation_site.py` so the exporter stays readable
as data-assembly code and the page stays readable as a page.  `render()`
substitutes four things: the JSON payload, the surulere.dev mark, and the two
background tiles.

Design notes, since they are choices rather than defaults:

  * The shell is the surulere.dev tool chrome — the same masthead, menu bar,
    sidebar, and status row as Archive 3D and the Artifacts Index, so this
    reads as another instrument on the same bench rather than a one-off.
    Tokens, borders, and breakpoints follow `design-system.md` in
    `suruleredotdev/african-artifacts`; the one house rule worth restating is
    that **dashed means available and solid means selected**, which is how
    every control on this page signals state.
  * Four views over one corpus: panels, motifs, families, synthesis.  Each is
    a gallery that opens into a detail, and the three regions stay in sync —
    centre holds the object, the right column holds what has been *said* about
    it, the bottom row holds what is *known* about it.
  * Prose is navigation.  Panel and object identifiers inside a reading or the
    synthesis are turned into links back to the plate they name, so a sentence
    like "six of nine members come from EBA-Div_00302" can be checked rather
    than taken on faith.  This is done over text nodes after insertion, never
    by pattern-matching HTML.
  * No webfonts and no CDN: the artifact CSP blocks external hosts, and the
    page is meant to survive being emailed or opened off a USB stick.  The
    design system's Tailwind utilities are therefore written out as plain CSS.

Substitution is a plain sentinel replace rather than `string.Template` or
`str.format`, because the page is full of JS `${...}` literals and CSS braces
that both of those would try to interpret.
"""

import json
from typing import Any

from .brand import TEXTURE_DARK, TEXTURE_LIGHT, favicon_data_uri, logo_svg

_PAYLOAD_SENTINEL = "/*__PAYLOAD__*/null"
_LOGO_SENTINEL = "<!--__LOGO__-->"
_TEXTURE_LIGHT_SENTINEL = "__TEXTURE_LIGHT__"
_TEXTURE_DARK_SENTINEL = "__TEXTURE_DARK__"

TITLE = "Symbolic Motif Analysis"

#: Wrapper for the standalone file.  An artifact host supplies its own
#: `<head>`, but a page opened off disk gets none — and without the viewport
#: meta every mobile breakpoint in the stylesheet is dead on arrival.  The
#: `<body>` ground is painted here too, so there is no white flash before the
#: layout's own background lands.
DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="icon" href="__FAVICON__">
<style>
  html, body { margin: 0; padding: 0; }
  body { background: #e5e5e5; }
  @media (prefers-color-scheme: dark) { body { background: #0e160c; } }
</style>
</head>
<body>
__PAGE__
</body>
</html>
"""

PAGE = r"""<div id="layout">
<style>
/* ══ Tokens ══════════════════════════════════════════════════════════════
   surulere.dev palette: warm grey ground, muted olive borders, burgundy
   accent in light and teal in dark.  Light is the bare :root default so the
   un-stamped "system" state can never render one theme's ink on the other's
   ground; dark is layered on three ways — the OS preference, an explicit
   data-theme stamp from a host page, and the in-page toggle. */
:root {
  --bg-size: 50px;
  --floater-offset: 12px;
  --floater-offset-mobile: 8px;

  /* Static references — these do not switch with the theme. */
  --bg-color-light:     #e5e5e5;
  --bg-color-dark:      #0e160c;
  --txt-color-light:    #4d4d4d;
  --txt-color-dark:     #a3a8a2;
  --accent-color-light: #751a42;
  --accent-color-dark:  #2ca7ad;
  --border-color-light: #3f493d2e;
  --border-color-dark:  #3f493d;
  --title-color:        #3f493d;

  /* Semantic tokens — every one of these is redefined per theme below. */
  --bg-color:           #e5e5e5;
  --surface:            #e5e5e5f2;
  --txt-color:          #4d4d4d;
  --txt-color-annotation: #535e52fa;
  --accent-color:       #751a42;
  --border-color:       #3f493d2e;
  --border-width:       1px;
  --texture:            url("__TEXTURE_LIGHT__");
  /* The house tile is a wordmark, and a wordmark at full strength competes
     with the reading. Held back to a watermark so it stays brand rather than
     becoming pattern. */
  --texture-opacity:    .4;
  --sunk:               #00000008;
  --fam-l:              38%;

  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
          "Helvetica Neue", "Noto Sans", sans-serif;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua",
           Georgia, serif;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg-color:           #0e160c;
    --surface:            #0e160cf2;
    --txt-color:          #a3a8a2;
    --txt-color-annotation: #7d867b;
    --accent-color:       #2ca7ad;
    --border-color:       #3f493d;
    --border-width:       0.5px;
    --texture:            url("__TEXTURE_DARK__");
    --texture-opacity:    .3;
    --sunk:               #ffffff08;
    --fam-l:              62%;
  }
}
:root[data-theme="dark"] {
  --bg-color:           #0e160c;
  --surface:            #0e160cf2;
  --txt-color:          #a3a8a2;
  --txt-color-annotation: #7d867b;
  --accent-color:       #2ca7ad;
  --border-color:       #3f493d;
  --border-width:       0.5px;
  --texture:            url("__TEXTURE_DARK__");
  --texture-opacity:    .3;
  --sunk:               #ffffff08;
  --fam-l:              62%;
}

/* The in-page toggle. Scoped to #layout so it beats both blocks above in
   either direction — the design system drives theme off .LIGHT / .DARK on
   the root layout element, and this is that hook. */
#layout.LIGHT {
  --bg-color:           #e5e5e5;
  --surface:            #e5e5e5f2;
  --txt-color:          #4d4d4d;
  --txt-color-annotation: #535e52fa;
  --accent-color:       #751a42;
  --border-color:       #3f493d2e;
  --border-width:       1px;
  --texture:            url("__TEXTURE_LIGHT__");
  --texture-opacity:    .4;
  --sunk:               #00000008;
  --fam-l:              38%;
}
#layout.DARK {
  --bg-color:           #0e160c;
  --surface:            #0e160cf2;
  --txt-color:          #a3a8a2;
  --txt-color-annotation: #7d867b;
  --accent-color:       #2ca7ad;
  --border-color:       #3f493d;
  --border-width:       0.5px;
  --texture:            url("__TEXTURE_DARK__");
  --texture-opacity:    .3;
  --sunk:               #ffffff08;
  --fam-l:              62%;
}

/* ══ Ground ══════════════════════════════════════════════════════════════ */
#layout, #layout *, #layout *::before, #layout *::after { box-sizing: border-box; }
#layout {
  position: relative;
  min-height: 100vh;
  background-color: var(--bg-color);
  color: var(--txt-color);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.5;
  padding-bottom: 3.2rem;          /* clears the fixed status row */
}
/* The tile rides in a pseudo-element so its opacity is independent of the
   content stacked over it. */
#layout::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image: var(--texture);
  background-size: var(--bg-size);
  opacity: var(--texture-opacity);
  pointer-events: none;
  z-index: 0;
}
#layout > * { position: relative; z-index: 1; }

#layout h1, #layout h2, #layout h3, #layout h4 {
  margin: 0; font-weight: 700; text-wrap: balance;
}
#layout p { margin: 0; }
#layout ul, #layout ol { margin: 0; }
#layout a { color: inherit; text-decoration: none; }
#layout button { font: inherit; color: inherit; background: none; border: 0;
                 margin: 0; padding: 0; cursor: pointer; }
#layout img { max-width: 100%; }
#layout :focus-visible { outline: 2px solid var(--accent-color); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  #layout *, #layout *::before { transition: none !important; animation: none !important; }
}

.b-color { border-color: var(--border-color); }
.mono { font-family: var(--mono); }
.dim { color: var(--txt-color-annotation); }
.empty { color: var(--txt-color-annotation); font-style: italic; }
.nowrap { white-space: nowrap; }

/* ══ Masthead ════════════════════════════════════════════════════════════ */
#header {
  position: sticky; top: 0; z-index: 50;
  display: flex; flex-direction: row; align-items: center;
  justify-content: space-between; gap: .5rem;
  padding: .75rem .75rem .25rem;
  background-color: var(--bg-color);          /* opaque: content scrolls under */
}
/* Its own copy of the tile, so the masthead is textured like the page rather
   than a flat band across it. */
#header::before {
  content: ""; position: absolute; inset: 0;
  background-image: var(--texture); background-size: var(--bg-size);
  opacity: var(--texture-opacity); pointer-events: none;
}
#header > * { position: relative; }
#title { display: flex; flex-direction: row; align-items: center; padding: .25rem; }
.brand { display: block; padding: .25rem; opacity: .9; transition: opacity .2s ease; }
.brand:hover { opacity: .6; }
.brand-mark { display: block; width: var(--bg-size); height: var(--bg-size); }
.title-box {
  font-size: 1.25rem; font-weight: 700; letter-spacing: .02em;
  border: var(--border-width) solid var(--border-color);
  background-color: var(--bg-color);
  padding: .25rem .5rem; margin-left: .5rem;
}

#menu-bar {
  display: flex; flex-direction: row; align-items: center;
  border: var(--border-width) solid var(--border-color);
  background-color: var(--bg-color);
}
.menu-bar-section-toggle { display: none; padding: .25rem; line-height: 0; }
.menu-bar-sections { display: flex; flex-direction: row; padding: .25rem; gap: .35rem; }
.menu-bar-section {
  display: flex; flex-direction: row; align-items: center; gap: .2rem;
  padding: 0 .35rem;
  border-right: 2px dashed var(--border-color);
}
.menu-bar-section:last-child { border-right: 0; }
.menu-bar-section[hidden] { display: none; }
.menu-bar-section-label {
  font-weight: 700; font-size: .7rem; letter-spacing: .06em;
  padding: .25rem; color: var(--txt-color-annotation); white-space: nowrap;
}
.menu-item {
  padding: .2rem .5rem; margin: .1rem;
  font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
  border: 1px dashed transparent; white-space: nowrap;
  transition: opacity .2s ease, border-color .2s ease;
}
.menu-item:hover { opacity: .55; }
/* Dashed = available, solid = selected. Never invert this. */
.menu-item[aria-pressed="true"], .menu-item[aria-current="page"] {
  border: 1px solid currentColor; font-weight: 600; color: var(--accent-color);
}

@media (max-width: 767px) {
  #header { align-items: flex-start; padding: .5rem .5rem .25rem; }
  .title-box { font-size: 1rem; }
  .brand-mark { width: 34px; height: 34px; }
  #menu-bar { border: 0; }
  .menu-bar-section-toggle { display: block; border: var(--border-width) solid var(--border-color); }
  .menu-bar-sections {
    display: none; position: absolute; top: 3.4rem; right: .5rem;
    flex-direction: column; align-items: stretch; min-width: 12rem;
    border: var(--border-width) solid var(--border-color);
    background-color: var(--bg-color); z-index: 50;
  }
  #menu-bar.is-open .menu-bar-sections { display: flex; }
  .menu-bar-section {
    flex-direction: column; align-items: stretch; padding: .25rem;
    border-right: 0; border-bottom: 2px dashed var(--border-color);
  }
  .menu-item { text-align: left; }
}

/* ══ Breadcrumbs ═════════════════════════════════════════════════════════ */
#crumbs {
  display: flex; flex-wrap: wrap; align-items: center; gap: .35rem;
  font-family: var(--mono); font-size: .72rem;
  padding: .35rem .9rem .1rem;
}
#crumbs .crumb { border-bottom: 1px dashed currentColor; cursor: pointer; }
#crumbs .crumb:hover { opacity: .55; }
#crumbs .crumb.is-current { border-bottom: 0; opacity: .6; cursor: default; }
#crumbs .sep { opacity: .4; }

/* ══ Content ═════════════════════════════════════════════════════════════ */
#content {
  display: flex; flex-direction: row; align-items: flex-start;
  gap: .75rem; padding: .25rem .75rem 1rem;
}
#main { flex: 1 1 auto; min-width: 0; }
#sidebar {
  flex: 0 0 24rem; width: 24rem; max-width: 24rem;
  position: sticky; top: 5.2rem; z-index: 10;
  max-height: calc(100vh - 7.5rem); overflow-y: auto;
  display: flex; flex-direction: column; gap: .6rem;
  padding-right: .2rem;
}
#sidebar[hidden] { display: none; }
@media (max-width: 1100px) { #sidebar { flex-basis: 19rem; width: 19rem; } }
@media (max-width: 767px) {
  #content { flex-direction: column; padding: .25rem .5rem 1rem; }
  #sidebar { position: static; width: 100%; max-width: 100%; flex-basis: auto;
             max-height: none; }
}

.view[hidden] { display: none; }

/* ══ Cards ═══════════════════════════════════════════════════════════════ */
.card {
  border: var(--border-width) dashed var(--border-color);
  background-color: var(--surface);
  padding: .6rem .7rem;
}
.card + .card { margin-top: 0; }
.card h3 { font-size: 1rem; display: flex; align-items: baseline; gap: .4rem;
           flex-wrap: wrap; }
.card p { margin-top: .45rem; }
.card p.serif { font-family: var(--serif); font-size: .95rem; line-height: 1.55; }
.eyebrow {
  font-family: var(--mono); font-size: .68rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--txt-color-annotation);
  margin-bottom: .3rem;
}
/* The masthead of a card: counts, then faces, then a rule the prose sits
   under. Ruled off rather than merely spaced, so the eye can tell at once
   where the measurements stop and the claims start. */
.summary {
  display: flex; flex-direction: column; gap: .45rem;
  margin: .5rem 0 .1rem; padding-bottom: .6rem;
  border-bottom: 1px dashed var(--border-color);
}
.summary + p, .summary + .serif, .summary + .empty { margin-top: .55rem; }
.tally {
  display: flex; flex-wrap: wrap; gap: .1rem .75rem;
  font-family: var(--mono); font-size: .72rem; line-height: 1.5;
}
.tally b { font-weight: 600; font-variant-numeric: tabular-nums; }
.tally .k { color: var(--txt-color-annotation); }

/* A legend, not a gallery — small enough that eight fit without scrolling. */
.fam-row {
  display: grid; gap: .35rem;
  grid-template-columns: repeat(auto-fill, minmax(3.4rem, 1fr));
  margin-top: .1rem;
}
.fam-row button {
  border: var(--border-width) dashed var(--border-color);
  padding: .2rem; text-align: left; min-width: 0;
  transition: border-color .2s ease;
}
.fam-row button:hover { border-style: solid; }
.fam-row button.is-on { border-style: solid; border-color: var(--accent-color); }
.fam-row img, .fam-row .blank {
  display: block; width: 100%; aspect-ratio: 1; object-fit: contain;
  background: var(--sunk);
}
.fam-row .cap {
  display: flex; align-items: center; gap: .25rem;
  font-family: var(--mono); font-size: .62rem; line-height: 1.3; padding-top: .15rem;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.fam-row .cap.dim { color: var(--txt-color-annotation); display: block; }

dl.meta {
  display: grid; grid-template-columns: auto 1fr; gap: .15rem .7rem;
  margin: .55rem 0 0; font-size: .74rem;
}
dl.meta dt { font-family: var(--mono); color: var(--txt-color-annotation);
             letter-spacing: .03em; }
dl.meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.notes { margin: .45rem 0 0; padding-left: 1rem; font-size: .78rem;
         color: var(--txt-color-annotation); }
.notes li { margin-bottom: .2rem; }
.swatch { display: inline-block; width: .62em; height: .62em; border-radius: 50%;
          flex: 0 0 auto; }

.chips { display: flex; flex-wrap: wrap; gap: .25rem; margin-top: .45rem; }
.chip {
  display: inline-flex; align-items: center; gap: .25rem;
  padding: .12rem .45rem; font-size: .7rem; letter-spacing: .03rem;
  font-family: var(--mono); line-height: 1.4;
  border: 1px dashed currentColor; opacity: .7;
  transition: opacity .15s ease;
}
.chip:hover { opacity: 1; }
.chip.is-on { border-style: solid; opacity: 1; font-weight: 600; }

.registers { list-style: none; margin: 0; padding: 0;
             display: flex; flex-direction: column; gap: .55rem; }
.registers li { border-left: 2px solid var(--border-color); padding-left: .6rem; }
.reg-head { font-family: var(--mono); font-size: .7rem;
            color: var(--txt-color-annotation); letter-spacing: .04em; }
.registers p { margin-top: .15rem; font-family: var(--serif); font-size: .9rem; }

/* Cross-reference links minted from prose. */
.ref {
  border-bottom: 1px dashed currentColor; cursor: pointer;
  color: var(--accent-color); font-weight: 500;
}
.ref:hover { opacity: .6; }

/* ══ Galleries ═══════════════════════════════════════════════════════════ */
.gallery-head {
  display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between;
  gap: .5rem; padding: .25rem .1rem .6rem;
}
.gallery-head h2 { font-size: 1rem; letter-spacing: .04em; text-transform: uppercase; }
.gallery-head .count { font-family: var(--mono); font-size: .72rem;
                       color: var(--txt-color-annotation); }

.grid-panels {
  display: grid; gap: .6rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 12rem), 1fr));
}
.grid-motifs {
  display: grid; gap: .45rem;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 6.5rem), 1fr));
}
/* A phone column wide enough for one plate is a list, not a gallery — fix the
   count instead of the tile width so both stay scannable. */
@media (max-width: 767px) {
  .grid-panels { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid-motifs { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
.tile {
  display: flex; flex-direction: column; text-align: left;
  border: var(--border-width) dashed var(--border-color);
  background-color: var(--surface);
  padding: .3rem; transition: all .3s ease; width: 100%;
}
.tile:hover, .tile:focus-visible { border-style: solid; }
.tile.is-active { border-style: solid; border-color: var(--accent-color); }
.tile figure { margin: 0; position: relative; line-height: 0;
               background: var(--sunk); }
/* `contain`, not `cover`: a cropped plate is a misleading thumbnail. */
.tile img { width: 100%; display: block; object-fit: contain; }
.tile-panel img { aspect-ratio: 3 / 4; }
.tile-motif img { aspect-ratio: 1; }
.tile figcaption {
  font-family: var(--mono); font-size: .64rem; line-height: 1.35;
  padding-top: .3rem; min-width: 0;
}
/* Both caption lines are clipped rather than wrapped, so a long stem cannot
   push one tile out of step with the rest of the row. */
.tile figcaption > .name {
  display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.tile .sub { display: block; color: var(--txt-color-annotation);
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tile .fam-bar { position: absolute; left: 0; right: 0; bottom: 0; height: 3px; }

.group { margin-bottom: 1.1rem; }
.group-head {
  display: flex; align-items: center; gap: .45rem; flex-wrap: wrap;
  padding: .2rem .3rem; margin-bottom: .4rem;
  border-bottom: 2px dashed var(--border-color);
  font-family: var(--mono); font-size: .74rem; letter-spacing: .04em;
  width: 100%; text-align: left;
}
.group-head:hover { opacity: .6; }
.group-head.is-active { border-bottom-style: solid; color: var(--accent-color); }
.group-head .n { color: var(--txt-color-annotation); }

/* ══ Plate (panel detail) ════════════════════════════════════════════════ */
.plate {
  border: var(--border-width) solid var(--border-color);
  background-color: var(--surface); padding: .4rem;
  display: flex; justify-content: center;
}
/* The figure shrink-wraps the image rather than the column, so a tall plate
   is bounded by the viewport — and the overlay, pinned to the figure, keeps
   registration with the pixels underneath it either way. */
.plate figure { margin: 0; position: relative; line-height: 0; width: fit-content;
                max-width: 100%; }
.plate img { display: block; width: auto; max-width: 100%;
             max-height: calc(100vh - 11rem); }
.plate svg { position: absolute; inset: 0; width: 100%; height: 100%; }
@media (max-width: 767px) { .plate img { max-height: none; width: 100%; } }
.band { fill: var(--accent-color); opacity: .07; }
.band-line { stroke: var(--accent-color); opacity: .4; stroke-dasharray: 4 3; }
.band-label { fill: var(--accent-color); font-family: var(--mono);
              opacity: .7; letter-spacing: .06em; }
/* Faint by default — the carving is the subject, the detections are apparatus. */
.box { fill: transparent; cursor: pointer; stroke-opacity: .38;
       transition: stroke-opacity .12s ease, fill-opacity .12s ease; }
.box:hover, .box:focus-visible { stroke-opacity: 1; fill: currentColor; fill-opacity: .14; }
.box.is-field { stroke-dasharray: 3 2.5; }
.box.is-active { stroke-opacity: 1; fill: currentColor; fill-opacity: .2; }
.box.is-dim { stroke-opacity: .12; }
.box-num { font-family: var(--mono); pointer-events: none; opacity: .75;
           font-variant-numeric: tabular-nums; font-weight: 600;
           paint-order: stroke; stroke: var(--bg-color); }

/* ══ Motif detail ════════════════════════════════════════════════════════ */
.motif-detail { display: flex; flex-wrap: wrap; gap: .75rem; align-items: flex-start; }
.motif-zoom {
  flex: 1 1 20rem; min-width: 0;
  border: var(--border-width) solid var(--border-color);
  background-color: var(--surface); padding: .4rem;
}
.motif-zoom img { width: 100%; display: block; background: var(--sunk); }
.zoom-face { width: 100%; margin-inline: auto;
             background-repeat: no-repeat; background-color: var(--sunk); }
@media (max-width: 767px) { .zoom-face { max-width: 100% !important; } }
.motif-context { flex: 0 1 14rem; }
.motif-context .plate { padding: .3rem; }
.strip { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .4rem; }
.strip button {
  width: 3.4rem; height: 3.4rem; padding: 2px;
  border: var(--border-width) dashed var(--border-color); background: var(--sunk);
}
.strip button:hover { border-style: solid; }
.strip button.is-active { border-style: solid; border-color: var(--accent-color); }
.strip img { width: 100%; height: 100%; object-fit: contain; display: block; }

/* ══ Synthesis ═══════════════════════════════════════════════════════════ */
.prose {
  max-width: 66ch; margin: 0 auto;
  font-family: var(--serif); font-size: 1.02rem; line-height: 1.68;
  border: var(--border-width) dashed var(--border-color);
  background-color: var(--surface); padding: 1rem 1.2rem;
}
.prose h1 { font-size: 1.7rem; margin: 1.6rem 0 .5rem; letter-spacing: -.01em; }
.prose h2 { font-size: 1.3rem; margin: 1.6rem 0 .4rem; }
.prose h3 { font-size: 1.08rem; margin: 1.2rem 0 .3rem; }
.prose h1:first-child, .prose h2:first-child { margin-top: 0; }
.prose p, .prose li { margin: .7rem 0; }
.prose ul, .prose ol { padding-left: 1.3rem; }
.prose code { font-family: var(--mono); font-size: .85em; background: var(--sunk);
              padding: .05em .3em; }
.prose hr { border: 0; border-top: 1px dashed var(--border-color); margin: 1.6rem 0; }

/* ══ Status row ══════════════════════════════════════════════════════════ */
#statusbar {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 30;
  display: flex; align-items: center; justify-content: space-between; gap: .75rem;
  padding: .3rem .75rem;
  border-top: var(--border-width) solid var(--border-color);
  background-color: var(--surface);
  font-family: var(--mono); font-size: .7rem; line-height: 1.4;
}
#status-meta { display: flex; flex-wrap: wrap; gap: .1rem .9rem; min-width: 0;
               overflow: hidden; }
#status-meta b { font-weight: 600; }
#status-meta .k { color: var(--txt-color-annotation); }
#statusbar .copyright { white-space: nowrap; opacity: .7; }
#statusbar .copyright:hover { opacity: 1; }
@media (max-width: 767px) {
  #layout { padding-bottom: 4.2rem; }        /* the row wraps to two lines here */
  #statusbar { font-size: .62rem; padding: .25rem .5rem; }
  #status-meta { gap: 0 .55rem; max-height: 2.4em; }
}

/* ══ About ═══════════════════════════════════════════════════════════════ */
#about[hidden] { display: none; }
#about {
  position: fixed; inset: 0; z-index: 60;
  display: flex; align-items: center; justify-content: center;
  background: #00000080; padding: 1rem;
}
#about .sheet {
  max-width: 44rem; width: 100%; max-height: 86vh; overflow-y: auto;
  border: var(--border-width) solid var(--border-color);
  background-color: var(--bg-color); padding: 1.1rem 1.3rem;
}
#about h2 { font-size: 1.2rem; margin-bottom: .5rem; }
#about p { margin-top: .7rem; max-width: 62ch; }
#about .close {
  border: var(--border-width) solid var(--border-color);
  padding: .3rem .7rem; margin-top: 1rem;
  font-size: .72rem; letter-spacing: .08em; text-transform: uppercase;
}
#about .close:hover { opacity: .6; }
</style>

<header id="header">
  <div id="title">
    <a class="brand" href="https://surulere.dev" target="_blank"
       rel="noopener noreferrer" title="surulere.dev"><!--__LOGO__--></a>
    <div class="title-box">Symbolic Motif Analysis</div>
  </div>

  <nav id="menu-bar" aria-label="Views and options">
    <button class="menu-bar-section-toggle" id="menu-toggle"
            aria-expanded="false" aria-controls="menu-bar-sections" aria-label="Menu">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <line x1="3" y1="6" x2="21" y2="6"></line>
        <line x1="3" y1="12" x2="21" y2="12"></line>
        <line x1="3" y1="18" x2="21" y2="18"></line>
      </svg>
    </button>
    <div class="menu-bar-sections" id="menu-bar-sections">
      <div class="menu-bar-section">
        <div class="menu-bar-section-label">VIEW</div>
        <button class="menu-item" data-view="panels">Panels</button>
        <button class="menu-item" data-view="motifs">Motifs</button>
        <button class="menu-item" data-view="families">Families</button>
        <button class="menu-item" data-view="synthesis">Synthesis</button>
      </div>
      <div class="menu-bar-section" id="menu-context">
        <div class="menu-bar-section-label">SHOW</div>
      </div>
      <div class="menu-bar-section">
        <div class="menu-bar-section-label">&middot;</div>
        <button class="menu-item" id="opt-notes">Notes</button>
        <button class="menu-item" id="opt-theme">Theme</button>
        <button class="menu-item" id="opt-about">About</button>
      </div>
    </div>
  </nav>
</header>

<nav id="crumbs" aria-label="Breadcrumb"></nav>

<div id="content">
  <main id="main">
    <section class="view" id="view-panels"></section>
    <section class="view" id="view-motifs" hidden></section>
    <section class="view" id="view-families" hidden></section>
    <section class="view" id="view-synthesis" hidden></section>
  </main>
  <aside id="sidebar" aria-label="Notes and interpretation"></aside>
</div>

<div id="statusbar">
  <div id="status-meta"></div>
  <a class="copyright" href="https://surulere.dev" target="_blank" rel="noopener noreferrer">
    &copy; SURULERE.DEV <span id="year"></span></a>
</div>

<div id="about" hidden role="dialog" aria-modal="true" aria-labelledby="about-title">
  <div class="sheet">
    <h2 id="about-title">About this reading</h2>
    <div id="about-body"></div>
    <button class="close" id="about-close">Close</button>
  </div>
</div>
</div>

<script>
const DATA = /*__PAYLOAD__*/null;

/* ══ Small helpers ════════════════════════════════════════════════════════ */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));
const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + "s")}`;

/* Family hues: evenly spaced, held at one saturation so a tint reads as an
   identity rather than as a position on a scale. Lightness follows the theme. */
const familyColor = (id) => id < 0 ? "var(--txt-color-annotation)"
  : `hsl(${(id * 47) % 360} 42% var(--fam-l))`;
const familyName = (cid) =>
  (DATA.families[String(cid)] || {}).name || (cid < 0 ? "unclustered" : `Cluster ${cid}`);

/* ══ Indexes ══════════════════════════════════════════════════════════════ */
const PANELS = DATA.panels;
const PANEL_BY_STEM = new Map(PANELS.map((p) => [p.stem, p]));
const MOTIF_BY_KEY = new Map();
for (const p of PANELS) for (const m of p.motifs) MOTIF_BY_KEY.set(m.key, { m, p });

const OBJECTS = new Map();                    /* object id → panel stems */
for (const p of PANELS) {
  if (!OBJECTS.has(p.object)) OBJECTS.set(p.object, []);
  OBJECTS.get(p.object).push(p.stem);
}

const cropOf = (key) => DATA.crops[key] || null;

/* ══ State ════════════════════════════════════════════════════════════════ */
const state = {
  view: "panels",
  panel: null,          /* stem, when a plate is open                     */
  motif: null,          /* "<stem>/<index>", when a motif is selected     */
  object: null,         /* object id, when the panel gallery is filtered  */
  family: null,         /* cluster id, when a family is selected          */
  group: "cluster",     /* motif gallery grouping: cluster | panel | label */
  boxes: true,
  registers: true,
  notes: true,
};

const VIEWS = ["panels", "motifs", "families", "synthesis"];

/* ══ Hash navigation ══════════════════════════════════════════════════════
   State lives in the hash, per the design system, so a reading can be linked
   to at the depth it was found. */
function writeHash() {
  const bits = [`view=${state.view}`];
  if (state.panel) bits.push(`panel=${state.panel}`);
  if (state.motif) bits.push(`motif=${state.motif}`);
  if (state.object) bits.push(`object=${state.object}`);
  if (state.family !== null) bits.push(`family=${state.family}`);
  if (state.view === "motifs") bits.push(`group=${state.group}`);
  const hash = "#" + bits.join("&");
  if (hash === location.hash) return;
  try { history.pushState(null, "", hash); }
  catch (e) { try { location.hash = hash; } catch (e2) { /* sandboxed */ } }
}

/* Reads *every* field, including the ones the hash leaves out — an absent key
   has to mean "back to the default", or Back out of a detail view would parse
   an empty hash and leave the detail on screen. */
function readHash() {
  const raw = (location.hash || "").replace(/^#/, "");
  const q = new Map(raw ? raw.split("&").map((kv) => {
    const i = kv.indexOf("=");
    return i < 0 ? [kv, ""] : [kv.slice(0, i), decodeURIComponent(kv.slice(i + 1))];
  }) : []);

  state.view = VIEWS.includes(q.get("view")) ? q.get("view") : "panels";
  state.panel = PANEL_BY_STEM.has(q.get("panel")) ? q.get("panel") : null;
  state.motif = MOTIF_BY_KEY.has(q.get("motif")) ? q.get("motif") : null;
  state.object = OBJECTS.has(q.get("object")) ? q.get("object") : null;
  const fam = q.get("family");
  state.family = fam !== undefined && DATA.families[fam] ? Number(fam) : null;
  state.group = ["cluster", "panel", "label"].includes(q.get("group"))
    ? q.get("group") : "cluster";
  if (state.motif && !state.panel) state.panel = MOTIF_BY_KEY.get(state.motif).p.stem;
}

/* Every navigation goes through here, so the hash, the three regions, and the
   menu can never disagree about where we are. */
function go(patch) {
  Object.assign(state, patch);
  if (state.motif && !MOTIF_BY_KEY.has(state.motif)) state.motif = null;
  if (state.motif) state.panel = MOTIF_BY_KEY.get(state.motif).p.stem;
  writeHash();
  render();
}

/* ══ Cross-references: prose → plates ═════════════════════════════════════
   Panel stems and object ids named in a reading become links to the plate
   they name.  Done over text nodes after insertion — never by rewriting HTML
   — so a match inside a tag or an existing link is impossible by construction. */
const REF_ALIASES = Object.keys(DATA.refs).sort((a, b) => b.length - a.length);
const REF_RE = REF_ALIASES.length
  ? new RegExp("(^|[^\\w-])(" +
      REF_ALIASES.map((a) => a.replace(/[.*+?^${}()|[\]\\-]/g, "\\$&")).join("|") +
      ")(?![\\w-])", "g")
  : null;

function refTarget(alias) {
  const stems = DATA.refs[alias] || [];
  if (stems.length === 1) return { panel: stems[0], motif: null, object: null };
  const obj = OBJECTS.has(alias) ? alias : (PANEL_BY_STEM.get(stems[0]) || {}).object;
  return { panel: null, motif: null, object: obj || null };
}

function linkifyRefs(root, panelStem) {
  const motifRe = panelStem ? /(^|[^\w#])#(\d{1,4})(?![\w])/g : null;
  if (!REF_RE && !motifRe) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const targets = [];
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.nodeValue.trim()) continue;
    if (node.parentElement.closest("a, button, code, .ref")) continue;
    targets.push(node);
  }
  for (const node of targets) replaceRefs(node, panelStem, motifRe);
}

function replaceRefs(node, panelStem, motifRe) {
  let html = esc(node.nodeValue);
  let touched = false;

  if (REF_RE) {
    REF_RE.lastIndex = 0;
    html = html.replace(REF_RE, (all, pre, alias) => {
      const t = refTarget(alias);
      if (!t.panel && !t.object) return all;
      touched = true;
      const attrs = t.panel ? `data-panel="${esc(t.panel)}"` : `data-object="${esc(t.object)}"`;
      const n = (DATA.refs[alias] || []).length;
      const title = t.panel ? `Open plate ${esc(alias)}` : `Show the ${n} plates of ${esc(alias)}`;
      return `${pre}<a class="ref" role="button" tabindex="0" ${attrs} title="${title}">${esc(alias)}</a>`;
    });
  }
  if (motifRe) {
    motifRe.lastIndex = 0;
    html = html.replace(motifRe, (all, pre, idx) => {
      const key = `${panelStem}/${Number(idx)}`;
      if (!MOTIF_BY_KEY.has(key)) return all;
      touched = true;
      return `${pre}<a class="ref" role="button" tabindex="0" data-motif="${esc(key)}"` +
             ` title="Motif ${esc(idx)} on this plate">#${esc(idx)}</a>`;
    });
  }
  if (!touched) return;

  const frag = document.createElement("span");
  frag.innerHTML = html;
  node.parentNode.replaceChild(frag, node);
}

/* One delegated handler serves every generated link, tile, and chip. */
function navFromEvent(e) {
  const el = e.target.closest("[data-motif], [data-panel], [data-object], [data-family]");
  if (!el) return false;
  if (el.dataset.motif) {
    const openDetail = el.dataset.zoom === "1" || state.view === "motifs";
    go({ view: openDetail ? "motifs" : "panels", motif: el.dataset.motif,
         family: MOTIF_BY_KEY.get(el.dataset.motif).m.cluster });
  } else if (el.dataset.panel) {
    go({ view: "panels", panel: el.dataset.panel, motif: null, object: null });
  } else if (el.dataset.object) {
    go({ view: "panels", panel: null, motif: null, object: el.dataset.object });
  } else if (el.dataset.family) {
    const cid = Number(el.dataset.family);
    go({ family: state.family === cid ? null : cid });
  }
  return true;
}
document.addEventListener("click", (e) => { if (navFromEvent(e)) e.preventDefault(); });
document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  if (!e.target.classList || !e.target.classList.contains("ref")) return;
  if (navFromEvent(e)) e.preventDefault();
});

/* ══ Menu ═════════════════════════════════════════════════════════════════ */
document.querySelectorAll(".menu-item[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const view = btn.dataset.view;
    /* The motif detail belongs to the motifs view, so switching away drops it —
       which is what makes the menu a way *out* of a detail as well as into a
       view. The open plate survives, because Plates is where it lives. */
    go({ view, motif: view === "motifs" ? state.motif : null });
  });
});

$("menu-toggle").addEventListener("click", () => {
  const bar = $("menu-bar");
  const open = bar.classList.toggle("is-open");
  $("menu-toggle").setAttribute("aria-expanded", String(open));
});

$("opt-notes").addEventListener("click", () => go({ notes: !state.notes }));

/* Theme cycles system → light → dark. `data-theme` on the root is what a host
   page stamps, so the toggle drives .LIGHT/.DARK on #layout instead and never
   fights it. */
const THEMES = ["system", "LIGHT", "DARK"];
let themeIndex = 0;
try {
  const saved = localStorage.getItem("sma-theme");
  if (saved && THEMES.includes(saved)) themeIndex = THEMES.indexOf(saved);
} catch (e) { /* storage may be blocked */ }

function applyTheme() {
  const theme = THEMES[themeIndex];
  $("layout").classList.remove("LIGHT", "DARK");
  if (theme !== "system") $("layout").classList.add(theme);
  const btn = $("opt-theme");
  btn.textContent = theme === "system" ? "Theme" : theme.toLowerCase();
  btn.setAttribute("aria-pressed", String(theme !== "system"));
  btn.title = `Theme: ${theme === "system" ? "follows your system" : theme.toLowerCase()}`;
  try { localStorage.setItem("sma-theme", theme); } catch (e) { /* ignore */ }
}
$("opt-theme").addEventListener("click", () => {
  themeIndex = (themeIndex + 1) % THEMES.length;
  applyTheme();
});

$("opt-about").addEventListener("click", () => { $("about").hidden = false; });
$("about-close").addEventListener("click", () => { $("about").hidden = true; });
$("about").addEventListener("click", (e) => {
  if (e.target === $("about")) $("about").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("about").hidden) $("about").hidden = true;
});

/* The second menu section changes with the view — it holds whatever controls
   the thing in the centre right now, and nothing else. */
function renderContextMenu() {
  const host = $("menu-context");
  const opts = [];
  if (state.view === "panels" && state.panel) {
    opts.push(["Boxes", state.boxes, () => go({ boxes: !state.boxes })]);
    opts.push(["Registers", state.registers, () => go({ registers: !state.registers })]);
  } else if (state.view === "motifs" && !state.motif) {
    for (const g of ["cluster", "panel", "label"]) {
      opts.push([`By ${g}`, state.group === g, () => go({ group: g })]);
    }
  } else if (state.view === "panels" && state.object) {
    opts.push([`All ${OBJECTS.size} objects`, false, () => go({ object: null })]);
  }
  host.innerHTML = `<div class="menu-bar-section-label">${
    state.view === "motifs" && !state.motif ? "GROUP" : "SHOW"}</div>`;
  host.hidden = opts.length === 0;
  for (const [label, on, fn] of opts) {
    const btn = document.createElement("button");
    btn.className = "menu-item";
    btn.textContent = label;
    btn.setAttribute("aria-pressed", String(on));
    btn.addEventListener("click", fn);
    host.appendChild(btn);
  }
}

/* ══ Breadcrumbs ══════════════════════════════════════════════════════════ */
function renderCrumbs() {
  const crumbs = [["Corpus", { view: "panels", panel: null, motif: null, object: null }]];
  const titleOf = { panels: "Panels", motifs: "Motifs", families: "Families",
                    synthesis: "Synthesis" }[state.view];
  crumbs.push([titleOf, { view: state.view, panel: null, motif: null }]);

  if (state.view === "panels" && state.object) {
    crumbs.push([state.object, { object: state.object, panel: null, motif: null }]);
  }
  if (state.view === "panels" && state.panel) {
    const p = PANEL_BY_STEM.get(state.panel);
    if (!state.object && OBJECTS.get(p.object).length > 1) {
      crumbs.push([p.object, { object: p.object, panel: null, motif: null }]);
    }
    crumbs.push([p.stem, { panel: p.stem, motif: null }]);
  }
  if (state.view === "motifs" && state.motif) {
    const { m, p } = MOTIF_BY_KEY.get(state.motif);
    crumbs.push([`${p.stem} #${m.index}`, { motif: m.key }]);
  }
  if (state.view === "panels" && state.motif) {
    crumbs.push([`#${MOTIF_BY_KEY.get(state.motif).m.index}`, { motif: state.motif }]);
  }

  $("crumbs").innerHTML = crumbs.map(([label, patch], i) => {
    const last = i === crumbs.length - 1;
    return (i ? '<span class="sep">/</span>' : "") +
      `<a class="crumb${last ? " is-current" : ""}" role="button" tabindex="${last ? -1 : 0}"` +
      ` data-crumb="${i}">${esc(label)}</a>`;
  }).join("");

  $("crumbs").querySelectorAll(".crumb").forEach((el, i) => {
    if (i === crumbs.length - 1) return;
    const patch = crumbs[i][1];
    el.addEventListener("click", () => go(patch));
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(patch); }
    });
  });
}

/* ══ Panels view ══════════════════════════════════════════════════════════ */
const els = {};

function panelTile(p) {
  const clusters = [...new Set(p.motifs.map((m) => m.cluster))].filter((c) => c >= 0);
  const bar = clusters.length
    ? `<span class="fam-bar" style="background:linear-gradient(to right,${
        clusters.map(familyColor).join(",")})"></span>` : "";
  return `<button class="tile tile-panel" data-panel="${esc(p.stem)}"
    title="${esc(p.stem)}${p.title && p.title !== p.stem ? " — " + esc(p.title) : ""}">
    <figure><img src="${p.image}" alt="Panel ${esc(p.stem)}" loading="lazy">${bar}</figure>
    <figcaption><span class="name">${esc(p.title && p.title !== p.stem ? p.title : p.stem)}</span>
      <span class="sub">${esc(p.object)}</span>
      <span class="sub">${plural(p.motifs.length, "motif")}
        &middot; ${p.reading ? "read" : "unread"}</span>
    </figcaption></button>`;
}

function renderPanelsView() {
  const host = $("view-panels");
  if (!els.panelsBuilt) {
    host.innerHTML = `<div id="panels-gallery">
        <div class="gallery-head"><h2 id="panels-head"></h2>
          <span class="count" id="panels-count"></span></div>
        <div class="grid-panels" id="panels-grid">${PANELS.map(panelTile).join("")}</div>
      </div>
      <div id="panels-detail" hidden>
        <div class="plate"><figure>
          <img id="plate-img" alt="">
          <svg id="plate-svg" viewBox="0 0 100 100" preserveAspectRatio="none"
               role="group" aria-label="Motif detections"></svg>
        </figure></div>
      </div>`;
    els.panelsBuilt = true;
  }

  const gallery = $("panels-gallery"), detail = $("panels-detail");
  gallery.hidden = !!state.panel;
  detail.hidden = !state.panel;
  if (state.panel) return drawPlate();

  const shown = state.object ? OBJECTS.get(state.object) : PANELS.map((p) => p.stem);
  const set = new Set(shown);
  $("panels-grid").querySelectorAll(".tile").forEach((t) => {
    t.hidden = !set.has(t.dataset.panel);
  });
  $("panels-head").textContent = state.object ? `Object ${state.object}` : "Plates";
  $("panels-count").textContent =
    `${plural(shown.length, "plate")} · ${plural(
      shown.reduce((n, s) => n + PANEL_BY_STEM.get(s).motifs.length, 0), "motif")}`;
}

function drawPlate() {
  const panel = PANEL_BY_STEM.get(state.panel);
  const img = $("plate-img");
  if (img.dataset.stem !== panel.stem) {
    img.src = panel.image;
    img.dataset.stem = panel.stem;
  }
  img.alt = `Panel ${panel.stem}, ${panel.motifs.length} detected motifs`;

  const svg = $("plate-svg");
  svg.setAttribute("viewBox", `0 0 ${panel.width} ${panel.height}`);
  const s = panel.width / 100;                  /* keeps strokes even across plates */
  const active = state.motif && MOTIF_BY_KEY.get(state.motif).p.stem === panel.stem
    ? MOTIF_BY_KEY.get(state.motif).m.index : null;
  let out = "";

  if (state.registers) {
    for (const reg of panel.registers) {
      const y = reg.y_top * panel.height, h = (reg.y_bottom - reg.y_top) * panel.height;
      out += `<rect class="band" x="0" y="${y}" width="${panel.width}" height="${h}"></rect>`
           + `<line class="band-line" x1="0" y1="${y}" x2="${panel.width}" y2="${y}"
                    stroke-width="${0.6 * s}"></line>`
           + `<text class="band-label" x="${0.6 * s}" y="${y + 3 * s}"
                    font-size="${2.2 * s}">REG ${reg.index}</text>`;
    }
  }
  if (state.boxes) {
    for (const m of panel.motifs) {
      const dim = active !== null && active !== m.index;
      const colour = familyColor(m.cluster);
      out += `<rect class="box${m.is_field ? " is-field" : ""}`
           + `${active === m.index ? " is-active" : ""}${dim ? " is-dim" : ""}"`
           + ` data-index="${m.index}" x="${m.x}" y="${m.y}" width="${m.w}" height="${m.h}"`
           + ` style="color:${colour}" stroke="${colour}" stroke-width="${0.7 * s}"`
           + ` tabindex="0" role="button"`
           + ` aria-label="Motif ${m.index}${m.label ? ": " + esc(m.label) : ""}"></rect>`
           + `<text class="box-num" x="${m.x + 0.9 * s}" y="${m.y + 3.1 * s}"`
           + ` font-size="${2.3 * s}" stroke-width="${0.7 * s}"`
           + ` fill="${colour}">${m.index}</text>`;
    }
  }
  svg.innerHTML = out;

  svg.querySelectorAll(".box").forEach((box) => {
    const key = `${panel.stem}/${box.dataset.index}`;
    const pick = () => go({ motif: state.motif === key ? null : key,
                            family: MOTIF_BY_KEY.get(key).m.cluster });
    box.addEventListener("click", pick);
    box.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
    });
    /* Hover names the motif in the status row and nowhere else.  It must not
       reach the notes column: the pointer sits wherever it last clicked, so a
       hover that rewrote the sidebar would silently replace the reading of a
       plate the moment you navigated to it. */
    box.addEventListener("mouseenter", () => renderStatus(key));
    box.addEventListener("mouseleave", () => renderStatus());
    box.addEventListener("focus", () => renderStatus(key));
    box.addEventListener("blur", () => renderStatus());
  });
}

/* ══ Motifs view ══════════════════════════════════════════════════════════ */
function motifTile(m, p) {
  const src = cropOf(m.key);
  const face = src
    ? `<img src="${src}" alt="${esc(p.stem)} motif ${m.index}" loading="lazy">`
    : `<div class="dim mono" style="aspect-ratio:1;display:grid;place-items:center">#${m.index}</div>`;
  return `<button class="tile tile-motif" data-motif="${esc(m.key)}" data-zoom="1"
     title="${esc(p.stem)} #${m.index}${m.label ? " — " + esc(m.label) : ""}">
     <figure>${face}<span class="fam-bar" style="background:${familyColor(m.cluster)}"></span></figure>
     <figcaption><span class="name">${esc(m.label || "unlabelled")}</span>
       <span class="sub">#${m.index} &middot; ${esc(p.object)}</span></figcaption></button>`;
}

function motifGroups() {
  const groups = new Map();
  const add = (k, entry) => {
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(entry);
  };
  for (const p of PANELS) for (const m of p.motifs) {
    if (state.group === "cluster") add(m.cluster, { m, p });
    else if (state.group === "panel") add(p.stem, { m, p });
    else add(m.label || "unlabelled", { m, p });
  }
  const keys = [...groups.keys()].sort((a, b) =>
    state.group === "cluster" ? a - b : String(a).localeCompare(String(b)));
  return keys.map((k) => [k, groups.get(k)]);
}

function renderMotifsView() {
  const host = $("view-motifs");
  if (state.motif) return renderMotifDetail(host);

  const signature = `${state.group}|${state.family}`;
  if (host.dataset.signature !== signature) {
    const groups = motifGroups();
    host.innerHTML = `<div class="gallery-head"><h2>Motifs</h2>
        <span class="count">${plural(DATA.scale.motifs, "motif")} in
          ${plural(groups.length, "group", "groups")}, by ${state.group}</span></div>`
      + groups.map(([k, members]) => {
          const isFamily = state.group === "cluster";
          const label = isFamily
            ? `<span class="swatch" style="background:${familyColor(k)}"></span>
               ${esc(familyName(k))}`
            : esc(String(k));
          const on = isFamily && state.family === Number(k);
          return `<section class="group">
            <button class="group-head${on ? " is-active" : ""}"
                    ${isFamily ? `data-family="${k}"` : ""}>
              ${label} <span class="n">${plural(members.length, "motif")}</span>
              ${isFamily ? `<span class="n">${on ? "— brief shown" : "— show brief"}</span>` : ""}
            </button>
            <div class="grid-motifs">${
              members.map(({ m, p }) => motifTile(m, p)).join("")}</div></section>`;
        }).join("");
    host.dataset.signature = signature;
  }
}

/* The zoom windows into the embedded panel rather than blowing up the crop.
   The panel is carried at `--max-dim` (1400px by default) and the crop at
   `--crop-dim` (160px), so this is several times the resolution for no extra
   bytes — the image is already in the page and already decoded. */
function zoomStyle(m, p) {
  const px = p.width === m.w ? 50 : (m.x / (p.width - m.w)) * 100;
  const py = p.height === m.h ? 50 : (m.y / (p.height - m.h)) * 100;
  return `background-image:url(${p.image});`
       + `background-size:${(p.width / m.w) * 100}% ${(p.height / m.h) * 100}%;`
       + `background-position:${px}% ${py}%;`
       + `aspect-ratio:${m.w} / ${m.h};`
       /* Bound by width rather than height so the ratio — and with it the
          background's registration — survives a tall motif. */
       + `max-width:calc((100vh - 13rem) * ${(m.w / m.h).toFixed(4)});`;
}

function renderMotifDetail(host) {
  const { m, p } = MOTIF_BY_KEY.get(state.motif);
  host.dataset.signature = "";                   /* gallery must rebuild on return */
  const siblings = [];
  for (const q of PANELS) for (const n of q.motifs) {
    if (n.cluster === m.cluster && m.cluster >= 0) siblings.push({ m: n, p: q });
  }
  const s = p.width / 100;

  host.innerHTML = `<div class="motif-detail">
    <div class="motif-zoom">
      <div class="zoom-face" role="img" style="${zoomStyle(m, p)}"
           aria-label="${esc(m.label || "Motif " + m.index)} on ${esc(p.stem)}"></div>
      <div class="eyebrow" style="margin:.5rem 0 0">${esc(p.stem)} &middot; #${m.index}
        &middot; ${esc(m.scale)}</div>
    </div>
    <div class="motif-context">
      <div class="eyebrow">On the plate</div>
      <div class="plate"><figure>
        <img src="${p.image}" alt="Panel ${esc(p.stem)}">
        <svg viewBox="0 0 ${p.width} ${p.height}" preserveAspectRatio="none" aria-hidden="true">
          <rect class="box is-active" x="${m.x}" y="${m.y}" width="${m.w}" height="${m.h}"
                style="color:${familyColor(m.cluster)}" stroke="${familyColor(m.cluster)}"
                stroke-width="${1.2 * s}"></rect>
        </svg>
      </figure></div>
      <button class="chip" data-panel="${esc(p.stem)}" style="margin-top:.4rem">
        Open the full plate</button>
      ${siblings.length > 1 ? `<div class="eyebrow" style="margin-top:.7rem">
        ${esc(familyName(m.cluster))} &middot; ${plural(siblings.length, "member")}</div>
        <div class="strip">${siblings.map(({ m: n, p: q }) => {
          const c = cropOf(n.key);
          return `<button data-motif="${esc(n.key)}" data-zoom="1"
            class="${n.key === m.key ? "is-active" : ""}" title="${esc(q.stem)} #${n.index}">
            ${c ? `<img src="${c}" alt="${esc(q.stem)} #${n.index}" loading="lazy">`
                : `<span class="mono">#${n.index}</span>`}</button>`;
        }).join("")}</div>` : ""}
    </div>
  </div>`;
}

/* ══ Families view ════════════════════════════════════════════════════════ */
function renderFamiliesView() {
  const host = $("view-families");
  const signature = String(state.family);
  if (host.dataset.signature === signature) return;
  host.dataset.signature = signature;

  const ids = Object.keys(DATA.families).sort((a, b) => a - b);
  if (!ids.length) {
    host.innerHTML = `<p class="empty">No motif families yet — run the clustering stage.</p>`;
    return;
  }
  host.innerHTML = `<div class="gallery-head"><h2>Motif families</h2>
      <span class="count">${plural(ids.length, "family", "families")} · select one for its brief</span></div>
    <div class="grid-panels" style="grid-template-columns:repeat(auto-fill,minmax(min(100%,22rem),1fr))">`
    + ids.map((cid) => {
        const f = DATA.families[cid];
        const on = state.family === Number(cid);
        return `<div class="card${on ? " is-active" : ""}"
                     style="${on ? "border-style:solid" : ""}">
          <div class="eyebrow">Cluster ${cid}</div>
          <h3><span class="swatch" style="background:${familyColor(Number(cid))}"></span>
            ${esc(f.name || "Unnamed family")}</h3>
          ${familySummary(Number(cid), f, 12)}
          ${f.visual_definition ? `<p class="serif">${esc(f.visual_definition)}</p>` : ""}
          ${f.iconographic_reading ? `<p class="serif"><em>${esc(f.iconographic_reading)}</em></p>` : ""}
          ${f.visual_definition || f.iconographic_reading ? "" :
            `<p class="empty">No brief written for this family yet.</p>`}
          <div class="chips"><button class="chip${on ? " is-on" : ""}" data-family="${cid}">
            ${on ? "brief shown" : "show brief"}</button></div>
        </div>`;
      }).join("") + `</div>`;
}

/* ══ Synthesis view ═══════════════════════════════════════════════════════ */
function renderSynthesisView() {
  const host = $("view-synthesis");
  if (host.dataset.built) return;
  host.dataset.built = "1";
  host.innerHTML = `<article class="prose">${
    DATA.synthesis || `<p class="empty">No corpus synthesis has been generated yet.</p>`
  }</article>`;
  linkifyRefs(host);
}

/* ══ Card summaries ═══════════════════════════════════════════════════════
   A brief is a wall of prose, and prose is the wrong thing to meet first.
   Every summarising card therefore opens with what it is made of — the counts,
   then the faces — and only then makes its claim.  Counts read value-first
   ("6 members · 5 plates") because at a glance the number is the news. */

const tally = (...pairs) =>
  `<div class="tally">` + pairs
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `<span><b>${v}</b> <span class="k">${k}</span></span>`)
    .join("") + `</div>`;

/* Motif crops, as a row of buttons that open the motif they show. */
function cropStrip(keys, limit) {
  const faces = keys.slice(0, limit).map((key) => {
    const src = cropOf(key), ref = MOTIF_BY_KEY.get(key);
    if (!src || !ref) return "";
    return `<button data-motif="${esc(key)}" data-zoom="1"
      title="${esc(ref.p.stem)} #${ref.m.index}${ref.m.label ? " — " + esc(ref.m.label) : ""}">
      <img src="${src}" alt="${esc(ref.p.stem)} motif ${ref.m.index}" loading="lazy"></button>`;
  }).join("");
  return faces ? `<div class="strip">${faces}</div>` : "";
}

/* Plates, likewise — the corpus's own thumbnails rather than its motifs'. */
function plateStrip(panels, limit) {
  const faces = panels.slice(0, limit).map((p) =>
    `<button data-panel="${esc(p.stem)}" title="${esc(p.stem)} — ${
      plural(p.motifs.length, "motif")}">
      <img src="${p.image}" alt="Panel ${esc(p.stem)}" loading="lazy"></button>`).join("");
  return faces ? `<div class="strip">${faces}</div>` : "";
}

function familySummary(cid, f, cropLimit) {
  return `<div class="summary">`
    + tally(["members", f.size], ["plates", f.panel_spread],
            ["cohesion", f.cohesion == null ? null : f.cohesion.toFixed(3)],
            ["confidence", f.confidence ? esc(f.confidence) : null])
    + cropStrip(f.exemplars || [], cropLimit) + `</div>`;
}

/* ══ Sidebar ══════════════════════════════════════════════════════════════
   Whatever is in the centre, this column holds what has been *said* about it:
   corpus brief → object → plate reading → motif annotation and its family. */
function renderSidebar() {
  const el = $("sidebar");
  el.hidden = !state.notes;
  if (!state.notes) return;

  const key = state.motif;
  let html;
  if (key && MOTIF_BY_KEY.has(key)) {
    const { m } = MOTIF_BY_KEY.get(key);
    html = motifCard(m) + familyCard(m.cluster);
  } else if (state.view === "panels" && state.panel) {
    html = panelCards(PANEL_BY_STEM.get(state.panel));
  } else if (state.family !== null && DATA.families[String(state.family)]) {
    html = familyCard(state.family, true) + corpusCard();
  } else if (state.view === "panels" && state.object) {
    html = objectCard(state.object);
  } else {
    html = corpusCard();
  }
  el.innerHTML = html;
  linkifyRefs(el, state.panel);
}

function corpusCard() {
  const s = DATA.scale;
  const top = Object.entries(DATA.families)
    .sort((a, b) => b[1].size - a[1].size).slice(0, 8);
  return `<div class="card">
    <div class="eyebrow">The corpus</div>
    <h3>${esc(DATA.title)}</h3>
    <div class="summary">
      ${tally(["plates", s.panels], ["objects", OBJECTS.size], ["motifs", s.motifs],
              ["families", s.clusters], ["labelled", s.labelled],
              ["unclustered", s.unclustered], ["read", `${s.readings}/${s.panels}`])}
      ${plateStrip(PANELS, 8)}
    </div>
    ${DATA.synthesis_lead
      ? `<div class="serif">${DATA.synthesis_lead}</div>`
      : `<p class="empty">No corpus synthesis has been generated yet — the plates,
         motifs, and families above are what the pipeline recovered on its own.</p>`}
    ${DATA.synthesis ? `<div class="chips"><button class="chip" id="go-synthesis">
      Read the full synthesis</button></div>` : ""}
  </div>` + (top.length ? `<div class="card">
    <div class="eyebrow">Largest families</div>
    <div class="fam-row">${top.map(([cid, f]) => {
      const on = state.family === Number(cid);
      const key = (f.exemplars || [])[0];
      const src = key ? cropOf(key) : null;
      return `<button data-family="${cid}" class="${on ? "is-on" : ""}"
        title="${esc(f.name || "cluster " + cid)} — ${plural(f.size, "member")} across
               ${plural(f.panel_spread, "plate")}">
        ${src ? `<img src="${src}" alt="" loading="lazy">`
              : `<span class="blank" style="background:${familyColor(Number(cid))}"></span>`}
        <span class="cap"><span class="swatch"
          style="background:${familyColor(Number(cid))}"></span>${f.size}</span>
        <span class="cap dim">${esc(f.name || "cluster " + cid)}</span></button>`;
    }).join("")}</div>
  </div>` : "");
}

function objectCard(object) {
  const stems = OBJECTS.get(object) || [];
  const motifs = stems.reduce((n, s) => n + PANEL_BY_STEM.get(s).motifs.length, 0);
  const fams = new Set();
  for (const s of stems) for (const m of PANEL_BY_STEM.get(s).motifs) {
    if (m.cluster >= 0) fams.add(m.cluster);
  }
  return `<div class="card">
    <div class="eyebrow">Object</div>
    <h3 class="mono">${esc(object)}</h3>
    <p>${plural(stems.length, "plate")} cut from this object, carrying
       ${plural(motifs, "motif")} across ${plural(fams.size, "family", "families")}.</p>
    <div class="chips">${stems.map((s) =>
      `<button class="chip" data-panel="${esc(s)}">${esc(s.replace(object, "").replace(/^_/, "") || s)}</button>`
    ).join("")}</div>
  </div>` + corpusCard();
}

function panelCards(panel) {
  const r = panel.reading;
  let html = `<div class="card"><div class="eyebrow">Plate reading</div>`;
  if (!r) {
    html += `<h3 class="mono">${esc(panel.stem)}</h3>`
          + `<p class="empty">No reading has been generated for this plate yet.
             Run the panels stage, or hover a box to read its motif.</p></div>`;
    return html + registersCard(panel) + `<div class="card">
      <div class="eyebrow">Object</div>
      <button class="chip" data-object="${esc(panel.object)}">${esc(panel.object)}</button></div>`;
  }
  html += `<h3>${esc(r.title || panel.stem)}</h3>`;
  if (r.confidence) {
    html += `<div class="eyebrow" style="margin:.3rem 0 0">Confidence — ${esc(r.confidence)}</div>`;
  }
  for (const field of ["summary", "composition", "narrative"]) {
    if (r[field]) html += `<p class="serif">${esc(r[field])}</p>`;
  }
  html += `<div class="chips"><button class="chip" data-object="${esc(panel.object)}">
    ${esc(panel.object)}</button></div></div>`;

  html += registersCard(panel);
  if (r.cross_panel_links?.length) {
    html += `<div class="card"><div class="eyebrow">Links across the corpus</div>`
          + `<ul class="notes">${r.cross_panel_links.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  }
  if (r.uncertainties?.length) {
    html += `<div class="card"><div class="eyebrow">Uncertainties</div>`
          + `<ul class="notes">${r.uncertainties.map((u) => `<li>${esc(u)}</li>`).join("")}</ul></div>`;
  }
  return html;
}

function registersCard(panel) {
  if (!panel.registers.length) return "";
  const readings = new Map(
    (panel.reading?.register_readings ?? []).map((r) => [r.register, r.reading]));
  return `<div class="card"><div class="eyebrow">Registers, top to bottom</div>`
    + `<ul class="registers">` + panel.registers.map((reg) => {
        /* `members` are placement keys, not indices — the number a reader
           recognises has to come back off the motif. */
        const members = reg.members.map((key) => {
          const ref = MOTIF_BY_KEY.get(key);
          return ref ? `<button class="chip" data-motif="${esc(key)}"
            title="${esc(ref.m.label || "unlabelled")}">#${ref.m.index}</button>` : "";
        }).join("");
        const text = readings.get(reg.index);
        return `<li><div class="reg-head">Register ${reg.index} — `
             + `${(reg.y_top * 100).toFixed(0)}%&ndash;${(reg.y_bottom * 100).toFixed(0)}%`
             + ` of height</div>`
             + (text ? `<p>${esc(text)}</p>` : "")
             + `<div class="chips">${members}</div></li>`;
      }).join("") + `</ul></div>`;
}

function motifCard(m) {
  const rows = [
    ["plate", `<button class="chip" data-panel="${esc(m.panel_stem)}">${esc(m.panel_stem)}</button>`],
    ["index", `#${m.index}`],
    ["family", m.cluster >= 0
      ? `<button class="chip" data-family="${m.cluster}"><span class="swatch"
         style="background:${familyColor(m.cluster)}"></span>${esc(familyName(m.cluster))}</button>`
      : "unclustered"],
    ["scale", esc(m.scale)],
    ["position", esc(m.zone)],
    ["area", `${(m.area_fraction * 100).toFixed(1)}% of plate`],
    ["register", m.register >= 0 ? `register ${m.register}` : "field / ground"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  const src = cropOf(m.key);
  let html = `<div class="card"><div class="eyebrow">Motif</div>`
    + `<h3>${esc(m.label || "Unlabelled detection")}</h3>`;
  if (src) {
    html += `<button class="tile tile-motif" data-motif="${esc(m.key)}" data-zoom="1"
      style="width:6rem;margin-top:.5rem"><figure>
      <img src="${src}" alt="${esc(m.label || "motif " + m.index)}"></figure></button>`;
  }
  if (m.description) html += `<p class="serif">${esc(m.description)}</p>`;
  if (m.iconography) html += `<p class="serif"><em>${esc(m.iconography)}</em></p>`;
  if (m.notes) html += `<p>${esc(m.notes)}</p>`;
  if (!m.label && !m.description) {
    html += `<p class="empty">No label recorded — this detection was never annotated.</p>`;
  }
  html += `<dl class="meta">${rows}</dl>`;
  if (m.label_source) {
    html += `<div class="eyebrow" style="margin:.6rem 0 0">Label source — ${esc(m.label_source)}</div>`;
  }
  return html + `</div>`;
}

function familyCard(cid, expanded) {
  const f = DATA.families[String(cid)];
  if (!f) return "";
  let html = `<div class="card"><div class="eyebrow">Motif family</div>`
    + `<h3><span class="swatch" style="background:${familyColor(cid)}"></span>`
    + `${esc(f.name || `Cluster ${cid}`)}</h3>`
    + familySummary(cid, f, expanded ? 12 : 6);

  const fields = expanded
    ? ["visual_definition", "variation", "distribution_note"] : ["visual_definition"];
  let said = false;
  for (const field of fields) {
    if (f[field]) { html += `<p class="serif">${esc(f[field])}</p>`; said = true; }
  }
  if (f.iconographic_reading) {
    html += `<p class="serif"><em>${esc(f.iconographic_reading)}</em></p>`;
    said = true;
  }
  if (!said) {
    html += `<p class="empty">No brief written for this family yet — the counts
      and members above are what the clustering found on its own.</p>`;
  }
  return html + `</div>`;
}

/* ══ Status row ═══════════════════════════════════════════════════════════ */
function renderStatus(hoverKey) {
  const bits = [];
  const add = (k, v) => bits.push(`<span><span class="k">${k}</span> <b>${v}</b></span>`);

  if (state.view === "panels" && state.panel) {
    const p = PANEL_BY_STEM.get(state.panel);
    add("file", `${esc(p.stem)}.png`);
    add("size", `${p.width}&times;${p.height}px`);
    add("motifs", p.motifs.length);
    add("registers", p.registers.length);
    add("reading", p.reading ? esc(p.reading.confidence || "yes") : "none");
    /* A hover over the motif already selected is not news — keep the stronger
       word rather than flickering between the two. */
    const hovering = hoverKey && hoverKey !== state.motif;
    const shown = hovering ? hoverKey : state.motif;
    if (shown && MOTIF_BY_KEY.has(shown)) {
      const { m } = MOTIF_BY_KEY.get(shown);
      add(hovering ? "under cursor" : "selected",
          `#${m.index} ${esc(m.label || "unlabelled")}`
          + (m.cluster >= 0 ? ` · ${esc(familyName(m.cluster))}` : ""));
    }
  } else if (state.view === "motifs" && state.motif) {
    const { m, p } = MOTIF_BY_KEY.get(state.motif);
    add("motif", `${esc(p.stem)} #${m.index}`);
    add("label", esc(m.label || "unlabelled"));
    add("family", esc(familyName(m.cluster)));
    add("position", esc(m.zone || "—"));
    add("area", `${(m.area_fraction * 100).toFixed(1)}%`);
  } else if (state.view === "motifs") {
    add("motifs", DATA.scale.motifs);
    add("grouped by", esc(state.group));
    add("labelled", `${DATA.scale.labelled} of ${DATA.scale.motifs}`);
  } else if (state.view === "families") {
    add("families", DATA.scale.clusters);
    add("unclustered", DATA.scale.unclustered);
    if (state.family !== null) add("selected", esc(familyName(state.family)));
  } else if (state.view === "synthesis") {
    add("source", "interpretation/corpus.md");
    add("plates read", `${DATA.scale.readings} of ${DATA.scale.panels}`);
  } else {
    add("plates", state.object ? OBJECTS.get(state.object).length : DATA.scale.panels);
    add("objects", OBJECTS.size);
    add("motifs", DATA.scale.motifs);
    add("families", DATA.scale.clusters);
  }
  $("status-meta").innerHTML = bits.join("");
}

/* ══ About ════════════════════════════════════════════════════════════════ */
$("about-body").innerHTML = `
  <p>This page is the readable end of a segmentation-and-clustering pipeline run
     over carved panel photographs from the Frobenius archive. Panels were split
     from source plates, motifs segmented with SAM-2, embedded, and clustered;
     the layout pass recovered registers and reading order from the bounding
     boxes alone; the interpretation pass joined all of it into the readings you
     see in the right-hand column.</p>
  <p>Everything here is <b>evidence plus a reading of it</b>, and the two are kept
     apart on purpose. Geometry, cluster membership, and statistics are what the
     pipeline measured. Titles, summaries, family names, and the synthesis are
     interpretation — generated or human-written, with the source of every label
     recorded on the motif itself. Confidence is stated where it was assessed.
     Nothing here is a museum determination.</p>
  <p>This run covers <b>${DATA.scale.panels}</b> plates from
     <b>${OBJECTS.size}</b> objects, <b>${DATA.scale.motifs}</b> motifs
     (<b>${DATA.scale.labelled}</b> labelled) in
     <b>${DATA.scale.clusters}</b> families, with
     <b>${DATA.scale.readings}</b> plates read.
     Built ${esc(DATA.generated)}.</p>
  <p class="dim">Identifiers named in a reading are links: click one to open the
     plate it refers to. Arrow keys walk a plate in the reading order the layout
     pass computed. Everything — images, crops, and text — is embedded in this
     one file; it needs no server and no network.</p>`;

/* ══ Keyboard ═════════════════════════════════════════════════════════════ */
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const panel = state.panel ? PANEL_BY_STEM.get(state.panel) : null;
  if (!panel || !panel.reading_order.length) return;
  const order = panel.reading_order;
  const at = state.motif ? order.indexOf(MOTIF_BY_KEY.get(state.motif).m.index) : -1;
  const step = e.key === "ArrowRight" ? 1 : -1;
  const from = at < 0 ? (step > 0 ? -1 : 0) : at;
  e.preventDefault();
  go({ motif: `${panel.stem}/${order[(from + step + order.length) % order.length]}` });
});

/* ══ Render ═══════════════════════════════════════════════════════════════ */
function render() {
  document.querySelectorAll(".menu-item[data-view]").forEach((b) => {
    if (b.dataset.view === state.view) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  $("opt-notes").setAttribute("aria-pressed", String(state.notes));
  renderContextMenu();
  renderCrumbs();

  VIEWS.forEach((v) => { $(`view-${v}`).hidden = v !== state.view; });
  if (state.view === "panels") renderPanelsView();
  else if (state.view === "motifs") renderMotifsView();
  else if (state.view === "families") renderFamiliesView();
  else renderSynthesisView();

  renderSidebar();
  renderStatus();
  const jump = $("go-synthesis");
  if (jump) jump.addEventListener("click", () => go({ view: "synthesis" }));
}

window.addEventListener("popstate", () => { readHash(); render(); });
window.addEventListener("hashchange", () => { readHash(); render(); });

$("year").textContent = String(new Date().getFullYear());
applyTheme();
readHash();
if (!state.panel && !PANELS.length) state.view = "synthesis";
render();
</script>
"""


def render(payload: dict[str, Any], standalone: bool = True) -> str:
    """Return the page with `payload` embedded as its data island.

    `standalone` wraps the page in a document shell — the right default,
    because the usual destination is a file someone opens off disk.  Pass
    False when the page is going somewhere that supplies its own `<head>`,
    such as an artifact host, which would otherwise end up with two.

    `</script>` inside the JSON would close the block early, so it is escaped —
    the sequence cannot appear in valid JSON string data any other way.
    """
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE
            .replace(_TEXTURE_LIGHT_SENTINEL, TEXTURE_LIGHT)
            .replace(_TEXTURE_DARK_SENTINEL, TEXTURE_DARK)
            .replace(_LOGO_SENTINEL, logo_svg())
            .replace(_PAYLOAD_SENTINEL, blob))
    if not standalone:
        return page
    return (DOCUMENT
            .replace("__TITLE__", TITLE)
            .replace("__FAVICON__", favicon_data_uri())
            .replace("__PAGE__", page))
