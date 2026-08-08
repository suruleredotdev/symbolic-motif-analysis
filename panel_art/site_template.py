"""
site_template.py — the HTML/CSS/JS shell for the interpretation site.

Kept apart from `export_interpretation_site.py` so the exporter stays readable
as data-assembly code and the page stays readable as a page.  `PAGE` is a
`string.Template` — the only substitution is `$payload`, the JSON blob of
panels, motifs, families, and readings.

Design notes, since they are choices rather than defaults:

  * The archive's own idiom is the annotated plate — a numbered figure with
    letterpress apparatus beside it — so the page is built as a plate and its
    apparatus, not as a dashboard.
  * Registers are drawn as bands across the plate and boxes are tinted by
    motif family. Both are things the pipeline discovered and neither is
    legible in the JSON; on the plate they can be checked at a glance.
  * Palette is mount board and iron-oxide ink. Neutrals carry a green bias so
    they read as chosen rather than as default grey.
  * No webfonts: the artifact CSP blocks font CDNs, and a silent fallback is
    worse than an honest system stack.

Substitution is a plain sentinel replace rather than `string.Template` or
`str.format`, because the page is full of JS `${...}` literals and CSS braces
that both of those would try to interpret.
"""

import json
from typing import Any

_PAYLOAD_SENTINEL = "/*__PAYLOAD__*/null"

PAGE = r"""<div id="app">
<style>
:root {
  /* Light: mount board, iron-oxide ink. Neutrals biased green so they read
     as chosen rather than inherited. */
  --board:      #E9EAE5;
  --plate:      #F4F5F1;
  --sunk:       #DEE0D9;
  --ink:        #1C1F1B;
  --ink-soft:   #575C54;
  --ink-faint:  #868B80;
  --rule:       #C3C6BD;
  --bole:       #9E4A2E;
  --bole-soft:  #B9714F;
  --verdigris:  #4A5D52;
  --shadow:     0 1px 2px rgba(28,31,27,.10), 0 8px 24px rgba(28,31,27,.06);
  --fam-l:      40%;

  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  --sans:  system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono:  ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;

  --step--1: clamp(.74rem, .72rem + .1vw, .80rem);
  --step-0:  clamp(.94rem, .90rem + .18vw, 1.02rem);
  --step-1:  clamp(1.12rem, 1.05rem + .3vw, 1.28rem);
  --step-2:  clamp(1.4rem, 1.25rem + .6vw, 1.75rem);
  --step-3:  clamp(1.85rem, 1.5rem + 1.2vw, 2.6rem);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --board:     #141613;
    --plate:     #1C1F1B;
    --sunk:      #242722;
    --ink:       #E4E6DF;
    --ink-soft:  #A9AEA3;
    --ink-faint: #767B71;
    --rule:      #33372F;
    --bole:      #C4643F;
    --bole-soft: #D98A66;
    --verdigris: #7E9C89;
    --shadow:    0 1px 2px rgba(0,0,0,.5), 0 10px 30px rgba(0,0,0,.35);
    --fam-l:     64%;
  }
}
:root[data-theme="dark"] {
  --board:     #141613;
  --plate:     #1C1F1B;
  --sunk:      #242722;
  --ink:       #E4E6DF;
  --ink-soft:  #A9AEA3;
  --ink-faint: #767B71;
  --rule:      #33372F;
  --bole:      #C4643F;
  --bole-soft: #D98A66;
  --verdigris: #7E9C89;
  --shadow:    0 1px 2px rgba(0,0,0,.5), 0 10px 30px rgba(0,0,0,.35);
  --fam-l:     64%;
}

* { box-sizing: border-box; }
#app { background: var(--board); color: var(--ink); font-family: var(--sans);
       font-size: var(--step-0); line-height: 1.55; min-height: 100vh; }
h1,h2,h3,h4 { text-wrap: balance; margin: 0; font-weight: 600; }
a { color: var(--bole); }
:focus-visible { outline: 2px solid var(--bole); outline-offset: 2px; border-radius: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }

/* ── Masthead ─────────────────────────────────────────────────────────── */
.masthead { border-bottom: 1px solid var(--rule); padding: 1.4rem clamp(1rem,4vw,2.5rem) 0;
            display: flex; flex-direction: column; gap: 1.1rem; }
.masthead-top { display: flex; flex-wrap: wrap; align-items: baseline;
                justify-content: space-between; gap: 1rem; }
.title { font-family: var(--serif); font-size: var(--step-2); letter-spacing: -.01em; }
.subtitle { font-family: var(--serif); color: var(--ink-soft); font-style: italic;
            font-size: var(--step-0); max-width: 62ch; }
.counts { font-family: var(--mono); font-size: var(--step--1); color: var(--ink-faint);
          font-variant-numeric: tabular-nums; display: flex; gap: 1.1rem; flex-wrap: wrap; }
.counts b { color: var(--ink-soft); font-weight: 600; }

.tabs { display: flex; gap: .2rem; }
.tab { appearance: none; background: none; border: 0; border-bottom: 2px solid transparent;
       color: var(--ink-faint); font: inherit; font-size: var(--step--1);
       letter-spacing: .09em; text-transform: uppercase; padding: .55rem .85rem;
       cursor: pointer; transition: color .15s, border-color .15s; }
.tab:hover { color: var(--ink); }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--bole); }

main { padding: clamp(1rem,3vw,2rem) clamp(1rem,4vw,2.5rem) 4rem; }
.view[hidden] { display: none; }

/* ── Plate + apparatus ────────────────────────────────────────────────── */
.plate-layout { display: grid; grid-template-columns: minmax(0,1fr) minmax(300px, 25rem);
                gap: clamp(1rem,2.5vw,2rem); align-items: start; }
@media (max-width: 62rem) { .plate-layout { grid-template-columns: minmax(0,1fr); } }

.picker { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
          margin-bottom: .9rem; }
select, .toggle { font: inherit; font-size: var(--step--1); color: var(--ink);
                  background: var(--plate); border: 1px solid var(--rule);
                  border-radius: 3px; padding: .35rem .55rem; }
select { max-width: min(100%, 34rem); }
.toggle { display: inline-flex; align-items: center; gap: .4rem; cursor: pointer;
          user-select: none; }
.toggle input { accent-color: var(--bole); margin: 0; }

.plate { position: relative; background: var(--plate); border: 1px solid var(--rule);
         box-shadow: var(--shadow); padding: .7rem; }
.plate figure { margin: 0; position: relative; line-height: 0; }
.plate img { width: 100%; height: auto; display: block; }
.plate svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.plate figcaption { font-family: var(--mono); font-size: var(--step--1);
                    color: var(--ink-faint); padding-top: .6rem; line-height: 1.4;
                    word-break: break-all; }

.band { fill: var(--verdigris); opacity: .13; }
.band-line { stroke: var(--verdigris); opacity: .55; stroke-dasharray: 4 3; }
.band-label { fill: var(--verdigris); font-family: var(--mono);
              opacity: .85; letter-spacing: .06em; }
.box { fill: transparent; stroke-width: .7; cursor: pointer;
       transition: fill .12s, stroke-width .12s; }
.box:hover { fill: currentColor; fill-opacity: .16; stroke-width: 1.3; }
.box.is-field { stroke-dasharray: 2.5 2; }
.box.is-active { fill: currentColor; fill-opacity: .22; stroke-width: 1.6; }
.box.is-dim { opacity: .22; }
.box-num { font-family: var(--mono); paint-order: stroke; stroke: var(--plate);
           pointer-events: none; font-variant-numeric: tabular-nums;
           font-weight: 600; }

/* ── Apparatus ────────────────────────────────────────────────────────── */
.apparatus { position: sticky; top: 1rem; display: flex; flex-direction: column;
             gap: 1rem; max-height: calc(100vh - 2rem); overflow-y: auto;
             padding-right: .3rem; }
@media (max-width: 62rem) { .apparatus { position: static; max-height: none; } }

.card { background: var(--plate); border: 1px solid var(--rule); padding: 1rem 1.1rem; }
.card h3 { font-family: var(--serif); font-size: var(--step-1); margin-bottom: .15rem; }
.eyebrow { font-family: var(--mono); font-size: var(--step--1); letter-spacing: .12em;
           text-transform: uppercase; color: var(--ink-faint); margin-bottom: .5rem; }
.card p { margin: .5rem 0 0; font-family: var(--serif); }
.card p.plain { font-family: var(--sans); }
dl.meta { display: grid; grid-template-columns: auto 1fr; gap: .25rem .8rem;
          margin: .7rem 0 0; font-size: var(--step--1); }
dl.meta dt { font-family: var(--mono); color: var(--ink-faint); letter-spacing: .04em; }
dl.meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.swatch { display: inline-block; width: .62em; height: .62em; border-radius: 50%;
          margin-right: .4em; vertical-align: baseline; }

.registers { list-style: none; margin: 0; padding: 0;
             display: flex; flex-direction: column; gap: .8rem; }
.registers li { border-left: 2px solid var(--verdigris); padding-left: .8rem; }
.registers .reg-head { font-family: var(--mono); font-size: var(--step--1);
                       color: var(--ink-faint); letter-spacing: .06em; }
.registers p { margin: .2rem 0 0; font-family: var(--serif); }

.chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .6rem; }
.chip { font-family: var(--mono); font-size: var(--step--1); border: 1px solid var(--rule);
        border-radius: 999px; padding: .1rem .55rem; cursor: pointer; background: none;
        color: var(--ink-soft); }
.chip:hover { border-color: currentColor; }
.notes { margin: .6rem 0 0; padding-left: 1.1rem; font-size: var(--step--1);
         color: var(--ink-soft); }
.notes li { margin-bottom: .25rem; }
.empty { color: var(--ink-faint); font-style: italic; font-family: var(--serif); }

/* ── Families ─────────────────────────────────────────────────────────── */
.families { display: grid; gap: 1rem;
            grid-template-columns: repeat(auto-fill, minmax(min(100%, 24rem), 1fr)); }
.family h3 { display: flex; align-items: baseline; gap: .5rem; }
.crops { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .8rem; }
.crops img { width: 62px; height: 62px; object-fit: cover; background: var(--sunk);
             border: 1px solid var(--rule); cursor: pointer; }
.crops img:hover { border-color: var(--bole); }

/* ── Synthesis ────────────────────────────────────────────────────────── */
.prose { max-width: 68ch; font-family: var(--serif); font-size: var(--step-1);
         line-height: 1.68; }
.prose h1 { font-size: var(--step-3); margin: 2rem 0 .6rem; letter-spacing: -.015em; }
.prose h2 { font-size: var(--step-2); margin: 2rem 0 .5rem; }
.prose h3 { font-size: var(--step-1); margin: 1.6rem 0 .4rem; }
.prose p, .prose li { margin: .8rem 0; }
.prose ul, .prose ol { padding-left: 1.4rem; }
.prose code { font-family: var(--mono); font-size: .88em; background: var(--sunk);
              padding: .08em .35em; border-radius: 2px; }
.prose hr { border: 0; border-top: 1px solid var(--rule); margin: 2rem 0; }
.prose strong { font-weight: 650; }
</style>

<header class="masthead">
  <div class="masthead-top">
    <div>
      <h1 class="title">Carved Panel Motifs — an Interpretation</h1>
      <p class="subtitle">Segmented, clustered, and read. Every box is a
        detection; every tint is a motif family recurring across the corpus.</p>
    </div>
    <div class="counts" id="counts"></div>
  </div>
  <nav class="tabs" role="tablist">
    <button class="tab" role="tab" data-view="plates"    aria-selected="true">Plates</button>
    <button class="tab" role="tab" data-view="families"  aria-selected="false">Families</button>
    <button class="tab" role="tab" data-view="synthesis" aria-selected="false">Synthesis</button>
  </nav>
</header>

<main>
  <section class="view" id="view-plates">
    <div class="picker">
      <label for="panel-select" class="eyebrow" style="margin:0">Plate</label>
      <select id="panel-select"></select>
      <label class="toggle"><input type="checkbox" id="show-registers" checked> Registers</label>
      <label class="toggle"><input type="checkbox" id="show-field" checked> Field detections</label>
    </div>
    <div class="plate-layout">
      <div class="plate">
        <figure>
          <img id="plate-img" alt="">
          <svg id="plate-svg" viewBox="0 0 100 100" preserveAspectRatio="none"
               role="group" aria-label="Motif detections"></svg>
        </figure>
        <figcaption id="plate-caption"></figcaption>
      </div>
      <aside class="apparatus" id="apparatus"></aside>
    </div>
  </section>

  <section class="view" id="view-families" hidden>
    <div class="families" id="families"></div>
  </section>

  <section class="view" id="view-synthesis" hidden>
    <article class="prose" id="synthesis"></article>
  </section>
</main>
</div>

<script>
const DATA = /*__PAYLOAD__*/null;

/* Family hues: evenly spaced, held at one saturation/lightness so the tint
   reads as an identity rather than as a heat scale. */
const familyColor = (id) => id < 0 ? "var(--ink-faint)"
  : `hsl(${(id * 47) % 360} 46% var(--fam-l, 42%))`;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c]));

const state = { panel: DATA.panels[0]?.stem ?? null, motif: null, view: "plates" };

/* ── Masthead counts ──────────────────────────────────────────────────── */
document.getElementById("counts").innerHTML = [
  ["plates", DATA.scale.panels], ["motifs", DATA.scale.motifs],
  ["families", DATA.scale.clusters], ["read", DATA.scale.readings],
].map(([k, v]) => `<span><b>${v}</b> ${k}</span>`).join("");

/* ── Tabs ─────────────────────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    state.view = tab.dataset.view;
    document.querySelectorAll(".tab").forEach((t) =>
      t.setAttribute("aria-selected", String(t === tab)));
    document.querySelectorAll(".view").forEach((v) =>
      v.hidden = v.id !== `view-${state.view}`);
  });
});

/* ── Plate select ─────────────────────────────────────────────────────── */
const select = document.getElementById("panel-select");
select.innerHTML = DATA.panels.map((p) =>
  `<option value="${esc(p.stem)}">${esc(p.title || p.stem)}${p.reading ? "" : "  (no reading)"}</option>`
).join("");
select.addEventListener("change", () => { state.panel = select.value; state.motif = null; drawPlate(); });

document.getElementById("show-registers").addEventListener("change", drawPlate);
document.getElementById("show-field").addEventListener("change", drawPlate);

const panelOf = (stem) => DATA.panels.find((p) => p.stem === stem);

/* ── Plate rendering ──────────────────────────────────────────────────── */
function drawPlate() {
  const panel = panelOf(state.panel);
  if (!panel) return;
  select.value = panel.stem;

  const img = document.getElementById("plate-img");
  img.src = panel.image;
  img.alt = `Panel ${panel.stem}, ${panel.motifs.length} detected motifs`;

  const svg = document.getElementById("plate-svg");
  svg.setAttribute("viewBox", `0 0 ${panel.width} ${panel.height}`);
  const scale = panel.width / 100;               /* keep stroke widths even */

  const showRegisters = document.getElementById("show-registers").checked;
  const showField = document.getElementById("show-field").checked;
  let out = "";

  if (showRegisters) {
    for (const reg of panel.registers) {
      const y = reg.y_top * panel.height, h = (reg.y_bottom - reg.y_top) * panel.height;
      out += `<rect class="band" x="0" y="${y}" width="${panel.width}" height="${h}"></rect>`
           + `<line class="band-line" x1="0" y1="${y}" x2="${panel.width}" y2="${y}"
                    stroke-width="${0.6 * scale}"></line>`
           + `<text class="band-label" x="${0.6 * scale}" y="${y + 4 * scale}"
                    font-size="${3 * scale}">REG ${reg.index}</text>`;
    }
  }

  for (const m of panel.motifs) {
    if (m.is_field && !showField) continue;
    const dim = state.motif !== null && state.motif !== m.index;
    out += `<rect class="box${m.is_field ? " is-field" : ""}`
         + `${state.motif === m.index ? " is-active" : ""}${dim ? " is-dim" : ""}"`
         + ` data-index="${m.index}" x="${m.x}" y="${m.y}" width="${m.w}" height="${m.h}"`
         + ` style="color:${familyColor(m.cluster)}" stroke="${familyColor(m.cluster)}"`
         + ` stroke-width="${0.7 * scale}" tabindex="0" role="button"`
         + ` aria-label="Motif ${m.index}${m.label ? ": " + esc(m.label) : ""}"></rect>`
         + `<text class="box-num" x="${m.x + 1.2 * scale}" y="${m.y + 4.2 * scale}"
                  font-size="${3.2 * scale}" stroke-width="${0.9 * scale}"
                  fill="${familyColor(m.cluster)}">${m.index}</text>`;
  }
  svg.innerHTML = out;

  svg.querySelectorAll(".box").forEach((box) => {
    const idx = Number(box.dataset.index);
    const pick = () => { state.motif = state.motif === idx ? null : idx; drawPlate(); };
    box.addEventListener("click", pick);
    box.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
    });
    box.addEventListener("mouseenter", () => renderApparatus(panel, idx));
    box.addEventListener("mouseleave", () => renderApparatus(panel, state.motif));
  });

  document.getElementById("plate-caption").textContent =
    `${panel.stem} · ${panel.width}x${panel.height}px · ${panel.motifs.length} detections · `
    + `${panel.registers.length} register${panel.registers.length === 1 ? "" : "s"}`;

  renderApparatus(panel, state.motif);
}

/* ── Apparatus: panel reading, or the selected motif ──────────────────── */
function renderApparatus(panel, motifIndex) {
  const el = document.getElementById("apparatus");
  const motif = motifIndex === null || motifIndex === undefined
    ? null : panel.motifs.find((m) => m.index === motifIndex);

  if (motif) { el.innerHTML = motifCard(motif) + familyCard(motif.cluster); return; }

  const r = panel.reading;
  let html = `<div class="card"><div class="eyebrow">Plate reading</div>`;
  if (!r) {
    html += `<h3>${esc(panel.stem)}</h3>`
          + `<p class="empty">No panel reading generated for this plate yet. `
          + `Run the panels stage, or hover a box to read its motif.</p></div>`;
    return void (el.innerHTML = html + registersCard(panel));
  }
  html += `<h3>${esc(r.title || panel.stem)}</h3>`;
  if (r.confidence) html += `<div class="eyebrow" style="margin:.35rem 0 0">Confidence — ${esc(r.confidence)}</div>`;
  if (r.summary) html += `<p>${esc(r.summary)}</p>`;
  if (r.composition) html += `<p>${esc(r.composition)}</p>`;
  if (r.narrative) html += `<p>${esc(r.narrative)}</p>`;
  html += `</div>`;

  html += registersCard(panel);

  if (r.cross_panel_links?.length) {
    html += `<div class="card"><div class="eyebrow">Links across the corpus</div>`
          + `<ul class="notes">${r.cross_panel_links.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></div>`;
  }
  if (r.uncertainties?.length) {
    html += `<div class="card"><div class="eyebrow">Uncertainties</div>`
          + `<ul class="notes">${r.uncertainties.map((u) => `<li>${esc(u)}</li>`).join("")}</ul></div>`;
  }
  el.innerHTML = html;
}

function registersCard(panel) {
  if (!panel.registers.length) return "";
  const readings = new Map(
    (panel.reading?.register_readings ?? []).map((r) => [r.register, r.reading]));
  return `<div class="card"><div class="eyebrow">Registers, top to bottom</div>`
    + `<ul class="registers">` + panel.registers.map((reg) => {
        const members = reg.members.map((i) =>
          `<button class="chip" data-goto="${i}">#${i}</button>`).join("");
        const text = readings.get(reg.index);
        return `<li><div class="reg-head">Register ${reg.index} — `
             + `${(reg.y_top * 100).toFixed(0)}%&ndash;${(reg.y_bottom * 100).toFixed(0)}% of height</div>`
             + (text ? `<p>${esc(text)}</p>` : "")
             + `<div class="chips">${members}</div></li>`;
      }).join("") + `</ul></div>`;
}

function motifCard(m) {
  const rows = [
    ["index", `#${m.index}`],
    ["family", m.cluster >= 0
      ? `<span class="swatch" style="background:${familyColor(m.cluster)}"></span>cluster ${m.cluster}`
      : "unclustered"],
    ["scale", esc(m.scale)],
    ["position", esc(m.zone)],
    ["area", `${(m.area_fraction * 100).toFixed(1)}% of plate`],
    ["register", m.register >= 0 ? `register ${m.register}` : "field / ground"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  let html = `<div class="card"><div class="eyebrow">Motif</div>`
    + `<h3>${esc(m.label || "Unlabelled detection")}</h3>`;
  if (m.description) html += `<p>${esc(m.description)}</p>`;
  if (m.iconography) html += `<p><em>${esc(m.iconography)}</em></p>`;
  if (m.notes) html += `<p class="plain">${esc(m.notes)}</p>`;
  if (!m.label && !m.description) {
    html += `<p class="empty">No label recorded — this detection was never annotated.</p>`;
  }
  html += `<dl class="meta">${rows}</dl>`;
  if (m.label_source) {
    html += `<div class="eyebrow" style="margin:.7rem 0 0">Label source — ${esc(m.label_source)}</div>`;
  }
  return html + `</div>`;
}

function familyCard(cid) {
  const fam = DATA.families[String(cid)];
  if (!fam) return "";
  let html = `<div class="card"><div class="eyebrow">Motif family</div>`
    + `<h3><span class="swatch" style="background:${familyColor(cid)}"></span>`
    + `${esc(fam.name || `Cluster ${cid}`)}</h3>`;
  if (fam.visual_definition) html += `<p>${esc(fam.visual_definition)}</p>`;
  if (fam.iconographic_reading) html += `<p><em>${esc(fam.iconographic_reading)}</em></p>`;
  const rows = [
    ["members", fam.size],
    ["plates", fam.panel_spread],
    ["cohesion", fam.cohesion === null || fam.cohesion === undefined
      ? "&mdash;" : fam.cohesion.toFixed(3)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  return html + `<dl class="meta">${rows}</dl></div>`;
}

/* Chips jump to a motif; crops jump to the plate that motif sits on. */
document.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-goto]");
  if (chip) { state.motif = Number(chip.dataset.goto); drawPlate(); return; }
  const crop = e.target.closest("[data-plate]");
  if (crop) {
    state.panel = crop.dataset.plate;
    state.motif = Number(crop.dataset.motif);
    document.querySelector('.tab[data-view="plates"]').click();
    drawPlate();
    document.getElementById("plate-img").scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

/* Arrow keys walk the plate in the reading order the pipeline computed. */
document.addEventListener("keydown", (e) => {
  if (state.view !== "plates" || e.target.matches("select, input")) return;
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const panel = panelOf(state.panel);
  const order = panel.reading_order;
  if (!order.length) return;
  const step = e.key === "ArrowRight" ? 1 : -1;
  const at = order.indexOf(state.motif);
  const from = at < 0 ? (step > 0 ? -1 : 0) : at;
  state.motif = order[(from + step + order.length) % order.length];
  e.preventDefault();
  drawPlate();
});

/* ── Families view ────────────────────────────────────────────────────── */
document.getElementById("families").innerHTML =
  Object.keys(DATA.families).sort((a, b) => a - b).map((cid) => {
    const f = DATA.families[cid];
    const crops = (f.crops ?? []).map((c) =>
      `<img src="${c.src}" data-plate="${esc(c.stem)}" data-motif="${c.index}"
            alt="${esc(c.stem)} motif ${c.index}" title="${esc(c.stem)} #${c.index}" loading="lazy">`
    ).join("");
    const rows = [
      ["members", f.size], ["plates", f.panel_spread],
      ["cohesion", f.cohesion == null ? "&mdash;" : f.cohesion.toFixed(3)],
      ["confidence", esc(f.confidence || "—")],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    return `<div class="card family">
      <div class="eyebrow">Cluster ${cid}</div>
      <h3><span class="swatch" style="background:${familyColor(Number(cid))}"></span>
          ${esc(f.name || "Unnamed family")}</h3>
      ${f.visual_definition ? `<p>${esc(f.visual_definition)}</p>` : ""}
      ${f.variation ? `<p>${esc(f.variation)}</p>` : ""}
      ${f.distribution_note ? `<p>${esc(f.distribution_note)}</p>` : ""}
      ${f.iconographic_reading ? `<p><em>${esc(f.iconographic_reading)}</em></p>` : ""}
      <dl class="meta">${rows}</dl>
      <div class="crops">${crops}</div>
    </div>`;
  }).join("") || `<p class="empty">No motif families yet.</p>`;

/* ── Synthesis ────────────────────────────────────────────────────────── */
document.getElementById("synthesis").innerHTML =
  DATA.synthesis || `<p class="empty">No corpus synthesis has been generated yet.</p>`;

drawPlate();
</script>
"""

def render(payload: dict[str, Any]) -> str:
    """Return the full page with `payload` embedded as its data island.

    `</script>` inside the JSON would close the block early, so it is escaped —
    the sequence cannot appear in valid JSON string data any other way.
    """
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace(_PAYLOAD_SENTINEL, blob)
