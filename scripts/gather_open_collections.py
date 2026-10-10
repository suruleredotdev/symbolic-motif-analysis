"""
gather_open_collections.py

Pulls panel-type records (doors, door panels, posts, Ifa boards, relief
plaques) from the open collection APIs that answered from a script in
October 2026, and tags each record with the latest year it can be shown
to have existed: its terminus ante quem (TAQ). Records whose TAQ falls
on or before --before (default 1910) are the ones we want for the archaic
corpus.

Why TAQ and not "date": museum dates for Yoruba carving are usually
"19th-20th century" or "late 1800s", which is a cataloguer's estimate.
The date it was collected, accessioned or photographed is a fact. Of the
dates a record carries, the earliest is the TAQ: an object accessioned in
1904 existed in 1904, whatever its label says.

Sources (see COLLECTIONS.md for what each one holds and its caveats):
  museum_digital  global.museum-digital.org — mirrors SMB Ethnologisches
                  Museum Berlin (incl. the Frobenius 1910-12 objects)
  europeana       api.europeana.eu — EUROPEANA_KEY, or the public demo key
  met             Met Open Access, v1.1 search (v1 search retired 2026-10-01)
  commons         Wikimedia Commons categories

Output is a list of records in the panel_art_dataset/a*.json shape, plus
source/TAQ fields, so it can be reviewed before images are downloaded.

Usage:
  python3 scripts/gather_open_collections.py --out panel_art_dataset/open_collections.json
  python3 scripts/gather_open_collections.py --sources museum_digital met --before 1912
  python3 scripts/gather_open_collections.py --keep-undated --no-filter
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "symbolic-motif-analysis/0.1 (+https://github.com/suruleredotdev/symbolic-motif-analysis)"
REQUEST_DELAY = 0.5  # seconds between calls; these are shared public APIs

# ─── Queries ────────────────────────────────────────────────────────────────

QUERIES = {
    # museum-digital's Berlin records are catalogued in German.
    "museum_digital": [
        "Yoruba Tür", "Yoruba Türflügel", "Yoruba Türpfosten", "Frobenius Yoruba",
        "Orakelbrett", "Ifa", "Benin Platte", "Benin Reliefplatte",
        "Kamerun Türrahmen", "Kamerun Türpfosten", "Dahome Relief", "Nupe Tür",
    ],
    "europeana": [
        "Yoruba door", "Yoruba Tür", "Yoruba porte", "Yoruba deur",
        "opon ifa", "Ifa board", "Benin plaque", "Abomey bas-relief",
    ],
    "met": ["Yoruba door", "Yoruba panel", "Ifa divination tray", "Benin plaque", "Ijebu"],
    "commons": [
        "Category:Yoruba doors", "Category:Olowe of Ise", "Category:Areogun",
        "Category:Benin Bronzes", "Category:Opon Ifa",
    ],
}

# Relevance filter: keep records that name a panel-type object.
PANEL_RE = re.compile(
    r"\b(door|doors|panel|panels|lintel|post|posts|board|tray|plaque|plaques|relief|"
    r"ilekun|opon|t[üu]r\w*|brett|platte|pfosten|deur\w*|porte\w*|panneau\w*|bas-relief)\b",
    re.IGNORECASE,
)

# Collectors whose collecting window is known even when the record carries
# no date. The value is the last year they could have acquired the object.
COLLECTOR_TAQ = {
    "frobenius": 1912,         # 4th DIAFE, Nigeria & Cameroon 1910-1912
    "northcote": 1915,         # N. W. Thomas surveys, Nigeria & Sierra Leone 1909-1915
    "konietzko": None,         # dealer active into the 1960s: no constraint
}

YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-2][0-9])(?!\d)")
CENTURY_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)[\s-]+century", re.IGNORECASE)
DECADE_RE = re.compile(r"(?<!\d)(1[0-9]{2}0)s\b")
SHORT_RANGE_RE = re.compile(r"(?<!\d)(1[0-9]{3})\s*[-–/]\s*(\d{2})(?!\d)")
BC_RE = re.compile(r"\bB\.?\s?C\.?(?:E\.?)?(?!\w)")
LIFESPAN_RE = re.compile(r"\((1[0-9]{3})\s*[-–]\s*(1[0-9]{3}|20[0-2][0-9])\)")


# ─── Dating ─────────────────────────────────────────────────────────────────

def latest_year(text: str | None) -> int | None:
    """Latest year a free-text date allows: '1890-1910' → 1910, '19th century' → 1900."""
    if not text:
        return None
    years = [int(y) for y in YEAR_RE.findall(text)]
    years += [int(d) + 9 for d in DECADE_RE.findall(text)]  # '1890s' → 1899
    # '1905–15' → 1915: the abbreviated end takes the start's century
    years += [int(a[:2] + b) for a, b in SHORT_RANGE_RE.findall(text)]
    centuries = [int(c) * 100 for c in CENTURY_RE.findall(text)]
    candidates = years + centuries
    if not candidates:
        return None
    if BC_RE.search(text):
        return -min(candidates)  # 'ca. 1961–1917 B.C.' ends in 1917 BC
    return max(candidates)


def collector_taq(collector: str | None) -> int | None:
    """Last year the collector could have acquired it: a known window, or their death."""
    if not collector:
        return None
    low = collector.lower()
    bounds = [year for name, year in COLLECTOR_TAQ.items() if name in low and year]
    # museum-digital writes 'Hans Glauning (1868-1908)': nothing he collected is later than 1908.
    bounds += [int(death) for _, death in LIFESPAN_RE.findall(collector)]
    return min(bounds) if bounds else None


def terminus_ante_quem(*, date_display=None, acquired_year=None, collector=None) -> int | None:
    """The earliest year by which the object is known to have existed.

    Each source bounds the object's existence from above; the tightest bound
    is the smallest. A cataloguer's date range bounds it at its *latest* year,
    because the object could have been made at any point in the range.
    """
    bounds = [b for b in (latest_year(date_display), acquired_year, collector_taq(collector)) if b]
    return min(bounds) if bounds else None


def classify(taq: int | None, before: int) -> str:
    if taq is None:
        return "unknown"
    return "yes" if taq <= before else "no"


# ─── HTTP ───────────────────────────────────────────────────────────────────

def get_json(url: str, params: dict | None = None, retries: int = 4) -> object:
    """GET and decode JSON, backing off on throttling. Returns None on 404."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(retries + 1):
        time.sleep(REQUEST_DELAY * 2 ** attempt)
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None  # museum-digital 404s an empty search; Met 404s stale IDs
            # The Met answers bursts with 403 rather than 429.
            if exc.code not in (403, 429, 503) or attempt == retries:
                raise
    return None


def record(**fields) -> dict:
    base = {
        "id": None, "source": None, "query": None, "title": None, "culture": None,
        "object_type": None, "country": None, "date_display": None, "acquired_year": None,
        "collector": None, "museum": None, "record_url": None, "primary_image_url": None,
        "licence": None, "description": None,
    }
    base.update(fields)
    return base


# ─── Sources ────────────────────────────────────────────────────────────────

def gather_museum_digital(queries: list[str], max_per_query: int) -> list[dict]:
    api = "https://global.museum-digital.org/json"
    out = []
    for q in queries:
        page_size, start, ids = 50, 0, []
        while len(ids) < max_per_query:
            hits = get_json(f"{api}/objects", {"s": q, "gbreitenat": page_size, "startwert": start})
            if not isinstance(hits, list) or not hits:
                break
            ids += [h["objekt_id"] for h in hits]
            if len(hits) < page_size:
                break
            start += page_size
        for oid in ids[:max_per_query]:
            d = get_json(f"{api}/object/{oid}")
            if not d:
                continue
            events = d.get("object_events") or []
            collectors = [e["people"]["displayname"] for e in events
                          if e.get("event_type_name") == "Collected" and e.get("people")]
            places = [e["place"]["place_name"] for e in events if e.get("place")]
            times = [t.get("time_name", "") for e in events for t in (e.get("time") or [])
                     if isinstance(t, dict)]
            images = d.get("object_images") or []
            out.append(record(
                id=f"md:{oid}", source="museum_digital", query=q,
                title=d.get("object_name"), object_type=d.get("object_type"),
                country=", ".join(places) or None,
                date_display="; ".join(t for t in times if t) or None,
                collector="; ".join(collectors) or None,
                museum=(d.get("object_institution") or {}).get("institution_name"),
                record_url=f"https://global.museum-digital.org/object/{oid}",
                primary_image_url=images[0].get("filename_loc") if images else None,
                licence=(d.get("licence") or {}).get("metadata_rights_status"),
                description=d.get("object_description"),
                inventory_number=d.get("object_inventory_number"),
            ))
    return out


def gather_europeana(queries: list[str], max_per_query: int) -> list[dict]:
    key = os.environ.get("EUROPEANA_KEY", "api2demo")
    out = []
    for q in queries:
        d = get_json("https://api.europeana.eu/record/v2/search.json",
                     {"wskey": key, "query": q, "rows": min(max_per_query, 100),
                      "qf": "TYPE:IMAGE", "profile": "standard"}) or {}
        for it in d.get("items", []):
            first = lambda k: (it.get(k) or [None])[0]  # noqa: E731
            out.append(record(
                id=f"europeana:{it['id']}", source="europeana", query=q,
                title=first("title"), country=first("country"),
                date_display=first("year"), museum=first("dataProvider"),
                record_url=first("edmIsShownAt") or it.get("guid"),
                primary_image_url=first("edmIsShownBy") or first("edmPreview"),
                licence=first("rights"),
            ))
    return out


MET_AFRICA_DEPT = 5  # Arts of Africa, Oceania, and the Americas


def gather_met(queries: list[str], max_per_query: int) -> list[dict]:
    api = "https://collectionapi.metmuseum.org/public/collection"
    out = []
    for q in queries:
        hits = get_json(f"{api}/v1.1/search", {"q": q, "hasImages": "true", "limit": max_per_query,
                                                  "departmentId": MET_AFRICA_DEPT}) or {}
        for oid in (hits.get("objectIDs") or [])[:max_per_query]:
            d = get_json(f"{api}/v1/objects/{oid}")
            if not d or not d.get("objectID"):
                continue  # IDs from search can lag the object endpoint
            acquired = d.get("accessionYear")
            out.append(record(
                id=f"met:{oid}", source="met", query=q, title=d.get("title"),
                culture=d.get("culture"), object_type=d.get("objectName"),
                country=d.get("country") or None, date_display=d.get("objectDate"),
                acquired_year=int(acquired) if str(acquired).isdigit() else None,
                museum="The Metropolitan Museum of Art", record_url=d.get("objectURL"),
                primary_image_url=d.get("primaryImage") or None,
                licence="CC0" if d.get("isPublicDomain") else "see record",
                artist=d.get("artistDisplayName") or None,
            ))
    return out


def gather_commons(categories: list[str], max_per_query: int) -> list[dict]:
    api = "https://commons.wikimedia.org/w/api.php"
    strip = lambda s: re.sub(r"<[^>]+>", " ", s or "").strip()  # noqa: E731
    out = []
    for cat in categories:
        d = get_json(api, {"action": "query", "generator": "categorymembers", "gcmtitle": cat,
                           "gcmtype": "file", "gcmlimit": min(max_per_query, 500),
                           "prop": "imageinfo", "iiprop": "url|extmetadata", "format": "json"}) or {}
        for page in (d.get("query") or {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            val = lambda k: strip((meta.get(k) or {}).get("value"))  # noqa: E731
            out.append(record(
                id=f"commons:{page['pageid']}", source="commons", query=cat,
                title=page.get("title"), date_display=val("DateTimeOriginal") or None,
                museum=val("Credit") or None, record_url=info.get("descriptionurl"),
                primary_image_url=info.get("url"), licence=val("LicenseShortName") or None,
                description=val("ImageDescription")[:1000] or None,
            ))
    return out


GATHERERS = {
    "museum_digital": gather_museum_digital,
    "europeana": gather_europeana,
    "met": gather_met,
    "commons": gather_commons,
}


# ─── Main ───────────────────────────────────────────────────────────────────

def is_panel(rec: dict) -> bool:
    text = " ".join(str(rec.get(k) or "") for k in ("title", "object_type", "description"))
    return bool(PANEL_RE.search(text))


def annotate(rec: dict, before: int) -> dict:
    rec["taq"] = terminus_ante_quem(date_display=rec.get("date_display"),
                                    acquired_year=rec.get("acquired_year"),
                                    collector=rec.get("collector"))
    rec["pre_cutoff"] = classify(rec["taq"], before)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", choices=sorted(GATHERERS), default=sorted(GATHERERS))
    ap.add_argument("--before", type=int, default=1910,
                    help="cutoff year for pre_cutoff, inclusive (1912 admits the Frobenius 1910-12 objects)")
    ap.add_argument("--max-per-query", type=int, default=100)
    ap.add_argument("--keep-undated", action="store_true", help="keep records with no TAQ")
    ap.add_argument("--keep-later", action="store_true", help="keep records with TAQ after --before")
    ap.add_argument("--no-filter", action="store_true", help="skip the panel-type relevance filter")
    ap.add_argument("--out", type=Path, default=Path("panel_art_dataset/open_collections.json"))
    args = ap.parse_args(argv)

    records, seen = [], set()
    for name in args.sources:
        try:
            got = GATHERERS[name](QUERIES[name], args.max_per_query)
        except Exception as exc:  # one dead API shouldn't sink the run
            print(f"[{name}] failed: {exc}", file=sys.stderr)
            continue
        kept = 0
        for rec in got:
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            annotate(rec, args.before)
            if not args.no_filter and not is_panel(rec):
                continue
            if rec["pre_cutoff"] == "unknown" and not args.keep_undated:
                continue
            if rec["pre_cutoff"] == "no" and not args.keep_later:
                continue
            records.append(rec)
            kept += 1
        print(f"[{name}] {len(got)} fetched, {kept} kept", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"wrote {len(records)} records → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
