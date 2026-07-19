"""
scrape_frobenius.py

Scrapes the Frobenius Institut Bildarchiv (http://bildarchiv.frobenius-katalog.de/)
for panel-type works: doors, boards, plaques, divination trays, and related
pictorial/relief objects from Yoruba and related West African traditions.

Starting point:
  http://bildarchiv.frobenius-katalog.de/rech.FAU?sid=66C606474&dm=1&auft=0
  (Search results for "yoruba", sorted by relevance)

Key URL patterns observed:
  - Search results list:
      http://bildarchiv.frobenius-katalog.de/rech.FAU?sid=...&dm=1&auft=0
  - Individual record (detail):
      http://bildarchiv.frobenius-katalog.de/hzeig.FAU?sid=...&dm=1&ind=1&zeig=FoA%2004-5578
  - Image URL:
      http://bildarchiv.frobenius-katalog.de/zvimg.FAU?sid=...&DM=1&qpos=...&ipos=1&erg=A&hst=1&rpos=...png

Usage:
  python3 src/python/scripts/scrape_frobenius.py

Output:
  panel_art_dataset/raw/frobenius_raw.json   — raw scraped records
  panel_art_dataset/parsed/frobenius.json    — parsed/normalised records

Notes:
  - The Frobenius site uses session IDs (sid=) in URLs; a fresh session must be
    obtained before scraping by hitting the search page first.
  - The site appears to use a CGI/FAU framework; each search result page lists
    ~25 records and provides navigation to next pages.
  - Image download URLs are constructed from qpos/rpos parameters embedded in
    the search result or detail HTML.
  - Respectful rate limiting is applied (1–2 s between requests).
  - Run with --dry-run to only print what would be fetched.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print(
        "ERROR: Install dependencies first:\n"
        "  pip install requests beautifulsoup4 lxml\n"
        "(or: pip install -r src/python/requirements.txt)"
    )
    sys.exit(1)

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "http://bildarchiv.frobenius-katalog.de"

# The known-working search-results entry point for "yoruba" search.
# A fresh session ID may need to be obtained; the script attempts this
# by first visiting the site root.
YORUBA_SEARCH_URL = (
    "http://bildarchiv.frobenius-katalog.de/rech.FAU?sid=66C606474&dm=1&auft=0"
)

# Terms we use to filter records that are panels, boards, doors, or plaques.
# These are matched case-insensitively against the title/description fields.
PANEL_KEYWORDS = [
    "panel", "plaque", "door", "board", "tür", "tafel", "brett",
    "divination", "opon", "ifa", "tray", "relief", "platte",
    "modakeke", "carved door", "palace door", "temple door",
]

# How many seconds to pause between HTTP requests
REQUEST_DELAY_S = 1.5

# ─── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]          # african-artifacts/
DATASET_DIR = PROJECT_ROOT / "panel_art_dataset"
RAW_DIR = DATASET_DIR / "raw"
PARSED_DIR = DATASET_DIR / "parsed"

# ─── HTTP session ──────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update(
    {
        # Mimic a real browser — some CGI systems reject non-browser UAs outright
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }
)


def get(url: str, *, timeout: int = 20) -> requests.Response | None:
    """
    GET with manual redirect handling.

    The Frobenius FAU CGI system sometimes emits a redirect whose Location
    header is pure whitespace (or contains leading/trailing whitespace).
    requests.get() URL-encodes that whitespace, producing a nonsense URL like
    http://HOST/%20%20%20...  which then 404s.

    We disable automatic redirects and follow them manually after stripping
    the Location value.
    """
    current_url = url
    for hop in range(10):
        try:
            resp = SESSION.get(current_url, timeout=timeout, allow_redirects=False)
        except requests.RequestException as exc:
            print(f"  [WARN] GET {current_url} failed: {exc}")
            return None

        if resp.is_redirect:
            raw_location = resp.headers.get("Location", "")
            location = raw_location.strip()
            if not location:
                print(f"  [WARN] Redirect from {current_url} has empty/whitespace Location — stopping redirect chain")
                # Return the redirect response itself so the caller can inspect content
                time.sleep(REQUEST_DELAY_S)
                return resp
            next_url = urljoin(current_url, location)
            if next_url == current_url:
                break
            print(f"  [→] {current_url}  →  {next_url}")
            current_url = next_url
            time.sleep(0.3)
            continue

        # Not a redirect — check for HTTP errors
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"  [WARN] GET {current_url} failed: {exc}")
            return None

        time.sleep(REQUEST_DELAY_S)
        return resp

    print(f"  [WARN] Too many redirects for {url}")
    return None


# ─── Session ID extraction ──────────────────────────────────────────────────────

def extract_sid(url_or_qs: str) -> str | None:
    """Extract the session ID (sid=...) from a Frobenius URL or query string."""
    # Try as a full URL first
    qs = parse_qs(urlparse(url_or_qs).query)
    sid = qs.get("sid", [None])[0]
    if sid:
        return sid
    # Try as a bare query string
    qs2 = parse_qs(url_or_qs.lstrip("?"))
    return qs2.get("sid", [None])[0]


def _find_sid_in_html(html: str) -> str | None:
    """Search HTML text for any sid= value in links, forms, or meta-refresh tags."""
    soup = BeautifulSoup(html, "html.parser")

    # <a href="...?sid=...">
    for tag in soup.find_all(href=True):
        sid = extract_sid(tag["href"])
        if sid:
            return sid

    # <form action="...?sid=...">
    for form in soup.find_all("form", action=True):
        sid = extract_sid(form["action"])
        if sid:
            return sid

    # <meta http-equiv="refresh" content="0; url=...?sid=...">
    for meta in soup.find_all("meta"):
        content = meta.get("content", "")
        if "sid=" in content:
            # content might be "0; url=...?sid=..." or just "?sid=..."
            sid = extract_sid(content)
            if sid:
                return sid

    # Raw text scan (sometimes the SID is in inline JS)
    match = re.search(r"[?&]sid=([A-Z0-9]+)", html)
    if match:
        return match.group(1)

    return None


def get_fresh_session(start_url: str) -> str | None:
    """
    Obtain a valid Frobenius session ID.

    Strategy:
      1. Visit the site root — the FAU system often issues a session on the
         very first page load (via meta-refresh or a link).
      2. If not found there, try the provided start_url.
      3. Fall back to the sid already embedded in start_url.

    The site does NOT use cookies for session management — the session ID
    travels entirely through URL parameters (sid=...).
    """
    existing_sid = extract_sid(start_url)

    for probe_url in [BASE_URL + "/", BASE_URL + "/index.html", start_url]:
        print(f"  Probing for session: {probe_url}")
        resp = get(probe_url)
        if resp is None:
            continue
        sid = _find_sid_in_html(resp.text)
        if sid:
            print(f"  Found session ID in {probe_url}: {sid}")
            return sid

    print(f"  No fresh session found — reusing existing sid={existing_sid}")
    return existing_sid


def replace_sid(url: str, new_sid: str) -> str:
    """Replace the sid= parameter in a URL with a fresh one."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["sid"] = [new_sid]
    new_query = urlencode({k: v[0] for k, v in qs.items()})
    return urlunparse(parsed._replace(query=new_query))


# ─── Search results parsing ────────────────────────────────────────────────────

def parse_search_results_page(
    html: str, page_url: str, *, debug: bool = False
) -> dict[str, Any]:
    """
    Parse a Frobenius search results page and extract:
      - list of record entries (registration number + title + detail URL)
      - link to next page (if any)
      - total result count (if visible)
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []
    next_url = None
    total_count = None

    page_text = soup.get_text(" ", strip=True)

    if debug:
        print(f"  [DEBUG] Page text snippet: {page_text[:400]!r}")
        all_links = [a.get("href", "") for a in soup.find_all("a", href=True)]
        print(f"  [DEBUG] All hrefs ({len(all_links)}): {all_links[:20]}")

    # ── Try to find total count ──
    # FAU shows e.g. "342 Treffer", "Treffer: 342", "1 - 25 von 342"
    for pattern in [
        r"(\d+)\s*Treffer",
        r"Treffer[:\s]+(\d+)",
        r"von\s+(\d+)",
        r"(\d+)\s*Ergebnis",
        r"gefunden[:\s]+(\d+)",
    ]:
        m = re.search(pattern, page_text, re.IGNORECASE)
        if m:
            total_count = int(m.group(1))
            break

    # ── Extract individual record links ──
    # FAU detail links contain "hzeig.FAU" with a zeig= parameter holding the record ID.
    # e.g. hzeig.FAU?sid=...&dm=1&ind=1&zeig=FoA%2004-5578
    seen_zeig: set[str] = set()
    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]
        if "hzeig.FAU" not in href and "zeig=" not in href:
            continue
        full_url = urljoin(BASE_URL, href)
        qs = parse_qs(urlparse(full_url).query)
        zeig_val = qs.get("zeig", [None])[0]
        dedup_key = zeig_val or full_url
        if dedup_key in seen_zeig:
            continue
        seen_zeig.add(dedup_key)

        # Grab link text and any nearby sibling text for a richer title snippet
        link_text = a_tag.get_text(" ", strip=True)
        parent_text = ""
        if a_tag.parent:
            parent_text = a_tag.parent.get_text(" ", strip=True)

        # Registration number patterns: FoA 04-5578, EB 26 123, MS-12345, etc.
        for text_src in [link_text, parent_text, zeig_val or ""]:
            reg_match = re.search(
                r"\b(Fo[A-Za-z]\s*[\w][\w\s\-]+\d+|[A-Z]{2,4}\s*\d+[-\s]\d+|[A-Z]+-\d+)\b",
                text_src,
            )
            if reg_match:
                break
        reg_number = (
            reg_match.group(1).strip() if reg_match
            else (zeig_val.replace("%20", " ") if zeig_val else "")
        )

        records.append(
            {
                "registration_number": reg_number,
                "title_snippet": (link_text or parent_text)[:200],
                "detail_url": full_url,
                "zeig_param": zeig_val,
            }
        )

    # ── Next page link ──
    # Strategy 1: explicit "weiter" / ">>" navigation links
    NAV_TEXTS = {"weiter", ">>", "next", "nächste", "vor", ">", "weiter »", "vorwärts"}
    for a_tag in soup.find_all("a", href=True):
        text = a_tag.get_text(strip=True).lower()
        if text in NAV_TEXTS or text.startswith("weiter"):
            href = a_tag["href"]
            if "rech.FAU" in href or "auft=" in href:
                next_url = urljoin(BASE_URL, href)
                break

    # Strategy 2: find any rech.FAU link with a higher auft= than current page
    if not next_url:
        current_auft = int(parse_qs(urlparse(page_url).query).get("auft", ["0"])[0])
        best_auft = current_auft
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            if "rech.FAU" not in href and "auft=" not in href:
                continue
            qs_link = parse_qs(urlparse(href).query)
            auft_val = qs_link.get("auft", [None])[0]
            if auft_val and int(auft_val) > best_auft:
                best_auft = int(auft_val)
                next_url = urljoin(BASE_URL, href)

    # Strategy 3: if we got records and found no next link, compute next auft
    # by advancing by the number of records on this page (FAU default page size = 25)
    if not next_url and records:
        current_auft = int(parse_qs(urlparse(page_url).query).get("auft", ["0"])[0])
        next_auft = current_auft + len(records)
        next_url = replace_sid_and_auft(page_url, next_auft)

    return {
        "page_url": page_url,
        "total_count": total_count,
        "records": records,
        "next_url": next_url,
    }


def replace_sid_and_auft(url: str, new_auft: int) -> str:
    """Build a next-page URL by updating auft= in an existing rech.FAU URL."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query, keep_blank_values=True)
    qs["auft"] = [str(new_auft)]
    new_query = urlencode({k: v[0] for k, v in qs.items()})
    return urlunparse(parsed._replace(query=new_query))


# ─── Detail page parsing ───────────────────────────────────────────────────────

def parse_detail_page(html: str, detail_url: str) -> dict[str, Any]:
    """
    Parse a Frobenius archive detail/record page.

    Fields captured:
      - registration_number (Signatur / Archiv-Nr)
      - title / object name
      - description (Beschreibung / Bildinhalt)
      - photographer / Fotograf
      - date (Aufnahmedatum / Datierung)
      - location (Aufnahmeort / Fundort)
      - ethnic_group / Ethnie
      - technique / Technik
      - dimensions / Maße
      - keywords / Schlagworte
      - image_url (constructed from embedded parameters)
      - detail_url (canonical)
      - raw_metadata (all table key-value pairs found)
    """
    soup = BeautifulSoup(html, "html.parser")
    record: dict[str, Any] = {
        "detail_url": detail_url,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
    }

    # ── Extract all label→value pairs from definition lists / tables ──
    raw_meta: dict[str, str] = {}

    # Pattern 1: <dt>/<dd> pairs
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            key = dt.get_text(" ", strip=True).rstrip(":").strip()
            val = dd.get_text(" ", strip=True)
            raw_meta[key] = val

    # Pattern 2: Table row key / value
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = cells[0].get_text(" ", strip=True).rstrip(":").strip()
            val = cells[1].get_text(" ", strip=True)
            if key and val:
                raw_meta[key] = val

    # Pattern 3: <span class="label"> or similar
    for span in soup.find_all(class_=re.compile(r"label|key|field", re.I)):
        sib = span.find_next_sibling()
        if sib:
            key = span.get_text(" ", strip=True).rstrip(":").strip()
            val = sib.get_text(" ", strip=True)
            if key and val:
                raw_meta[key] = val

    record["raw_metadata"] = raw_meta

    # ── Map known German/English field names to structured fields ──
    FIELD_MAP = {
        "registration_number": [
            "Signatur", "Archiv-Nr", "Inv.-Nr", "Nummer", "Nr", "Reg.-Nr",
            "Registration", "Call number",
        ],
        "title": [
            "Titel", "Bildtitel", "Objekt", "Objektbezeichnung", "Name",
            "Title", "Object",
        ],
        "description": [
            "Bildinhalt", "Beschreibung", "Inhalt", "Kommentar",
            "Description", "Notes",
        ],
        "photographer": [
            "Fotograf", "Photograph", "Aufnahme von", "Autor", "Photographer",
        ],
        "date": [
            "Aufnahmedatum", "Datierung", "Datum", "Jahr", "Date", "Year",
        ],
        "location": [
            "Aufnahmeort", "Fundort", "Ort", "Standort", "Location", "Place",
        ],
        "ethnic_group": ["Ethnie", "Volk", "Gruppe", "Culture", "Ethnic group"],
        "technique": ["Technik", "Material", "Technique"],
        "dimensions": ["Maße", "Format", "Größe", "Dimensions", "Size"],
        "keywords": ["Schlagworte", "Stichwörter", "Keywords", "Tags"],
        "collection": ["Sammlung", "Bestand", "Collection", "Series"],
        "rights": ["Rechte", "Copyright", "Rights", "Nutzungsrechte"],
    }

    for field, candidates in FIELD_MAP.items():
        for candidate in candidates:
            for key, val in raw_meta.items():
                if key.lower() == candidate.lower() or key.startswith(candidate):
                    record[field] = val
                    break
            if field in record:
                break

    # Registration number fallback: extract from URL zeig= param
    if "registration_number" not in record:
        qs = parse_qs(urlparse(detail_url).query)
        zeig = qs.get("zeig", [None])[0]
        if zeig:
            record["registration_number"] = zeig.replace("%20", " ")

    # ── Extract image URL ──
    # The image URL pattern uses zvimg.FAU with qpos/rpos parameters.
    # These appear as <img> src, <a> href, or embedded in JS.
    image_url = None

    # Direct img src
    for img in soup.find_all("img", src=True):
        src: str = img["src"]
        if "zvimg" in src or "img" in src.lower():
            image_url = urljoin(BASE_URL, src)
            break

    # Links with zvimg
    if not image_url:
        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            if "zvimg" in href:
                image_url = urljoin(BASE_URL, href)
                break

    # Embedded in inline JS
    if not image_url:
        scripts = " ".join(
            tag.get_text() for tag in soup.find_all("script")
        )
        js_match = re.search(
            r"(zvimg\.FAU[^\"'\s]+)", scripts
        )
        if js_match:
            image_url = urljoin(BASE_URL + "/", js_match.group(1))

    record["image_url"] = image_url

    # ── Extract page title as fallback title ──
    if "title" not in record:
        page_title = soup.find("title")
        if page_title:
            record["title"] = page_title.get_text(strip=True)

    # ── Full page text (for debugging / fallback parsing) ──
    record["page_text_snippet"] = soup.get_text(" ", strip=True)[:1000]

    return record


# ─── Panel-type filtering ──────────────────────────────────────────────────────

def is_panel_type(record: dict[str, Any]) -> bool:
    """
    Return True if the record's title/description/keywords suggest it is a
    door panel, board, plaque, relief, divination tray, or similar flat
    pictorial/narrative object.
    """
    text_fields = [
        record.get("title", ""),
        record.get("description", ""),
        record.get("title_snippet", ""),
        " ".join(str(v) for v in record.get("raw_metadata", {}).values()),
    ]
    combined = " ".join(text_fields).lower()
    return any(kw.lower() in combined for kw in PANEL_KEYWORDS)


# ─── Main scraping logic ───────────────────────────────────────────────────────

def scrape(
    start_url: str,
    *,
    max_pages: int = 50,
    filter_panel_types: bool = True,
    dry_run: bool = False,
    debug: bool = False,
    save_html_dir: Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Crawl the Frobenius search results starting from `start_url`, then fetch
    detail pages for each record (optionally filtering to panel-type objects).

    Returns (raw_records, parsed_records).
    """
    print(f"Starting Frobenius scrape from: {start_url}")
    print(f"Max pages: {max_pages}  |  Filter panel types: {filter_panel_types}")
    if dry_run:
        print("[DRY RUN] — no HTTP requests will be made")
    if save_html_dir:
        save_html_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving raw HTML to: {save_html_dir}")

    # Obtain a fresh session ID
    sid: str | None = None
    if not dry_run:
        print("Obtaining session...")
        sid = get_fresh_session(start_url)
        if sid:
            start_url = replace_sid(start_url, sid)
            print(f"  Session ID: {sid}")
        else:
            print("  Could not obtain fresh session — using original URL as-is")

    # Crawl search result pages
    all_result_stubs: list[dict] = []
    current_url: str | None = start_url
    pages_fetched = 0

    while current_url and pages_fetched < max_pages:
        if sid:
            current_url = replace_sid(current_url, sid)

        print(f"\n[Page {pages_fetched + 1}] {current_url}")
        if dry_run:
            print("  [DRY RUN] Skipping fetch")
            break

        resp = get(current_url)
        if resp is None:
            print("  Failed to fetch search results page; stopping.")
            break

        if save_html_dir:
            html_file = save_html_dir / f"results_page_{pages_fetched + 1}.html"
            html_file.write_text(resp.text, encoding="utf-8")
            print(f"  Saved HTML → {html_file}")

        page_data = parse_search_results_page(resp.text, current_url, debug=debug)
        stubs = page_data["records"]
        print(
            f"  Found {len(stubs)} records on this page  "
            f"(total reported: {page_data['total_count']})"
        )

        if debug and not stubs:
            print(f"  [DEBUG] Page text (first 600 chars):\n{resp.text[:600]!r}")

        all_result_stubs.extend(stubs)

        current_url = page_data["next_url"]
        pages_fetched += 1

        if not stubs:
            print("  No records found on page; stopping pagination.")
            break

    print(f"\nTotal stubs collected: {len(all_result_stubs)}")

    # Fetch detail pages
    raw_records: list[dict] = []
    parsed_records: list[dict] = []

    for i, stub in enumerate(all_result_stubs, 1):
        detail_url = stub["detail_url"]
        if sid:
            detail_url = replace_sid(detail_url, sid)

        print(f"  [{i}/{len(all_result_stubs)}] {stub.get('registration_number', '?')} — {detail_url}")

        if dry_run:
            print("    [DRY RUN] Skipping fetch")
            continue

        resp = get(detail_url)
        if resp is None:
            continue

        detail = parse_detail_page(resp.text, detail_url)
        detail["stub"] = stub

        raw_records.append(detail)

        # Filter to panel-type objects if requested
        if filter_panel_types and not is_panel_type(detail):
            print(f"    → Skipped (not panel-type): {detail.get('title', '?')[:80]}")
            continue

        parsed = {
            "source": "Frobenius Institut Bildarchiv",
            "registration_number": detail.get("registration_number"),
            "title": detail.get("title"),
            "description": detail.get("description"),
            "photographer": detail.get("photographer"),
            "date": detail.get("date"),
            "location": detail.get("location"),
            "ethnic_group": detail.get("ethnic_group"),
            "technique": detail.get("technique"),
            "dimensions": detail.get("dimensions"),
            "keywords": detail.get("keywords"),
            "collection": detail.get("collection"),
            "rights": detail.get("rights"),
            "detail_url": detail_url,
            "image_url": detail.get("image_url"),
            "scraped_at": detail.get("scraped_at"),
        }

        print(
            f"    ✓ Panel-type: {parsed['title'] or parsed['registration_number'] or '(no title)'}"
        )
        parsed_records.append(parsed)

    return raw_records, parsed_records


# ─── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Frobenius Bildarchiv for panel-type African art"
    )
    parser.add_argument(
        "--start-url",
        default=YORUBA_SEARCH_URL,
        help="Search results page URL to start from",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum search result pages to crawl (default: 50)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Capture all records, not just panel-type objects",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making HTTP requests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Root output directory (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extra diagnostic output (page text, all hrefs, etc.)",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Save raw HTML of each fetched page to panel_art_dataset/debug_html/",
    )
    args = parser.parse_args()

    out_raw = args.output_dir / "raw"
    out_parsed = args.output_dir / "parsed"
    out_raw.mkdir(parents=True, exist_ok=True)
    out_parsed.mkdir(parents=True, exist_ok=True)

    save_html_dir = (args.output_dir / "debug_html") if args.save_html else None

    raw_records, parsed_records = scrape(
        args.start_url,
        max_pages=args.max_pages,
        filter_panel_types=not args.no_filter,
        dry_run=args.dry_run,
        debug=args.debug,
        save_html_dir=save_html_dir,
    )

    if not args.dry_run:
        raw_path = out_raw / "frobenius_raw.json"
        parsed_path = out_parsed / "frobenius.json"

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                    "start_url": args.start_url,
                    "count": len(raw_records),
                    "records": raw_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\nRaw records saved:    {raw_path}  ({len(raw_records)} records)")

        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                    "start_url": args.start_url,
                    "panel_type_count": len(parsed_records),
                    "records": parsed_records,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Panel records saved:  {parsed_path}  ({len(parsed_records)} records)")

    else:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
