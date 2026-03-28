"""
filter_frobenius_panel_art.py

Produces frobenius_panel_art.json — a filtered subset of frobenius_all.json
containing only records that are:
  - door_panel : carved wooden doors, door panels, wall boards
  - ifa_board  : Ifa divination boards (opon Ifa), oracle boards, and
                 closely related Ifa cult objects (iroke, bowls)
  - figurine   : detached figurines, bronze figures, ancestor carvings,
                 Ogboni edan, temple-pillar figures with religious function

Each output record preserves all original fields and adds:
  "categories": ["door_panel", ...]   — one or more matched categories
  "matched_terms": ["door", ...]      — the specific terms that triggered

Usage:
  python3 src/python/scripts/filter_frobenius_panel_art.py
"""

import json
import re
import sys
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "src" / "typescript" / "backend" / "lib" / "data"
INPUT = DATA_DIR / "frobenius_all.json"
OUTPUT = DATA_DIR / "frobenius_panel_art.json"

# ─── Filter rules ────────────────────────────────────────────────────────────
#
# Each category is a list of (label, regex) pairs.
# A record matches a category if ANY pattern matches its combined text.
# Patterns use word boundaries (\b) to avoid substring false positives.

CATEGORIES: dict[str, list[tuple[str, re.Pattern]]] = {
    "door_panel": [
        ("door",           re.compile(r"\bdoors?\b",             re.I)),
        ("tür",            re.compile(r"\btüren?\b",             re.I)),
        ("türflügel",      re.compile(r"\btürflügel\b",          re.I)),
        ("wandbrett",      re.compile(r"\bwandbretter?\b",       re.I)),
        ("portal",         re.compile(r"\bportal\b",             re.I)),
        ("tempeltor",      re.compile(r"\btempeltor\b",          re.I)),
        ("carved door",    re.compile(r"carved\s+doors?",        re.I)),
        ("carvings from temple doors",
                           re.compile(r"carvings?\s+from\s+temple\s+doors?", re.I)),
        ("palace door",    re.compile(r"palace\s+doors?",        re.I)),
        ("geschnitzte tür",re.compile(r"geschnitzte\s+tür",      re.I)),
    ],

    "ifa_board": [
        # Word-boundary on 'ifa' so it doesn't match "Olokuntempels/Burg Ifas" etc.
        ("ifa oracle",     re.compile(r"\bifa\s+(oracle|orakel)\b", re.I)),
        ("ifa board",      re.compile(r"\bifa\s+board\b",           re.I)),
        ("ifabrett",       re.compile(r"\bifabretter?\b",           re.I)),
        ("opon ifa",       re.compile(r"\bopon\s+ifa\b",            re.I)),
        ("divination board",
                           re.compile(r"divination\s+(board|tray)", re.I)),
        ("oracle board",   re.compile(r"oracle\s+board",            re.I)),
        # iroke = ivory tapper, essential Ifa tool
        ("iroke",          re.compile(r"\biroke\b",                 re.I)),
        # Ifa cult objects (bowls, containers)
        ("ifa cult",       re.compile(r"\bifa[- ]cult\b",           re.I)),
        ("ifa orakel",     re.compile(r"\bifa[- ]orakel\b",         re.I)),
        # "Boards for the Ifa oracle" pattern
        ("boards for ifa", re.compile(r"boards?\s+for\s+the\s+ifa", re.I)),
        ("die 16 odu",     re.compile(r"sechzehn\s+odu|sixteen\s+odu", re.I)),
    ],

    "figurine": [
        # Detached figures / statuettes (not just people in a photograph)
        ("bronzefigur",    re.compile(r"\bbronzefiguren?\b",        re.I)),
        ("holzfigur",      re.compile(r"\bholzfiguren?\b",          re.I)),
        ("kultfigur",      re.compile(r"\bkultfiguren?\b",          re.I)),
        ("ahnenfigur",     re.compile(r"\bahnenfiguren?\b",         re.I)),
        ("idenafigur",     re.compile(r"\bidenafiguren?\b",         re.I)),
        ("edan",           re.compile(r"\bedan\b",                  re.I)),
        ("bronze figure",  re.compile(r"bronze\s+figures?",         re.I)),
        ("ivory carving",  re.compile(r"ivory\s+carvings?",         re.I)),
        ("elfenbeinschnitz",re.compile(r"\belfenbeinschnitz",       re.I)),
        # "heilige Bronzefiguren" (sacred bronze figures, Jebba)
        ("heilige bronzefiguren",
                           re.compile(r"heilige[n]?\s+bronzefiguren", re.I)),
        # Wooden figures serving as roof supports or altar pieces
        ("dachstützen mit figuren",
                           re.compile(r"dachpfeiler\s+mit\s+weiblichen\s+holzfiguren", re.I)),
        # Ogboni edan figures (specifically named)
        ("ogboni figur",   re.compile(r"gelbgußfiguren|gelb-\s*gußfiguren|anthropomorphe.*gußfigur", re.I)),
        # altar with carved cult figures
        ("kultfiguren altar",
                           re.compile(r"kultfiguren.*altar|altar.*kultfiguren", re.I)),
        # "anthropomorphe Holzfiguren" in temple windows (Shango)
        ("holzfiguren fenster",
                           re.compile(r"holzfiguren.*fenster|fenster.*holzfiguren|holzfiguren.*wand|wand.*holzfiguren", re.I)),
    ],
}

# ─── Text fields to search (both German and English variants) ────────────────

TEXT_FIELDS = [
    "title", "titel",
    "register_title", "registertitel",
    "historical_image_description", "histor._bildbeschreibung",
    "historical_keywords", "histor._stichworte",
    "keywords", "schlagworte",
    "historical_location", "histor._lokalisierung",
    "descriptive_editing", "deskriptive_bearbeitung",
    "remarks", "bemerkungen",
]


def record_text(r: dict) -> str:
    return " ".join(str(r[f]) for f in TEXT_FIELDS if f in r and r[f])


def classify(r: dict) -> dict[str, list[str]]:
    """
    Returns {category: [matched_term, ...]} for every category that fires.
    Empty dict = no match.
    """
    text = record_text(r)
    matched: dict[str, list[str]] = {}
    for cat, patterns in CATEGORIES.items():
        hits = [label for label, pat in patterns if pat.search(text)]
        if hits:
            matched[cat] = hits
    return matched


# ─── Run ─────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(INPUT, encoding="utf-8") as f:
        records: list[dict] = json.load(f)

    print(f"Loaded {len(records)} records from {INPUT.name}")

    output_records = []
    category_counts: dict[str, int] = {c: 0 for c in CATEGORIES}

    for r in records:
        match = classify(r)
        if not match:
            continue

        categories = sorted(match.keys())
        matched_terms = sorted({t for terms in match.values() for t in terms})

        enriched = dict(r)
        enriched["categories"] = categories
        enriched["matched_terms"] = matched_terms
        output_records.append(enriched)

        for cat in categories:
            category_counts[cat] += 1

    # Deduplicate by registration_number (keep the one with more fields)
    seen: dict[str, dict] = {}
    for r in output_records:
        reg = r.get("registration_number") or r.get("register_nr.", "")
        if not reg or reg == "?":
            output_records_deduped = output_records  # keep as-is if no reg
            break
        if reg not in seen or len(r) > len(seen[reg]):
            seen[reg] = r
    else:
        output_records_deduped = list(seen.values())

    output = {
        "generated_from": "frobenius_all.json",
        "filter_description": (
            "Records specifically depicting: door panels (carved wooden doors, "
            "wall boards), Ifa divination boards (opon Ifa, oracle boards, iroke), "
            "or figurines (bronze figures, wooden cult figures, Ogboni edan, "
            "ivory carvings)."
        ),
        "total_input_records": len(records),
        "total_output_records": len(output_records_deduped),
        "counts_by_category": {
            cat: sum(1 for r in output_records_deduped if cat in r["categories"])
            for cat in CATEGORIES
        },
        "records": output_records_deduped,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {OUTPUT.name}")
    print(f"  Total matched: {len(output_records_deduped)} / {len(records)}")
    for cat, count in output["counts_by_category"].items():
        print(f"  {cat:15s}: {count}")


if __name__ == "__main__":
    main()
