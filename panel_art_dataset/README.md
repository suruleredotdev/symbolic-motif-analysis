# Panel Art Dataset

Focused dataset of panel/plaque-type African artwork — doors, divination boards,
relief plaques — for comparative pictorial analysis.

## Focus objects

| Category | Examples |
|---|---|
| Yoruba door panels | Carved wooden palace/temple doors, often by known carvers (Olowe of Ise etc.) |
| Yoruba Ifa divination boards (opon Ifa) | Circular/rectangular wooden trays with figurative borders |
| Benin bronze/brass relief plaques | Court scenes and individual warrior/official figures |

## Sources covered

| Source | Script | Key |
|---|---|---|
| Art Institute of Chicago (ARTIC) | `gather_panel_art.ts` | No API key needed |
| Staatliche Museen zu Berlin (SMB) | `gather_panel_art.ts` | No API key needed |
| British Museum | `gather_panel_art.ts` | No API key needed |
| Frobenius Institut Bildarchiv | `scrape_frobenius.py` | No key; session-based |

## Directory layout

```
panel_art_dataset/
├── README.md          — this file
├── db_queries.sql     — SQL queries for searching the local DB
├── summary.json       — run summary from gather_panel_art.ts (generated)
├── raw/               — raw API JSON responses, one file per query
│   ├── artic__yoruba_door_panel.json
│   ├── artic__ifa_divination_board.json
│   ├── artic__benin_plaque.json
│   ├── smb__benin_reliefplatte.json
│   ├── smb__yoruba_tuer.json
│   ├── british__yoruba_door.json
│   ├── frobenius_raw.json        ← from scrape_frobenius.py
│   └── ...
└── parsed/            — normalised MuseumObject[] JSON, one file per query
    ├── artic__yoruba_door_panel.json
    ├── artic__ifa_divination_board.json
    ├── artic__benin_plaque.json
    ├── smb__benin_reliefplatte.json
    ├── smb__yoruba_tuer.json
    ├── british__yoruba_door.json
    ├── frobenius.json            ← from scrape_frobenius.py
    └── ...
```

## Running the gather script (museum APIs)

```bash
# From the src/typescript directory:
cd src/typescript
npx tsx backend/scripts/gather_panel_art.ts
```

No environment variables or API keys are required for the default museums.
Raw JSON is written to `panel_art_dataset/raw/` and parsed output to
`panel_art_dataset/parsed/` — nothing is written to the database.

## Running the Frobenius scraper

```bash
# Install deps if needed:
pip install -r src/python/requirements.txt

# Run the scraper (fetches all panel-type records from the Yoruba search):
python3 src/python/scripts/scrape_frobenius.py

# Options:
python3 src/python/scripts/scrape_frobenius.py --help
python3 src/python/scripts/scrape_frobenius.py --dry-run       # see what would be fetched
python3 src/python/scripts/scrape_frobenius.py --no-filter     # capture all records
python3 src/python/scripts/scrape_frobenius.py --max-pages 10  # limit crawl depth
```

### Notes on the Frobenius site

- URL: http://bildarchiv.frobenius-katalog.de/
- Uses session IDs (`sid=`) — the scraper obtains a fresh one on startup
- ~25 records per search results page; `--max-pages 50` covers ~1250 stubs
- Images are served via `zvimg.FAU` URLs embedded in detail pages
- The scraper filters to panel-type objects (doors, boards, trays, plaques)
  using `PANEL_KEYWORDS`; use `--no-filter` to capture everything

#### Example URLs

```
# Search results (starting point):
http://bildarchiv.frobenius-katalog.de/rech.FAU?sid=66C606474&dm=1&auft=0

# Detail record (Yoruba temple doors, Modakeke):
http://bildarchiv.frobenius-katalog.de/hzeig.FAU?sid=66C6064746&dm=1&ind=1&zeig=FoA%2004-5578

# Image:
http://bildarchiv.frobenius-katalog.de/zvimg.FAU?sid=66C60647&DM=1&qpos=48633&ipos=1&erg=A&hst=1&rpos=48633.png
```

## DB queries

See `db_queries.sql` for ready-to-run SQL covering:

- **A. Keyword / full-text** — no embedding required; fast
  - A1. Yoruba door panels
  - A2. Ifa divination boards
  - A3. Benin relief plaques — individual figures
  - A4. Benin palace scenes — multi-figure
  - A5. Combined broad panel/plaque/board/door search
  - A6. SMB-specific Reliefplatte search

- **B. Semantic embedding** — requires `text_embedding` populated
  - B1. Embedding-only search (swap in your vector)
  - B2. Keyword pre-filter + embedding re-rank (hybrid, more efficient)

- **C. Utility / diagnostics**
  - C1. Count panel-type objects by museum
  - C2. Similarity to a known reference object
  - C3. Yoruba objects grouped by type
  - C4. Benin objects with images

### Generating embedding vectors for query B

```typescript
// In a tsx script (see generateTextEmbeddings.ts for reference):
import { generateEmbedding } from "../lib/embeddings";
const queryText = "Yoruba carved wooden door panel figurative relief Ifa";
const vector = await generateEmbedding(queryText);
// Paste as the literal vector in the SQL query above
```

## SMB example records

The following SMB detail pages are confirmed Benin relief plaques:

| ID | URL | Notes |
|---|---|---|
| 212077 | https://recherche.smb.museum/detail/212077/fisch | Individual figure (Fisch/fish motif) |
| 492257 | https://recherche.smb.museum/detail/492257/palasthof | Palace court scene, multiple figures |

Their object numbers / identNumbers can be used to search the SMB API:
```
https://api.smb.museum/search?q=Reliefplatte+Benin&size=100&page=1
```
