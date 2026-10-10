# Expanding the corpus: panel art from 1910 and earlier

Where to find more carved door panels, door posts, Ifa boards and relief
plaques made by 1910, which sources can be scraped, and whom to ask for
the rest. Researched October 2026; every "verified" note below means the
endpoint answered from a script on 2026-10-10.

The current corpus is 108 Frobenius Institut records from the 1910–12
expedition (`panel_art_dataset/frobenius_panel_art.json`). The aim is a
corpus large enough to test whether motif families persist **across
places and carvers**, not only within one expedition's photographs.

---

## The dating rule: terminus ante quem, not the label

Museum dates for Yoruba carving are mostly stylistic guesses ("19th–20th
century", "late 1800s"). The year an object was collected, accessioned or
photographed is evidence. So the cutoff is applied to the **terminus ante
quem (TAQ)**: the earliest of

- the end of the catalogued date range ("1890s" → 1899, "1905–15" → 1915),
- the accession or acquisition year (`Af1924,…` at the British Museum means 1924),
- the end of the collector's collecting window (Frobenius in Nigeria/Cameroon → 1912),
- the collector's death (Hans Glauning, d. 1908 → 1908),
- the date of a field photograph showing the object in situ.

The Met lists most of its Ifa trays as "19th–20th century", but it
acquired them in 1978–1991, so nothing in those records proves they
predate 1910. Glauning's Cameroon door posts in Berlin carry no date at
all, but he died in 1908, so they are pre-1910.

`scripts/gather_open_collections.py` applies this rule automatically.
Use `--before 1912` to admit the Frobenius-era objects alongside the
earlier ones.

---

## Tier 1: scrape now

| Source | What's there | Access | Verified |
|---|---|---|---|
| **museum-digital** (mirrors SMB Ethnologisches Museum Berlin, plus Übersee-Museum Bremen, Museum Ulm and others) | 202 "Yoruba" objects; 78 "Frobenius + Yoruba" incl. Ifa boards III C 27124, 27126; Frobenius door leaf **III C 27243 a** (288 cm); 10 Cameroon door posts and a gate collected by **Hans Glauning (d. 1908)**; Benin plaques III C 7655, 8054–8056, 8265, 8348–8395 | JSON API, no key: `global.museum-digital.org/json/objects?s=…`, `/json/object/<id>`. Images via `id.smb.museum`. Metadata licence CC BY-NC-SA | ✅ 12 panel objects with TAQ ≤ 1912 from one run |
| **SMB search portal** ([search.smb.museum](https://search.smb.museum/)) | ~270,000 Berlin objects; public-domain images released under open licences, some mirrored to Wikimedia Commons ([SMB announcement](https://www.smb.museum/en/whats-new/detail/collections-online-open-access-wikimedia-commons-the-staatliche-museen-zu-berlin-expands-its-online-content/)) | `api.smb.museum` reset the connection from this sandbox. Retry it locally; museum-digital covers the same records in the meantime | ⚠️ portal up, API unconfirmed |
| **The Met** | Benin plaques ca. 1540–70 (23 with images, Dept. 5); Yoruba Ifa trays (accessioned after 1978, so they fail the cutoff) | `collectionapi.metmuseum.org/public/collection/v1.1/search`. **v1 search was retired 2026-10-01** and now returns 410 | ✅ |
| **Europeana** | Aggregates German, Dutch and French ethnographic museums; noisy for this query set | `api.europeana.eu/record/v2/search.json`. The `api2demo` key works; [register a key](https://pro.europeana.eu/page/get-api) for real runs | ✅ 61 candidates, all needing manual dating |
| **Wikimedia Commons** | `Category:Yoruba doors` (14 files: an Areogun palace door at Museum Rietberg, and Tropenmuseum panels TM 4416-x, **commissioned 1977–78**, so comparanda only) | MediaWiki API | ⚠️ HTTP 429 from this sandbox; runs fine from a normal connection |
| **Art Institute of Chicago, British Museum** | Already gathered by `gather_panel_art.ts` | — | — |
| **Smithsonian** | NMAfA objects and the **Eliot Elisofon Photographic Archives** (EEPA). Metadata is CC0 with IIIF manifests; images need EEPA permission | The Open Access API (`api.si.edu`) only returns CC0 items: NMAfA/EEPA door queries returned **0**. Use [collections.si.edu](https://collections.si.edu) or an EDAN key | ⚠️ |
| **Digital Benin** ([digitalbenin.org](https://digitalbenin.org/catalogue)) | 5,246 Benin objects from 131 institutions; the authoritative plaque census | Was showing "Under Maintenance" (HTTP 503) on 2026-10-10. Ask MARKK Hamburg about a data export | ⚠️ |

---

## Tier 2: catalogued, but ask the holder

These are the strongest pre-1910 sources. Most have a searchable
catalogue, but full-resolution images or the full set need a request.

### Field photographs (in situ, firmly dated)

- **Frobenius Institut, Frankfurt.** We have the search-result records.
  The next step is research-resolution files and the drawings (EBA). The
  [order page](https://frobenius-institut.de/en/datenbanken/online-bildarchiv/bildbestellung)
  routes EBA/FBA/KBA/PBA (ethnographic drawings, rock art) to
  **Dr Richard Kuba** and FoA photos to **Jennifer Markwirth**. A 2025
  institute press release gives Kuba@em.uni-frankfurt.de. There is a €28
  provision fee per publishable file and €30 where no file exists yet;
  scientific use generally carries no copyright fee.
- **MAA Cambridge: the Northcote W. Thomas collection, 1909–1915.** About
  7,000 glass negatives and 6,200 prints from Edo, Igbo and Sierra Leone
  surveys ([MAA project page](https://maa.cam.ac.uk/collections/anthropology/northcote-w-thomas-project),
  [[Re:]Entanglements](https://re-entanglements.net/)). Edo shrine and
  palace carving is the closest neighbour to the Yoruba corpus. Contacts:
  MAA collections staff; **Dr Eva Namusoke**, Senior Curator, African
  Collections Futures ([profile](https://www.arch.cam.ac.uk/staff/dr-eva-namusoke));
  Prof. Paul Basu (SOAS), who led the project. **Ask Dr Steele for the
  introduction.** She is at Cambridge and hosted the VIEWS/EWRN seminar.
- **Cambridge University Library.** The Royal Commonwealth Society
  photographs (100,000+ images, 1850s–1980s; includes Sir Walter
  Buchanan-Smith's Nigeria collection, c. 1909, ref. Y3043U)
  ([RCS project](https://www.repository.cam.ac.uk/handle/1810/752)), and
  the **Church Missionary Society photograph collection**, series CMS 28,
  1892–1914 ([ArchiveSearch](https://archivesearch.lib.cam.ac.uk/repositories/2/resources/95/digitized)).
  Search both for Abeokuta, Ibadan, Oyo, Ijebu, Ekiti, "door", "palace".
- **The National Archives (UK), CO 1069.** Colonial Office photographs,
  partly on Flickr Commons. CO 1069/80 "Lagos, general views – interior
  expedition, c. 1890s" is catalogued but not online: it can be ordered.
- **Basel Mission Archives.** Cameroon photographs of carved doors in
  situ, e.g. E-30.12.047 (Jakob Stutz, 1904–1912), at [bmarchives.org](https://www.bmarchives.org).
  Publication fees apply: info@bmarchives.org.
- **Yale–Van Rijn Archive of African Art.** About 150,000 photographs of
  African art made before the mid-20th century. Open by appointment at
  Yale or in Brussels, and partly through JSTOR
  ([Yale page](https://artgallery.yale.edu/research-and-learning/curatorial-areas/african-art/african-art-digital-archives)).

### Objects

- **British Museum.** Af1924,-.135.a-c, Olowe of Ise's door and lintel
  for the Ogoga of Ikere, c. 1910–14, depicting the arrival of Captain
  Ambrose around 1895
  ([record](https://www.britishmuseum.org/collection/object/E_Af1924-135-a-c)).
  It sits on the boundary and is the anchor for named-carver comparison.
  Also Af1954,23.208.a-c (panels shown at Wembley, 1924). Ask the Africa
  department for a list of Yoruba, Ekiti and Ijebu doors **registered
  before 1912**. Registration-number prefixes make that a simple query on
  their side.
- **US museums with 19th-century attributions.** These still need
  provenance, because each date is stylistic:
  Brooklyn Museum 22.1526, the "Master of Ikare" door, late 19th c.,
  accessioned 1922 ([record](https://www.brooklynmuseum.org/objects/4670));
  Speed Art Museum, a pair of Ijebu-Ode doors dated to the 1890s;
  Nelson-Atkins, a door from a Shango shrine, 19th c.
  ([record](https://art.nelson-atkins.org/objects/29388/door-from-shrine-for-deity-shango));
  Denver Art Museum 1973.357; NCMA, an Osugbo lodge panel from Ijebu.
  For each, ask the registrar: when did it leave Nigeria, and who held
  it before the museum?
- **Other German ethnographic museums** (colonial-era collecting, mostly
  pre-1914): GRASSI Leipzig / SKD, Linden-Museum Stuttgart, Übersee-Museum
  Bremen (Ifa board B14779 is already on museum-digital), Museum Ulm
  (Ifa-Brett AB 1486), MARKK Hamburg, Weltkulturen Museum Frankfurt (the
  Frobenius Institute's object collection went there in 1925). For
  Cameroon door frames, start from *Atlas der Abwesenheit: Kameruns
  Kulturerbe in Deutschland* (TU Berlin / Université de Dschang).
- **Wereldmuseum (NL).** [collectie.wereldmuseum.nl](https://collectie.wereldmuseum.nl/),
  collectie@wereldculturen.nl.
- **Abomey palace doors and reliefs (Dahomey).** Taken by French forces
  in 1892–93 and held at the Quai Branly until their return to Benin in
  2021–22. They are a pre-1910 panel tradition next door to the Yoruba,
  now with the Republic of Benin's museums.
- **Nigeria.** NCMM / National Museum Lagos has the largest in-country
  holding. The Met–NCMM digitisation partnership (announced April 2023)
  is the most likely route to images
  ([Met release](https://www.metmuseum.org/press-releases/ncmm-project-announcement-2023-news)).
  Prof. Ogundiran is the natural introduction here, and to Ibadan and Ife
  faculty.

### Rejected or post-1910, but useful as a control set

- MAA 1947.28 B and 1947.29 A: panels attributed to Areogun of Osi
  (c. 1880–1954), acquired in 1946.
- Tropenmuseum TM 4416-x: panels commissioned 1977–78.
- The Met's Ifa trays: accessioned 1978–1991.

These belong in the **named-carver control set** (below), not the archaic
corpus.

---

## Tier 3: printed sources

Plates in early publications are the drawings and photographs collectors
made at the time, and many are on archive.org:

- Frobenius, *Und Afrika sprach* (1912) / *The Voice of Africa* (1913).
  Already linked from the README.
- To check for plates of doors and boards: A. B. Ellis, *The
  Yoruba-Speaking Peoples* (1894); Read & Dalton, *Antiquities from the
  City of Benin* (1899); H. Ling Roth, *Great Benin* (1903); von Luschan,
  *Die Altertümer von Benin* (1919, objects removed in 1897).

The literature Prof. Ogundiran pointed to, for documented motif meanings
and attributions: Carroll, *Yoruba Religious Carving* (1967); Fagg &
Pemberton, *Yoruba Sculpture of West Africa* (1982); Drewal, Pemberton &
Abiodun, *Yoruba: Nine Centuries* (1989); Abiodun, Drewal & Pemberton
(eds.), *The Yoruba Artist* (1994); Walker, *Olowe of Ise* (1998);
Fọlárànmí on the Ọ̀yọ́ palace doors; Arnoldi's MSU thesis on Yoruba
doors, which notes that their iconography is "still to be explored".

---

## How this extends the work so far

Talking points for outreach, so each request is concrete:

- **What exists.** An open-source pipeline
  ([repo](https://github.com/suruleredotdev/symbolic-motif-analysis))
  that segments carved motifs, embeds and clusters them into families, and
  supports human and LLM-assisted labeling and readings. A first browsable
  motif archive is live at
  [symbolic-motif-analysis.surulere.dev](https://symbolic-motif-analysis.surulere.dev/),
  built from the Frobenius 1910–12 photographs.
- **Cambridge.** Presented as "Yoruba Symbolic Systems" at the VIEWS
  (Visual Interactions in Early Writing Systems) / Endangered Writing
  Research Network seminar, University of Cambridge, 28 September 2026,
  organised by VIEWS (Dr Philippa Steele). The writing-systems question it raises
  is whether a sign keeps its form and its position across carvers,
  places and decades. That needs exactly the cross-collection, dated
  corpus this document is about.
- **Northwestern.** Prof. Akin Ogundiran (Material History Lab) advised
  being systematic: start from documented carvers (Areogun to Lamidi
  Fakeye) and the publications that record motif meanings, build an
  easy-to-navigate archive first, and consult Prof. Henry Drewal. He also
  flagged Oyo pottery, calabash decoration and embroidery as the next
  media.
- **The design that answers both.** Run two corpora through the same
  pipeline. An **archaic corpus** with TAQ ≤ 1912 (Frobenius, Glauning,
  the 1890s photographs, early accessions). A **named-carver control
  set**: Olowe's Ikere door, then Bamgbose, Areogun, Fakeye, whose
  published meanings serve as partial ground truth. Families that span
  both are the candidates for stable signs. Families that appear only in
  the archaic set are what the field photographs are for.

---

## Outreach

### What to ask every holder

1. A list of door panels, door posts, Ifa boards and relief plaques from
   West Africa **accessioned or photographed before 1912**, with
   accession numbers and collector names.
2. Research-resolution images of those objects (TIFF or high-quality
   JPEG), including **field or accession photographs**.
3. Permission for **computational analysis**, and to publish derived
   motif crops at low resolution with credit, in a non-commercial open
   research archive.
4. Any documentation recorded at collection: carver, town, the owner's
   explanation of the scenes.

Asking for (3) explicitly saves a second round: most image licences don't
mention crops or embeddings.

### Template: collection manager / registrar

> Subject: Research request: West African carved door panels and Ifa boards collected before 1912
>
> Dear [Name],
>
> I am an independent researcher building an open, computational archive
> of the carved motifs on Yoruba and neighbouring West African door
> panels, Ifa divination boards and relief plaques. The pipeline
> segments each motif, groups recurring forms across objects, and
> supports scholars in labeling and interpreting them. A first version,
> built from the Frobenius Institut's 1910–12 expedition photographs, is
> at symbolic-motif-analysis.surulere.dev, and the code is open source.
>
> I recently presented this work, "Yoruba Symbolic Systems", at the
> VIEWS / Endangered Writing Research Network seminar at the University
> of Cambridge (Dr Philippa Steele), and it is being developed in
> conversation with Prof. Akin Ogundiran (Northwestern, Material History
> Lab).
>
> I am now extending the corpus to objects with firm early dates, and
> your collection holds [object / series, e.g. "the Glauning door posts,
> III C 20327–21124"]. Could you help with:
>
> 1. a list of comparable objects accessioned or photographed before 1912;
> 2. research-resolution images, including any field or accession photographs;
> 3. permission to analyse them computationally and to publish
>    low-resolution motif crops with full credit in a non-commercial
>    research archive.
>
> I am glad to share the results back with the museum, including motif
> annotations for your records.
>
> With thanks,
> Korede Aderele
> surulere.dev

### Specific asks

| To | Ask | Why them |
|---|---|---|
| Dr Philippa Steele (Cambridge, VIEWS/EWRN) | Introductions to MAA (Northcote Thomas, Areogun panels) and to the CUL RCS/CMS photograph curators. Also whether VIEWS has a framework for sign variation that the clustering should report against | Follow-up to the 28 Sept seminar; Cambridge holds three of the Tier 2 sources |
| Prof. Akin Ogundiran (Northwestern) | Introductions at NCMM / National Museum Lagos, Ibadan and Ife. Whether Material History Lab students could take on the literature pass (Carroll, Fagg & Pemberton, Walker) as RAs | His own suggestion of a systematic, carver-first approach; he asked how to scale |
| Dr Richard Kuba (Frobenius Institut) | Research-resolution files for the 108 records already identified, plus the EBA drawings of carved doors; terms for computational use | We already depend on this archive; formalising it unblocks publication |
| Ethnologisches Museum Berlin | High-resolution images and acquisition years for III C 27243 a (Frobenius door leaf), the Glauning door posts, and the III C 8xxx Benin plaques | They're already identified, so the request is narrow |
| British Museum, Africa department | A query of Yoruba, Ekiti and Ijebu doors and boards with registration year ≤ 1912 | Registration numbers encode the year; trivial for them, impossible for us |
| MARKK Hamburg (Digital Benin) | A bulk export of plaque records and images | The site was down on 2026-10-10; an export is better than scraping anyway |
| Prof. Henry Drewal | Already emailed (Aug 2026). Follow up with the dated-corpus design and ask which doors he would treat as anchor examples | Recommended by Prof. Ogundiran |

---

## Running the gatherer

```bash
# All open sources, records with TAQ ≤ 1912
python3 scripts/gather_open_collections.py --before 1912 \
  --out panel_art_dataset/open_collections.json

# Keep undated records for manual dating (Europeana mostly lands here)
python3 scripts/gather_open_collections.py --keep-undated

# One source, everything, no relevance filter
python3 scripts/gather_open_collections.py --sources museum_digital --no-filter --keep-undated --keep-later
```

Each record keeps its `source`, `query`, `collector`, `date_display`,
`acquired_year`, computed `taq`, and `pre_cutoff` (`yes` / `no` /
`unknown`). Review the `yes` and `unknown` records before downloading
images. Europeana and Commons return comparanda and modern photographs
alongside historic objects. Set `EUROPEANA_KEY` for anything beyond a
trial run.
