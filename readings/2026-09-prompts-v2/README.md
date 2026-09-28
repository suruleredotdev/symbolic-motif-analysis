# Readings, September 2026 (prompts v2)

Motif, panel and corpus readings for the 37 Frobenius panels, written under
the v2 prompts in `panel_art/interpret.py`. A Claude session with no API key
wrote them by looking at each photograph and drawing, so they did not come
from a pipeline run. They are kept here because they cannot be regenerated.
Anything a pipeline run writes stays out of git.

- `<panel_stem>.json`: `{"motifs": {index: reading}, "panel": reading}`, in
  the `MOTIF_READING_SCHEMA` and `PANEL_SCHEMA` shapes. Every motif reading
  carries its `bbox`, so it still lands on the right motif after boxes are
  edited.
- `corpus.md`: the collection essay.

These readings match the approved boxes as of 26 Sep 2026, including the
FoA_04-5947 re-segmentation of 25 Sep and the three FoA_04-5580 boxes added on
25 Aug. The two labels by a person (EBA-Div_00302 #0 and #1) are not included,
and the importer leaves them untouched.

## Apply them

```bash
python3 scripts/import_readings.py --analysis-dir frobenius_artifacts/analysis \
  --readings readings/2026-09-prompts-v2 \
  --corpus   readings/2026-09-prompts-v2/corpus.md \
  --model    "in-session reading (prompts v2)" --dry-run   # check, then drop --dry-run
python3 scripts/run_interpretation.py --analysis-dir frobenius_artifacts/analysis --only site
```

## Regenerate through the API instead

```bash
python3 scripts/run_interpretation.py --analysis-dir frobenius_artifacts/analysis \
  --embeddings motif_embeddings_edges.npy --paths motif_paths_edges.txt
```
