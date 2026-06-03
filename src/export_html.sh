#!/usr/bin/env bash
# export_html.sh — convert executed notebooks to self-contained HTML
#
# Usage:
#   ./src/python/export_html.sh                  # convert all notebooks
#   ./src/python/export_html.sh motif_similarity  # convert one by name
#
# Output: frobenius_artifacts/site/<notebook>.html
#
# Notes:
#   • Runs with --no-input by default (hides code cells); pass --with-code to show them.
#   • Widgets render as their last saved state (not interactive).
#     For best results: run the notebook fully in Jupyter, then save
#     (File → Save Notebook) before running this script.
#   • All images are base64-embedded → single portable .html file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NB_DIR="$REPO_ROOT/src/python"
SITE_DIR="$REPO_ROOT/frobenius_artifacts/site"
mkdir -p "$SITE_DIR"

# ── Parse args ─────────────────────────────────────────────────────────────
SHOW_CODE=0
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --with-code)  SHOW_CODE=1 ;;
        --*)          echo "Unknown flag: $arg" >&2; exit 1 ;;
        *)            FILTER="$arg" ;;
    esac
done

INPUT_FLAG="--no-input"
[[ $SHOW_CODE -eq 1 ]] && INPUT_FLAG=""

# ── Notebooks to export ────────────────────────────────────────────────────
declare -a NOTEBOOKS=(
    "motif_similarity"
    "bbox_review"
)

convert_one() {
    local name="$1"
    local src="$NB_DIR/${name}.ipynb"
    local dst="$SITE_DIR/${name}.html"

    if [[ ! -f "$src" ]]; then
        echo "  SKIP: $src not found"
        return
    fi

    echo "  → $name ..."
    uv run --project "$NB_DIR" jupyter nbconvert \
        --to html \
        --embed-images \
        $INPUT_FLAG \
        --output "$dst" \
        "$src"

    # Inject a minimal nav/style header so it looks reasonable standalone
    TITLE=$(echo "$name" | tr '_' ' ' | sed 's/\b\(.\)/\u\1/g')
    python3 - "$dst" "$TITLE" <<'PYEOF'
import sys, re
path, title = sys.argv[1], sys.argv[2]
html = open(path).read()
banner = f"""<div style="font-family:sans-serif;background:#111;color:#eee;
  padding:12px 20px;display:flex;align-items:center;gap:16px">
  <span style="font-size:18px;font-weight:600">Frobenius Panel Art</span>
  <span style="opacity:.5">·</span>
  <span style="opacity:.8">{title}</span>
</div>"""
html = html.replace("<body>", f"<body>\n{banner}", 1)
open(path, "w").write(html)
PYEOF

    size=$(du -sh "$dst" | cut -f1)
    echo "    saved $dst  ($size)"
}

if [[ -n "$FILTER" ]]; then
    convert_one "$FILTER"
else
    echo "Exporting notebooks → $SITE_DIR/"
    for nb in "${NOTEBOOKS[@]}"; do
        convert_one "$nb"
    done
fi

echo ""
echo "Done. Host frobenius_artifacts/site/ on any static server."
echo "  Quick local preview:"
echo "    python3 -m http.server 8080 --directory $SITE_DIR"
