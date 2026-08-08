#!/usr/bin/env bash
# make_subset.sh — stage a shareable subset of the panel corpus for collaborators
#
# Usage:
#   ./make_subset.sh                          # use subset_panels.txt
#   ./make_subset.sh --list my_panels.txt     # use a different stem list
#   ./make_subset.sh --out /tmp/share         # staging dir (default: subset_share/)
#   ./make_subset.sh --tar                    # also write <out>.tar.gz
#   ./make_subset.sh --no-source              # omit source photographs (derived only)
#
# Reads each panel stem from the list and copies, into a mirror of the
# frobenius_artifacts layout:
#
#   analysis/panels/<stem>.png              panel crop (and _cropped variant)
#   analysis/motifs/<stem>/*.png            motif crops
#   analysis/motifs/<stem>_motif_*.svg      vectorised motifs (stored flat, not nested)
#   analysis/motifs_norm/<stem>/*.png       normalised crops
#   analysis/annotated/<stem>_*             annotated jpg + detections/approved json
#   images/<source>.png                     source photograph, deduplicated
#
# Source photographs carry the tightest licensing restrictions — several panels
# usually share one original, so the image count is much lower than the panel
# count. Use --no-source when sharing with collaborators who only need crops.

set -euo pipefail
shopt -s nullglob

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$REPO_ROOT/frobenius_artifacts"

LIST="$REPO_ROOT/subset_panels.txt"
OUT="$REPO_ROOT/subset_share"
MAKE_TAR=0
WITH_SOURCE=1

# ── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)       LIST="$2"; shift 2 ;;
        --out)        OUT="$2";  shift 2 ;;
        --tar)        MAKE_TAR=1; shift ;;
        --no-source)  WITH_SOURCE=0; shift ;;
        -h|--help)    sed -n '2,26p' "$0"; exit 0 ;;
        *)            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -e "$SRC" ]] || {
    echo "ERROR: $SRC not found." >&2
    echo "  It is gitignored — symlink or copy it from the african-artifacts checkout:" >&2
    echo "    ln -s ../african-artifacts/frobenius_artifacts frobenius_artifacts" >&2
    exit 1
}
[[ -f "$LIST" ]] || { echo "ERROR: stem list not found: $LIST" >&2; exit 1; }

ANA="$SRC/analysis"
IMAGES="$SRC/images"

# ── Read stems (skip blanks and # comments) ────────────────────────────────
declare -a STEMS=()
while IFS= read -r line; do
    line="${line%%$'\r'}"                       # tolerate CRLF
    [[ -z "${line// }" ]] && continue
    [[ "$line" == \#* ]]  && continue
    STEMS+=("$line")
done < "$LIST"

[[ ${#STEMS[@]} -gt 0 ]] || { echo "ERROR: no panel stems in $LIST" >&2; exit 1; }

echo "Staging ${#STEMS[@]} panel(s) → $OUT/"
echo ""

rm -rf "$OUT"
mkdir -p "$OUT/analysis"/{panels,motifs,motifs_norm,annotated}
[[ $WITH_SOURCE -eq 1 ]] && mkdir -p "$OUT/images"

# copy_glob <destination dir> <file...> — copies only what exists, counts hits
copied=0
copy_glob() {
    local dest="$1"; shift
    local f
    for f in "$@"; do
        [[ -e "$f" ]] || continue
        cp -p "$f" "$dest/"
        copied=$((copied + 1))
    done
}

declare -a MISSING=()
declare -A SOURCES=()

for stem in "${STEMS[@]}"; do
    before=$copied

    # Panel crop — some panels are stored with a _cropped suffix instead
    copy_glob "$OUT/analysis/panels" \
        "$ANA/panels/$stem".png "$ANA/panels/$stem"_cropped.png

    # Motif crops + normalised crops live in per-panel subdirectories
    if [[ -d "$ANA/motifs/$stem" ]]; then
        mkdir -p "$OUT/analysis/motifs/$stem"
        copy_glob "$OUT/analysis/motifs/$stem" "$ANA/motifs/$stem"/*.png
    fi
    if [[ -d "$ANA/motifs_norm/$stem" ]]; then
        mkdir -p "$OUT/analysis/motifs_norm/$stem"
        copy_glob "$OUT/analysis/motifs_norm/$stem" "$ANA/motifs_norm/$stem"/*.png
    fi

    # Vectorised SVGs sit flat in motifs/, named <stem>_motif_NNN.svg
    copy_glob "$OUT/analysis/motifs" "$ANA/motifs/$stem"_motif_*.svg

    # Annotation sidecars: _annotated.jpg, _detections.json, _approved.json
    copy_glob "$OUT/analysis/annotated" "$ANA/annotated/$stem"_*

    # Source photograph: strip the trailing _panel_NN to get the original stem
    src_stem="${stem%_panel_*}"
    SOURCES["$src_stem"]=1

    if [[ $copied -eq $before ]]; then
        MISSING+=("$stem")
        echo "  !  $stem — nothing found"
    else
        printf '  ok %-52s %3d file(s)\n' "${stem:0:52}" "$((copied - before))"
    fi
done

# ── Source photographs, deduplicated across panels ─────────────────────────
img_count=0
if [[ $WITH_SOURCE -eq 1 ]]; then
    echo ""
    echo "Source photographs (deduplicated):"
    for src_stem in "${!SOURCES[@]}"; do
        hits=("$IMAGES/$src_stem".*)
        if [[ ${#hits[@]} -eq 0 ]]; then
            echo "  !  $src_stem — no source image found"
            continue
        fi
        for f in "${hits[@]}"; do
            cp -p "$f" "$OUT/images/"
            img_count=$((img_count + 1))
            echo "  ok $(basename "$f")"
        done
    done
fi

# ── Provenance manifest ────────────────────────────────────────────────────
{
    echo "# Subset bundle"
    echo ""
    echo "Generated: $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo "Source:    frobenius_artifacts (Frobenius Institut Bildarchiv)"
    echo ""
    echo "## Panels (${#STEMS[@]})"
    echo ""
    printf '%s\n' "${STEMS[@]}" | sed 's/^/  - /'
    echo ""
    if [[ $WITH_SOURCE -eq 1 ]]; then
        echo "## Source photographs ($img_count)"
        echo ""
        printf '%s\n' "${!SOURCES[@]}" | sort | sed 's/^/  - /'
    else
        echo "## Source photographs"
        echo ""
        echo "  Omitted (--no-source). Derived crops only."
    fi
    echo ""
    echo "## Terms"
    echo ""
    echo "  Shared for non-commercial research use with named collaborators."
    echo "  Do not redistribute. Source images remain the property of the"
    echo "  Frobenius Institut; see TERMS.md before sharing onward."
} > "$OUT/MANIFEST.md"

# ── Summary ────────────────────────────────────────────────────────────────
total=$((copied + img_count))
size=$(du -sh "$OUT" | cut -f1)

echo ""
echo "─────────────────────────────────────────────"
printf 'derived files : %d\n' "$copied"
printf 'source images : %d\n' "$img_count"
printf 'total         : %d files, %s\n' "$total" "$size"

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "WARNING: ${#MISSING[@]} stem(s) matched nothing — check for typos:"
    printf '  %s\n' "${MISSING[@]}"
fi

if [[ $MAKE_TAR -eq 1 ]]; then
    tar_path="$OUT.tar.gz"
    tar -czf "$tar_path" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    echo ""
    echo "archive: $tar_path ($(du -sh "$tar_path" | cut -f1))"
fi

echo ""
echo "Next: upload to the shared Drive folder, e.g."
echo "  rclone sync '$OUT' gdrive:motif-subset --progress"
