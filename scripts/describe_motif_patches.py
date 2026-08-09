"""
describe_motif_patches.py

Sends each extracted motif patch to Claude (claude-haiku-4-5) to generate
a structured visual description, then writes results to JSON.

Input:   *_patches.json  OR  *_embeddings.json  (from previous pipeline stages)
Output:  *_described.json  — patches enriched with 'description' field (JSON object)

Usage:
  # Describe patches from a manifest
  python3 describe_motif_patches.py path/to/image_patches.json

  # From embeddings manifest (adds descriptions alongside vectors)
  python3 describe_motif_patches.py image_embeddings.json

  # Dry run — show prompts and first Claude response, don't write output
  python3 describe_motif_patches.py patches.json --dry-run

  # Resume — skip patches that already have a description
  python3 describe_motif_patches.py patches.json --resume

  # Use a different model
  python3 describe_motif_patches.py patches.json --model claude-opus-4-6

Environment:
  ANTHROPIC_API_KEY  — required

Dependencies:
  pip install anthropic pillow
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panel_art.interpret import text_of  # noqa: E402

# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an art historian specialising in West African visual culture, \
with deep expertise in Yoruba carved wooden objects — door panels, Ifa \
divination boards, figurines, and architectural sculpture. \
You analyse details from early 20th-century ink illustrations and drawings \
produced by Leo Frobenius's expeditions (1910–1912, Nigeria/Benin). \
Be precise, concise, and grounded in what is visually present.\
"""

USER_PROMPT_TEMPLATE = """\
This is a detail extracted from a Yoruba carved panel illustration in the \
Frobenius archive ({source_context}).

Analyse this motif and respond with a JSON object using exactly these keys:

{{
  "subject": "One sentence — what figure, object, or pattern is depicted",
  "composition": "One sentence — symmetry, orientation, density, spatial organisation",
  "technique": "Brief note on the carved medium visible (relief, openwork, incised line, etc.)",
  "symbolism": "Any recognisable iconographic elements (Edschu/Eshu, Ifa, Shango, Ogboni, \
geometric cosmogram, etc.) — write null if uncertain",
  "keywords": ["array", "of", "3–6", "short", "descriptive", "terms"],
  "confidence": "high | medium | low — your confidence in the subject identification"
}}

Output only the JSON object, no markdown fences, no extra commentary.\
"""

# ─── Image encoding ───────────────────────────────────────────────────────────

def encode_image_b64(image_path: Path, max_dim: int = 1024) -> tuple[str, str]:
    """
    Load image, resize if needed (API limit), return (base64_data, media_type).
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return data, "image/jpeg"


# ─── Claude call ──────────────────────────────────────────────────────────────

def describe_patch(
    client: "anthropic.Anthropic",
    patch: dict,
    base_dir: Path,
    model: str,
    max_retries: int = 3,
) -> dict | None:
    """
    Send a patch image to Claude and return the parsed description dict.
    Returns None if the image file cannot be found or Claude returns invalid JSON.
    """
    pf = patch.get("patch_file", "")
    candidates = [Path(pf), base_dir / pf]
    image_path = next((c for c in candidates if c.exists()), None)

    if image_path is None:
        print(f"    SKIP: patch file not found: {pf}")
        return None

    # Build source context string for the prompt
    source_reg  = patch.get("source_reg") or patch.get("registration_number", "")
    source_img  = patch.get("source_image", "")
    area_pct    = patch.get("area_pct", "?")
    source_ctx  = f"reg. {source_reg}" if source_reg else source_img
    source_ctx += f", patch covering ~{area_pct}% of the image"

    img_b64, media_type = encode_image_b64(image_path)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model   = model,
                max_tokens = 512,
                system  = SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type":       "base64",
                                "media_type": media_type,
                                "data":       img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": USER_PROMPT_TEMPLATE.format(source_context=source_ctx),
                        },
                    ],
                }],
            )
            # Not content[0]: on models with thinking on by default the
            # first block is a thinking block, which has no .text.
            raw = text_of(response)

            # Parse JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Claude occasionally wraps in backticks — strip them
                clean = raw.strip("`").lstrip("json").strip()
                try:
                    return json.loads(clean)
                except json.JSONDecodeError:
                    print(f"    WARNING: could not parse JSON response:\n      {raw[:120]}…")
                    return {"raw_response": raw}

        except Exception as e:
            wait = 2 ** attempt
            print(f"    API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in {wait}s…")
                time.sleep(wait)

    return None


# ─── Batch processing ─────────────────────────────────────────────────────────

def process_patches(
    patches: list[dict],
    client: "anthropic.Anthropic",
    base_dir: Path,
    model: str,
    resume: bool,
    dry_run: bool,
    delay: float,
) -> list[dict]:
    results = [dict(p) for p in patches]

    for idx, patch in enumerate(results):
        pf   = Path(patch.get("patch_file", f"patch_{idx}")).name
        print(f"\n  [{idx+1}/{len(results)}] {pf}")

        if resume and patch.get("description"):
            print("    Already described, skipping (--resume)")
            continue

        if dry_run:
            if idx == 0:
                print(f"    DRY RUN — prompt preview:")
                print(f"    System: {SYSTEM_PROMPT[:100]}…")
                src_ctx = patch.get("source_image", "?")
                print(f"    User:   {USER_PROMPT_TEMPLATE.format(source_context=src_ctx)[:200]}…")
                # Call Claude once to show example output
                desc = describe_patch(client, patch, base_dir, model)
                print(f"    Response:\n{json.dumps(desc, indent=6)}")
            else:
                print("    DRY RUN — skipping")
            continue

        desc = describe_patch(client, patch, base_dir, model)
        if desc:
            patch["description"] = desc
            # Flatten keywords for easy text search later
            patch["description_keywords"] = desc.get("keywords", [])
            subject = desc.get("subject", "")[:80]
            print(f"    → {subject}")
        else:
            print("    → description failed")

        # Polite rate-limiting (Haiku tier is generous but avoid bursting)
        if delay > 0 and idx < len(results) - 1:
            time.sleep(delay)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Claude claude-haiku-4-5 motif descriptions for extracted patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("manifests", nargs="+", metavar="PATCHES_JSON",
                   help="*_patches.json or *_embeddings.json file(s)")
    p.add_argument("--out-dir",  metavar="OUT_DIR",
                   help="Output directory (default: same as input)")
    p.add_argument("--model",    default="claude-haiku-4-5-20251001",
                   help="Claude model to use")
    p.add_argument("--resume",   action="store_true",
                   help="Skip patches that already have a description field")
    p.add_argument("--dry-run",  action="store_true",
                   help="Process only the first patch and print output, don't write")
    p.add_argument("--delay",    type=float, default=0.3,
                   help="Seconds to wait between API calls (rate-limit buffer)")
    p.add_argument("--max-dim",  type=int,   default=1024,
                   help="Max image dimension before resizing for API upload")
    return p


def main() -> None:
    if not ANTHROPIC_AVAILABLE:
        print("ERROR: anthropic not installed.  pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    parser = build_parser()
    args   = parser.parse_args()

    client = anthropic.Anthropic(api_key=api_key)

    for manifest_str in args.manifests:
        manifest_path = Path(manifest_str)
        if not manifest_path.exists():
            print(f"WARNING: {manifest_path} not found, skipping")
            continue

        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        patches = data if isinstance(data, list) else data.get("patches", [])
        print(f"\n{'═'*60}")
        print(f"Describing {len(patches)} patches from {manifest_path.name}")
        print(f"Model: {args.model}")

        results = process_patches(
            patches,
            client      = client,
            base_dir    = manifest_path.parent,
            model       = args.model,
            resume      = args.resume,
            dry_run     = args.dry_run,
            delay       = args.delay,
        )

        if args.dry_run:
            print("\nDry run complete — no output written")
            continue

        out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem     = manifest_path.stem.replace("_patches", "").replace("_embeddings", "")
        out_path = out_dir / f"{stem}_described.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        described = sum(1 for r in results if r.get("description"))
        print(f"\n{'─'*60}")
        print(f"Described {described}/{len(results)} patches → {out_path}")


if __name__ == "__main__":
    main()
