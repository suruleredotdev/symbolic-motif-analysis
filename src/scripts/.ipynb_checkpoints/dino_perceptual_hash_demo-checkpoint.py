import argparse
from PIL import Image
import torch
from dinohash import DINOHash
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute DINO perceptual hashes for one or more images.")
    parser.add_argument(
        "image_paths",
        nargs="*",
        help="Path(s) to image file(s) to hash."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    image_paths = args.image_paths
    if not image_paths:
        env_path = os.environ.get("DINOHASH_IMAGE_PATH")
        if env_path:
            image_paths = [env_path]
        else:
            raise ValueError(
                "Please provide at least one image path as an argument or set the DINOHASH_IMAGE_PATH environment variable."
            )

    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path))
        except Exception as e:
            print(f"Error opening {path}: {e}")

    if not images:
        raise RuntimeError("No valid images to process.")

    dinohash = DINOHash()
    hashes = dinohash.hash(images)

    for i, h in enumerate(hashes):
        print(f"Perceptual hash for '{image_paths[i]}': {h.hex}")

if __name__ == "__main__":
    main()