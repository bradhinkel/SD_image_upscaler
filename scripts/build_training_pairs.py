"""Build (LR_128, HR_512) JPEG training pairs from raw source images.

Walks every image under each --source-dirs root, takes 3 random 512x512
crops per image, generates a realistic LR_128 via the degradation pipeline,
saves both to data/pairs/ as {idx:06d}_(hr|lr).jpg.

Usage:
    python -m scripts.build_training_pairs
    python -m scripts.build_training_pairs --crops-per-image 4 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from upscaler.dataset import build_pairs, iter_source_images

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIRS = [
    Path.home() / "datasets" / "div2k",
    REPO_ROOT / "data" / "raw" / "unsplash_landscape",
    REPO_ROOT / "data" / "raw" / "unsplash_cityscape",
    REPO_ROOT / "data" / "raw" / "unsplash_animals",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dirs", nargs="+", type=Path, default=DEFAULT_SOURCE_DIRS)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "pairs")
    parser.add_argument("--hr-size", type=int, default=512)
    parser.add_argument("--lr-size", type=int, default=128)
    parser.add_argument("--crops-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("Source dirs:", file=sys.stderr)
    for d in args.source_dirs:
        n = sum(1 for _ in iter_source_images([d]))
        print(f"  {d}: {n} images", file=sys.stderr)

    sources = list(iter_source_images(args.source_dirs))
    print(f"\nTotal sources: {len(sources)}", file=sys.stderr)

    stats = build_pairs(
        sources,
        args.out,
        hr_size=args.hr_size,
        lr_size=args.lr_size,
        crops_per_image=args.crops_per_image,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"\nResult: {stats}", file=sys.stderr)
    print(f"Pairs saved to: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
