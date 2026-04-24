"""Migrate the source test set from Images/Test_Images/ (1024x1024 JPEGs)
to data/test_images/ (1000x1000 HR + _100/_200/_250 LR variants via bicubic).

Idempotent: running twice produces the same output. Overwrites existing files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

HR_SIZE = 1000
LR_SIZES = (100, 200, 250)
JPEG_QUALITY = 95


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        raise ValueError(f"Image {w}x{h} smaller than crop target {size}x{size}")
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def bicubic_downsample(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.Resampling.BICUBIC)


def migrate(src_dir: Path, dst_dir: Path, metadata_name: str = "metadata.json") -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_metadata = src_dir / metadata_name
    if not src_metadata.is_file():
        raise FileNotFoundError(f"Expected metadata at {src_metadata}")
    metadata = json.loads(src_metadata.read_text())

    stats = {"hr_written": 0, "lr_written": 0, "skipped_missing": [], "size_warnings": []}

    for filename in sorted(metadata):
        src_path = src_dir / filename
        if not src_path.is_file():
            stats["skipped_missing"].append(filename)
            print(f"WARN: missing source {src_path}", file=sys.stderr)
            continue

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if img.size != (1024, 1024):
                stats["size_warnings"].append((filename, img.size))
                print(f"WARN: {filename} is {img.size}, expected (1024, 1024)", file=sys.stderr)

            hr = center_crop(img, HR_SIZE)
            hr.save(dst_dir / filename, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
            stats["hr_written"] += 1

            stem = Path(filename).stem
            suffix = Path(filename).suffix
            for lr_size in LR_SIZES:
                lr = bicubic_downsample(hr, lr_size)
                lr_name = f"{stem}_{lr_size}{suffix}"
                lr.save(dst_dir / lr_name, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
                stats["lr_written"] += 1

    shutil.copy(src_metadata, dst_dir / metadata_name)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("Images/Test_Images"),
        help="Source directory with 1024x1024 JPEGs and metadata.json",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/test_images"),
        help="Output directory for 1000x1000 HR + LR variants",
    )
    args = parser.parse_args()

    stats = migrate(args.src, args.dst)

    print(f"HR written: {stats['hr_written']}")
    print(f"LR written: {stats['lr_written']}")
    if stats["skipped_missing"]:
        print(f"Missing sources ({len(stats['skipped_missing'])}): {stats['skipped_missing']}")
    if stats["size_warnings"]:
        print(f"Size warnings ({len(stats['size_warnings'])}): {stats['size_warnings']}")
    return 0 if not stats["skipped_missing"] else 1


if __name__ == "__main__":
    sys.exit(main())
