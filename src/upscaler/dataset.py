"""Training-data utilities: source image iteration, random cropping, pair
building, and a torch Dataset for the LoRA training loop.

Phase 4a writes (LR, HR) JPEG pairs to disk via `build_pairs`. Phase 4b's
training loop consumes those pairs (plus BLIP-2 captions) via
`PairCaptionDataset`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from upscaler.degradations import DegradationConfig, degrade

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def iter_source_images(
    roots: list[Path], extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
) -> Iterator[Path]:
    """Yield every image-file path under each root (recursively)."""
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for ext in extensions:
            for p in sorted(root.rglob(f"*{ext}")):
                if p not in seen:
                    seen.add(p)
                    yield p


def random_crops(
    img: Image.Image,
    crop_size: int,
    n: int,
    rng: np.random.Generator,
) -> list[Image.Image]:
    """n random crop_size x crop_size patches; empty list if img < crop_size."""
    if img.width < crop_size or img.height < crop_size:
        return []
    out: list[Image.Image] = []
    for _ in range(n):
        x = int(rng.integers(0, img.width - crop_size + 1))
        y = int(rng.integers(0, img.height - crop_size + 1))
        out.append(img.crop((x, y, x + crop_size, y + crop_size)))
    return out


def build_pairs(
    source_paths: list[Path],
    out_dir: Path,
    hr_size: int = 512,
    lr_size: int = 128,
    crops_per_image: int = 3,
    seed: int = 42,
    overwrite: bool = False,
    quality: int = 95,
) -> dict[str, Any]:
    """Build (LR, HR) JPEG pairs and write to out_dir.

    Output filenames are {idx:06d}_hr.jpg and {idx:06d}_lr.jpg, where idx
    is a sequential index across all sources x crops. Idempotent: skips
    pairs whose files already exist on disk unless overwrite=True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if hr_size % lr_size != 0:
        raise ValueError(f"hr_size {hr_size} must be a multiple of lr_size {lr_size}")
    scale = hr_size // lr_size
    cfg = DegradationConfig(scale=scale)
    rng = np.random.default_rng(seed)

    written = 0
    skipped_small = 0
    skipped_existing = 0
    failed_open = 0
    idx = 0

    for src in tqdm(source_paths, desc="pairs"):
        try:
            with Image.open(src) as im:
                im_rgb = im.convert("RGB").copy()
        except Exception as e:
            failed_open += 1
            print(f"WARN: failed to open {src}: {e}")
            continue

        crops = random_crops(im_rgb, hr_size, crops_per_image, rng)
        if not crops:
            skipped_small += 1
            continue

        for hr in crops:
            hr_path = out_dir / f"{idx:06d}_hr.jpg"
            lr_path = out_dir / f"{idx:06d}_lr.jpg"
            if not overwrite and hr_path.is_file() and lr_path.is_file():
                idx += 1
                skipped_existing += 1
                continue
            # Per-pair seed so re-runs reproduce the same LR for a given idx.
            pair_rng = np.random.default_rng(seed + idx)
            lr = degrade(hr, config=cfg, rng=pair_rng)
            hr.save(hr_path, format="JPEG", quality=quality, subsampling=0)
            lr.save(lr_path, format="JPEG", quality=quality, subsampling=0)
            written += 1
            idx += 1

    return {
        "written": written,
        "skipped_small": skipped_small,
        "skipped_existing": skipped_existing,
        "failed_open": failed_open,
        "total_idx": idx,
    }


class PairCaptionDataset(Dataset):
    """LoRA training dataset: (LR_tensor, HR_tensor, caption_str) per index.

    Reads pairs from pairs_dir and captions from a JSONL where each row is
    `{"idx": int, "caption": str}`. Tensors are (3, H, W) float32 in [-1, 1].
    """

    def __init__(self, pairs_dir: Path, captions_path: Path | None = None):
        self.pairs_dir = pairs_dir
        self.indices: list[int] = []
        for p in sorted(pairs_dir.glob("*_hr.jpg")):
            idx = int(p.stem.split("_")[0])
            if (pairs_dir / f"{idx:06d}_lr.jpg").is_file():
                self.indices.append(idx)

        self.captions: dict[int, str] = {}
        if captions_path is not None and captions_path.is_file():
            with captions_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.captions[int(rec["idx"])] = rec["caption"]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        idx = self.indices[i]
        hr = Image.open(self.pairs_dir / f"{idx:06d}_hr.jpg").convert("RGB")
        lr = Image.open(self.pairs_dir / f"{idx:06d}_lr.jpg").convert("RGB")
        hr_t = torch.from_numpy(np.asarray(hr, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        lr_t = torch.from_numpy(np.asarray(lr, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        return lr_t, hr_t, self.captions.get(idx, "")


__all__ = [
    "DEFAULT_EXTENSIONS",
    "PairCaptionDataset",
    "build_pairs",
    "iter_source_images",
    "random_crops",
]
