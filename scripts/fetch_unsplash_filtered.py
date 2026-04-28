"""Filter Unsplash Lite metadata by AI tags + min resolution, download
~600 images per category to data/raw/unsplash_{landscape,cityscape,animals}/.

Reads photos.tsv000 + keywords.tsv000 from the Unsplash Lite dataset (already
on disk under ~/datasets/unsplash_lite/). Selects photos with min(width,
height) >= 2048 whose AI keyword tags match one of three category sets.
Excludes photos with high-confidence person/human tags.

Each download goes through Unsplash's CDN with ?w=2048&q=92 to deliver a
clean 2K JPEG. Idempotent: skips files already on disk.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

LANDSCAPE_KW = {
    "nature",
    "landscape",
    "mountain",
    "mountain range",
    "forest",
    "tree",
    "sea",
    "ocean",
    "water",
    "sunset",
    "sunrise",
    "sky",
    "cloud",
    "beach",
    "lake",
    "river",
    "valley",
    "desert",
    "meadow",
    "field",
    "snow",
    "plant",
    "flower",
    "grass",
    "scenery",
    "coast",
    "ice",
    "fir",
    "conifer",
    "blossom",
    "vegetation",
    "flora",
    "land",
    "outdoors",
}
CITYSCAPE_KW = {
    "city",
    "cityscape",
    "urban",
    "building",
    "architecture",
    "skyscraper",
    "street",
    "downtown",
    "bridge",
    "road",
    "town",
    "tower",
    "monument",
    "metropolis",
    "office building",
}
ANIMAL_KW = {
    "animal",
    "wildlife",
    "mammal",
    "bird",
    "dog",
    "cat",
    "horse",
    "deer",
    "wolf",
    "fox",
    "butterfly",
    "squirrel",
    "rabbit",
    "tiger",
    "lion",
    "bear",
    "elephant",
    "zebra",
    "giraffe",
    "owl",
    "eagle",
    "fish",
    "reptile",
}
EXCLUDE_KW = {"person", "human", "people", "man", "woman", "girl", "boy", "child", "face"}

CATEGORIES = {
    "landscape": LANDSCAPE_KW,
    "cityscape": CITYSCAPE_KW,
    "animals": ANIMAL_KW,
}


def select_photos(
    unsplash_root: Path,
    per_category: int,
    min_side: int,
    confidence: float,
    seed: int,
) -> dict[str, list[str]]:
    """Return {category: [photo_id, ...]} after filtering and sampling."""
    photos = pd.read_csv(
        unsplash_root / "photos.tsv000",
        sep="\t",
        usecols=["photo_id", "photo_image_url", "photo_width", "photo_height"],
    )
    kw = pd.read_csv(
        unsplash_root / "keywords.tsv000",
        sep="\t",
        usecols=["photo_id", "keyword", "ai_service_1_confidence"],
    )

    ok = photos[(photos.photo_width >= min_side) & (photos.photo_height >= min_side)]
    print(f"Photos with min(w,h) >= {min_side}: {len(ok)} / {len(photos)}", file=sys.stderr)

    confident = kw[(kw.photo_id.isin(ok.photo_id)) & (kw.ai_service_1_confidence >= confidence)]
    excluded = set(confident[confident.keyword.isin(EXCLUDE_KW)].photo_id.unique())
    print(f"Excluding {len(excluded)} photos with people/face tags", file=sys.stderr)

    rng = np.random.default_rng(seed)
    used: set[str] = set()
    selected: dict[str, list[str]] = {}
    for name, kws in CATEGORIES.items():
        cands = confident[confident.keyword.isin(kws)].photo_id.unique()
        pool = [p for p in cands if p not in excluded and p not in used]
        if len(pool) < per_category:
            print(
                f"WARN: {name} has only {len(pool)} candidates (wanted {per_category})",
                file=sys.stderr,
            )
        chosen = rng.choice(pool, size=min(per_category, len(pool)), replace=False).tolist()
        selected[name] = chosen
        used.update(chosen)
        print(f"  {name}: {len(chosen)} selected (pool {len(pool)})", file=sys.stderr)
    return selected


def download_one(
    photo_id: str,
    image_url: str,
    out_dir: Path,
    width: int,
    quality: int,
    timeout: int = 30,
) -> tuple[str, str]:
    out = out_dir / f"{photo_id}.jpg"
    if out.is_file() and out.stat().st_size > 0:
        return photo_id, "cached"
    sized_url = f"{image_url}?w={width}&q={quality}&fm=jpg"
    last_err = ""
    for _ in range(3):
        try:
            resp = requests.get(sized_url, timeout=timeout)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            return photo_id, "ok"
        except Exception as e:
            last_err = str(e)
    return photo_id, f"err: {last_err}"


def download_batch(
    photo_ids: list[str],
    url_map: dict[str, str],
    out_dir: Path,
    width: int,
    quality: int,
    workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fails: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(download_one, pid, url_map[pid], out_dir, width, quality) for pid in photo_ids
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=out_dir.name):
            pid, status = fut.result()
            if status not in ("ok", "cached"):
                fails.append((pid, status))
    if fails:
        print(f"  {out_dir.name}: {len(fails)} failures (first 3): {fails[:3]}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unsplash-root",
        type=Path,
        default=Path.home() / "datasets" / "unsplash_lite",
    )
    parser.add_argument("--out-root", type=Path, default=Path("data") / "raw")
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--min-side", type=int, default=2048)
    parser.add_argument(
        "--confidence", type=float, default=50.0, help="AI tag confidence threshold"
    )
    parser.add_argument("--width", type=int, default=2048, help="URL ?w= param")
    parser.add_argument("--quality", type=int, default=92, help="URL ?q= param")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection counts and exit without downloading.",
    )
    args = parser.parse_args()

    selected = select_photos(
        args.unsplash_root, args.per_category, args.min_side, args.confidence, args.seed
    )

    if args.dry_run:
        return 0

    photos = pd.read_csv(
        args.unsplash_root / "photos.tsv000",
        sep="\t",
        usecols=["photo_id", "photo_image_url"],
    )
    url_map = photos.set_index("photo_id").photo_image_url.to_dict()

    for name, pids in selected.items():
        download_batch(
            pids,
            url_map,
            args.out_root / f"unsplash_{name}",
            args.width,
            args.quality,
            args.workers,
        )

    print("\nDone. Files under:", args.out_root, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
