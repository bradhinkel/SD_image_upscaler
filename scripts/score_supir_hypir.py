"""Score Phase 4.6 SUPIR + HYPIR outputs against HR.

Reads outputs from outputs/supir/ and outputs/hypir/, computes LPIPS + DISTS
vs the matching HR test image, and merges with the existing 4x leaderboard
rows from outputs/eval/leaderboard_phase3.csv (lr_size=250). The result is
outputs/eval/leaderboard_phase4_6.csv with one row per (image, method) at 4x
for the 12-image subset.

Methods scored:
  - supir          : generic-prompt SUPIR Ultra mode (12 images)
  - hypir          : generic-prompt HYPIR (12 images)
  - hypir_caption  : HYPIR with per-image BLIP captions (12 images)

The hypir_caption rows enable a direct caption-quality A/B since HYPIR was
the only engine Brad ran both modes on (SUPIR credits preserved).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from upscaler import eval_metrics, testset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supir-dir", type=Path, default=REPO_ROOT / "outputs" / "supir")
    parser.add_argument("--hypir-dir", type=Path, default=REPO_ROOT / "outputs" / "hypir")
    parser.add_argument(
        "--leaderboard-in",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "leaderboard_phase3.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "leaderboard_phase4_6.csv",
    )
    parser.add_argument(
        "--subset", type=Path, default=REPO_ROOT / "outputs" / "supir" / "subset.json"
    )
    args = parser.parse_args()

    subset = json.loads(args.subset.read_text())
    subset_images = [s["image"] for s in subset["subset"]]
    print(f"Scoring {len(subset_images)} subset images at 4x", file=sys.stderr)

    ts = testset.load(REPO_ROOT / "data" / "test_images")
    by_name = {img.name: img for img in ts.images}

    # Reuse Phase 3 4x rows for these 12 images so the final leaderboard has
    # bicubic / lanczos / realesrgan / x4_upscaler / two_stage already scored.
    lb = pd.read_csv(args.leaderboard_in)
    base = lb[(lb.lr_size == 250) & (lb.image.isin(subset_images))].copy()
    base["lr_size"] = 250  # 4x ratio
    base["dists"] = base.get("dists", float("nan"))
    print(f"Merged {len(base)} existing rows from {args.leaderboard_in.name}", file=sys.stderr)

    new_rows: list[dict] = []
    for name in tqdm(subset_images, desc="score"):
        img = by_name[name]
        hr_pil = Image.open(img.hr_path).convert("RGB")
        stem = Path(name).stem
        for method, dirpath, suffix in (
            ("supir", args.supir_dir, "4x"),
            ("hypir", args.hypir_dir, "4x"),
            ("hypir_caption", args.hypir_dir, "4x_caption"),
        ):
            pred_path = dirpath / f"{stem}_{suffix}.jpg"
            if not pred_path.is_file():
                pred_path = dirpath / f"{stem}_{suffix}.png"
            if not pred_path.is_file():
                if method == "hypir_caption":
                    continue  # caption variant is optional
                print(f"WARN: missing {pred_path}", file=sys.stderr)
                continue
            pred = Image.open(pred_path).convert("RGB")
            if pred.size != hr_pil.size:
                pred = pred.resize(hr_pil.size, Image.Resampling.BICUBIC)
            new_rows.append(
                {
                    "method": method,
                    "image": name,
                    "category": img.category,
                    "subcategory": img.subcategory or "",
                    "challenges": "|".join(img.challenges),
                    "lr_size": 250,
                    "ratio": 4,
                    "lpips": round(eval_metrics.lpips(pred, hr_pil), 6),
                    "dists": round(eval_metrics.dists(pred, hr_pil), 6),
                }
            )

    new_df = pd.DataFrame(new_rows)
    print(f"Scored {len(new_df)} new rows", file=sys.stderr)

    # Concatenate, with consistent column set
    all_df = pd.concat([base, new_df], ignore_index=True, sort=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(args.out, index=False)
    print(f"Wrote {len(all_df)} rows to {args.out}", file=sys.stderr)

    # Quick summary
    print("\n=== Mean LPIPS at 4x by method (12-image subset) ===", file=sys.stderr)
    print(
        all_df.groupby("method")
        .lpips.agg(["mean", "count"])
        .sort_values("mean")
        .round(4)
        .to_string(),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
