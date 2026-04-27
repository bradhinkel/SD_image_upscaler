"""Phase 3 benchmark driver.

Walks the cached upscales produced by Phase 1 (baselines + x4-upscaler) and
Phase 2 (two-stage at denoise=0.30 / steps=20), scores each output against the
HR reference with LPIPS / DISTS / PSNR / SSIM, and writes the leaderboard CSV.

Idempotent and CPU-aware. Uses GPU for LPIPS/DISTS/SSIM when available.

Usage:
    python -m scripts.benchmark_pipeline --methods all --out outputs/eval/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from upscaler import eval_metrics, testset
from upscaler.testset import TestImage

REPO_ROOT = Path(__file__).resolve().parent.parent

PHASE1_CACHE = REPO_ROOT / "outputs" / "phase1" / "upscales"
PHASE2_CACHE = REPO_ROOT / "outputs" / "phase2" / "upscales"


def _phase1_path(method: str, img: TestImage, lr_size: int) -> Path:
    stem = Path(img.name).stem
    return PHASE1_CACHE / f"{stem}_{lr_size}_{method}.jpg"


def _phase2_path(img: TestImage, lr_size: int) -> Path:
    """Phase 2 uses denoise=0.30 / steps=20 in its main run."""
    stem = Path(img.name).stem
    return PHASE2_CACHE / f"{stem}_{lr_size}_d030_s20.jpg"


def _make_loader(path_fn):
    def loader(img: TestImage, lr_size: int) -> Image.Image:
        path = path_fn(img, lr_size)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached upscale: {path}")
        return Image.open(path)

    return loader


METHODS = {
    "bicubic": _make_loader(lambda img, lr: _phase1_path("bicubic", img, lr)),
    "lanczos": _make_loader(lambda img, lr: _phase1_path("lanczos", img, lr)),
    "realesrgan": _make_loader(lambda img, lr: _phase1_path("realesrgan", img, lr)),
    "x4_upscaler": _make_loader(lambda img, lr: _phase1_path("x4_upscaler", img, lr)),
    "two_stage": _make_loader(_phase2_path),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval",
        help="Output directory for the leaderboard CSV",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        default=REPO_ROOT / "data" / "test_images",
        help="Path to data/test_images/",
    )
    parser.add_argument(
        "--name",
        default="leaderboard_phase3",
        help="CSV filename stem (default: leaderboard_phase3)",
    )
    args = parser.parse_args()

    if args.methods == "all":
        chosen = list(METHODS)
    else:
        chosen = [m.strip() for m in args.methods.split(",")]
        unknown = [m for m in chosen if m not in METHODS]
        if unknown:
            parser.error(f"unknown methods: {unknown}; available: {list(METHODS)}")

    args.out.mkdir(parents=True, exist_ok=True)
    ts = testset.load(args.testset)
    print(f"loaded {len(ts)} test images", file=sys.stderr)

    frames: list[pd.DataFrame] = []
    for method in chosen:
        print(f"  scoring {method}...", file=sys.stderr)
        df = eval_metrics.evaluate_method(method, METHODS[method], ts)
        frames.append(df)

    leaderboard = pd.concat(frames, ignore_index=True)
    csv_path = args.out / f"{args.name}.csv"
    leaderboard.to_csv(csv_path, index=False)
    print(f"wrote {len(leaderboard)} rows to {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
