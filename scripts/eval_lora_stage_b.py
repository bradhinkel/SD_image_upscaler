"""Stage-B LoRA evaluator.

Runs the full Phase 2 two-stage pipeline (x4-upscaler -> SD 1.5 + ControlNet
Tile img2img, tiled) with the trained LoRA injected into stage B's UNet.
Scores LPIPS at 5x against the 60-image test set, then reports both:

  (a) win-rate vs Phase 3's BASELINE two_stage at the same ratio
      (i.e., did the LoRA improve our own pipeline?)
  (b) win-rate vs Real-ESRGAN at the same ratio
      (the SDXL gate criterion as originally written)

Usage:
    python -m scripts.eval_lora_stage_b \\
        --lora outputs/loras/sd15_stage_b \\
        --out outputs/eval/lora_stage_b_gate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from upscaler import eval_metrics, pipeline, testset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora", type=Path, default=REPO_ROOT / "outputs" / "loras" / "sd15_stage_b"
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "leaderboard_phase3.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "eval" / "lora_stage_b_gate.csv"
    )
    parser.add_argument("--lr-size", type=int, default=200, help="5x ratio LR")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--denoise", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cn-weight", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default="a high-resolution photograph, sharp detail, professional quality",
    )
    args = parser.parse_args()

    if not args.lora.is_dir():
        print(f"LoRA dir not found: {args.lora}", file=sys.stderr)
        return 1

    print("Loading UpscalerPipeline (stage A + stage B) ...", file=sys.stderr)
    up = pipeline.UpscalerPipeline()
    up.load()
    up.load_stage_b()

    print(f"Injecting LoRA into stage B's UNet from {args.lora} ...", file=sys.stderr)
    from peft import PeftModel

    up._stage_b.unet = PeftModel.from_pretrained(up._stage_b.unet, str(args.lora))
    up._stage_b.unet.eval()
    # Stage B's VAE in fp32 for safety (matches training fix).
    up._stage_b.vae.to(torch.float32)

    ts = testset.load(REPO_ROOT / "data" / "test_images")
    print(f"Test set: {len(ts)} images at {args.target // args.lr_size}x", file=sys.stderr)

    rows: list[dict] = []
    for img in tqdm(ts.images, desc="LoRA stage-B eval"):
        lr_pil = Image.open(img.lr_path(args.lr_size)).convert("RGB")
        hr_pil = Image.open(img.hr_path).convert("RGB")
        with torch.no_grad():
            up_img = up.upscale_two_stage(
                lr_pil,
                target_size=args.target,
                denoise=args.denoise,
                steps=args.steps,
                cn_weight=args.cn_weight,
                prompt=args.prompt,
            )
        score = eval_metrics.lpips(up_img, hr_pil)
        rows.append(
            {
                "image": img.name,
                "category": img.category,
                "subcategory": img.subcategory or "",
                "challenges": "|".join(img.challenges),
                "method": "sd15_stage_b_lora",
                "lr_size": args.lr_size,
                "lpips_lora": round(score, 6),
            }
        )
    df = pd.DataFrame(rows)

    leaderboard = pd.read_csv(args.leaderboard)
    baseline_two_stage = leaderboard[
        (leaderboard.method == "two_stage") & (leaderboard.lr_size == args.lr_size)
    ][["image", "lpips"]].rename(columns={"lpips": "lpips_baseline_twostage"})
    realesrgan = leaderboard[
        (leaderboard.method == "realesrgan") & (leaderboard.lr_size == args.lr_size)
    ][["image", "lpips"]].rename(columns={"lpips": "lpips_realesrgan"})
    df = df.merge(baseline_two_stage, on="image", how="left").merge(
        realesrgan, on="image", how="left"
    )
    df["beats_baseline_twostage"] = df.lpips_lora < df.lpips_baseline_twostage
    df["beats_realesrgan"] = df.lpips_lora < df.lpips_realesrgan

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} rows to {args.out}\n", file=sys.stderr)
    print(f"Mean LPIPS at {args.target // args.lr_size}x:", file=sys.stderr)
    print(f"  stage-B + LoRA:      {df.lpips_lora.mean():.4f}", file=sys.stderr)
    print(f"  baseline two_stage:  {df.lpips_baseline_twostage.mean():.4f}", file=sys.stderr)
    print(f"  Real-ESRGAN:         {df.lpips_realesrgan.mean():.4f}", file=sys.stderr)
    print()
    base_win = df.beats_baseline_twostage.mean()
    re_win = df.beats_realesrgan.mean()
    print(
        f"vs baseline two_stage: {base_win:.1%}  "
        f"({int(df.beats_baseline_twostage.sum())}/{len(df)})",
        file=sys.stderr,
    )
    print(
        f"vs Real-ESRGAN:        {re_win:.1%}  "
        f"({int(df.beats_realesrgan.sum())}/{len(df)})  <- gate criterion (>=50%)",
        file=sys.stderr,
    )
    print(
        f"\nSDXL gate: {'PASS' if re_win >= 0.5 else 'FAIL'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
