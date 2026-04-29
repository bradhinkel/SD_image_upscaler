"""SDXL gate quick eval for the Phase 4c SD 1.5 LoRA.

Loads the LoRA into the x4-upscaler pipeline, runs every test image at the
5x ratio (LR_200 -> 800 stage A, bicubic-fit to 1000 for parity with the
existing leaderboard rows), scores LPIPS vs HR, and computes the win rate
vs Real-ESRGAN at the same ratio.

Gate (CLAUDE.md Phase 4c):
  Proceed to 4d only if SD 1.5+LoRA beats Real-ESRGAN on LPIPS for >=50%
  of test images at 5x.

Usage:
    python -m scripts.eval_lora_sd15 \\
        --lora outputs/loras/sd15_main \\
        --out outputs/eval/lora_gate.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

from upscaler import eval_metrics, testset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora", type=Path, default=REPO_ROOT / "outputs" / "loras" / "sd15_main")
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=REPO_ROOT / "outputs" / "eval" / "leaderboard_phase3.csv",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "outputs" / "eval" / "lora_gate.csv"
    )
    parser.add_argument("--lr-size", type=int, default=200, help="5x ratio LR")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prompt", default="a high-resolution photograph, sharp detail")
    args = parser.parse_args()

    if not args.lora.is_dir():
        print(f"LoRA dir not found: {args.lora}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading x4-upscaler + LoRA on {device} ...", file=sys.stderr)
    from diffusers import StableDiffusionUpscalePipeline
    from peft import PeftModel

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=dtype
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.unet = PeftModel.from_pretrained(pipe.unet, str(args.lora))
    pipe.unet.eval()
    pipe.vae.to(torch.float32)  # avoid fp16 NaN in VAE (same as training)

    ts = testset.load(REPO_ROOT / "data" / "test_images")
    print(f"Test set: {len(ts)} images", file=sys.stderr)

    # Run inference + LPIPS for each image.
    rows: list[dict] = []
    for img in tqdm(ts.images, desc="LoRA eval"):
        lr_pil = Image.open(img.lr_path(args.lr_size)).convert("RGB")
        hr_pil = Image.open(img.hr_path).convert("RGB")
        with torch.no_grad():
            up = pipe(
                prompt=args.prompt,
                image=lr_pil,
                num_inference_steps=args.steps,
            ).images[0]
        # Stage A produces 4x = 800 from 200; bicubic-fit to 1000 for parity
        # with leaderboard_phase3 rows.
        if up.size != (args.target, args.target):
            up = up.resize((args.target, args.target), Image.Resampling.BICUBIC)
        score = eval_metrics.lpips(up, hr_pil)
        rows.append(
            {
                "image": img.name,
                "category": img.category,
                "subcategory": img.subcategory or "",
                "challenges": "|".join(img.challenges),
                "method": "sd15_lora",
                "lr_size": args.lr_size,
                "lpips_lora": round(score, 6),
            }
        )
    df = pd.DataFrame(rows)

    # Pull Real-ESRGAN baseline at the same lr_size for the win-rate check.
    leaderboard = pd.read_csv(args.leaderboard)
    realesrgan = leaderboard[
        (leaderboard.method == "realesrgan") & (leaderboard.lr_size == args.lr_size)
    ][["image", "lpips"]].rename(columns={"lpips": "lpips_realesrgan"})
    df = df.merge(realesrgan, on="image", how="left")
    df["lora_wins"] = df.lpips_lora < df.lpips_realesrgan

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    win_rate = df.lora_wins.mean()
    print(f"\nWrote {len(df)} rows to {args.out}", file=sys.stderr)
    print(f"\nMean LPIPS at {1000 // args.lr_size}x:", file=sys.stderr)
    print(f"  SD 1.5 + LoRA:  {df.lpips_lora.mean():.4f}", file=sys.stderr)
    print(f"  Real-ESRGAN:    {df.lpips_realesrgan.mean():.4f}", file=sys.stderr)
    print(
        f"\nWin rate vs Real-ESRGAN: {win_rate:.1%}  ({int(df.lora_wins.sum())}/{len(df)})",
        file=sys.stderr,
    )
    print(
        f"Gate: {'PASS' if win_rate >= 0.5 else 'FAIL'}  (criterion: >=50%)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
