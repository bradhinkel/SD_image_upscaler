"""Push the working stage-B SD 1.5 LoRA to HF Hub at
bradhinkel/sd-image-upscaler-sd15-lora — overwriting the broken Phase 4c v1
weights that were pushed earlier.

Model card is intentionally honest: this LoRA is a research artifact from
the SD_image_upscaler 50-hour capability study. It does NOT improve the
two-stage pipeline on average; it produces a small night-scene preference
shift but mildly hurts daylight categories. Anyone using it should read
the project's Phase 4c writeup first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "bradhinkel/sd-image-upscaler-sd15-lora"

MODEL_CARD = """\
---
license: openrail
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags:
  - lora
  - super-resolution
  - upscaler
  - stable-diffusion
  - research-artifact
library_name: peft
---

# SD Image Upscaler — Stage-B SD 1.5 LoRA (research artifact)

A LoRA from the **SD Image Upscaler** capability study at
[github.com/bradhinkel/SD_image_upscaler](https://github.com/bradhinkel/SD_image_upscaler).
Phase 4c. **Read this card before using.**

## What this LoRA actually does (and doesn't)

This is a **research artifact**, not a production-grade weight.

Tested through the project's full Phase 2 two-stage upscaling pipeline
(`stable-diffusion-x4-upscaler` -> `stable-diffusion-v1-5` + ControlNet Tile,
img2img, tiled) on a 60-image frozen test set at 5x:

| Pipeline | Mean LPIPS at 5x (lower = better) |
|---|---|
| Real-ESRGAN baseline | 0.299 |
| Two-stage (no LoRA) | 0.433 |
| **Two-stage + this LoRA** | **0.443** |

**The LoRA does not improve the pipeline on average.** It does shift the
texture/color prior in interesting per-category ways:

| Test slice | LoRA win rate vs no-LoRA two-stage | Δ mean LPIPS |
|---|---|---|
| traditional / landscape | 30% | +0.0145 (worse) |
| traditional / cityscape | 20% | +0.0156 (worse) |
| traditional / animals | 30% | +0.0040 |
| hard / fine_architecture | 25% | +0.0150 (worse) |
| hard / hf_texture | 50% | +0.0093 (worse) |
| **hard / night** | **62.5%** | **-0.0056 (better)** |
| hard / reflection | 60% | +0.0019 |
| hard / noise | 50% | +0.0109 (worse) |
| hard / text | 46% | +0.0092 (worse) |

The LoRA learned a slight darken-and-saturate prior that genuinely helps
**night scenes** but mildly hurts daylight categories — even the very
domains it was trained on (landscape / cityscape / animals).

## Why publish a non-improving LoRA?

This is the **third** training attempt in the study. The first two targeted
LoRA on `stable-diffusion-x4-upscaler`'s cross-attention modules and produced
catastrophically destructive deltas (output LPIPS 0.78-0.92, vs base 0.33)
regardless of recipe. Detailed failure analysis is in the project's Phase 4c
writeup; the short version is that x4-upscaler's denoising trajectory is
unusually fragile to U-Net perturbations and the SUPIR paper's
architectural choices (zero-init additive adapters on intermediate
ResBlocks, NOT LoRA on attention) are validated by our negative result.

This stage-B LoRA is **technically functional** (does not destabilise the
pipeline), and that's noteworthy on its own. The honest finding is that
small-scale cross-attention LoRA on SD 1.5 isn't sufficient to close the
gap to dedicated SR architectures (Real-ESRGAN, SUPIR) when LR is clean
bicubic-downsampled HR.

## Architecture

- Base model: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Adapter: PEFT LoRA, **rank 16, alpha 8** (effective scale 0.5)
- Targets: `to_q / to_k / to_v / to_out.0` in the UNet cross-attention
- Trainable params: ~3.2M (~0.7% of base)

## Training

- 7786 (LR_128, HR_512) pairs from DIV2K + Unsplash Lite
  (`bradhinkel/sd-image-upscaler-pairs`, private dataset)
- BLIP-large captions (one per HR tile)
- 8000 steps, batch 4, lr 1e-4, fp16 UNet + fp32 VAE
- Loss flat at ~0.15 across all 8000 steps (base SD 1.5 already achieves
  this on photographs at step 0; LoRA's contribution is the small prior
  shift, not loss reduction)
- RunPod RTX 5090 32 GB, ~63 min, ~$0.71

## Usage

```python
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from peft import PeftModel
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")
pipe.unet = PeftModel.from_pretrained(pipe.unet, "bradhinkel/sd-image-upscaler-sd15-lora")

# Use as the stage-B refinement step in a two-stage upscale pipeline.
# See github.com/bradhinkel/SD_image_upscaler for the full inference path.
```

## Recommended usage

If you want to use this artifact at all, **night scenes only**, at LoRA
scale 0.5-1.0. For daylight imagery, prefer the same pipeline without
the LoRA — it's slightly better.

## Source attributions

- Training dataset: DIV2K (research-only license) + Unsplash Lite
  (ML training permitted under the Lite Dataset terms)
- Captions: `Salesforce/blip-image-captioning-large`

## License

OpenRAIL inherited from SD 1.5 base.
"""


def load_hf_token() -> str:
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("HF_TOKEN=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lora", type=Path, default=REPO_ROOT / "outputs" / "loras" / "sd15_stage_b"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--execute", action="store_true", help="Actually push (default is dry-run)."
    )
    args = parser.parse_args()

    if not args.lora.is_dir():
        print(f"LoRA dir not found: {args.lora}", file=sys.stderr)
        return 1
    token = load_hf_token()
    if not token:
        print("HF_TOKEN missing.", file=sys.stderr)
        return 1

    print(f"Repo:    {args.repo_id}", file=sys.stderr)
    print(f"LoRA:    {args.lora}", file=sys.stderr)
    print(f"Files:   {sorted(p.name for p in args.lora.iterdir())}", file=sys.stderr)
    if not args.execute:
        print("\nDRY RUN — pass --execute to push.", file=sys.stderr)
        return 0

    (args.lora / "README.md").write_text(MODEL_CARD)

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type="model", private=False, token=token, exist_ok=True)
    api.upload_folder(
        folder_path=str(args.lora),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Phase 4c closure: stage-B LoRA + honest model card",
    )
    print(f"\nDone. https://huggingface.co/{args.repo_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
