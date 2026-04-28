"""LoRA training loop for `stabilityai/stable-diffusion-x4-upscaler`.

The x4-upscaler UNet has 7 input channels (4 latent + 3 LR pixels) and uses
`class_labels` for noise-level conditioning. Both quirks are handled here so
the same script works for the Phase 4b rehearsal and the Phase 4c/4d cloud
runs — only the YAML config changes.

Usage:
    python -m upscaler.lora_train --config configs/rehearsal.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from upscaler.dataset import PairCaptionDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_pipeline(model_id: str, device: str, dtype: torch.dtype) -> Any:
    from diffusers import StableDiffusionUpscalePipeline

    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def _attach_lora(unet: torch.nn.Module, rank: int, target_modules: list[str]) -> torch.nn.Module:
    cfg = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(unet, cfg)


def _save_sample(
    pipe: Any,
    lora_unet: torch.nn.Module,
    lr_pil: Image.Image,
    prompt: str,
    out_path: Path,
    steps: int = 20,
) -> None:
    """Render one sample from the current LoRA weights for visual sanity."""
    original = pipe.unet
    pipe.unet = lora_unet
    lora_unet.eval()
    try:
        with torch.no_grad():
            image = pipe(prompt=prompt, image=lr_pil, num_inference_steps=steps).images[0]
        image.save(out_path)
    finally:
        pipe.unet = original


def train(config_path: Path) -> int:
    cfg = yaml.safe_load(config_path.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    out_dir = REPO_ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))

    print(f"Loading {cfg['base_model']} on {device} ({dtype}) ...", file=sys.stderr)
    pipe = _load_pipeline(cfg["base_model"], device, dtype)

    # Freeze every component; LoRA layers will be the only trainable parameters.
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.unet.requires_grad_(False)

    # The VAE in fp16 produces NaN latents on a non-trivial fraction of inputs
    # (overflow during sampling from the latent distribution). Diffusers emits
    # an `upcast_vae` deprecation warning steering us to do this manually.
    # Keep VAE in fp32 throughout — UNet stays in fp16 for memory savings.
    pipe.vae.to(torch.float32)

    print(
        f"Attaching LoRA (rank={cfg['lora_rank']}, targets={cfg['lora_target_modules']}) ...",
        file=sys.stderr,
    )
    lora_unet = _attach_lora(pipe.unet, cfg["lora_rank"], cfg["lora_target_modules"])
    lora_unet.train()

    trainable = sum(p.numel() for p in lora_unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_unet.parameters())
    print(
        f"Trainable: {trainable:,} / {total:,}  ({trainable / total * 100:.3f}%)",
        file=sys.stderr,
    )
    if trainable == 0:
        print(
            "FATAL: no trainable LoRA parameters were created. "
            "The target_modules list does not match any module in the UNet.",
            file=sys.stderr,
        )
        return 1

    pairs_dir = REPO_ROOT / cfg["pairs_dir"]
    captions_path = REPO_ROOT / cfg["captions_path"]
    ds = PairCaptionDataset(pairs_dir, captions_path)
    if len(ds) == 0:
        print(f"FATAL: empty dataset at {pairs_dir}", file=sys.stderr)
        return 1
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(ds)} pairs, {len(dl)} batches/epoch", file=sys.stderr)

    optimizer = torch.optim.AdamW(
        [p for p in lora_unet.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None

    # Training scheduler (DDPM); separate from the inference scheduler the pipe holds.
    from diffusers import DDPMScheduler

    train_scheduler = DDPMScheduler.from_pretrained(cfg["base_model"], subfolder="scheduler")

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Fixed sample we render every cfg["sample_every"] steps for visual sanity.
    sample_idx = ds.indices[0]
    sample_lr_pil = Image.open(ds.pairs_dir / f"{sample_idx:06d}_lr.jpg").convert("RGB")
    sample_caption = ds.captions.get(sample_idx, "a high resolution photograph, sharp detail")
    print(f"Sample image: idx={sample_idx}  caption={sample_caption!r}", file=sys.stderr)

    try:
        from torch.utils.tensorboard import SummaryWriter

        tb: Any = SummaryWriter(out_dir / "tb")
    except Exception as e:
        print(f"(tensorboard disabled: {e})", file=sys.stderr)
        tb = None

    losses: list[dict] = []
    step = 0
    start_time = time.time()
    pbar = tqdm(total=cfg["max_steps"], desc="train")

    while step < cfg["max_steps"]:
        for lr_t, hr_t, captions in dl:
            if step >= cfg["max_steps"]:
                break
            lr_t = lr_t.to(device, dtype, non_blocking=True)
            hr_t = hr_t.to(device, dtype, non_blocking=True)
            bsz = hr_t.shape[0]

            # VAE runs in fp32, so cast HR up first then bring latents back to UNet dtype.
            with torch.no_grad():
                latents_fp32 = (
                    pipe.vae.encode(hr_t.float()).latent_dist.sample()
                    * pipe.vae.config.scaling_factor
                )
                latents = latents_fp32.to(dtype)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, train_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)

            # x4-upscaler's noise-level conditioning. Inference uses up to 350;
            # for training rehearsal we keep it modest so loss isn't dominated
            # by extreme augmentations.
            noise_level = torch.randint(0, cfg.get("max_noise_level", 100), (bsz,), device=device)

            # Concat LR pixels (3 channels) with noisy latents (4 channels) at the
            # same spatial resolution — this is exactly what the upscale pipeline
            # does at inference, so the geometry already matches.
            unet_input = torch.cat([noisy_latents, lr_t], dim=1)

            tok = tokenizer(
                list(captions),
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
            with torch.no_grad():
                text_emb = text_encoder(tok)[0]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, dtype=dtype):
                noise_pred = lora_unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=text_emb,
                    class_labels=noise_level,
                ).sample
                loss = F.mse_loss(noise_pred.float(), noise.float())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lora_unet.parameters() if p.requires_grad], max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lora_unet.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()

            losses.append(
                {"step": step, "loss": float(loss.item()), "elapsed": time.time() - start_time}
            )
            if tb is not None:
                tb.add_scalar("train/loss", loss.item(), step)

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if step % cfg.get("sample_every", 200) == 0 or step == cfg["max_steps"]:
                _save_sample(
                    pipe,
                    lora_unet,
                    sample_lr_pil,
                    sample_caption,
                    samples_dir / f"step_{step:05d}.jpg",
                    steps=cfg.get("sample_inference_steps", 20),
                )
                lora_unet.train()

            if step % cfg.get("checkpoint_every", 500) == 0:
                lora_unet.save_pretrained(ckpt_dir / f"step_{step:05d}")

    pbar.close()
    lora_unet.save_pretrained(ckpt_dir / "final")
    (out_dir / "losses.jsonl").write_text("\n".join(json.dumps(line) for line in losses))
    elapsed = time.time() - start_time
    print(
        f"\nDone. {len(losses)} steps in {elapsed:.0f}s "
        f"({elapsed / max(1, len(losses)):.2f} s/step). Final LoRA: {ckpt_dir / 'final'}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    return train(args.config)


if __name__ == "__main__":
    sys.exit(main())
