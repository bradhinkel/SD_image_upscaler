"""LoRA training for stage-B SD 1.5 — domain-aesthetic adaptation.

This is the SUPIR-informed pivot from the failed x4-upscaler LoRA attempt.
The job of *this* LoRA is much smaller and better-precedented:

  - Base model: stable-diffusion-v1-5 (NOT x4-upscaler).
  - Training task: standard SD 1.5 text-to-image fine-tuning on our HR_512
    crops + BLIP-large captions. No 7-channel concat, no class_labels.
  - Inference: the trained LoRA loads into stage B of UpscalerPipeline,
    where ControlNet Tile already provides the structural conditioning.
    The LoRA only nudges the texture/color prior toward our domain
    (landscape / cityscape / animals).

This is the well-trodden SD 1.5 LoRA pattern. Cross-attention targets are
appropriate here (unlike on the x4-upscaler) because the LoRA's job is to
shift the text-conditioned aesthetic prior, not to teach restoration.

Usage:
    python -m upscaler.lora_train_sd15_stage_b --config configs/sd15_stage_b.yaml
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from upscaler.dataset import PairCaptionDataset

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _load_pipeline(model_id: str, device: str, dtype: torch.dtype) -> Any:
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


def _attach_lora(
    unet: torch.nn.Module,
    rank: int,
    target_modules: list[str],
    alpha: int | None = None,
) -> torch.nn.Module:
    if alpha is None:
        alpha = rank
    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(unet, cfg)


def _save_sample(
    pipe: Any,
    lora_unet: torch.nn.Module,
    prompt: str,
    out_path: Path,
    steps: int = 25,
    seed: int = 0,
) -> None:
    """txt2img sample for visual sanity. Tests text conditioning + LoRA.

    The training loop keeps the VAE in fp32 for numerical safety, but the
    standard SD 1.5 pipeline doesn't auto-upcast latents during VAE decode
    (unlike the x4-upscaler pipeline). We temporarily cast the VAE back to
    the UNet's dtype for sampling, then restore fp32 for training.
    """
    original_unet = pipe.unet
    pipe.unet = lora_unet
    lora_unet.eval()
    unet_dtype = next(lora_unet.parameters()).dtype
    orig_vae_dtype = next(pipe.vae.parameters()).dtype
    pipe.vae.to(unet_dtype)
    try:
        with torch.no_grad():
            gen = torch.Generator(device=pipe.device).manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                generator=gen,
                height=512,
                width=512,
            ).images[0]
        image.save(out_path)
    finally:
        pipe.unet = original_unet
        pipe.vae.to(orig_vae_dtype)


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

    model_id = cfg.get("base_model", DEFAULT_BASE_MODEL)
    print(f"Loading {model_id} on {device} ({dtype}) ...", file=sys.stderr)
    pipe = _load_pipeline(model_id, device, dtype)

    # Freeze every component; LoRA layers are the only trainable parameters.
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    pipe.unet.requires_grad_(False)

    # SD 1.5 VAE in fp32 for numerical safety, same fix as the x4 path.
    pipe.vae.to(torch.float32)

    rank = cfg["lora_rank"]
    alpha = cfg.get("lora_alpha", rank)
    print(
        f"Attaching LoRA (rank={rank}, alpha={alpha} -> scale {alpha / rank:.2f}, "
        f"targets={cfg['lora_target_modules']}) ...",
        file=sys.stderr,
    )
    lora_unet = _attach_lora(pipe.unet, rank, cfg["lora_target_modules"], alpha=alpha)
    lora_unet.train()

    trainable = sum(p.numel() for p in lora_unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_unet.parameters())
    print(
        f"Trainable: {trainable:,} / {total:,}  ({trainable / total * 100:.3f}%)",
        file=sys.stderr,
    )
    if trainable == 0:
        print("FATAL: no trainable LoRA parameters were created.", file=sys.stderr)
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

    from diffusers import DDPMScheduler

    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Sample prompt for periodic visual sanity.
    sample_prompt = cfg.get("sample_prompt", "a high-resolution landscape photograph, sharp detail")
    print(f"Sample prompt: {sample_prompt!r}", file=sys.stderr)

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
        for _lr_t, hr_t, captions in dl:
            if step >= cfg["max_steps"]:
                break
            # We only use the HR; the LR isn't part of stage-B training.
            hr_t = hr_t.to(device, dtype, non_blocking=True)
            bsz = hr_t.shape[0]

            # Encode HR -> latent (VAE in fp32, then cast to UNet dtype).
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

            # Standard SD 1.5 t2i forward — text only.
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
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_emb,
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

            if step % cfg.get("sample_every", 500) == 0 or step == cfg["max_steps"]:
                _save_sample(
                    pipe,
                    lora_unet,
                    sample_prompt,
                    samples_dir / f"step_{step:05d}.jpg",
                    steps=cfg.get("sample_inference_steps", 25),
                )
                lora_unet.train()

            if step % cfg.get("checkpoint_every", 1000) == 0:
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

    # Post-training sanity render — txt2img with our sample prompt.
    print("Post-training sanity sample (txt2img) ...", file=sys.stderr)
    _save_sample(
        pipe,
        lora_unet,
        sample_prompt,
        samples_dir / "post_training_sanity.jpg",
        steps=cfg.get("sample_inference_steps", 25),
    )
    print(f"  saved {samples_dir / 'post_training_sanity.jpg'}", file=sys.stderr)
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
