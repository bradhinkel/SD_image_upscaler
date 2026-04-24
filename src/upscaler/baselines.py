"""Non-diffusion baseline upscalers: bicubic, Lanczos, Real-ESRGAN."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Real-ESRGAN weights (general x4 model from xinntao's official release).
_REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)
_CACHE_DIR = Path.home() / ".cache" / "upscaler"
_REALESRGAN_WEIGHT = _CACHE_DIR / "RealESRGAN_x4plus.pth"
_REALESRGAN_NATIVE_SCALE = 4

_model_cache: dict[str, torch.nn.Module] = {}


def _target_size(img: Image.Image, scale: float) -> tuple[int, int]:
    w, h = img.size
    return (round(w * scale), round(h * scale))


def bicubic(img: Image.Image, scale: float) -> Image.Image:
    return img.resize(_target_size(img, scale), resample=Image.Resampling.BICUBIC)


def lanczos(img: Image.Image, scale: float) -> Image.Image:
    return img.resize(_target_size(img, scale), resample=Image.Resampling.LANCZOS)


def _download_realesrgan_weights() -> Path:
    if _REALESRGAN_WEIGHT.is_file():
        return _REALESRGAN_WEIGHT
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_REALESRGAN_URL, _REALESRGAN_WEIGHT)
    return _REALESRGAN_WEIGHT


def _load_realesrgan() -> torch.nn.Module:
    if "realesrgan" in _model_cache:
        return _model_cache["realesrgan"]
    from spandrel import ModelLoader

    weight_path = _download_realesrgan_weights()
    desc = ModelLoader().load_from_file(weight_path)
    model = desc.model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = model.to(device=device, dtype=dtype)
    _model_cache["realesrgan"] = model
    return model


def realesrgan(img: Image.Image, scale: float) -> Image.Image:
    """Upsample via Real-ESRGAN (x4 model), then bicubic-fit to the target scale.

    scale=4 hits the model's native ratio directly. Other scales apply the x4
    model then bicubic-resize to the requested size.
    """
    model = _load_realesrgan()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    with torch.no_grad():
        out = model(t)
    out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    out_img = Image.fromarray((out * 255).round().astype(np.uint8))

    if scale != _REALESRGAN_NATIVE_SCALE:
        out_img = out_img.resize(_target_size(img, scale), resample=Image.Resampling.BICUBIC)
    return out_img


__all__ = ["bicubic", "lanczos", "realesrgan"]
