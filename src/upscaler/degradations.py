"""Real-ESRGAN-style realistic degradation pipeline for synthesising LR images
from HR sources during training-pair construction.

The goal isn't to perfectly model real-world bad images — it's to give the
LoRA a varied diet of artifacts (blur, downsample interpolation, sensor noise,
JPEG quantisation) so it learns to invert them rather than overfitting to one
specific degradation. Pure-function, no PyTorch, no GPU.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Default ranges. Conservative on the heavy side because our base model
# (x4-upscaler) already handles the worst cases — we just need diversity.
DEFAULT_BLUR_SIGMA = (0.2, 2.0)
DEFAULT_NOISE_SIGMA = (0.0, 8.0)  # 0..8 / 255 std-dev
DEFAULT_JPEG_QUALITY = (60, 95)
DEFAULT_RESAMPLE_METHODS = ("bicubic", "bilinear", "lanczos")

_RESAMPLE_PIL = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}


@dataclass(frozen=True)
class DegradationConfig:
    scale: int = 4
    blur_sigma: tuple[float, float] = DEFAULT_BLUR_SIGMA
    noise_sigma: tuple[float, float] = DEFAULT_NOISE_SIGMA
    jpeg_quality: tuple[int, int] = DEFAULT_JPEG_QUALITY
    resample_methods: tuple[str, ...] = DEFAULT_RESAMPLE_METHODS


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def degrade(
    hr: Image.Image,
    config: DegradationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Synthesise a realistic LR image from `hr`.

    Returns a PIL JPEG-equivalent image at HR_size / config.scale.
    Pipeline: blur -> downsample -> additive Gaussian noise -> JPEG roundtrip.
    """
    if config is None:
        config = DegradationConfig()
    if rng is None:
        rng = np.random.default_rng()

    img = hr.convert("RGB")

    # 1. Random isotropic Gaussian blur. Slight blur softens the resampling
    # before downsampling, mimicking lens/optics effects.
    sigma = float(rng.uniform(*config.blur_sigma))
    if sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 2. Random downsample method.
    method = rng.choice(config.resample_methods)
    target = (img.width // config.scale, img.height // config.scale)
    img = img.resize(target, resample=_RESAMPLE_PIL[method])

    # 3. Random additive Gaussian noise (sensor / shot noise).
    n_sigma = float(rng.uniform(*config.noise_sigma))
    if n_sigma > 0:
        arr = np.asarray(img, dtype=np.float32)
        noise = rng.normal(0.0, n_sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 4. JPEG compression.
    q = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
    img = _jpeg_roundtrip(img, q)
    return img


def degrade_seeded(
    hr: Image.Image, seed: int, config: DegradationConfig | None = None
) -> Image.Image:
    """Convenience: deterministic degradation given a seed."""
    return degrade(hr, config=config, rng=np.random.default_rng(seed))


__all__ = ["DegradationConfig", "degrade", "degrade_seeded"]
