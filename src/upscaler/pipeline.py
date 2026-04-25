"""UpscalerPipeline: wraps stage A (stable-diffusion-x4-upscaler) and stage B
(SD 1.5 + ControlNet Tile img2img, tiled). Lazy-loads each stage on first use.

Stage A handles the native 4x lift. Stage B refines the bicubic-to-target
result via tiled ControlNet Tile img2img so fine detail is invented in a
scene-structure-preserving way. Together they cover 4x / 5x / 10x targets.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from upscaler import tiling

_DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"
_DEFAULT_SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
_DEFAULT_CONTROLNET_TILE_ID = "lllyasviel/control_v11f1e_sd15_tile"

# Test hooks: tests set these to mock callables to avoid importing diffusers.
_PIPELINE_CLASS: type | None = None
_STAGE_B_LOADER: Callable[[str, torch.dtype], Any] | None = None


def _get_pipeline_class() -> type:
    global _PIPELINE_CLASS
    if _PIPELINE_CLASS is None:
        from diffusers import StableDiffusionUpscalePipeline

        _PIPELINE_CLASS = StableDiffusionUpscalePipeline
    return _PIPELINE_CLASS


def _default_stage_b_loader(device: str, dtype: torch.dtype) -> Any:
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    controlnet = ControlNetModel.from_pretrained(_DEFAULT_CONTROLNET_TILE_ID, torch_dtype=dtype)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        _DEFAULT_SD15_ID,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


class UpscalerPipeline:
    """Two-stage upscaler: x4-upscaler + ControlNet Tile img2img with tiling."""

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: Any = None
        self._stage_b: Any = None

    def _dtype(self) -> torch.dtype:
        return torch.float16 if self.device == "cuda" else torch.float32

    def load(self) -> None:
        """Load stage A (x4-upscaler). Idempotent."""
        if self._pipe is not None:
            return
        cls = _get_pipeline_class()
        pipe = cls.from_pretrained(self.model_id, torch_dtype=self._dtype())
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        self._pipe = pipe

    def load_stage_b(self) -> None:
        """Load stage B (SD 1.5 + ControlNet Tile). Idempotent."""
        if self._stage_b is not None:
            return
        loader = _STAGE_B_LOADER or _default_stage_b_loader
        self._stage_b = loader(self.device, self._dtype())

    def close(self) -> None:
        if self._pipe is None and self._stage_b is None:
            return
        self._pipe = None
        self._stage_b = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def upscale_x4(
        self,
        image: Image.Image,
        prompt: str = "",
        noise_level: int = 20,
        steps: int = 20,
    ) -> Image.Image:
        if self._pipe is None:
            self.load()
        result = self._pipe(
            prompt=prompt,
            image=image.convert("RGB"),
            num_inference_steps=steps,
            noise_level=noise_level,
        )
        return result.images[0]

    def upscale_two_stage(
        self,
        image: Image.Image,
        target_size: int,
        denoise: float = 0.3,
        steps: int = 20,
        cn_weight: float = 1.0,
        prompt: str = "",
        tile_size: int = 512,
        overlap: int = 128,
    ) -> Image.Image:
        """Two-stage upscale: x4-upscaler then ControlNet-Tile img2img (tiled)."""
        if self._pipe is None:
            self.load()
        if self._stage_b is None:
            self.load_stage_b()

        stage_a_out = self.upscale_x4(image, prompt=prompt)
        if stage_a_out.size != (target_size, target_size):
            stage_a_out = stage_a_out.resize((target_size, target_size), Image.Resampling.BICUBIC)

        img_np = np.array(stage_a_out.convert("RGB"))

        def process_tile(tile_np: np.ndarray) -> np.ndarray:
            tile_pil = Image.fromarray(tile_np)
            h, w = tile_np.shape[:2]
            refined = self._stage_b(
                prompt=prompt,
                image=tile_pil,
                control_image=tile_pil,
                strength=denoise,
                num_inference_steps=steps,
                controlnet_conditioning_scale=cn_weight,
                height=h,
                width=w,
            ).images[0]
            return np.array(refined.convert("RGB"))

        out_np = tiling.tile_and_process(img_np, process_tile, tile_size=tile_size, overlap=overlap)
        return Image.fromarray(out_np)

    def __enter__(self) -> UpscalerPipeline:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["UpscalerPipeline"]
