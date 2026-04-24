"""UpscalerPipeline: wraps the stable-diffusion-x4-upscaler diffusers pipeline.

Stage A of the two-stage upscale path. Lazy-loads the model on first use so
import costs stay low; `.close()` releases GPU memory.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

_DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"

# Overridable by tests: set to a mock class to avoid importing diffusers.
_PIPELINE_CLASS: type | None = None


def _get_pipeline_class() -> type:
    global _PIPELINE_CLASS
    if _PIPELINE_CLASS is None:
        from diffusers import StableDiffusionUpscalePipeline

        _PIPELINE_CLASS = StableDiffusionUpscalePipeline
    return _PIPELINE_CLASS


class UpscalerPipeline:
    """Wraps `stable-diffusion-x4-upscaler` with lazy load + idempotent close."""

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe: Any = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        cls = _get_pipeline_class()
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        pipe = cls.from_pretrained(self.model_id, torch_dtype=dtype)
        pipe = pipe.to(self.device)
        pipe.enable_attention_slicing()
        self._pipe = pipe

    def close(self) -> None:
        if self._pipe is None:
            return
        self._pipe = None
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

    def __enter__(self) -> UpscalerPipeline:
        self.load()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["UpscalerPipeline"]
