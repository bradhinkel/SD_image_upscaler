"""Image captioning wrapper for training-pair conditioning.

Uses BLIP (Salesforce/blip-image-captioning-large by default) — adequate
description quality for upscaler conditioning and dramatically faster than
BLIP-2 OPT-2.7B which would also be tight on 8 GB VRAM. The model id is
overridable to swap in BLIP-2 variants if needed.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-large"


class BLIPCaptioner:
    """Lazy-loaded image captioner. Greedy decoding by default for speed.

    Usage:
        cap = BLIPCaptioner()
        cap.caption(pil_image)
        cap.caption_batch([img1, img2, img3])  # GPU-batched
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = BlipForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    def caption(self, image: Image.Image, max_new_tokens: int = 30) -> str:
        return self.caption_batch([image], max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def caption_batch(self, images: list[Image.Image], max_new_tokens: int = 30) -> list[str]:
        if not images:
            return []
        self._load()
        rgb = [img.convert("RGB") for img in images]
        inputs = self._processor(images=rgb, return_tensors="pt")
        if self.device == "cuda":
            inputs = {
                k: v.to(self.device).half() if v.dtype.is_floating_point else v.to(self.device)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,  # greedy: faster and quality is fine for short scene captions
            do_sample=False,
        )
        return [self._processor.decode(o, skip_special_tokens=True).strip() for o in out]

    def close(self) -> None:
        self._model = None
        self._processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()


__all__ = ["DEFAULT_MODEL_ID", "BLIPCaptioner"]
