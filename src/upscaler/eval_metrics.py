"""Evaluation metrics for upscaler outputs.

Phase 2.5 only needs LPIPS. Phase 3 will extend this module with DISTS, PSNR,
SSIM, and an `evaluate_method` driver. Keeping the surface minimal until then.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

_LPIPS_NET: Any = None


def _load_lpips_net() -> Any:
    """Lazy-load the AlexNet-backbone LPIPS model. Cached at module level."""
    global _LPIPS_NET
    if _LPIPS_NET is not None:
        return _LPIPS_NET
    import lpips as lpips_pkg

    net = lpips_pkg.LPIPS(net="alex", verbose=False)
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    _LPIPS_NET = net
    return net


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> 1x3xHxW tensor in LPIPS's expected [-1, 1] range."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def lpips(pred: Image.Image, target: Image.Image) -> float:
    """LPIPS distance (Alex backbone) between two PIL images. Lower is better."""
    if pred.size != target.size:
        raise ValueError(f"size mismatch: pred={pred.size}, target={target.size}")
    net = _load_lpips_net()
    p = _to_tensor(pred)
    t = _to_tensor(target)
    if torch.cuda.is_available():
        p = p.cuda()
        t = t.cuda()
    with torch.no_grad():
        d = net(p, t)
    return float(d.item())


__all__ = ["lpips"]
