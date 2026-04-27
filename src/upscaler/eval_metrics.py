"""Evaluation metrics for upscaler outputs.

Wrappers around `lpips` (Alex backbone), `piq.DISTS`, `piq.ssim`, and a hand-
rolled PSNR. Plus `evaluate_method` that walks a Testset, calls a method
function for each (image, ratio) pair, and returns a per-image DataFrame.

Lower-is-better: LPIPS, DISTS. Higher-is-better: PSNR, SSIM.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from upscaler.testset import LR_SIZES, TestImage, Testset

_LPIPS_NET: Any = None
_DISTS_NET: Any = None


def _load_lpips_net() -> Any:
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


def _load_dists_net() -> Any:
    global _DISTS_NET
    if _DISTS_NET is not None:
        return _DISTS_NET
    from piq import DISTS

    net = DISTS()
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()
    _DISTS_NET = net
    return net


def _pil_to_unit_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> 1x3xHxW tensor in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _check_same_size(pred: Image.Image, target: Image.Image) -> None:
    if pred.size != target.size:
        raise ValueError(f"size mismatch: pred={pred.size}, target={target.size}")


def lpips(pred: Image.Image, target: Image.Image) -> float:
    """LPIPS (Alex backbone). Lower is better; identical images give ~0."""
    _check_same_size(pred, target)
    net = _load_lpips_net()
    p = _pil_to_unit_tensor(pred) * 2.0 - 1.0  # LPIPS expects [-1, 1]
    t = _pil_to_unit_tensor(target) * 2.0 - 1.0
    if torch.cuda.is_available():
        p, t = p.cuda(), t.cuda()
    with torch.no_grad():
        d = net(p, t)
    return float(d.item())


def dists(pred: Image.Image, target: Image.Image) -> float:
    """DISTS (deep image structure & texture similarity). Lower is better."""
    _check_same_size(pred, target)
    net = _load_dists_net()
    p = _pil_to_unit_tensor(pred)  # piq expects [0, 1]
    t = _pil_to_unit_tensor(target)
    if torch.cuda.is_available():
        p, t = p.cuda(), t.cuda()
    with torch.no_grad():
        d = net(p, t)
    return float(d.item())


def psnr(pred: Image.Image, target: Image.Image) -> float:
    """Peak signal-to-noise ratio (dB). Higher is better; identical -> +inf."""
    _check_same_size(pred, target)
    p = np.asarray(pred.convert("RGB"), dtype=np.float64)
    t = np.asarray(target.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((p - t) ** 2))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mse)


def ssim(pred: Image.Image, target: Image.Image) -> float:
    """Structural similarity (mean over channels). Higher is better; identical -> 1."""
    _check_same_size(pred, target)
    from piq import ssim as piq_ssim

    p = _pil_to_unit_tensor(pred)
    t = _pil_to_unit_tensor(target)
    if torch.cuda.is_available():
        p, t = p.cuda(), t.cuda()
    with torch.no_grad():
        s = piq_ssim(p, t, data_range=1.0)
    return float(s.item())


def evaluate_method(
    method_name: str,
    method_fn: Callable[[TestImage, int], Image.Image],
    test_set: Testset,
    ratios: Iterable[int] = LR_SIZES,
) -> pd.DataFrame:
    """Run a method over a Testset and return per-image metrics.

    method_fn receives a TestImage and an LR size, and returns the upscaled
    prediction at the canonical 1000x1000 target. Use a closure over a cache
    directory to evaluate already-computed outputs.
    """
    rows: list[dict] = []
    ratios = tuple(ratios)
    for img in test_set:
        hr = Image.open(img.hr_path).convert("RGB")
        for lr in ratios:
            pred = method_fn(img, lr).convert("RGB")
            rows.append(
                {
                    "method": method_name,
                    "image": img.name,
                    "category": img.category,
                    "subcategory": img.subcategory or "",
                    "challenges": "|".join(img.challenges),
                    "lr_size": lr,
                    "ratio": 1000 // lr if 1000 % lr == 0 else 1000 / lr,
                    "lpips": lpips(pred, hr),
                    "dists": dists(pred, hr),
                    "psnr": psnr(pred, hr),
                    "ssim": ssim(pred, hr),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["dists", "evaluate_method", "lpips", "psnr", "ssim"]
