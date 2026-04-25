"""Pure-function tile/merge logic for processing large images in chunks.

Used by `UpscalerPipeline.upscale_two_stage` to run stage B on tile-sized
chunks when the full output exceeds VRAM. No PyTorch dependency — numpy only.

Design: linear-ramp blend weights where adjacent tile weights sum to exactly
1.0 in the overlap region (ramp[i] = (i+1)/(overlap+1)). Identity round-trip
is byte-for-byte exact after rounding back to uint8.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

import numpy as np


def tile_positions(dim: int, tile: int, overlap: int) -> list[int]:
    """Evenly-spaced tile starts that cover [0, dim) with the given tile size and overlap.

    For `dim <= tile`, returns `[0]` (one tile, possibly cropped).
    """
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    if overlap < 0 or overlap >= tile:
        raise ValueError(f"overlap {overlap} must be in [0, {tile})")
    if dim <= tile:
        return [0]
    stride = tile - overlap
    n = math.ceil((dim - overlap) / stride)
    if n <= 1:
        return [0]
    return [round(i * (dim - tile) / (n - 1)) for i in range(n)]


def _ramp(overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.empty(0, dtype=np.float32)
    # (i+1)/(overlap+1) so a right ramp of length O mirrored against a left ramp
    # of length O sums to exactly 1 at each position in the overlap region.
    return np.arange(1, overlap + 1, dtype=np.float32) / (overlap + 1)


def blend_weights_1d(tile: int, overlap: int) -> np.ndarray:
    """1D tile weight: linear ramps at each edge, flat 1 in the middle."""
    w = np.ones(tile, dtype=np.float32)
    ramp = _ramp(overlap)
    if ramp.size > 0 and tile >= 2 * overlap:
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def blend_weights_2d(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """2D weight map: outer product of two 1D ramps."""
    return np.outer(blend_weights_1d(tile_h, overlap), blend_weights_1d(tile_w, overlap))


def iter_tiles(
    img: np.ndarray, tile_size: int, overlap: int
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield (y, x, tile_view) for every tile in a scan over `img`."""
    h, w = img.shape[:2]
    for y in tile_positions(h, tile_size, overlap):
        for x in tile_positions(w, tile_size, overlap):
            yield y, x, img[y : y + tile_size, x : x + tile_size]


def stitch_tiles(
    shape: tuple[int, ...],
    tiles: list[tuple[int, int, np.ndarray]],
    overlap: int,
) -> np.ndarray:
    """Merge overlapping tiles into one image via feathered-weight averaging."""
    h, w = shape[:2]
    is_color = len(shape) == 3
    accum_shape = (h, w, shape[2]) if is_color else (h, w)
    accum = np.zeros(accum_shape, dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)

    for y, x, t in tiles:
        th, tw = t.shape[:2]
        w2d = blend_weights_2d(th, tw, overlap).astype(np.float64)
        if is_color:
            accum[y : y + th, x : x + tw] += t.astype(np.float64) * w2d[..., None]
        else:
            accum[y : y + th, x : x + tw] += t.astype(np.float64) * w2d
        weight_sum[y : y + th, x : x + tw] += w2d

    # Guard against zero-weight pixels (shouldn't happen with ramp[0] > 0,
    # but cheap insurance against floating-point underflow).
    weight_sum = np.maximum(weight_sum, 1e-12)
    out = accum / weight_sum[..., None] if is_color else accum / weight_sum
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def tile_and_process(
    img: np.ndarray,
    process_fn: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 512,
    overlap: int = 128,
) -> np.ndarray:
    """Tile the image, apply `process_fn` to each tile, stitch back.

    `process_fn` must return an array with the same shape as its input. Useful
    when `process_fn` is a GPU-heavy operation that doesn't fit at full size.
    """
    processed: list[tuple[int, int, np.ndarray]] = []
    for y, x, tile in iter_tiles(img, tile_size, overlap):
        out_tile = process_fn(tile)
        if out_tile.shape != tile.shape:
            raise ValueError(
                f"process_fn output shape {out_tile.shape} != input shape {tile.shape}"
            )
        processed.append((y, x, out_tile))
    return stitch_tiles(img.shape, processed, overlap)


__all__ = [
    "blend_weights_1d",
    "blend_weights_2d",
    "iter_tiles",
    "stitch_tiles",
    "tile_and_process",
    "tile_positions",
]
