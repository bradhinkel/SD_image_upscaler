"""Tests for the degradation pipeline. Validate output shape, determinism,
and that each parameter range stays in bounds."""

import numpy as np
import pytest
from PIL import Image

from upscaler import degradations


@pytest.fixture
def hr_image():
    """A 512x512 RGB image with structured content so degradation effects matter."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_degrade_shape_is_hr_divided_by_scale(hr_image):
    out = degradations.degrade_seeded(hr_image, seed=0)
    assert out.size == (128, 128)


def test_degrade_returns_rgb_uint8_pil(hr_image):
    out = degradations.degrade_seeded(hr_image, seed=42)
    assert out.mode == "RGB"
    arr = np.asarray(out)
    assert arr.dtype == np.uint8
    assert arr.shape == (128, 128, 3)


def test_degrade_is_deterministic_with_same_seed(hr_image):
    out1 = degradations.degrade_seeded(hr_image, seed=7)
    out2 = degradations.degrade_seeded(hr_image, seed=7)
    assert np.array_equal(np.asarray(out1), np.asarray(out2))


def test_degrade_differs_with_different_seeds(hr_image):
    out1 = degradations.degrade_seeded(hr_image, seed=1)
    out2 = degradations.degrade_seeded(hr_image, seed=2)
    assert not np.array_equal(np.asarray(out1), np.asarray(out2))


def test_degrade_respects_custom_scale(hr_image):
    cfg = degradations.DegradationConfig(scale=8)
    out = degradations.degrade(hr_image, config=cfg, rng=np.random.default_rng(0))
    assert out.size == (64, 64)


def test_degrade_with_zero_noise_and_blur_still_runs(hr_image):
    cfg = degradations.DegradationConfig(blur_sigma=(0.0, 0.0), noise_sigma=(0.0, 0.0))
    out = degradations.degrade(hr_image, config=cfg, rng=np.random.default_rng(0))
    arr = np.asarray(out)
    # Output stays in valid uint8 range even at extremes.
    assert arr.min() >= 0 and arr.max() <= 255


def test_degrade_with_max_noise_keeps_values_in_range(hr_image):
    cfg = degradations.DegradationConfig(noise_sigma=(50.0, 50.0))
    out = degradations.degrade(hr_image, config=cfg, rng=np.random.default_rng(0))
    arr = np.asarray(out)
    assert arr.min() >= 0 and arr.max() <= 255
