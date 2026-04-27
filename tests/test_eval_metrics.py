"""Tests for eval_metrics. Two cases per metric (identical, random) catch
higher-vs-lower-is-better bugs and confirm sane edge behaviour.
"""

import math

import numpy as np
import pytest
from PIL import Image

from upscaler import eval_metrics


@pytest.fixture(scope="module")
def img_a():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def img_b():
    rng = np.random.default_rng(99)
    return Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))


# ---------- LPIPS (lower is better) ----------


def test_lpips_identical_is_zero(img_a):
    assert eval_metrics.lpips(img_a, img_a) == pytest.approx(0.0, abs=1e-3)


def test_lpips_random_pair_above_threshold(img_a, img_b):
    # Random images should produce a clearly non-zero distance.
    assert eval_metrics.lpips(img_a, img_b) > 0.05


# ---------- DISTS (lower is better) ----------


def test_dists_identical_is_zero(img_a):
    assert eval_metrics.dists(img_a, img_a) == pytest.approx(0.0, abs=1e-3)


def test_dists_random_pair_above_threshold(img_a, img_b):
    assert eval_metrics.dists(img_a, img_b) > 0.02


# ---------- PSNR (higher is better) ----------


def test_psnr_identical_is_infinity(img_a):
    assert eval_metrics.psnr(img_a, img_a) == math.inf


def test_psnr_random_pair_is_low(img_a, img_b):
    # Random uint8 images give PSNR around 7-8 dB; very far from "good" (>30).
    score = eval_metrics.psnr(img_a, img_b)
    assert math.isfinite(score)
    assert score < 12.0


# ---------- SSIM (higher is better) ----------


def test_ssim_identical_is_one(img_a):
    assert eval_metrics.ssim(img_a, img_a) == pytest.approx(1.0, abs=1e-4)


def test_ssim_random_pair_is_low(img_a, img_b):
    assert eval_metrics.ssim(img_a, img_b) < 0.05


# ---------- Shape-mismatch guards ----------


@pytest.mark.parametrize(
    "metric_fn",
    [eval_metrics.lpips, eval_metrics.dists, eval_metrics.psnr, eval_metrics.ssim],
)
def test_each_metric_rejects_size_mismatch(metric_fn):
    a = Image.new("RGB", (64, 64))
    b = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="size mismatch"):
        metric_fn(a, b)


# ---------- evaluate_method ----------


def test_evaluate_method_returns_one_row_per_image_per_ratio(tmp_path, img_a):
    import json

    from upscaler import testset

    metadata = {
        "alpha.jpg": {"category": "traditional", "subcategory": "landscape"},
        "beta.jpg": {"category": "hard", "challenges": ["text"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    # Save HR + LR variants so the testset paths exist.
    for stem in ("alpha", "beta"):
        img_a.resize((1000, 1000)).save(tmp_path / f"{stem}.jpg")
        for sz in (100, 200, 250):
            img_a.resize((sz, sz)).save(tmp_path / f"{stem}_{sz}.jpg")
    ts = testset.load(tmp_path)

    def fake_method(img, lr_size):
        return Image.open(img.hr_path)  # identity = perfect score

    df = eval_metrics.evaluate_method("identity", fake_method, ts)
    assert len(df) == 2 * 3  # 2 images x 3 ratios
    assert set(df["method"]) == {"identity"}
    # Identity method => LPIPS/DISTS ~ 0, SSIM ~ 1
    assert (df["lpips"].abs() < 1e-3).all()
    assert (df["dists"].abs() < 1e-3).all()
    assert (df["ssim"] > 0.999).all()
