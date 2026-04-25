"""Smoke tests for UpscalerPipeline using a mocked diffusers class.

CPU-safe: no GPU, no model weights. Exercises the control flow and
parameter forwarding without touching the real diffusers.
"""

from unittest.mock import MagicMock

import pytest
from PIL import Image

from upscaler import pipeline as pipeline_mod
from upscaler.pipeline import UpscalerPipeline


def _install_mock_pipeline_class(monkeypatch, output_image):
    """Patch the module-level pipeline class with a mock that returns output_image."""
    inner = MagicMock()
    inner.to.return_value = inner
    inner.return_value = MagicMock(images=[output_image])

    cls = MagicMock()
    cls.from_pretrained.return_value = inner

    monkeypatch.setattr(pipeline_mod, "_PIPELINE_CLASS", cls)
    return cls, inner


@pytest.fixture
def out_image():
    return Image.new("RGB", (1000, 1000), color=(42, 42, 42))


def test_upscale_x4_auto_loads_when_called_without_explicit_load(monkeypatch, out_image):
    cls, _inner = _install_mock_pipeline_class(monkeypatch, out_image)
    up = UpscalerPipeline(device="cpu")

    result = up.upscale_x4(Image.new("RGB", (250, 250)), prompt="a test")

    assert result is out_image
    cls.from_pretrained.assert_called_once()


def test_upscale_x4_forwards_prompt_noise_and_steps(monkeypatch, out_image):
    _, inner = _install_mock_pipeline_class(monkeypatch, out_image)
    up = UpscalerPipeline(device="cpu")

    up.upscale_x4(Image.new("RGB", (200, 200)), prompt="forest", noise_level=55, steps=17)

    assert inner.call_args.kwargs["prompt"] == "forest"
    assert inner.call_args.kwargs["noise_level"] == 55
    assert inner.call_args.kwargs["num_inference_steps"] == 17


def test_load_is_idempotent_and_close_releases_pipe(monkeypatch, out_image):
    cls, _ = _install_mock_pipeline_class(monkeypatch, out_image)
    up = UpscalerPipeline(device="cpu")

    up.load()
    up.load()
    assert cls.from_pretrained.call_count == 1

    up.close()
    assert up._pipe is None

    up.close()  # second close is a no-op


def test_context_manager_loads_on_enter_and_closes_on_exit(monkeypatch, out_image):
    _cls, _inner = _install_mock_pipeline_class(monkeypatch, out_image)

    with UpscalerPipeline(device="cpu") as up:
        assert up._pipe is not None
    assert up._pipe is None


def test_upscale_coerces_non_rgb_input_to_rgb(monkeypatch, out_image):
    _, inner = _install_mock_pipeline_class(monkeypatch, out_image)
    up = UpscalerPipeline(device="cpu")

    rgba_input = Image.new("RGBA", (250, 250))
    up.upscale_x4(rgba_input)

    forwarded = inner.call_args.kwargs["image"]
    assert forwarded.mode == "RGB"


def _install_mock_stage_b(monkeypatch):
    """Install a mock stage B that returns the input tile unchanged (identity)."""
    mock_stage_b = MagicMock()

    def identity_call(**kwargs):
        return MagicMock(images=[kwargs["image"]])

    mock_stage_b.side_effect = identity_call
    monkeypatch.setattr(pipeline_mod, "_STAGE_B_LOADER", lambda device, dtype: mock_stage_b)
    return mock_stage_b


def test_upscale_two_stage_returns_target_size_and_forwards_denoise(monkeypatch, out_image):
    _install_mock_pipeline_class(monkeypatch, out_image)
    mock_stage_b = _install_mock_stage_b(monkeypatch)

    up = UpscalerPipeline(device="cpu")
    result = up.upscale_two_stage(
        Image.new("RGB", (250, 250)),
        target_size=1000,
        denoise=0.42,
        steps=15,
        cn_weight=0.7,
        prompt="a mountain",
    )

    assert result.size == (1000, 1000)
    # At least one stage-B call was made with our params.
    assert mock_stage_b.call_args.kwargs["strength"] == 0.42
    assert mock_stage_b.call_args.kwargs["num_inference_steps"] == 15
    assert mock_stage_b.call_args.kwargs["controlnet_conditioning_scale"] == 0.7
    assert mock_stage_b.call_args.kwargs["prompt"] == "a mountain"
