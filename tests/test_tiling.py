import numpy as np
import pytest

from upscaler import tiling


def test_tile_positions_evenly_spaces_three_tiles_over_1000px():
    # 1000 / (512, overlap 128) should give three evenly-spaced tiles.
    assert tiling.tile_positions(1000, 512, 128) == [0, 244, 488]


def test_tile_positions_single_tile_when_dim_le_tile():
    assert tiling.tile_positions(512, 512, 128) == [0]
    assert tiling.tile_positions(256, 512, 128) == [0]


def test_tile_positions_covers_non_divisible_dim():
    # Every pixel in a 777-wide image is inside some tile.
    pos = tiling.tile_positions(777, 512, 128)
    assert pos[0] == 0
    assert pos[-1] == 777 - 512  # last tile pinned to the right edge
    for y in range(777):
        assert any(p <= y < p + 512 for p in pos), f"pixel {y} uncovered"


def test_tile_positions_rejects_overlap_ge_tile():
    with pytest.raises(ValueError):
        tiling.tile_positions(1000, 256, 256)


def test_blend_weights_sum_to_exactly_one_in_overlap():
    # Two adjacent 1D windows. Tile 1 at position 0 (covers [0..7]),
    # tile 2 at position 6 (stride 8-2=6, covers [6..13]).
    # Overlap region is image positions [6..7] = tile1[6..7] + tile2[0..1].
    w = tiling.blend_weights_1d(8, 2)
    assert np.allclose(w[6:8] + w[0:2], 1.0, atol=1e-7)


def test_blend_weights_2d_is_outer_product_of_1d():
    w2d = tiling.blend_weights_2d(16, 16, 4)
    w1d = tiling.blend_weights_1d(16, 4)
    assert np.allclose(w2d, np.outer(w1d, w1d))
    # Center is flat 1; corners are smallest.
    assert w2d[8, 8] == 1.0
    assert w2d[0, 0] == pytest.approx(w1d[0] ** 2)


def test_round_trip_identity_byte_for_byte_large_random_image():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(1000, 1000, 3), dtype=np.uint8)
    out = tiling.tile_and_process(img, lambda t: t, tile_size=512, overlap=128)
    assert np.array_equal(out, img)


def test_round_trip_identity_on_edge_tile_dim():
    # Width 600 forces a last tile pinned to 88 (not divisible by stride).
    assert tiling.tile_positions(600, 512, 128) == [0, 88]
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(600, 600, 3), dtype=np.uint8)
    out = tiling.tile_and_process(img, lambda t: t, tile_size=512, overlap=128)
    assert np.array_equal(out, img)


def test_round_trip_identity_on_non_square_input():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(800, 1200, 3), dtype=np.uint8)
    out = tiling.tile_and_process(img, lambda t: t, tile_size=512, overlap=128)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_tile_and_process_raises_on_shape_change():
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        tiling.tile_and_process(img, lambda t: t[:10, :10], tile_size=512, overlap=128)
