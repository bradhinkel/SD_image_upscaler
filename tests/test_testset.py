import json

import pytest

from upscaler import testset


@pytest.fixture
def mini_testset(tmp_path):
    """Build a tiny 4-image fake test set on disk: 2 traditional + 2 hard."""
    metadata = {
        "forest.jpg": {"category": "traditional", "subcategory": "landscape"},
        "lion.jpg": {"category": "traditional", "subcategory": "animals"},
        "cathedral.jpg": {"category": "hard", "challenges": ["fine_architecture"]},
        "neon_night.jpg": {"category": "hard", "challenges": ["text", "night"]},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_load_reads_every_metadata_entry(mini_testset):
    ts = testset.load(mini_testset)
    assert len(ts) == 4
    names = {img.name for img in ts}
    assert names == {"forest.jpg", "lion.jpg", "cathedral.jpg", "neon_night.jpg"}


def test_slice_by_category_splits_traditional_and_hard(mini_testset):
    ts = testset.load(mini_testset)
    assert len(ts.slice(category="traditional")) == 2
    assert len(ts.slice(category="hard")) == 2


def test_slice_by_subcategory(mini_testset):
    ts = testset.load(mini_testset)
    landscapes = ts.slice(subcategory="landscape")
    assert [i.name for i in landscapes] == ["forest.jpg"]


def test_slice_by_challenge_matches_multi_tagged_images(mini_testset):
    ts = testset.load(mini_testset)
    night = ts.slice(challenge="night")
    assert [i.name for i in night] == ["neon_night.jpg"]
    text = ts.slice(challenge="text")
    assert [i.name for i in text] == ["neon_night.jpg"]


def test_slice_filters_combine(mini_testset):
    ts = testset.load(mini_testset)
    # Hard + fine_architecture should isolate the cathedral.
    result = ts.slice(category="hard", challenge="fine_architecture")
    assert [i.name for i in result] == ["cathedral.jpg"]


def test_lr_path_uses_expected_naming(mini_testset):
    ts = testset.load(mini_testset)
    img = next(i for i in ts if i.name == "forest.jpg")
    assert img.lr_path(100).name == "forest_100.jpg"
    assert img.lr_path(200).name == "forest_200.jpg"
    assert img.lr_path(250).name == "forest_250.jpg"


def test_lr_path_rejects_unknown_size(mini_testset):
    ts = testset.load(mini_testset)
    img = ts.images[0]
    with pytest.raises(ValueError):
        img.lr_path(123)


def test_enumerate_subcategories_and_challenges(mini_testset):
    ts = testset.load(mini_testset)
    assert set(ts.subcategories) == {"landscape", "animals"}
    assert set(ts.challenges) == {"fine_architecture", "text", "night"}
