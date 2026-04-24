"""Frozen 60-image test set: metadata loading + slice queries."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("data/test_images")
LR_SIZES: tuple[int, ...] = (100, 200, 250)


@dataclass(frozen=True)
class TestImage:
    name: str
    category: str
    subcategory: str | None = None
    challenges: tuple[str, ...] = ()
    root: Path = DEFAULT_ROOT

    @property
    def hr_path(self) -> Path:
        return self.root / self.name

    def lr_path(self, size: int) -> Path:
        if size not in LR_SIZES:
            raise ValueError(f"LR size {size} not in {LR_SIZES}")
        stem = Path(self.name).stem
        suffix = Path(self.name).suffix
        return self.root / f"{stem}_{size}{suffix}"


@dataclass(frozen=True)
class Testset:
    images: tuple[TestImage, ...]
    root: Path = DEFAULT_ROOT

    def __iter__(self) -> Iterator[TestImage]:
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)

    def slice(
        self,
        *,
        category: str | None = None,
        subcategory: str | None = None,
        challenge: str | None = None,
    ) -> list[TestImage]:
        out = list(self.images)
        if category is not None:
            out = [i for i in out if i.category == category]
        if subcategory is not None:
            out = [i for i in out if i.subcategory == subcategory]
        if challenge is not None:
            out = [i for i in out if challenge in i.challenges]
        return out

    @property
    def subcategories(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for img in self.images:
            if img.subcategory:
                seen[img.subcategory] = None
        return tuple(seen)

    @property
    def challenges(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for img in self.images:
            for c in img.challenges:
                seen[c] = None
        return tuple(seen)


def load(root: Path | str = DEFAULT_ROOT) -> Testset:
    root = Path(root)
    metadata_path = root / "metadata.json"
    raw = json.loads(metadata_path.read_text())
    images = tuple(
        TestImage(
            name=name,
            category=entry["category"],
            subcategory=entry.get("subcategory"),
            challenges=tuple(entry.get("challenges", ())),
            root=root,
        )
        for name, entry in raw.items()
    )
    return Testset(images=images, root=root)


__all__ = ["DEFAULT_ROOT", "LR_SIZES", "TestImage", "Testset", "load"]
