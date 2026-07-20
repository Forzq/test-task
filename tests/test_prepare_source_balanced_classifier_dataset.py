"""Tests for deterministic source-balanced classifier crop preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.prepare_source_balanced_classifier_dataset import (
    BalanceConfig,
    prepare_balanced_dataset,
)


SPLITS = ("train", "val", "test")
CLASSES = ("bolt", "nut", "other")


def _write_crop(path: Path, color: tuple[int, int, int], marker: int) -> None:
    """Write one visually distinct valid crop image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((marker, marker, marker + 4, marker + 4), fill=(255, 255, 255))
    image.save(path)


def _fixture_root(root: Path, source_offset: int) -> None:
    """Create two unique crops for every split and class."""
    for split_index, split in enumerate(SPLITS):
        for class_index, class_name in enumerate(CLASSES):
            base = 20 + split_index * 60 + class_index * 15
            for item_index in range(2):
                _write_crop(
                    root / split / class_name / f"{item_index}.png",
                    (base, 30 + source_offset, 50 + item_index * 30),
                    marker=2 + source_offset + item_index * 8,
                )


def test_balancer_removes_source_overlap_and_preserves_equal_quotas(
    tmp_path: Path,
) -> None:
    """Each output group must contain unique crops from both sources."""
    legacy = tmp_path / "legacy"
    dataset3 = tmp_path / "dataset3"
    output = tmp_path / "balanced"
    _fixture_root(legacy, source_offset=0)
    _fixture_root(dataset3, source_offset=1)
    shared_source = legacy / "train" / "bolt" / "0.png"
    shared_target = dataset3 / "train" / "bolt" / "0.png"
    shared_target.write_bytes(shared_source.read_bytes())

    manifest = prepare_balanced_dataset(
        BalanceConfig(
            legacy=legacy,
            dataset3=dataset3,
            output=output,
            train_per_source_class=1,
            val_per_source_class=1,
            test_per_source_class=1,
            seed=42,
            replace=False,
        )
    )

    assert manifest["cross_source_exact_overlaps_removed"]["train"]["bolt"] == 1
    assert manifest["selected_totals"]["train"] == {
        "bolt": 2,
        "nut": 2,
        "other": 2,
    }
    digests: list[str] = []
    for path in output.rglob("*.png"):
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(digests) == 18
    assert len(digests) == len(set(digests))
