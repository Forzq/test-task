"""Tests for reproducible source-balanced detector preparation."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.prepare_balanced_detector_dataset import (
    ImageRecord,
    _bucket,
    _stratified_positive_sample,
    _write_combined_yaml,
)


def _record(index: int, class_name: str, objects: int, area: float) -> ImageRecord:
    """
    Create an in-memory positive record for deterministic sampling tests.

    Parameters
    ----------
    index : int
        Unique filename component.
    class_name : str
        ``bolt`` or ``nut``.
    objects : int
        Number of boxes assigned to the selected class.
    area : float
        Median normalized box area.

    Returns
    -------
    ImageRecord
        Synthetic source record; referenced paths do not need to exist.
    """
    return ImageRecord(
        image_path=Path(f"screw_nut_bolt__{index}.jpg"),
        label_path=Path(f"screw_nut_bolt__{index}.txt"),
        split="train",
        source="screw_nut_bolt",
        bolt_boxes=objects if class_name == "bolt" else 0,
        nut_boxes=objects if class_name == "nut" else 0,
        median_area=area,
    )


def test_stratified_sample_is_deterministic_and_scale_diverse() -> None:
    """The same seed must preserve a quota and multiple density/scale strata."""
    records = [
        _record(index, "bolt", 1 if index < 5 else 25, 0.001 if index % 2 else 0.1)
        for index in range(20)
    ]

    first = _stratified_positive_sample(records, 8, 42, "train", "source")
    second = _stratified_positive_sample(records, 8, 42, "train", "source")

    assert [record.image_path for record in first] == [record.image_path for record in second]
    assert len(first) == 8
    assert len({_bucket(record) for record in first}) >= 3


def test_balanced_yaml_replaces_only_raw_roboflow_source(tmp_path: Path) -> None:
    """Balanced composition must retain existing sources and exclude the raw addition."""
    source = tmp_path / "combined.yaml"
    output = tmp_path / "combined_balanced.yaml"
    balanced = tmp_path / "roboflow_balanced"
    source.write_text(
        yaml.safe_dump(
            {
                "train": ["train/images", "roboflow_additions/train/images", "negatives/train/images"],
                "val": ["valid/images", "roboflow_additions/valid/images"],
                "test": ["test/images", "roboflow_additions/test/images"],
                "names": ["bolt", "nut"],
            }
        ),
        encoding="utf-8",
    )

    _write_combined_yaml(source, output, balanced)
    document = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert "train/images" in document["train"]
    assert "negatives/train/images" in document["train"]
    assert all("roboflow_additions" not in value for value in document["train"])
    assert any("roboflow_balanced/train/images" in value for value in document["train"])
