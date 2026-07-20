"""Tests for the local mechanical-parts dataset converter."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from scripts.prepare_mechanical_parts_dataset import (
    PreparationConfig,
    prepare_dataset,
)


SOURCE_NAMES = [
    "bearing",
    "bolt",
    "flange",
    "gear",
    "nut",
    "shaft",
    "snap_ring",
    "spacer",
    "spring",
    "washer",
]


def _write_example(
    root: Path, split: str, name: str, color: tuple[int, int, int], labels: list[str]
) -> None:
    """
    Write one small source image and matching YOLO label.

    Parameters
    ----------
    root : Path
        Temporary source root.
    split : str
        Publisher split.
    name : str
        Image filename.
    color : tuple[int, int, int]
        Solid RGB test color.
    labels : list[str]
        Source YOLO annotation lines.
    """
    image_root = root / split / "images"
    label_root = root / split / "labels"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(image_root / name)
    (label_root / f"{Path(name).stem}.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )


def test_prepare_dataset_maps_targets_and_deduplicates_families(tmp_path: Path) -> None:
    """The converter must retain one family and preserve ignored boxes."""
    source = tmp_path / "dataset3"
    output = tmp_path / "prepared"
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
    (source / "data.yaml").write_text(
        yaml.safe_dump({"names": SOURCE_NAMES}), encoding="utf-8"
    )
    _write_example(
        source,
        "train",
        "part_a_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
        (255, 0, 0),
        ["1 0.5 0.5 0.4 0.4", "9 0.2 0.2 0.1 0.1"],
    )
    _write_example(
        source,
        "valid",
        "part_a_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg",
        (200, 0, 0),
        ["1 0.5 0.5 0.4 0.4", "9 0.2 0.2 0.1 0.1"],
    )
    _write_example(
        source,
        "test",
        "part_b_jpg.rf.cccccccccccccccccccccccccccccccc.jpg",
        (0, 255, 0),
        ["4 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"],
    )

    manifest = prepare_dataset(
        PreparationConfig(
            source=source,
            output=output,
            existing_data=None,
            benchmark_images=None,
            validation_percent=10,
            max_variants_per_family=1,
            seed=42,
            replace=False,
        )
    )

    images = list(output.glob("*/images/*"))
    labels = list(output.glob("*/labels/*.txt"))
    assert len(images) == len(labels) == 2
    assert manifest["raw_images"] == 3
    assert manifest["source_families"] == 2
    assert manifest["offline_variants_not_retained"] == 1
    assert manifest["source_polygon_annotations_converted"] == 1
    label_classes = sorted(
        line.split()[0]
        for label in labels
        for line in label.read_text(encoding="utf-8").splitlines()
        if line
    )
    assert label_classes == ["0", "1"]
    ignored = [
        json.loads(line)
        for line in (output / "ignored_boxes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert ignored[0]["boxes"][0]["class"] == "washer"


def test_prepare_dataset_reports_retiring_benchmark_overlap(tmp_path: Path) -> None:
    """Source-family overlap with benchmark v1 must be explicit in the manifest."""
    source = tmp_path / "dataset3"
    output = tmp_path / "prepared"
    benchmark = tmp_path / "benchmark"
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
    benchmark.mkdir()
    (source / "data.yaml").write_text(
        yaml.safe_dump({"names": SOURCE_NAMES}), encoding="utf-8"
    )
    _write_example(
        source,
        "train",
        "bolt_001_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
        (1, 2, 3),
        ["1 0.5 0.5 0.4 0.4"],
    )
    Image.new("RGB", (16, 16), (4, 5, 6)).save(
        benchmark
        / "mechanical_parts_bolt_001_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg"
    )

    manifest = prepare_dataset(
        PreparationConfig(
            source=source,
            output=output,
            existing_data=None,
            benchmark_images=benchmark,
            validation_percent=10,
            max_variants_per_family=1,
            seed=42,
            replace=False,
        )
    )

    assert manifest["retiring_benchmark"]["matching_source_families"] == 1
