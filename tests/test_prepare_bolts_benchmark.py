"""Tests for the source-disjoint BOLTS benchmark converter."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from scripts.prepare_bolts_benchmark import BenchmarkConfig, prepare_benchmark


SOURCE_NAMES = ["bolt", "lockwasher", "nut", "other", "screw", "washer"]


def _write_source_example(
    root: Path,
    split: str,
    name: str,
    color: tuple[int, int, int],
    labels: list[str],
    pattern: str = "square",
) -> None:
    """Write one temporary benchmark-source example."""
    image_root = root / split / "images"
    label_root = root / split / "labels"
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(image)
    if pattern == "square":
        draw.rectangle((8, 8, 30, 30), fill=(0, 0, 0))
    elif pattern == "diagonal":
        draw.polygon(((0, 0), (18, 0), (64, 46), (64, 64)), fill=(0, 0, 0))
    image.save(image_root / name)
    (label_root / f"{Path(name).stem}.txt").write_text(
        "\n".join(labels) + "\n", encoding="utf-8"
    )


def _write_metadata(root: Path) -> None:
    """Write immutable source metadata expected by the converter."""
    document = {
        "names": SOURCE_NAMES,
        "roboflow": {
            "workspace": "fastener-urest",
            "project": "bolts-v5qf3",
            "version": 1,
            "license": "CC BY 4.0",
        },
    }
    (root / "data.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")


def test_prepare_benchmark_maps_classes_and_deduplicates_family(tmp_path: Path) -> None:
    """One family must be retained and non-target boxes must become background."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    for split in ("train", "valid", "test"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
    _write_metadata(source)
    _write_source_example(
        source,
        "test",
        "family_a_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
        (255, 0, 0),
        ["0 0.5 0.5 0.4 0.4", "5 0.2 0.2 0.1 0.1"],
    )
    _write_source_example(
        source,
        "train",
        "family_a_jpg.rf.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.jpg",
        (200, 0, 0),
        ["0 0.5 0.5 0.4 0.4", "5 0.2 0.2 0.1 0.1"],
    )
    _write_source_example(
        source,
        "valid",
        "family_b_jpg.rf.cccccccccccccccccccccccccccccccc.jpg",
        (0, 255, 0),
        ["2 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"],
        pattern="diagonal",
    )

    manifest = prepare_benchmark(
        BenchmarkConfig(
            source=source,
            output=output,
            training_data=None,
            additional_exclusion_roots=(),
            perceptual_distance=4,
            max_variants_per_family=1,
            replace=False,
        )
    )

    assert manifest["source"]["raw_images"] == 3
    assert manifest["source"]["source_families"] == 2
    assert manifest["deduplication"]["offline_family_variants_removed"] == 1
    assert manifest["source_polygon_annotations_converted"] == 1
    labels = list((output / "test" / "labels").glob("*.txt"))
    assert len(labels) == 2
    assert sorted(
        line.split()[0]
        for label in labels
        for line in label.read_text(encoding="utf-8").splitlines()
        if line
    ) == ["0", "1"]
    ignored = [
        json.loads(line)
        for line in (output / "ignored_boxes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert ignored[0]["boxes"][0]["class"] == "washer"


def test_prepare_benchmark_applies_reviewed_family_exclusions(tmp_path: Path) -> None:
    """Reviewed annotation failures must be excluded before model inference."""
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "test" / "images").mkdir(parents=True)
    (source / "test" / "labels").mkdir(parents=True)
    for split in ("train", "valid"):
        (source / split / "images").mkdir(parents=True)
        (source / split / "labels").mkdir(parents=True)
    _write_metadata(source)
    _write_source_example(
        source,
        "test",
        "bad_family_jpg.rf.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.jpg",
        (255, 0, 0),
        ["0 0.5 0.5 0.4 0.4"],
    )
    exclusions = source / "benchmark_exclusions.json"
    exclusions.write_text(
        json.dumps({"bad_family_jpg": "Incorrect source annotation."}),
        encoding="utf-8",
    )

    manifest = prepare_benchmark(
        BenchmarkConfig(
            source=source,
            output=output,
            training_data=None,
            additional_exclusion_roots=(),
            perceptual_distance=4,
            max_variants_per_family=1,
            replace=False,
            manual_exclusions=exclusions,
        )
    )

    assert manifest["deduplication"]["manual_quality_exclusions"] == 1
    assert list((output / "test" / "images").iterdir()) == []
    excluded = json.loads(
        (output / "excluded_manual_quality.json").read_text(encoding="utf-8")
    )
    assert excluded == [
        {"family": "bad_family_jpg", "reason": "Incorrect source annotation."}
    ]
