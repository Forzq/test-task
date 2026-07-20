"""Tests for strict conversion of additional Roboflow sources."""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from scripts.import_roboflow_sources import (
    SourceSpec,
    _convert_label,
    _convert_sources,
    _update_combined,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    """
    Write a small valid image used by the synthetic dataset.

    Parameters
    ----------
    path : Path
        Destination image path.
    color : tuple[int, int, int]
        Solid RGB fill used to produce distinct file digests.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def test_convert_label_maps_targets_and_ignores_background(tmp_path: Path) -> None:
    """Bolt and nut boxes must be remapped while screw remains background."""
    label = tmp_path / "example.txt"
    label.write_text(
        "0 0.5 0.5 0.4 0.4\n"
        "1 0.4 0.4 0.2 0.2\n"
        "2 0.5 0.5 0.1 0.1\n",
        encoding="utf-8",
    )

    lines, counts, ignored = _convert_label(
        label,
        {0: "Bolt", 1: "Nut", 2: "Screw"},
        {"bolt": "bolt", "nut": "nut"},
    )

    assert [line.split()[0] for line in lines] == ["0", "1"]
    assert counts == {"Bolt": 1, "Nut": 1, "Screw": 1}
    assert ignored == 1


def test_convert_sources_builds_dataset_and_updates_combined(tmp_path: Path) -> None:
    """A train-only source must become a valid strict dataset without training."""
    source_root = tmp_path / "source"
    image_root = source_root / "train" / "images"
    label_root = source_root / "train" / "labels"
    label_root.mkdir(parents=True)
    _write_image(image_root / "bolt.jpg", (220, 20, 20))
    _write_image(image_root / "screw.jpg", (20, 220, 20))
    (label_root / "bolt.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n"
        "1 0.25 0.25 0.2 0.2\n",
        encoding="utf-8",
    )
    (label_root / "screw.txt").write_text(
        "2 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    data_path = source_root / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
                "names": ["Bolt", "Nut", "Screw"],
            }
        ),
        encoding="utf-8",
    )

    existing_root = tmp_path / "existing"
    for split in ("train", "valid", "test"):
        (existing_root / split / "images").mkdir(parents=True)
    combined = tmp_path / "combined.yaml"
    combined.write_text(
        yaml.safe_dump(
            {
                "train": "existing/train/images",
                "val": "existing/valid/images",
                "test": "existing/test/images",
                "names": ["bolt", "nut"],
            }
        ),
        encoding="utf-8",
    )
    spec = SourceSpec(
        key="synthetic",
        workspace="workspace",
        project="project",
        version=1,
        title="Synthetic",
        url="https://example.test",
        license_name="CC BY 4.0",
        target_aliases={"bolt": "bolt", "nut": "nut"},
    )
    output = tmp_path / "normalized"

    manifest = _convert_sources(
        selected=(spec,),
        data_paths={spec.key: data_path},
        output=output,
        combined=combined,
        benchmark=tmp_path / "benchmark",
        max_variants_per_family=1,
        replace=False,
    )
    _update_combined(combined, output)

    assert sum(split.get("images", 0) for split in manifest["splits"].values()) == 2
    assert sum(split.get("bolt_boxes", 0) for split in manifest["splits"].values()) == 1
    assert sum(split.get("nut_boxes", 0) for split in manifest["splits"].values()) == 1
    assert sum(split.get("negative_images", 0) for split in manifest["splits"].values()) == 1
    combined_document = yaml.safe_load(combined.read_text(encoding="utf-8"))
    assert any("normalized/train/images" in path for path in combined_document["train"])
    assert (output / "manifest.json").is_file()
    assert (output / "review.csv").is_file()
