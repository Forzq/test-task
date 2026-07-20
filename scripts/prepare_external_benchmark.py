"""Build an immutable out-of-distribution benchmark from a held-out public source."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_CLASS_NAMES = {0: "bearing", 1: "bolt", 2: "gear", 3: "nut"}
TARGET_CLASS_IDS = {1: 0, 3: 1}


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line arguments for benchmark preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with paths for the extracted source and generated benchmark.
    """
    parser = argparse.ArgumentParser(
        description="Convert the Mechanical Parts test split into an external benchmark."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/external_benchmark_sources/Mechanical Parts Dataset"
        ),
    )
    parser.add_argument("--source-split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--training-data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/external_benchmark"))
    parser.add_argument("--replace", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate a file SHA-256 digest.

    Parameters
    ----------
    path : Path
        File whose bytes are hashed.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_images(data_path: Path) -> list[Path]:
    """
    Resolve all image files referenced by a YOLO dataset YAML.

    Parameters
    ----------
    data_path : Path
        Existing training dataset YAML.

    Returns
    -------
    list[Path]
        Unique sorted train, validation, and test image paths.
    """
    document: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = data_path.parent
    if document.get("path"):
        declared_root = Path(str(document["path"]))
        dataset_root = (
            declared_root if declared_root.is_absolute() else dataset_root / declared_root
        )
    paths: set[Path] = set()
    for split in ("train", "val", "valid", "test"):
        declared = document.get(split)
        values = declared if isinstance(declared, list) else [declared]
        for value in values:
            if value is None:
                continue
            root = Path(str(value))
            root = root if root.is_absolute() else dataset_root / root
            if root.is_dir():
                paths.update(
                    path.resolve()
                    for path in root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
    return sorted(paths)


def _parse_source_label(path: Path) -> tuple[list[str], Counter[str]]:
    """
    Convert one four-class source label to two-class YOLO lines.

    Parameters
    ----------
    path : Path
        Source YOLO label path.

    Returns
    -------
    tuple[list[str], collections.Counter]
        Converted target lines and counts for every original class.

    Raises
    ------
    ValueError
        Raised for malformed coordinates or unsupported class identifiers.
    """
    converted: list[str] = []
    source_counts: Counter[str] = Counter()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        values = raw_line.split()
        if len(values) != 5:
            raise ValueError(f"Expected YOLO box at {path}:{line_number}")
        try:
            class_id = int(values[0])
            coordinates = [float(value) for value in values[1:]]
        except ValueError as error:
            raise ValueError(f"Malformed label at {path}:{line_number}") from error
        if class_id not in SOURCE_CLASS_NAMES:
            raise ValueError(f"Unknown source class at {path}:{line_number}")
        if any(not 0.0 <= value <= 1.0 for value in coordinates):
            raise ValueError(f"Coordinates outside [0, 1] at {path}:{line_number}")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"Non-positive box at {path}:{line_number}")
        source_counts[SOURCE_CLASS_NAMES[class_id]] += 1
        if class_id in TARGET_CLASS_IDS:
            converted.append(
                " ".join([str(TARGET_CLASS_IDS[class_id]), *values[1:]])
            )
    return converted, source_counts


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create an empty benchmark output directory.

    Parameters
    ----------
    output : Path
        Generated benchmark root.
    replace : bool
        Whether an existing generated directory may be removed.
    """
    if output.exists():
        if not replace:
            raise ValueError(f"Output already exists: {output}; pass --replace")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir(parents=True)


def prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    """
    Convert, deduplicate, and document the independent source split.

    Parameters
    ----------
    arguments : argparse.Namespace
        Validated benchmark preparation settings.

    Returns
    -------
    dict[str, Any]
        Provenance, class balance, and duplicate-audit manifest.
    """
    source_images = arguments.source / arguments.source_split / "images"
    source_labels = arguments.source / arguments.source_split / "labels"
    training_hashes = {
        _sha256(path): str(path) for path in _resolve_dataset_images(arguments.training_data)
    }
    _prepare_output(arguments.output, arguments.replace)

    source_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    within_benchmark: set[str] = set()
    duplicate_training: list[dict[str, str]] = []
    duplicate_benchmark: list[str] = []
    review_rows: list[dict[str, object]] = []

    images = sorted(
        path
        for path in source_images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise ValueError(f"Missing source label: {label_path}")
        digest = _sha256(image_path)
        if digest in training_hashes:
            duplicate_training.append(
                {"source": str(image_path), "training": training_hashes[digest]}
            )
            continue
        if digest in within_benchmark:
            duplicate_benchmark.append(str(image_path))
            continue
        within_benchmark.add(digest)

        with Image.open(image_path) as image:
            image.verify()
        converted, original_counts = _parse_source_label(label_path)
        source_counts.update(original_counts)
        target_counts["bolt"] += sum(line.startswith("0 ") for line in converted)
        target_counts["nut"] += sum(line.startswith("1 ") for line in converted)
        category = "positive" if converted else "negative_bearing_or_gear"
        image_counts[category] += 1

        destination_name = f"mechanical_parts_{image_path.name}"
        shutil.copy2(image_path, arguments.output / "images" / destination_name)
        (arguments.output / "labels" / f"{Path(destination_name).stem}.txt").write_text(
            "\n".join(converted) + ("\n" if converted else ""),
            encoding="utf-8",
        )
        review_rows.append(
            {
                "image": destination_name,
                "bolt_boxes": sum(line.startswith("0 ") for line in converted),
                "nut_boxes": sum(line.startswith("1 ") for line in converted),
                "source_bearing_boxes": original_counts["bearing"],
                "source_gear_boxes": original_counts["gear"],
                "review_status": "source_annotation",
            }
        )

    benchmark_yaml = {
        "path": ".",
        "test": "images",
        "names": ["bolt", "nut"],
    }
    (arguments.output / "benchmark.yaml").write_text(
        yaml.safe_dump(benchmark_yaml, sort_keys=False), encoding="utf-8"
    )
    with (arguments.output / "review.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    manifest: dict[str, Any] = {
        "benchmark_policy": {
            "purpose": "evaluation_only",
            "included_in_training": False,
            "included_in_threshold_calibration": False,
            "source_split": arguments.source_split,
        },
        "source": {
            "title": "Mechanical Parts Dataset 2022",
            "doi": "10.5281/zenodo.7504801",
            "url": "https://zenodo.org/records/7504801",
            "license": "CC BY 4.0",
            "archive_md5": "fc2bfa33c4d0439188cb5974694a4520",
            "source_classes": SOURCE_CLASS_NAMES,
            "mapping": {"Bolt": "bolt", "Nut": "nut"},
            "ignored_as_background": ["Bearing", "Gear"],
        },
        "images": len(review_rows),
        "image_counts": dict(image_counts),
        "target_boxes": dict(target_counts),
        "source_boxes": dict(source_counts),
        "duplicates_removed": {
            "against_training": duplicate_training,
            "within_benchmark": duplicate_benchmark,
        },
        "manual_review": {
            "status": "source annotations retained; spot-check recommended",
            "worksheet": "review.csv",
        },
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Validate paths, build the benchmark, and print its summary."""
    arguments = build_parser().parse_args()
    source_split = arguments.source / arguments.source_split
    if not source_split.is_dir():
        raise ValueError(f"Source split was not found: {source_split}")
    if not arguments.training_data.is_file():
        raise ValueError(f"Training YAML was not found: {arguments.training_data}")
    manifest = prepare(arguments)
    print(json.dumps(manifest, indent=2))
    print(f"External benchmark: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
