"""Build a single YOLO dataset YAML from compatible source datasets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CLASS_NAMES = ("bolt", "nut")
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """
    Validation summary for one dataset split.

    Parameters
    ----------
    image_count : int
        Number of image files in the split.
    label_count : int
        Number of matching YOLO text-label files.
    object_count : int
        Number of annotated objects in the split.
    """

    image_count: int
    label_count: int
    object_count: int


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """
    Verified local YOLO dataset that can be combined with another source.

    Parameters
    ----------
    root : Path
        Root directory containing the source dataset YAML and split folders.
    splits : dict[str, SplitSummary]
        Per-split file and annotation statistics.
    """

    root: Path
    splits: dict[str, SplitSummary]


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line parser for combining two local datasets.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with the repository's two dataset locations.
    """
    parser = argparse.ArgumentParser(
        description="Create a combined YOLO YAML without copying source images."
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path("data"),
        help="Root directory of the original YOLO dataset.",
    )
    parser.add_argument(
        "--secondary",
        type=Path,
        default=Path("data/dataset2"),
        help="Root directory of the additional YOLO dataset.",
    )
    parser.add_argument(
        "--additional",
        type=Path,
        action="append",
        default=[],
        help="Optional extra positive dataset; repeat for multiple sources.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/combined.yaml"),
        help="Combined YAML file to create or replace.",
    )
    parser.add_argument(
        "--negatives",
        type=Path,
        help="Optional empty-label negative dataset prepared by prepare_negative_dataset.py.",
    )
    parser.add_argument(
        "--negative-additional",
        type=Path,
        action="append",
        default=[],
        help="Optional extra empty-label dataset; repeat for multiple hard-negative sources.",
    )
    return parser


def _read_class_names(dataset_root: Path) -> tuple[str, ...]:
    """
    Read and normalize the class names declared in a source dataset YAML.

    Parameters
    ----------
    dataset_root : Path
        Directory containing ``data.yaml``.

    Returns
    -------
    tuple[str, ...]
        Class names ordered by their YOLO numeric identifiers.

    Raises
    ------
    ValueError
        Raised when the source YAML is absent or does not contain class names.
    """
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.is_file():
        raise ValueError(f"Dataset YAML was not found: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file) or {}

    names = payload.get("names")
    if isinstance(names, list):
        return tuple(str(name).lower() for name in names)
    if isinstance(names, dict):
        return tuple(str(names[index]).lower() for index in range(len(names)))
    raise ValueError(f"Dataset YAML has no valid names field: {yaml_path}")


def _validate_label_file(label_path: Path) -> int:
    """
    Validate one YOLO label file and count its annotated objects.

    Parameters
    ----------
    label_path : Path
        Text file with one YOLO bounding-box or polygon annotation per line.

    Returns
    -------
    int
        Number of non-empty annotation lines in the file.

    Raises
    ------
    ValueError
        Raised when a line has an unsupported class identifier or malformed box.
    """
    object_count = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Malformed label at {label_path}:{line_number}")
        if values[0] not in {"0", "1"}:
            raise ValueError(
                f"Unsupported class id {values[0]!r} at {label_path}:{line_number}; "
                "expected only 0 (bolt) or 1 (nut)."
            )
        object_count += 1
    return object_count


def _inspect_split(dataset_root: Path, split: str) -> SplitSummary:
    """
    Validate image and annotation files for one standard YOLO split.

    Parameters
    ----------
    dataset_root : Path
        Root directory of a downloaded dataset.
    split : str
        Dataset partition name: ``train``, ``valid``, or ``test``.

    Returns
    -------
    SplitSummary
        Image, label, and object counts for the inspected partition.

    Raises
    ------
    ValueError
        Raised when the expected images or labels directories are missing.
    """
    image_directory = dataset_root / split / "images"
    label_directory = dataset_root / split / "labels"
    if not image_directory.is_dir() or not label_directory.is_dir():
        raise ValueError(f"Expected YOLO folders are missing for {dataset_root / split}")

    images = sorted(
        path for path in image_directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    labels = sorted(path for path in label_directory.glob("*.txt") if path.is_file())
    image_stems = {path.stem for path in images}
    orphan_labels = [path.name for path in labels if path.stem not in image_stems]
    if orphan_labels:
        raise ValueError(
            f"Found labels without images in {label_directory}: {', '.join(orphan_labels[:3])}"
        )

    return SplitSummary(
        image_count=len(images),
        label_count=len(labels),
        object_count=sum(_validate_label_file(label) for label in labels),
    )


def inspect_dataset(dataset_root: Path) -> DatasetSummary:
    """
    Verify that a source is a two-class YOLO dataset compatible with this project.

    Parameters
    ----------
    dataset_root : Path
        Directory containing ``data.yaml`` and the standard split folders.

    Returns
    -------
    DatasetSummary
        Checked dataset metadata used to write the combined configuration.

    Raises
    ------
    ValueError
        Raised when classes, folders, or annotation class IDs are incompatible.
    """
    root = dataset_root.resolve()
    class_names = _read_class_names(root)
    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Incompatible class order in {root / 'data.yaml'}: {class_names}. "
            f"Expected {CLASS_NAMES}."
        )

    return DatasetSummary(
        root=root,
        splits={split: _inspect_split(root, split) for split in SPLITS},
    )


def inspect_negative_dataset(dataset_root: Path) -> DatasetSummary:
    """
    Verify a prepared negative dataset contains images with empty YOLO labels only.

    Parameters
    ----------
    dataset_root : Path
        Directory containing ``train``, ``valid``, and ``test`` negative splits.

    Returns
    -------
    DatasetSummary
        Validated negative source with zero annotated objects in every split.

    Raises
    ------
    ValueError
        Raised when a negative image has no label file or contains an object box.
    """
    root = dataset_root.resolve()
    summary = DatasetSummary(
        root=root,
        splits={split: _inspect_split(root, split) for split in SPLITS},
    )
    for split, split_summary in summary.splits.items():
        if split_summary.image_count != split_summary.label_count:
            raise ValueError(
                f"Negative split {root / split} must have one empty label file per image."
            )
        if split_summary.object_count:
            raise ValueError(
                f"Negative split {root / split} contains object annotations; labels must be empty."
            )
    return summary


def _relative_image_directory(output_path: Path, dataset_root: Path, split: str) -> str:
    """
    Express a source image directory relative to the combined YAML directory.

    Parameters
    ----------
    output_path : Path
        Destination combined dataset YAML path.
    dataset_root : Path
        Root directory of one verified source dataset.
    split : str
        Source split name.

    Returns
    -------
    str
        Portable relative path for use in a YOLO dataset YAML.
    """
    return Path(os.path.relpath(dataset_root / split / "images", start=output_path.parent.resolve())).as_posix()


def write_combined_yaml(
    primary: DatasetSummary,
    secondary: DatasetSummary,
    output_path: Path,
    negatives: DatasetSummary | None = None,
    additional: list[DatasetSummary] | None = None,
    additional_negatives: list[DatasetSummary] | None = None,
) -> None:
    """
    Write a YOLO YAML which references positive and optional negative images.

    Parameters
    ----------
    primary : DatasetSummary
        Verified original dataset.
    secondary : DatasetSummary
        Verified additional dataset.
    output_path : Path
        Location of the resulting combined YAML file.
    negatives : DatasetSummary, optional
        Verified empty-label images used to teach background rejection.
    additional : list[DatasetSummary], optional
        Further verified positive datasets appended before the negative source.
    additional_negatives : list[DatasetSummary], optional
        Further verified empty-label datasets, such as mined hard negatives.
    """
    output = output_path.resolve()
    sources = [primary, secondary, *(additional or [])]
    if negatives is not None:
        sources.append(negatives)
    sources.extend(additional_negatives or [])
    payload: dict[str, Any] = {
        "train": [_relative_image_directory(output, source.root, "train") for source in sources],
        "val": [_relative_image_directory(output, source.root, "valid") for source in sources],
        "test": [_relative_image_directory(output, source.root, "test") for source in sources],
        "names": list(CLASS_NAMES),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("# Generated by scripts/combine_datasets.py. Do not edit source labels.\n")
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def _format_summary(dataset: DatasetSummary, title: str) -> str:
    """
    Format a concise per-split validation summary for terminal output.

    Parameters
    ----------
    dataset : DatasetSummary
        Validated source dataset.
    title : str
        Human-readable source dataset label.

    Returns
    -------
    str
        One line suitable for a reproducible training log.
    """
    details = ", ".join(
        f"{split}: {summary.image_count} images, {summary.object_count} objects"
        for split, summary in dataset.splits.items()
    )
    return f"{title}: {details}"


def main() -> None:
    """
    Validate source datasets and generate the combined YAML configuration.

    Raises
    ------
    ValueError
        Raised when either source dataset is not compatible with the task labels.
    """
    arguments = build_parser().parse_args()
    primary = inspect_dataset(arguments.primary)
    secondary = inspect_dataset(arguments.secondary)
    additional = [inspect_dataset(path) for path in arguments.additional]
    negatives = inspect_negative_dataset(arguments.negatives) if arguments.negatives else None
    additional_negatives = [
        inspect_negative_dataset(path) for path in arguments.negative_additional
    ]
    write_combined_yaml(
        primary,
        secondary,
        arguments.output,
        negatives,
        additional,
        additional_negatives,
    )
    print(_format_summary(primary, "Primary dataset"))
    print(_format_summary(secondary, "Secondary dataset"))
    for index, dataset in enumerate(additional, start=1):
        print(_format_summary(dataset, f"Additional dataset {index}"))
    if negatives is not None:
        print(_format_summary(negatives, "Negative dataset"))
    for index, dataset in enumerate(additional_negatives, start=1):
        print(_format_summary(dataset, f"Additional negative dataset {index}"))
    print(f"Combined YAML: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
