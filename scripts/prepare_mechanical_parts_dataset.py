"""Normalize the local Roboflow mechanical-parts export for bolt/nut training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "valid", "test")
SOURCE_CLASSES = (
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
)
TARGET_CLASS_IDS = {1: 0, 4: 1}
TARGET_CLASS_NAMES = ("bolt", "nut")
GENERATED_MARKER = ".generated-mechanical-parts"
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SourceBox:
    """
    One validated source annotation.

    Parameters
    ----------
    class_id : int
        Numeric class identifier declared by the source dataset.
    coordinates : tuple[float, float, float, float]
        Normalized YOLO ``x_center, y_center, width, height`` coordinates.
    converted_from_polygon : bool
        Whether the source line was a polygon converted to an enclosing box.
    """

    class_id: int
    coordinates: tuple[float, float, float, float]
    converted_from_polygon: bool = False


@dataclass(frozen=True, slots=True)
class SourceImage:
    """
    One image and its matching annotations from the Roboflow export.

    Parameters
    ----------
    image_path : Path
        Source image path.
    label_path : Path
        Matching YOLO label path.
    declared_split : str
        Split declared by the publisher.
    family : str
        Pre-augmentation Roboflow filename family.
    boxes : tuple[SourceBox, ...]
        Validated source annotations.
    """

    image_path: Path
    label_path: Path
    declared_split: str
    family: str
    boxes: tuple[SourceBox, ...]


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """
    Configuration for producing the normalized two-class dataset.

    Parameters
    ----------
    source : Path
        Local ten-class Roboflow export.
    output : Path
        Generated two-class dataset directory.
    existing_data : Path or None
        Existing combined YAML used for exact-byte deduplication.
    benchmark_images : Path or None
        Retiring benchmark directory used only to report source-family overlap.
    validation_percent : int
        Percentage of unique source families assigned to validation.
    max_variants_per_family : int
        Maximum retained offline variants for one original family.
    seed : int
        Stable split-assignment seed.
    replace : bool
        Whether an existing generated output may be replaced.
    """

    source: Path
    output: Path
    existing_data: Path | None
    benchmark_images: Path | None
    validation_percent: int
    max_variants_per_family: int
    seed: int
    replace: bool


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for the local dataset conversion.

    Returns
    -------
    argparse.ArgumentParser
        Parser with project-local defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert data/dataset3 from ten source classes to bolt/nut, "
            "remove offline-augmentation families, and make a safe train/val split."
        )
    )
    parser.add_argument("--source", type=Path, default=Path("data/dataset3"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/mechanical_parts_roboflow")
    )
    parser.add_argument(
        "--existing-data",
        type=Path,
        default=Path("data/combined.yaml"),
        help="Existing combined YAML used to skip byte-identical images.",
    )
    parser.add_argument(
        "--benchmark-images",
        type=Path,
        default=Path("data/external_benchmark/images"),
        help=(
            "Current benchmark used only for overlap reporting. It becomes retired "
            "after training on this prepared source."
        ),
    )
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--max-variants-per-family", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 digest of one file.

    Parameters
    ----------
    path : Path
        File to hash.

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


def _roboflow_family(path: Path) -> str:
    """
    Remove the generated Roboflow content suffix from a filename.

    Parameters
    ----------
    path : Path
        Exported image path.

    Returns
    -------
    str
        Stable pre-augmentation filename family.
    """
    return ROBOFLOW_SUFFIX.sub("", path.stem)


def _benchmark_family(path: Path) -> str:
    """
    Normalize the existing benchmark filename to its original source family.

    Parameters
    ----------
    path : Path
        Existing benchmark image path.

    Returns
    -------
    str
        Family comparable with the raw Roboflow export.
    """
    stem = path.stem.removeprefix("mechanical_parts_")
    return ROBOFLOW_SUFFIX.sub("", stem)


def _read_source_names(source: Path) -> tuple[str, ...]:
    """
    Read and validate the source class order.

    Parameters
    ----------
    source : Path
        Source dataset root.

    Returns
    -------
    tuple[str, ...]
        Normalized class names in identifier order.

    Raises
    ------
    ValueError
        Raised when the dataset YAML or class order is incompatible.
    """
    data_path = source / "data.yaml"
    if not data_path.is_file():
        raise ValueError(f"Source data.yaml was not found: {data_path}")
    with data_path.open(encoding="utf-8") as file:
        document: dict[str, Any] = yaml.safe_load(file) or {}
    names = document.get("names")
    if isinstance(names, list):
        normalized = tuple(str(name).strip().casefold() for name in names)
    elif isinstance(names, dict):
        normalized = tuple(
            str(names[index]).strip().casefold() for index in range(len(names))
        )
    else:
        raise ValueError(f"Source class names are missing: {data_path}")
    if normalized != SOURCE_CLASSES:
        raise ValueError(
            f"Unexpected source classes: {normalized}; expected {SOURCE_CLASSES}"
        )
    return normalized


def _parse_boxes(label_path: Path) -> tuple[SourceBox, ...]:
    """
    Parse and validate one source YOLO label file.

    Parameters
    ----------
    label_path : Path
        Source label path.

    Returns
    -------
    tuple[SourceBox, ...]
        Validated annotations.

    Raises
    ------
    ValueError
        Raised for malformed classes, polygons, or box coordinates.
    """
    boxes: list[SourceBox] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(
                f"Malformed annotation at {label_path}:{line_number}; got {len(values)} values"
            )
        try:
            class_id = int(values[0])
            raw_coordinates = tuple(float(value) for value in values[1:])
        except ValueError as error:
            raise ValueError(
                f"Non-numeric annotation at {label_path}:{line_number}"
            ) from error
        if class_id < 0 or class_id >= len(SOURCE_CLASSES):
            raise ValueError(
                f"Unsupported class {class_id} at {label_path}:{line_number}"
            )
        if not all(0.0 <= value <= 1.0 for value in raw_coordinates):
            raise ValueError(
                f"Coordinates outside [0, 1] at {label_path}:{line_number}"
            )
        converted_from_polygon = len(values) > 5
        if converted_from_polygon:
            if len(raw_coordinates) < 6 or len(raw_coordinates) % 2:
                raise ValueError(f"Malformed polygon at {label_path}:{line_number}")
            x_values = raw_coordinates[0::2]
            y_values = raw_coordinates[1::2]
            x1, x2 = min(x_values), max(x_values)
            y1, y2 = min(y_values), max(y_values)
            coordinates = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
            )
        else:
            coordinates = raw_coordinates
        x_center, y_center, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise ValueError(f"Empty box at {label_path}:{line_number}")
        tolerance = 1e-6
        if (
            x_center - width / 2.0 < -tolerance
            or x_center + width / 2.0 > 1.0 + tolerance
            or y_center - height / 2.0 < -tolerance
            or y_center + height / 2.0 > 1.0 + tolerance
        ):
            raise ValueError(f"Box exceeds image bounds at {label_path}:{line_number}")
        boxes.append(SourceBox(class_id, coordinates, converted_from_polygon))
    return tuple(boxes)


def _source_images(source: Path) -> list[SourceImage]:
    """
    Enumerate every image-label pair in all publisher splits.

    Parameters
    ----------
    source : Path
        Source dataset root.

    Returns
    -------
    list[SourceImage]
        Sorted validated source records.

    Raises
    ------
    ValueError
        Raised when a split, image, or matching label is missing.
    """
    records: list[SourceImage] = []
    for split in SOURCE_SPLITS:
        image_root = source / split / "images"
        label_root = source / split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise ValueError(f"Expected source split folders below {source / split}")
        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
        for image_path in images:
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing source label for {image_path}")
            records.append(
                SourceImage(
                    image_path=image_path,
                    label_path=label_path,
                    declared_split=split,
                    family=_roboflow_family(image_path),
                    boxes=_parse_boxes(label_path),
                )
            )
    return records


def _declared_image_roots(data_path: Path) -> list[Path]:
    """
    Resolve all image roots declared by an existing combined YAML.

    Parameters
    ----------
    data_path : Path
        Existing dataset YAML.

    Returns
    -------
    list[Path]
        Unique existing image directories.
    """
    with data_path.open(encoding="utf-8") as file:
        document: dict[str, Any] = yaml.safe_load(file) or {}
    roots: list[Path] = []
    for split_key in ("train", "val", "test"):
        values = document.get(split_key, [])
        if not isinstance(values, list):
            values = [values]
        for value in values:
            path = Path(str(value))
            resolved = path if path.is_absolute() else data_path.parent / path
            resolved = resolved.resolve()
            if resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
    return roots


def _existing_digests(data_path: Path | None) -> set[str]:
    """
    Hash images already referenced by the current combined dataset.

    Parameters
    ----------
    data_path : Path or None
        Existing combined YAML, or ``None`` to disable cross-source deduplication.

    Returns
    -------
    set[str]
        SHA-256 digests already present in training data.
    """
    if data_path is None:
        return set()
    if not data_path.is_file():
        raise ValueError(f"Existing dataset YAML was not found: {data_path}")
    digests: set[str] = set()
    for image_root in _declared_image_roots(data_path):
        for path in image_root.iterdir():
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES:
                digests.add(_sha256(path))
    return digests


def _stable_split(family: str, validation_percent: int, seed: int) -> str:
    """
    Assign one original family to train or validation reproducibly.

    Parameters
    ----------
    family : str
        Original image-family identifier.
    validation_percent : int
        Percentage assigned to validation.
    seed : int
        Stable split seed.

    Returns
    -------
    str
        ``train`` or ``valid``.
    """
    identity = f"{seed}:{family}".encode("utf-8")
    bucket = int(hashlib.sha256(identity).hexdigest()[:8], 16) % 100
    return "valid" if bucket < validation_percent else "train"


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Safely create or replace the generated output directory.

    Parameters
    ----------
    output : Path
        Generated dataset directory.
    replace : bool
        Whether a previously marked generated directory may be replaced.
    """
    resolved = output.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT not in resolved.parents:
        raise ValueError(f"Output must stay below the project root: {resolved}")
    if output.exists():
        if not replace:
            raise FileExistsError(f"Output already exists; use --replace: {output}")
        if not (output / GENERATED_MARKER).is_file():
            raise ValueError(f"Refusing to replace an unmarked directory: {output}")
        shutil.rmtree(output)
    for split in OUTPUT_SPLITS:
        (output / split / "images").mkdir(parents=True)
        (output / split / "labels").mkdir(parents=True)
    (output / GENERATED_MARKER).write_text("generated\n", encoding="utf-8")


def _target_label_lines(boxes: tuple[SourceBox, ...]) -> list[str]:
    """
    Convert retained bolt/nut boxes to the project's two-class order.

    Parameters
    ----------
    boxes : tuple[SourceBox, ...]
        Source annotations.

    Returns
    -------
    list[str]
        YOLO detection-label lines for target objects only.
    """
    lines: list[str] = []
    for box in boxes:
        target_id = TARGET_CLASS_IDS.get(box.class_id)
        if target_id is None:
            continue
        coordinates = " ".join(f"{value:.10g}" for value in box.coordinates)
        lines.append(f"{target_id} {coordinates}")
    return lines


def _output_name(record: SourceImage, content_digest: str) -> str:
    """
    Build a stable collision-resistant output image name.

    Parameters
    ----------
    record : SourceImage
        Selected representative.
    content_digest : str
        SHA-256 digest of its image bytes.

    Returns
    -------
    str
        Output filename retaining the original extension.
    """
    family_digest = hashlib.sha256(record.family.encode("utf-8")).hexdigest()[:12]
    return (
        f"mechanical_parts_{family_digest}_{content_digest[:12]}"
        f"{record.image_path.suffix.casefold()}"
    )


def _write_metadata(
    output: Path,
    manifest: dict[str, Any],
    review_rows: list[dict[str, str | int]],
    ignored_records: list[dict[str, Any]],
) -> None:
    """
    Persist provenance, review worksheet, and ignored-object annotations.

    Parameters
    ----------
    output : Path
        Generated dataset root.
    manifest : dict
        Reproducibility and count metadata.
    review_rows : list[dict]
        Per-image audit worksheet rows.
    ignored_records : list[dict]
        Non-target boxes retained for future classifier ``other`` crops.
    """
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with (output / "review.csv").open("w", encoding="utf-8", newline="") as file:
        fields = (
            "split",
            "image",
            "source_family",
            "bolt_boxes",
            "nut_boxes",
            "ignored_boxes",
            "review_status",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    with (output / "ignored_boxes.jsonl").open("w", encoding="utf-8") as file:
        for record in ignored_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dataset(config: PreparationConfig) -> dict[str, Any]:
    """
    Convert, deduplicate, split, and persist the mechanical-parts source.

    Parameters
    ----------
    config : PreparationConfig
        Validated preparation options.

    Returns
    -------
    dict[str, Any]
        Generated manifest.
    """
    _read_source_names(config.source)
    records = _source_images(config.source)
    existing = _existing_digests(config.existing_data)
    benchmark_families = (
        {
            _benchmark_family(path)
            for path in config.benchmark_images.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        }
        if config.benchmark_images and config.benchmark_images.is_dir()
        else set()
    )

    grouped: defaultdict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    for family_records in grouped.values():
        family_records.sort(
            key=lambda item: (
                SOURCE_SPLITS.index(item.declared_split),
                item.image_path.name,
            )
        )

    _prepare_output(config.output, config.replace)
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in OUTPUT_SPLITS
    }
    ignored_class_counts: Counter[str] = Counter()
    review_rows: list[dict[str, str | int]] = []
    ignored_records: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    selected_families: set[str] = set()

    for family in sorted(grouped):
        retained = grouped[family][: config.max_variants_per_family]
        for record in retained:
            content_digest = _sha256(record.image_path)
            if content_digest in existing:
                skipped_existing.append(str(record.image_path.resolve()))
                continue
            split = _stable_split(family, config.validation_percent, config.seed)
            output_name = _output_name(record, content_digest)
            image_destination = config.output / split / "images" / output_name
            label_destination = (
                config.output / split / "labels" / f"{Path(output_name).stem}.txt"
            )
            shutil.copy2(record.image_path, image_destination)
            target_lines = _target_label_lines(record.boxes)
            label_destination.write_text(
                "\n".join(target_lines) + ("\n" if target_lines else ""),
                encoding="utf-8",
            )

            source_counts = Counter(
                SOURCE_CLASSES[box.class_id] for box in record.boxes
            )
            ignored_boxes = [
                {
                    "class": SOURCE_CLASSES[box.class_id],
                    "xywh": list(box.coordinates),
                }
                for box in record.boxes
                if box.class_id not in TARGET_CLASS_IDS
            ]
            for item in ignored_boxes:
                ignored_class_counts[item["class"]] += 1
            if ignored_boxes:
                ignored_records.append(
                    {
                        "split": split,
                        "image": f"{split}/images/{output_name}",
                        "source_family": family,
                        "boxes": ignored_boxes,
                    }
                )

            stats = split_counts[split]
            stats["images"] += 1
            stats["bolt_boxes"] += source_counts["bolt"]
            stats["nut_boxes"] += source_counts["nut"]
            stats["ignored_boxes"] += len(ignored_boxes)
            if target_lines:
                stats["positive_images"] += 1
            else:
                stats["negative_images"] += 1
            review_rows.append(
                {
                    "split": split,
                    "image": f"{split}/images/{output_name}",
                    "source_family": family,
                    "bolt_boxes": source_counts["bolt"],
                    "nut_boxes": source_counts["nut"],
                    "ignored_boxes": len(ignored_boxes),
                    "review_status": "pending",
                }
            )
            selected_families.add(family)

    yaml_document = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": list(TARGET_CLASS_NAMES),
    }
    (config.output / "data.yaml").write_text(
        yaml.safe_dump(yaml_document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "source": {
            "title": "mechanical parts detection",
            "workspace": "martin-3efbd",
            "project": "mechanical-parts-detection-glf33-waais",
            "version": 3,
            "url": (
                "https://universe.roboflow.com/martin-3efbd/"
                "mechanical-parts-detection-glf33-waais/dataset/3"
            ),
            "license": "CC BY 4.0",
            "raw_directory": str(config.source.resolve()),
            "source_classes": list(SOURCE_CLASSES),
        },
        "mapping": {
            "bolt": "bolt",
            "nut": "nut",
            "other_source_classes": "detector background; retained in ignored_boxes.jsonl",
        },
        "configuration": {
            "validation_percent": config.validation_percent,
            "max_variants_per_family": config.max_variants_per_family,
            "seed": config.seed,
            "existing_data": (
                str(config.existing_data.resolve()) if config.existing_data else None
            ),
        },
        "raw_images": len(records),
        "source_polygon_annotations_converted": sum(
            box.converted_from_polygon for record in records for box in record.boxes
        ),
        "source_families": len(grouped),
        "multi_variant_families": sum(
            1 for family_records in grouped.values() if len(family_records) > 1
        ),
        "offline_variants_not_retained": sum(
            max(0, len(family_records) - config.max_variants_per_family)
            for family_records in grouped.values()
        ),
        "byte_duplicates_skipped_against_existing_data": len(skipped_existing),
        "selected_families": len(selected_families),
        "split_counts": {
            split: dict(sorted(split_counts[split].items())) for split in OUTPUT_SPLITS
        },
        "ignored_class_boxes": dict(sorted(ignored_class_counts.items())),
        "retiring_benchmark": {
            "path": (
                str(config.benchmark_images.resolve())
                if config.benchmark_images and config.benchmark_images.is_dir()
                else None
            ),
            "matching_source_families": len(set(grouped) & benchmark_families),
            "policy": (
                "The current benchmark must not be used for model selection after "
                "training on this prepared dataset. Build benchmark v2 for A/B comparison."
            ),
        },
        "review": {
            "status": "pending",
            "worksheet": "review.csv",
        },
    }
    _write_metadata(config.output, manifest, review_rows, ignored_records)
    return manifest


def parse_config() -> PreparationConfig:
    """
    Parse and validate preparation configuration.

    Returns
    -------
    PreparationConfig
        Validated immutable configuration.
    """
    arguments = build_parser().parse_args()
    if not arguments.source.is_dir():
        raise ValueError(f"Source dataset was not found: {arguments.source}")
    if not 1 <= arguments.validation_percent <= 50:
        raise ValueError("validation-percent must be between 1 and 50")
    if arguments.max_variants_per_family <= 0:
        raise ValueError("max-variants-per-family must be positive")
    return PreparationConfig(
        source=arguments.source,
        output=arguments.output,
        existing_data=arguments.existing_data,
        benchmark_images=arguments.benchmark_images,
        validation_percent=arguments.validation_percent,
        max_variants_per_family=arguments.max_variants_per_family,
        seed=arguments.seed,
        replace=arguments.replace,
    )


def main() -> None:
    """Prepare the source and print the generated manifest."""
    manifest = prepare_dataset(parse_config())
    print(json.dumps(manifest, indent=2))
    print("Dataset preparation is complete. Training has not been started.")


if __name__ == "__main__":
    main()
