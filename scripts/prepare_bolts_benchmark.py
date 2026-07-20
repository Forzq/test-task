"""Prepare a source-disjoint bolt/nut benchmark from the local BOLTS export."""

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

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_SPLITS = ("test", "valid", "train")
SOURCE_CLASSES = ("bolt", "lockwasher", "nut", "other", "screw", "washer")
TARGET_CLASS_IDS = {0: 0, 2: 1}
TARGET_CLASS_NAMES = ("bolt", "nut")
GENERATED_MARKER = ".generated-bolts-benchmark"
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{32}$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SourceBox:
    """
    One validated source annotation.

    Parameters
    ----------
    class_id : int
        Source class identifier.
    coordinates : tuple[float, float, float, float]
        Normalized YOLO ``x_center, y_center, width, height`` coordinates.
    converted_from_polygon : bool
        Whether a source polygon was converted to an enclosing rectangle.
    """

    class_id: int
    coordinates: tuple[float, float, float, float]
    converted_from_polygon: bool = False


@dataclass(frozen=True, slots=True)
class SourceImage:
    """
    One source image and its annotations.

    Parameters
    ----------
    image_path : Path
        Source image path.
    label_path : Path
        Matching source label path.
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
class BenchmarkConfig:
    """
    Configuration for generating the frozen benchmark candidate.

    Parameters
    ----------
    source : Path
        Local six-class BOLTS Roboflow export.
    output : Path
        Generated benchmark directory.
    training_data : Path or None
        Combined dataset YAML whose images must not overlap the benchmark.
    additional_exclusion_roots : tuple[Path, ...]
        Additional raw source roots checked for perceptual overlap.
    perceptual_distance : int
        Maximum pHash Hamming distance treated as a training overlap.
    max_variants_per_family : int
        Maximum retained variants of one Roboflow source family.
    replace : bool
        Whether a marked generated output may be replaced.
    manual_exclusions : Path or None
        Reviewed source families excluded because their annotations are ambiguous
        or objectively incorrect.
    """

    source: Path
    output: Path
    training_data: Path | None
    additional_exclusion_roots: tuple[Path, ...]
    perceptual_distance: int
    max_variants_per_family: int
    replace: bool
    manual_exclusions: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for benchmark preparation.

    Returns
    -------
    argparse.ArgumentParser
        Parser with project-local defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert the local BOLTS export into a two-class, source-disjoint "
            "test-only benchmark with review sheets and provenance."
        )
    )
    parser.add_argument("--source", type=Path, default=Path("data/benchmark-dataset"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/external_benchmark_v2")
    )
    parser.add_argument(
        "--training-data", type=Path, default=Path("data/combined_dataset3.yaml")
    )
    parser.add_argument(
        "--additional-exclusion-root",
        type=Path,
        action="append",
        default=[Path("data/dataset3")],
        help="Additional raw image tree checked for source overlap; repeatable.",
    )
    parser.add_argument("--perceptual-distance", type=int, default=4)
    parser.add_argument("--max-variants-per-family", type=int, default=1)
    parser.add_argument(
        "--manual-exclusions",
        type=Path,
        default=None,
        help="JSON object mapping manually rejected source families to reasons.",
    )
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


def _perceptual_hash(path: Path) -> int:
    """
    Calculate a 64-bit DCT perceptual hash for one image.

    Parameters
    ----------
    path : Path
        Image path.

    Returns
    -------
    int
        Integer bit representation of the perceptual hash.
    """
    with Image.open(path) as opened:
        grayscale = ImageOps.grayscale(opened).resize(
            (32, 32), Image.Resampling.LANCZOS
        )
    pixels = np.asarray(grayscale, dtype=np.float32)
    low_frequency = cv2.dct(pixels)[:8, :8]
    values = low_frequency.flatten()
    median = float(np.median(values[1:]))
    result = 0
    for value in values:
        result = (result << 1) | int(value > median)
    return result


def _hash_distance(first: int, second: int) -> int:
    """
    Return the Hamming distance between two perceptual hashes.

    Parameters
    ----------
    first : int
        First perceptual hash.
    second : int
        Second perceptual hash.

    Returns
    -------
    int
        Number of differing bits.
    """
    return (first ^ second).bit_count()


def _family(path: Path) -> str:
    """
    Return the pre-augmentation Roboflow filename family.

    Parameters
    ----------
    path : Path
        Source image path.

    Returns
    -------
    str
        Stable family name.
    """
    return ROBOFLOW_SUFFIX.sub("", path.stem)


def _read_source_classes(source: Path) -> None:
    """
    Validate the immutable source class order and license metadata.

    Parameters
    ----------
    source : Path
        Source dataset directory.

    Raises
    ------
    ValueError
        Raised when metadata does not match the expected public dataset version.
    """
    data_path = source / "data.yaml"
    if not data_path.is_file():
        raise ValueError(f"Source data.yaml was not found: {data_path}")
    document = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    names = document.get("names")
    if not isinstance(names, list):
        raise ValueError("The BOLTS source must declare list-style class names")
    normalized = tuple(str(name).strip().casefold() for name in names)
    if normalized != SOURCE_CLASSES:
        raise ValueError(
            f"Unexpected source classes: {normalized}; expected {SOURCE_CLASSES}"
        )
    roboflow = document.get("roboflow") or {}
    expected = {
        "workspace": "fastener-urest",
        "project": "bolts-v5qf3",
        "version": 1,
        "license": "CC BY 4.0",
    }
    for key, value in expected.items():
        if roboflow.get(key) != value:
            raise ValueError(
                f"Unexpected Roboflow metadata {key}={roboflow.get(key)!r}; expected {value!r}"
            )


def _parse_boxes(label_path: Path) -> tuple[SourceBox, ...]:
    """
    Parse detection boxes and convert source polygons to enclosing boxes.

    Parameters
    ----------
    label_path : Path
        Source YOLO label path.

    Returns
    -------
    tuple[SourceBox, ...]
        Validated source annotations.
    """
    boxes: list[SourceBox] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Malformed annotation at {label_path}:{line_number}")
        try:
            class_id = int(values[0])
            raw = tuple(float(value) for value in values[1:])
        except ValueError as error:
            raise ValueError(
                f"Non-numeric annotation at {label_path}:{line_number}"
            ) from error
        if class_id < 0 or class_id >= len(SOURCE_CLASSES):
            raise ValueError(f"Unsupported class at {label_path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in raw):
            raise ValueError(f"Coordinates outside [0, 1] at {label_path}:{line_number}")
        polygon = len(raw) > 4
        if polygon:
            if len(raw) < 6 or len(raw) % 2:
                raise ValueError(f"Malformed polygon at {label_path}:{line_number}")
            x_values, y_values = raw[0::2], raw[1::2]
            x1, x2 = min(x_values), max(x_values)
            y1, y2 = min(y_values), max(y_values)
            coordinates = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                x2 - x1,
                y2 - y1,
            )
        else:
            coordinates = raw
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
        boxes.append(SourceBox(class_id, coordinates, polygon))
    return tuple(boxes)


def _source_images(source: Path) -> list[SourceImage]:
    """
    Enumerate and validate every source image-label pair.

    Parameters
    ----------
    source : Path
        Source dataset directory.

    Returns
    -------
    list[SourceImage]
        Validated source records.
    """
    records: list[SourceImage] = []
    for split in SOURCE_SPLITS:
        image_root = source / split / "images"
        label_root = source / split / "labels"
        if not image_root.is_dir() or not label_root.is_dir():
            raise ValueError(f"Missing source folders below {source / split}")
        for image_path in sorted(image_root.iterdir()):
            if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_SUFFIXES:
                continue
            label_path = label_root / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise ValueError(f"Missing source label for {image_path}")
            records.append(
                SourceImage(
                    image_path=image_path,
                    label_path=label_path,
                    declared_split=split,
                    family=_family(image_path),
                    boxes=_parse_boxes(label_path),
                )
            )
    return records


def _image_roots_from_yaml(data_path: Path | None) -> list[Path]:
    """
    Resolve image directories declared by an existing dataset YAML.

    Parameters
    ----------
    data_path : Path or None
        Dataset YAML or ``None``.

    Returns
    -------
    list[Path]
        Existing unique image roots.
    """
    if data_path is None:
        return []
    if not data_path.is_file():
        raise ValueError(f"Training data YAML was not found: {data_path}")
    document: dict[str, Any] = yaml.safe_load(
        data_path.read_text(encoding="utf-8")
    ) or {}
    dataset_root = data_path.parent
    if document.get("path"):
        declared_root = Path(str(document["path"]))
        dataset_root = (
            declared_root if declared_root.is_absolute() else dataset_root / declared_root
        )
    roots: list[Path] = []
    for split in ("train", "val", "valid", "test"):
        declared = document.get(split)
        values = declared if isinstance(declared, list) else [declared]
        for value in values:
            if value is None:
                continue
            path = Path(str(value))
            resolved = (path if path.is_absolute() else dataset_root / path).resolve()
            if resolved.is_dir() and resolved not in roots:
                roots.append(resolved)
    return roots


def _images_below(root: Path) -> list[Path]:
    """
    Recursively enumerate supported images below one root.

    Parameters
    ----------
    root : Path
        Image tree or direct image directory.

    Returns
    -------
    list[Path]
        Sorted supported image paths.
    """
    if not root.is_dir():
        raise ValueError(f"Exclusion image root was not found: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in IMAGE_SUFFIXES
        and "images" in {part.casefold() for part in path.parts}
    )


def _exclusion_index(config: BenchmarkConfig) -> tuple[dict[str, Path], list[tuple[int, Path]]]:
    """
    Build exact and perceptual hash indices for every training source.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration.

    Returns
    -------
    tuple[dict[str, Path], list[tuple[int, Path]]]
        Exact digest map and perceptual-hash records.
    """
    paths: set[Path] = set()
    for root in _image_roots_from_yaml(config.training_data):
        paths.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
    for root in config.additional_exclusion_roots:
        paths.update(path.resolve() for path in _images_below(root))
    exact: dict[str, Path] = {}
    perceptual: list[tuple[int, Path]] = []
    for path in sorted(paths):
        exact.setdefault(_sha256(path), path)
        perceptual.append((_perceptual_hash(path), path))
    return exact, perceptual


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Safely create or replace a generated benchmark directory.

    Parameters
    ----------
    output : Path
        Generated benchmark directory.
    replace : bool
        Whether a marked output may be removed.
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
    (output / "test" / "images").mkdir(parents=True)
    (output / "test" / "labels").mkdir(parents=True)
    (output / "review_sheets").mkdir(parents=True)
    (output / GENERATED_MARKER).write_text("generated\n", encoding="utf-8")


def _target_lines(boxes: tuple[SourceBox, ...]) -> list[str]:
    """
    Map target source boxes to the project's class identifiers.

    Parameters
    ----------
    boxes : tuple[SourceBox, ...]
        Source annotations.

    Returns
    -------
    list[str]
        Two-class YOLO label lines.
    """
    lines: list[str] = []
    for box in boxes:
        target_id = TARGET_CLASS_IDS.get(box.class_id)
        if target_id is None:
            continue
        values = " ".join(f"{value:.10g}" for value in box.coordinates)
        lines.append(f"{target_id} {values}")
    return lines


def _output_name(record: SourceImage, digest: str) -> str:
    """
    Build a stable benchmark image filename.

    Parameters
    ----------
    record : SourceImage
        Selected source representative.
    digest : str
        SHA-256 digest of its bytes.

    Returns
    -------
    str
        Collision-resistant output filename.
    """
    family_digest = hashlib.sha256(record.family.encode("utf-8")).hexdigest()[:12]
    return f"benchmark_v2_{family_digest}_{digest[:12]}{record.image_path.suffix.casefold()}"


def _draw_review_tile(
    image_path: Path,
    boxes: tuple[SourceBox, ...],
    caption: str,
    tile_size: int = 300,
) -> Image.Image:
    """
    Draw source annotations on a square contact-sheet tile.

    Parameters
    ----------
    image_path : Path
        Source image path.
    boxes : tuple[SourceBox, ...]
        Source annotations to draw.
    caption : str
        Short review identifier.
    tile_size : int
        Edge size reserved for image pixels.

    Returns
    -------
    PIL.Image.Image
        Annotated tile with caption.
    """
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    canvas = Image.new("RGB", (tile_size, tile_size + 28), "white")
    fitted = ImageOps.contain(source, (tile_size, tile_size))
    offset_x = (tile_size - fitted.width) // 2
    offset_y = (tile_size - fitted.height) // 2
    canvas.paste(fitted, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    colors = {0: "#ff3030", 2: "#00b050"}
    for box in boxes:
        x_center, y_center, width, height = box.coordinates
        x1 = offset_x + (x_center - width / 2.0) * fitted.width
        y1 = offset_y + (y_center - height / 2.0) * fitted.height
        x2 = offset_x + (x_center + width / 2.0) * fitted.width
        y2 = offset_y + (y_center + height / 2.0) * fitted.height
        color = colors.get(box.class_id, "#ffd000")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((max(0, x1), max(0, y1 - 12)), SOURCE_CLASSES[box.class_id], fill=color)
    draw.text((4, tile_size + 5), caption, fill="black", font=ImageFont.load_default())
    return canvas


def _write_review_sheets(
    output: Path, selected: list[tuple[SourceImage, str]], columns: int = 4, rows: int = 4
) -> list[str]:
    """
    Create paginated visual review sheets covering every retained image.

    Parameters
    ----------
    output : Path
        Generated benchmark root.
    selected : list[tuple[SourceImage, str]]
        Selected records and output names.
    columns : int
        Number of tiles per row.
    rows : int
        Number of tile rows per sheet.

    Returns
    -------
    list[str]
        Relative paths of written review sheets.
    """
    per_sheet = columns * rows
    tile_width, tile_height = 300, 328
    paths: list[str] = []
    for page, start in enumerate(range(0, len(selected), per_sheet), start=1):
        sheet = Image.new(
            "RGB", (columns * tile_width, rows * tile_height), "#dddddd"
        )
        for local_index, (record, output_name) in enumerate(
            selected[start : start + per_sheet]
        ):
            absolute_index = start + local_index + 1
            caption = f"{absolute_index:03d} {Path(output_name).stem[-12:]}"
            tile = _draw_review_tile(record.image_path, record.boxes, caption)
            x = (local_index % columns) * tile_width
            y = (local_index // columns) * tile_height
            sheet.paste(tile, (x, y))
        path = output / "review_sheets" / f"sheet_{page:02d}.jpg"
        sheet.save(path, format="JPEG", quality=92)
        paths.append(path.relative_to(output).as_posix())
    return paths


def _read_manual_exclusions(path: Path | None) -> dict[str, str]:
    """
    Read reviewed source-family exclusions.

    Parameters
    ----------
    path : Path or None
        JSON object whose keys are source families and values are review reasons.

    Returns
    -------
    dict[str, str]
        Validated family-to-reason mapping.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"Manual exclusion file was not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not all(
        isinstance(family, str) and family.strip() and isinstance(reason, str)
        and reason.strip()
        for family, reason in document.items()
    ):
        raise ValueError("Manual exclusions must be a JSON object of string reasons")
    return {family.strip(): reason.strip() for family, reason in document.items()}


def prepare_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """
    Generate a deduplicated, source-disjoint two-class benchmark candidate.

    Parameters
    ----------
    config : BenchmarkConfig
        Validated benchmark configuration.

    Returns
    -------
    dict[str, Any]
        Benchmark manifest and audit counts.
    """
    _read_source_classes(config.source)
    records = _source_images(config.source)
    manual_exclusions = _read_manual_exclusions(config.manual_exclusions)
    grouped: defaultdict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        grouped[record.family].append(record)
    for family_records in grouped.values():
        family_records.sort(
            key=lambda item: (
                SOURCE_SPLITS.index(item.declared_split), item.image_path.name
            )
        )

    exact_exclusions, perceptual_exclusions = _exclusion_index(config)
    _prepare_output(config.output, config.replace)
    selected: list[tuple[SourceImage, str]] = []
    selected_hashes: list[tuple[int, SourceImage]] = []
    excluded_training: list[dict[str, Any]] = []
    excluded_internal: list[dict[str, Any]] = []
    excluded_manual: list[dict[str, str]] = []
    review_rows: list[dict[str, str | int]] = []
    ignored_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    image_digests: list[str] = []
    label_digests: list[str] = []

    for family in sorted(grouped):
        if family in manual_exclusions:
            excluded_manual.append(
                {"family": family, "reason": manual_exclusions[family]}
            )
            continue
        for record in grouped[family][: config.max_variants_per_family]:
            digest = _sha256(record.image_path)
            perceptual = _perceptual_hash(record.image_path)
            exact_path = exact_exclusions.get(digest)
            nearest_distance = 65
            nearest_path: Path | None = None
            for candidate_hash, candidate_path in perceptual_exclusions:
                distance = _hash_distance(perceptual, candidate_hash)
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_path = candidate_path
                    if distance == 0:
                        break
            if exact_path or nearest_distance <= config.perceptual_distance:
                excluded_training.append(
                    {
                        "source": str(record.image_path.resolve()),
                        "family": family,
                        "reason": "exact" if exact_path else "perceptual",
                        "distance": 0 if exact_path else nearest_distance,
                        "matched": str((exact_path or nearest_path).resolve()),
                    }
                )
                continue
            internal_match: SourceImage | None = None
            internal_distance = 65
            for selected_hash, selected_record in selected_hashes:
                distance = _hash_distance(perceptual, selected_hash)
                if distance < internal_distance:
                    internal_distance = distance
                    internal_match = selected_record
            if internal_match is not None and internal_distance <= 2:
                excluded_internal.append(
                    {
                        "source": str(record.image_path.resolve()),
                        "family": family,
                        "distance": internal_distance,
                        "matched": str(internal_match.image_path.resolve()),
                    }
                )
                continue

            output_name = _output_name(record, digest)
            image_destination = config.output / "test" / "images" / output_name
            label_destination = (
                config.output / "test" / "labels" / f"{Path(output_name).stem}.txt"
            )
            shutil.copy2(record.image_path, image_destination)
            lines = _target_lines(record.boxes)
            label_destination.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            image_digests.append(_sha256(image_destination))
            label_digests.append(_sha256(label_destination))
            selected_hashes.append((perceptual, record))
            selected.append((record, output_name))

            source_class_counts = Counter(
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
                ignored_counts[item["class"]] += 1
            if ignored_boxes:
                ignored_records.append(
                    {
                        "image": f"test/images/{output_name}",
                        "source_family": family,
                        "boxes": ignored_boxes,
                    }
                )
            counts["images"] += 1
            counts["bolt_boxes"] += source_class_counts["bolt"]
            counts["nut_boxes"] += source_class_counts["nut"]
            counts["ignored_boxes"] += len(ignored_boxes)
            if lines:
                counts["positive_images"] += 1
            else:
                counts["negative_images"] += 1
            review_rows.append(
                {
                    "index": len(selected),
                    "image": f"test/images/{output_name}",
                    "source_family": family,
                    "bolt_boxes": source_class_counts["bolt"],
                    "nut_boxes": source_class_counts["nut"],
                    "ignored_boxes": len(ignored_boxes),
                    "nearest_training_phash_distance": nearest_distance,
                    "review_status": "pending",
                    "review_notes": "",
                }
            )

    benchmark_yaml = {
        "path": ".",
        "train": None,
        "val": None,
        "test": "test/images",
        "names": list(TARGET_CLASS_NAMES),
    }
    (config.output / "benchmark.yaml").write_text(
        yaml.safe_dump(benchmark_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    with (config.output / "ignored_boxes.jsonl").open("w", encoding="utf-8") as file:
        for item in ignored_records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    fields = (
        "index",
        "image",
        "source_family",
        "bolt_boxes",
        "nut_boxes",
        "ignored_boxes",
        "nearest_training_phash_distance",
        "review_status",
        "review_notes",
    )
    with (config.output / "review.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    (config.output / "excluded_training_overlaps.json").write_text(
        json.dumps(excluded_training, indent=2), encoding="utf-8"
    )
    (config.output / "excluded_internal_duplicates.json").write_text(
        json.dumps(excluded_internal, indent=2), encoding="utf-8"
    )
    (config.output / "excluded_manual_quality.json").write_text(
        json.dumps(excluded_manual, indent=2), encoding="utf-8"
    )
    review_sheets = _write_review_sheets(config.output, selected)

    fingerprint = hashlib.sha256(
        "\n".join(sorted(image_digests + label_digests)).encode("utf-8")
    ).hexdigest().upper()
    manifest: dict[str, Any] = {
        "benchmark_policy": {
            "purpose": "source-disjoint final evaluation",
            "included_in_training": False,
            "included_in_threshold_calibration": False,
            "selection_used_model_predictions": False,
            "frozen": False,
        },
        "source": {
            "title": "BOLTS",
            "workspace": "fastener-urest",
            "project": "bolts-v5qf3",
            "version": 1,
            "url": "https://universe.roboflow.com/fastener-urest/bolts-v5qf3/dataset/1",
            "license": "CC BY 4.0",
            "raw_images": len(records),
            "source_families": len(grouped),
        },
        "mapping": {
            "bolt": "bolt",
            "nut": "nut",
            "lockwasher": "background",
            "other": "background",
            "screw": "background",
            "washer": "background",
        },
        "deduplication": {
            "max_variants_per_family": config.max_variants_per_family,
            "offline_family_variants_removed": sum(
                max(0, len(items) - config.max_variants_per_family)
                for items in grouped.values()
            ),
            "perceptual_hash_distance": config.perceptual_distance,
            "training_overlaps_removed": len(excluded_training),
            "internal_near_duplicates_removed": len(excluded_internal),
            "manual_quality_exclusions": len(excluded_manual),
            "manual_exclusion_file": (
                str(config.manual_exclusions.resolve())
                if config.manual_exclusions
                else None
            ),
            "training_data": (
                str(config.training_data.resolve()) if config.training_data else None
            ),
            "additional_exclusion_roots": [
                str(path.resolve()) for path in config.additional_exclusion_roots
            ],
        },
        "source_polygon_annotations_converted": sum(
            box.converted_from_polygon for record in records for box in record.boxes
        ),
        "counts": dict(sorted(counts.items())),
        "ignored_class_boxes": dict(sorted(ignored_counts.items())),
        "dataset_fingerprint_sha256": fingerprint,
        "review": {
            "status": "pending",
            "worksheet": "review.csv",
            "sheets": review_sheets,
        },
    }
    (config.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def parse_config() -> BenchmarkConfig:
    """
    Parse and validate benchmark preparation options.

    Returns
    -------
    BenchmarkConfig
        Validated immutable configuration.
    """
    arguments = build_parser().parse_args()
    if not arguments.source.is_dir():
        raise ValueError(f"Source dataset was not found: {arguments.source}")
    if not 0 <= arguments.perceptual_distance <= 16:
        raise ValueError("perceptual-distance must be between 0 and 16")
    if arguments.max_variants_per_family <= 0:
        raise ValueError("max-variants-per-family must be positive")
    manual_exclusions = arguments.manual_exclusions
    default_exclusions = arguments.source / "benchmark_exclusions.json"
    if manual_exclusions is None and default_exclusions.is_file():
        manual_exclusions = default_exclusions
    return BenchmarkConfig(
        source=arguments.source,
        output=arguments.output,
        training_data=arguments.training_data,
        additional_exclusion_roots=tuple(arguments.additional_exclusion_root),
        perceptual_distance=arguments.perceptual_distance,
        max_variants_per_family=arguments.max_variants_per_family,
        replace=arguments.replace,
        manual_exclusions=manual_exclusions,
    )


def main() -> None:
    """Prepare the benchmark candidate without running model inference."""
    manifest = prepare_benchmark(parse_config())
    print(json.dumps(manifest, indent=2))
    print("Benchmark preparation is complete. Model inference has not been run.")


if __name__ == "__main__":
    main()
