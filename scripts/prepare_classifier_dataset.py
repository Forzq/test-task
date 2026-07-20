"""Build a bolt, nut, and other crop-classification dataset from YOLO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.image_utils import extract_square_crop

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "bolt", 1: "nut"}
OUTPUT_CLASSES = ("bolt", "nut", "other")
OUTPUT_SPLITS = ("train", "val", "test")
GENERATED_MARKER = ".generated-classifier-dataset"


@dataclass(frozen=True, slots=True)
class CropCandidate:
    """
    A source image region selected for crop-classifier training.

    Parameters
    ----------
    source : Path
        Original image path.
    bbox : tuple[float, float, float, float]
        Crop region in source-image pixel coordinates.
    class_name : str
        Target class: ``bolt``, ``nut``, or ``other``.
    split : str
        Output split: ``train``, ``val``, or ``test``.
    origin : str
        Selection method used for provenance reporting.
    context_ratio : float
        Context added around the candidate when writing its crop.
    """

    source: Path
    bbox: tuple[float, float, float, float]
    class_name: str
    split: str
    origin: str
    context_ratio: float = 0.20


@dataclass(frozen=True, slots=True)
class PreparationConfig:
    """
    Immutable settings for classifier dataset generation.

    Parameters
    ----------
    data : Path
        Combined YOLO detection dataset YAML.
    weights : Path
        Detector checkpoint used to mine false candidate regions.
    output : Path
        Destination classification dataset root.
    image_size : int
        Saved square crop size.
    context_ratios : tuple[float, ...]
        Tight and contextual crop ratios generated per candidate.
    random_crops_per_negative : int
        Random ``other`` crops generated per known-negative image.
    proposals_per_negative : int
        Maximum low-confidence detector proposals retained per negative image.
    proposal_confidence : float
        Low detector confidence used for hard-negative proposal mining.
    max_train_per_class : int
        Maximum balanced training crops retained per class.
    max_eval_per_class : int
        Maximum balanced validation and test crops retained per class.
    seed : int
        Reproducible sampling seed.
    device : str
        Ultralytics inference device.
    batch_size : int
        Detector inference batch size.
    chunk_size : int
        Maximum negative images submitted in one inference call.
    replace : bool
        Whether a previously generated output may be replaced.
    """

    data: Path
    weights: Path
    output: Path
    image_size: int
    context_ratios: tuple[float, ...]
    random_crops_per_negative: int
    proposals_per_negative: int
    proposal_confidence: float
    max_train_per_class: int
    max_eval_per_class: int
    seed: int
    device: str
    batch_size: int
    chunk_size: int
    replace: bool


def build_parser() -> argparse.ArgumentParser:
    """
    Create the classifier-dataset command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with safe project-local defaults.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced bolt/nut/other crop dataset and mine detector "
            "proposals from known-negative images."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("data/classifier"))
    parser.add_argument("--imgsz", type=int, default=224, help="Saved crop edge size.")
    parser.add_argument(
        "--contexts",
        type=float,
        nargs="+",
        default=(0.20, 1.00),
        help="Crop context ratios, for example: --contexts 0.2 1.0",
    )
    parser.add_argument("--random-crops", type=int, default=4)
    parser.add_argument("--proposals", type=int, default=5)
    parser.add_argument("--proposal-confidence", type=float, default=0.01)
    parser.add_argument("--max-train-per-class", type=int, default=4000)
    parser.add_argument("--max-eval-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true")
    return parser


def parse_config() -> PreparationConfig:
    """
    Parse and validate classifier-dataset options.

    Returns
    -------
    PreparationConfig
        Validated immutable configuration.
    """
    arguments = build_parser().parse_args()
    config = PreparationConfig(
        data=arguments.data,
        weights=arguments.weights,
        output=arguments.output,
        image_size=arguments.imgsz,
        context_ratios=tuple(dict.fromkeys(arguments.contexts)),
        random_crops_per_negative=arguments.random_crops,
        proposals_per_negative=arguments.proposals,
        proposal_confidence=arguments.proposal_confidence,
        max_train_per_class=arguments.max_train_per_class,
        max_eval_per_class=arguments.max_eval_per_class,
        seed=arguments.seed,
        device=arguments.device,
        batch_size=arguments.batch,
        chunk_size=arguments.chunk_size,
        replace=arguments.replace,
    )
    _validate_config(config)
    return config


def _validate_config(config: PreparationConfig) -> None:
    """
    Validate paths and numeric dataset-preparation parameters.

    Parameters
    ----------
    config : PreparationConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        Raised for missing inputs or invalid numeric values.
    """
    if not config.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {config.data}")
    if not config.weights.is_file():
        raise ValueError(f"Detector weights were not found: {config.weights}")
    if min(
        config.image_size,
        config.proposals_per_negative,
        config.max_train_per_class,
        config.max_eval_per_class,
        config.batch_size,
        config.chunk_size,
    ) <= 0:
        raise ValueError("sizes, limits, batch, and chunk-size must be positive")
    if config.random_crops_per_negative < 0:
        raise ValueError("random-crops cannot be negative")
    if not config.context_ratios or any(value < 0.0 for value in config.context_ratios):
        raise ValueError("contexts must contain non-negative values")
    if not 0.0 < config.proposal_confidence <= 1.0:
        raise ValueError("proposal-confidence must be in (0, 1]")


def _dataset_entries(data_path: Path) -> dict[str, list[Path]]:
    """
    Resolve every detection image directory declared by a YOLO YAML file.

    Parameters
    ----------
    data_path : Path
        Combined YOLO dataset YAML.

    Returns
    -------
    dict[str, list[Path]]
        Resolved image directories keyed by output split.
    """
    document: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = data_path.parent
    if document.get("path"):
        declared_root = Path(str(document["path"]))
        dataset_root = declared_root if declared_root.is_absolute() else dataset_root / declared_root

    entries: dict[str, list[Path]] = {}
    for split in OUTPUT_SPLITS:
        declared = document.get(split)
        if declared is None and split == "val":
            declared = document.get("valid")
        values = declared if isinstance(declared, list) else [declared]
        directories: list[Path] = []
        for value in values:
            if value is None:
                continue
            path = Path(str(value))
            directories.append((path if path.is_absolute() else dataset_root / path).resolve())
        entries[split] = directories
    return entries


def _image_paths(root: Path) -> list[Path]:
    """
    Enumerate supported image files below one declared image directory.

    Parameters
    ----------
    root : Path
        Image directory.

    Returns
    -------
    list[Path]
        Sorted image paths.
    """
    if not root.is_dir():
        raise ValueError(f"Declared image directory was not found: {root}")
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _label_path(image_path: Path, image_root: Path) -> Path:
    """
    Resolve the YOLO label paired with one image.

    Parameters
    ----------
    image_path : Path
        Image below the declared image root.
    image_root : Path
        Directory corresponding to a sibling ``labels`` directory.

    Returns
    -------
    Path
        Expected text-label path.
    """
    relative = image_path.relative_to(image_root).with_suffix(".txt")
    return image_root.parent / "labels" / relative


def _parse_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[str, tuple[float, float, float, float]]]:
    """
    Convert YOLO box or polygon labels to pixel-coordinate rectangles.

    Parameters
    ----------
    label_path : Path
        YOLO label text file.
    width : int
        Source image width.
    height : int
        Source image height.

    Returns
    -------
    list[tuple[str, tuple[float, float, float, float]]]
        Canonical class names and pixel bounding boxes.
    """
    if not label_path.is_file():
        raise ValueError(f"Missing YOLO label: {label_path}")
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            values = [float(value) for value in raw_line.split()]
            class_id = int(values[0])
            if class_id not in CLASS_NAMES or values[0] != class_id:
                raise ValueError("unsupported class id")
            coordinates = values[1:]
            if len(coordinates) == 4:
                centre_x, centre_y, box_width, box_height = coordinates
                x1 = (centre_x - box_width / 2.0) * width
                y1 = (centre_y - box_height / 2.0) * height
                x2 = (centre_x + box_width / 2.0) * width
                y2 = (centre_y + box_height / 2.0) * height
            elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
                x_values = coordinates[0::2]
                y_values = coordinates[1::2]
                x1, x2 = min(x_values) * width, max(x_values) * width
                y1, y2 = min(y_values) * height, max(y_values) * height
            else:
                raise ValueError("expected one box or polygon")
        except (IndexError, ValueError) as error:
            raise ValueError(f"Malformed label {label_path}:{line_number}") from error

        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 >= 2.0 and y2 - y1 >= 2.0:
            boxes.append((CLASS_NAMES[class_id], (x1, y1, x2, y2)))
    return boxes


def _collect_source_examples(
    config: PreparationConfig,
) -> tuple[dict[str, dict[str, list[CropCandidate]]], dict[str, list[Path]]]:
    """
    Collect annotated positive boxes and known-negative images.

    Parameters
    ----------
    config : PreparationConfig
        Dataset-generation settings.

    Returns
    -------
    tuple[dict[str, dict[str, list[CropCandidate]]], dict[str, list[Path]]]
        Positive candidates by split/class and negative paths by split.
    """
    positives = {
        split: {"bolt": [], "nut": []} for split in OUTPUT_SPLITS
    }
    negatives = {split: [] for split in OUTPUT_SPLITS}
    seen: set[Path] = set()

    for split, image_roots in _dataset_entries(config.data).items():
        for image_root in image_roots:
            for image_path in _image_paths(image_root):
                resolved_image = image_path.resolve()
                if resolved_image in seen:
                    continue
                seen.add(resolved_image)
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes = _parse_yolo_boxes(_label_path(image_path, image_root), width, height)
                if not boxes:
                    negatives[split].append(resolved_image)
                    continue
                for class_name, bbox in boxes:
                    positives[split][class_name].append(
                        CropCandidate(resolved_image, bbox, class_name, split, "annotation")
                    )
    return positives, negatives


def _random_other_candidates(
    negatives: dict[str, list[Path]], config: PreparationConfig
) -> dict[str, list[CropCandidate]]:
    """
    Generate deterministic full-image and random crops from negative images.

    Parameters
    ----------
    negatives : dict[str, list[Path]]
        Known-negative image paths by split.
    config : PreparationConfig
        Dataset-generation settings.

    Returns
    -------
    dict[str, list[CropCandidate]]
        Generic ``other`` candidates by split.
    """
    random_generator = random.Random(config.seed)
    candidates = {split: [] for split in OUTPUT_SPLITS}
    for split, paths in negatives.items():
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            candidates[split].append(
                CropCandidate(path, (0.0, 0.0, float(width), float(height)), "other", split, "full_negative")
            )
            minimum_edge = min(width, height)
            for _ in range(config.random_crops_per_negative):
                side = max(8, round(minimum_edge * random_generator.uniform(0.25, 0.85)))
                left = random_generator.randint(0, max(0, width - side))
                top = random_generator.randint(0, max(0, height - side))
                candidates[split].append(
                    CropCandidate(
                        path,
                        (float(left), float(top), float(min(width, left + side)), float(min(height, top + side))),
                        "other",
                        split,
                        "random_negative",
                    )
                )
    return candidates


def _mine_detector_proposals(
    negatives: dict[str, list[Path]], config: PreparationConfig
) -> dict[str, list[CropCandidate]]:
    """
    Mine low-confidence detector boxes on known-negative images.

    Parameters
    ----------
    negatives : dict[str, list[Path]]
        Images whose labels prove they contain no target object.
    config : PreparationConfig
        Detector and batching settings.

    Returns
    -------
    dict[str, list[CropCandidate]]
        Hard-negative proposals labelled as ``other``.
    """
    from ultralytics import YOLO

    model = YOLO(str(config.weights))
    candidates = {split: [] for split in OUTPUT_SPLITS}
    for split, paths in negatives.items():
        for start in range(0, len(paths), config.chunk_size):
            chunk = paths[start : start + config.chunk_size]
            results = model.predict(
                source=[str(path) for path in chunk],
                conf=config.proposal_confidence,
                imgsz=640,
                iou=0.60,
                max_det=config.proposals_per_negative,
                batch=config.batch_size,
                device=config.device,
                stream=False,
                verbose=False,
            )
            for path, result in zip(chunk, results, strict=True):
                if result.boxes is None:
                    continue
                ranked = sorted(
                    result.boxes,
                    key=lambda box: float(box.conf[0].item()),
                    reverse=True,
                )[: config.proposals_per_negative]
                for box in ranked:
                    bbox = tuple(float(value) for value in box.xyxy[0].tolist())
                    x1, y1, x2, y2 = bbox
                    if (
                        not all(math.isfinite(value) for value in bbox)
                        or x2 - x1 < 2.0
                        or y2 - y1 < 2.0
                    ):
                        continue
                    candidates[split].append(
                        CropCandidate(path, bbox, "other", split, "detector_proposal")
                    )
    return candidates


def _balanced_selection(
    positives: dict[str, dict[str, list[CropCandidate]]],
    others: dict[str, list[CropCandidate]],
    config: PreparationConfig,
) -> list[CropCandidate]:
    """
    Sample equal class counts independently within every dataset split.

    Parameters
    ----------
    positives : dict[str, dict[str, list[CropCandidate]]]
        Annotated target-object candidates.
    others : dict[str, list[CropCandidate]]
        Generic and hard-negative candidates.
    config : PreparationConfig
        Per-class limits and random seed.

    Returns
    -------
    list[CropCandidate]
        Reproducibly shuffled balanced candidates.
    """
    random_generator = random.Random(config.seed)
    selected: list[CropCandidate] = []
    for split in OUTPUT_SPLITS:
        class_candidates = {
            "bolt": positives[split]["bolt"],
            "nut": positives[split]["nut"],
            "other": others[split],
        }
        split_limit = (
            config.max_train_per_class if split == "train" else config.max_eval_per_class
        )
        class_limit = min(split_limit, *(len(items) for items in class_candidates.values()))
        if class_limit <= 0:
            raise ValueError(f"Split {split!r} cannot provide all three classifier classes")
        for class_name in OUTPUT_CLASSES:
            candidates = list(class_candidates[class_name])
            if class_name == "other":
                hard_negatives = [
                    item for item in candidates if item.origin == "detector_proposal"
                ]
                generic_negatives = [
                    item for item in candidates if item.origin != "detector_proposal"
                ]
                random_generator.shuffle(hard_negatives)
                random_generator.shuffle(generic_negatives)
                candidates = hard_negatives + generic_negatives
            else:
                random_generator.shuffle(candidates)
            selected.extend(candidates[:class_limit])
    random_generator.shuffle(selected)
    return selected


def _expand_contexts(
    positives: dict[str, dict[str, list[CropCandidate]]],
    others: dict[str, list[CropCandidate]],
    config: PreparationConfig,
) -> None:
    """
    Expand base candidates into tight and contextual crop variants in place.

    Parameters
    ----------
    positives : dict[str, dict[str, list[CropCandidate]]]
        Annotated positive candidates grouped by split and class.
    others : dict[str, list[CropCandidate]]
        Generic and detector-proposal negative candidates.
    config : PreparationConfig
        Context ratios requested for the generated dataset.

    Notes
    -----
    Complete negative images use zero added context to avoid redundant padded
    copies. All object-sized regions receive every configured context ratio.
    """
    for split in OUTPUT_SPLITS:
        for class_name in ("bolt", "nut"):
            positives[split][class_name] = [
                replace(candidate, context_ratio=context_ratio)
                for candidate in positives[split][class_name]
                for context_ratio in config.context_ratios
            ]
        others[split] = [
            replace(candidate, context_ratio=context_ratio)
            for candidate in others[split]
            for context_ratio in (
                (0.0,) if candidate.origin == "full_negative" else config.context_ratios
            )
        ]


def _prepare_output(output: Path, replace: bool) -> None:
    """
    Create an empty generated classification dataset directory.

    Parameters
    ----------
    output : Path
        Destination root.
    replace : bool
        Whether an existing generated dataset may be removed.
    """
    resolved_output = output.resolve()
    if resolved_output == PROJECT_ROOT or PROJECT_ROOT not in resolved_output.parents:
        raise ValueError(f"Output must stay below the project root: {resolved_output}")
    if output.exists():
        if not replace:
            raise FileExistsError(f"Output already exists; use --replace: {output}")
        marker = output / GENERATED_MARKER
        if not marker.is_file():
            raise ValueError(f"Refusing to replace an unmarked directory: {output}")
        shutil.rmtree(output)
    for split in OUTPUT_SPLITS:
        for class_name in OUTPUT_CLASSES:
            (output / split / class_name).mkdir(parents=True, exist_ok=True)
    (output / GENERATED_MARKER).write_text("generated\n", encoding="utf-8")


def _candidate_name(candidate: CropCandidate, index: int) -> str:
    """
    Build a stable collision-resistant JPEG filename for a crop.

    Parameters
    ----------
    candidate : CropCandidate
        Candidate whose source and geometry identify it.
    index : int
        Selection-order index used to distinguish repeated candidates.

    Returns
    -------
    str
        Generated JPEG filename.
    """
    identity = (
        f"{candidate.source}|{candidate.bbox}|{candidate.class_name}|"
        f"{candidate.origin}|{candidate.context_ratio}|{index}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"{candidate.class_name}_{digest}.jpg"


def _write_crops(
    candidates: list[CropCandidate], config: PreparationConfig
) -> Counter[tuple[str, str, str]]:
    """
    Extract selected candidates and write the classification directory tree.

    Parameters
    ----------
    candidates : list[CropCandidate]
        Balanced candidates to persist.
    config : PreparationConfig
        Crop geometry and output settings.

    Returns
    -------
    collections.Counter
        Counts keyed by split, class, and candidate origin.
    """
    grouped: defaultdict[Path, list[tuple[int, CropCandidate]]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        grouped[candidate.source].append((index, candidate))

    counts: Counter[tuple[str, str, str]] = Counter()
    for source, source_candidates in grouped.items():
        with Image.open(source) as opened_image:
            image = opened_image.convert("RGB")
        for index, candidate in source_candidates:
            crop = extract_square_crop(
                image,
                candidate.bbox,
                context_ratio=candidate.context_ratio,
                output_size=config.image_size,
            )
            destination = (
                config.output
                / candidate.split
                / candidate.class_name
                / _candidate_name(candidate, index)
            )
            crop.save(destination, format="JPEG", quality=95, optimize=True)
            counts[(candidate.split, candidate.class_name, candidate.origin)] += 1
    return counts


def _write_manifest(
    config: PreparationConfig,
    positives: dict[str, dict[str, list[CropCandidate]]],
    negatives: dict[str, list[Path]],
    selected_counts: Counter[tuple[str, str, str]],
    selected: list[CropCandidate],
) -> None:
    """
    Record dataset inputs, settings, and selected class counts.

    Parameters
    ----------
    config : PreparationConfig
        Generation settings.
    positives : dict[str, dict[str, list[CropCandidate]]]
        Available positive instances before balancing.
    negatives : dict[str, list[Path]]
        Available negative images before crop generation.
    selected_counts : collections.Counter
        Written crop counts grouped by split, class, and origin.
    selected : list[CropCandidate]
        Selected candidates used to report context-ratio coverage.
    """
    selected_by_split_class: dict[str, dict[str, int]] = {
        split: {class_name: 0 for class_name in OUTPUT_CLASSES}
        for split in OUTPUT_SPLITS
    }
    selected_by_origin: Counter[str] = Counter()
    for (split, class_name, origin), count in selected_counts.items():
        selected_by_split_class[split][class_name] += count
        selected_by_origin[origin] += count

    manifest = {
        "data": str(config.data.resolve()),
        "detector_weights": str(config.weights.resolve()),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "available": {
            split: {
                "bolt_instances": len(positives[split]["bolt"]),
                "nut_instances": len(positives[split]["nut"]),
                "negative_images": len(negatives[split]),
            }
            for split in OUTPUT_SPLITS
        },
        "selected_by_split_and_class": selected_by_split_class,
        "selected_by_origin": dict(sorted(selected_by_origin.items())),
        "selected_by_context": {
            str(context): count
            for context, count in sorted(
                Counter(candidate.context_ratio for candidate in selected).items()
            )
        },
    }
    (config.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def prepare_dataset(config: PreparationConfig) -> dict[str, dict[str, int]]:
    """
    Build the complete balanced crop-classification dataset.

    Parameters
    ----------
    config : PreparationConfig
        Validated generation settings.

    Returns
    -------
    dict[str, dict[str, int]]
        Written image counts by split and class.
    """
    positives, negatives = _collect_source_examples(config)
    other_candidates = _random_other_candidates(negatives, config)
    proposal_candidates = _mine_detector_proposals(negatives, config)
    for split in OUTPUT_SPLITS:
        other_candidates[split].extend(proposal_candidates[split])
    _expand_contexts(positives, other_candidates, config)
    selected = _balanced_selection(positives, other_candidates, config)
    _prepare_output(config.output, config.replace)
    selected_counts = _write_crops(selected, config)
    _write_manifest(config, positives, negatives, selected_counts, selected)
    return {
        split: {
            class_name: sum(
                count
                for (count_split, count_class, _), count in selected_counts.items()
                if count_split == split and count_class == class_name
            )
            for class_name in OUTPUT_CLASSES
        }
        for split in OUTPUT_SPLITS
    }


def main() -> None:
    """Generate the classifier dataset and print its balanced class counts."""
    counts = prepare_dataset(parse_config())
    print(json.dumps(counts, indent=2))
    print("Crop-classifier dataset is ready. Do not train it with this script.")


if __name__ == "__main__":
    main()
