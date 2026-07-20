"""Audit YOLO annotations, class balance, geometry, and split leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_classifier_dataset import (
    OUTPUT_SPLITS,
    _dataset_entries,
    _image_paths,
    _label_path,
    _parse_yolo_boxes,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Create the dataset-audit command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with project-local dataset and report defaults.
    """
    parser = argparse.ArgumentParser(
        description="Audit YOLO class balance, annotations, and split leakage."
    )
    parser.add_argument("--data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/dataset_audit.json")
    )
    parser.add_argument(
        "--tiny-area-ratio",
        type=float,
        default=0.001,
        help="Relative box area considered operationally tiny.",
    )
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate the SHA-256 digest of one image.

    Parameters
    ----------
    path : Path
        Image file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal content digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    """
    Calculate stable area-ratio quantiles for a list of boxes.

    Parameters
    ----------
    values : list[float]
        Relative box areas.

    Returns
    -------
    dict[str, float]
        Minimum and selected empirical quantiles.
    """
    if not values:
        return {}
    ordered = sorted(values)
    return {
        name: ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]
        for name, fraction in (
            ("min", 0.0),
            ("p01", 0.01),
            ("p05", 0.05),
            ("p50", 0.50),
            ("p95", 0.95),
            ("max", 1.0),
        )
    }


def _source_name(image_root: Path, data_path: Path) -> str:
    """
    Return a concise source name for one declared image directory.

    Parameters
    ----------
    image_root : Path
        Declared split image directory.
    data_path : Path
        Combined dataset YAML.

    Returns
    -------
    str
        Relative dataset source root.
    """
    source_root = image_root.parent.parent
    try:
        relative = source_root.relative_to(data_path.parent.resolve())
        return relative.as_posix() or "primary"
    except ValueError:
        return str(source_root)


def audit(data_path: Path, tiny_area_ratio: float) -> dict[str, Any]:
    """
    Audit all images and annotations referenced by a combined YOLO YAML.

    Parameters
    ----------
    data_path : Path
        Combined YOLO dataset configuration.
    tiny_area_ratio : float
        Relative area used to flag very small boxes.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable audit report.
    """
    split_stats: dict[str, Counter[str]] = {
        split: Counter() for split in OUTPUT_SPLITS
    }
    source_stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
    areas: dict[str, list[float]] = {"bolt": [], "nut": []}
    hash_records: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    invalid: list[dict[str, str]] = []
    seen_paths: set[Path] = set()

    for split, roots in _dataset_entries(data_path).items():
        for image_root in roots:
            source = _source_name(image_root, data_path)
            for image_path in _image_paths(image_root):
                resolved = image_path.resolve()
                if resolved in seen_paths:
                    split_stats[split]["duplicate_path_references"] += 1
                    continue
                seen_paths.add(resolved)
                stats = split_stats[split]
                source_counter = source_stats[f"{split}:{source}"]
                stats["images"] += 1
                source_counter["images"] += 1
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                    label_path = _label_path(image_path, image_root)
                    raw_lines = [
                        line.split()
                        for line in label_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    boxes = _parse_yolo_boxes(label_path, width, height)
                    content_hash = _sha256(image_path)
                except (OSError, ValueError, UnidentifiedImageError) as error:
                    invalid.append({"image": str(resolved), "error": str(error)})
                    continue

                hash_records[content_hash].append(
                    {"split": split, "source": source, "path": str(resolved)}
                )
                if not boxes:
                    stats["negative_images"] += 1
                    source_counter["negative_images"] += 1
                for values in raw_lines:
                    stats["polygon_labels" if len(values) > 5 else "box_labels"] += 1
                for class_name, (x1, y1, x2, y2) in boxes:
                    relative_area = ((x2 - x1) * (y2 - y1)) / (width * height)
                    stats[f"{class_name}_boxes"] += 1
                    source_counter[f"{class_name}_boxes"] += 1
                    areas[class_name].append(relative_area)
                    if relative_area < tiny_area_ratio:
                        stats[f"tiny_{class_name}_boxes"] += 1

    duplicate_groups = [records for records in hash_records.values() if len(records) > 1]
    cross_split_groups = [
        records for records in duplicate_groups if len({item["split"] for item in records}) > 1
    ]
    train_bolts = split_stats["train"]["bolt_boxes"]
    train_nuts = split_stats["train"]["nut_boxes"]
    imbalance_ratio = (
        max(train_bolts, train_nuts) / min(train_bolts, train_nuts)
        if min(train_bolts, train_nuts) > 0
        else None
    )
    recommendations: list[str] = []
    if cross_split_groups:
        recommendations.append(
            "Remove content-identical images shared across train, validation, and test."
        )
    if imbalance_ratio and imbalance_ratio > 1.5:
        recommendations.append(
            "Increase nut diversity or use class-aware sampling; detector boxes are imbalanced."
        )
    if any(split_stats[split]["polygon_labels"] for split in OUTPUT_SPLITS):
        recommendations.append(
            "Convert mixed polygon and box labels to one consistent detection format."
        )
    recommendations.append(
        "Review tiny boxes and source-level outliers before using area filters for training metrics."
    )

    return {
        "dataset": str(data_path.resolve()),
        "tiny_area_ratio": tiny_area_ratio,
        "splits": {split: dict(stats) for split, stats in split_stats.items()},
        "sources": {name: dict(stats) for name, stats in sorted(source_stats.items())},
        "train_class_imbalance_ratio": imbalance_ratio,
        "box_area_quantiles": {
            class_name: _quantiles(class_areas)
            for class_name, class_areas in areas.items()
        },
        "invalid_images_or_labels": invalid,
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_split_duplicate_groups": len(cross_split_groups),
        "cross_split_duplicate_examples": cross_split_groups[:20],
        "recommendations": recommendations,
    }


def main() -> None:
    """Run the audit and persist a concise JSON report."""
    arguments = build_parser().parse_args()
    if not arguments.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {arguments.data}")
    if not 0.0 <= arguments.tiny_area_ratio <= 1.0:
        raise ValueError("tiny-area-ratio must be between zero and one")
    report = audit(arguments.data, arguments.tiny_area_ratio)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "splits": report["splits"],
        "train_class_imbalance_ratio": report["train_class_imbalance_ratio"],
        "invalid": len(report["invalid_images_or_labels"]),
        "exact_duplicate_groups": report["exact_duplicate_groups"],
        "cross_split_duplicate_groups": report["cross_split_duplicate_groups"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Audit report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
