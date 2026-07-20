"""Benchmark the complete detector, crop-classifier, and postprocessing pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier import UltralyticsCropClassifier
from app.detector import UltralyticsDetector
from app.service import DetectionService
from scripts.prepare_classifier_dataset import (
    _dataset_entries,
    _image_paths,
    _label_path,
    _parse_yolo_boxes,
)

TARGET_CLASSES = ("bolt", "nut")


def build_parser() -> argparse.ArgumentParser:
    """
    Create the end-to-end benchmark argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser exposing models, thresholds, contexts, and output report.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the complete two-stage pipeline on a YOLO split."
    )
    parser.add_argument("--data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--detector", type=Path, default=Path("models/best.pt"))
    parser.add_argument(
        "--classifier", type=Path, default=Path("models/classifier_best.pt")
    )
    parser.add_argument(
        "--disable-classifier",
        action="store_true",
        help="Benchmark detector and postprocessing without crop verification.",
    )
    parser.add_argument("--bolt-detector-threshold", type=float, default=0.45)
    parser.add_argument("--nut-detector-threshold", type=float, default=0.45)
    parser.add_argument("--bolt-classifier-threshold", type=float, default=0.54)
    parser.add_argument("--nut-classifier-threshold", type=float, default=0.54)
    parser.add_argument("--contexts", type=float, nargs="+", default=(0.20,))
    parser.add_argument("--min-box-area-ratio", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/benchmarks/pipeline_baseline_test.json"),
    )
    return parser


def _sha256(path: Path) -> str:
    """
    Calculate a model file digest for benchmark provenance.

    Parameters
    ----------
    path : Path
        Model checkpoint.

    Returns
    -------
    str
        Uppercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _image_bytes(path: Path) -> bytes:
    """
    Return service-compatible JPEG or PNG bytes for one dataset image.

    Parameters
    ----------
    path : Path
        Dataset image path.

    Returns
    -------
    bytes
        Original bytes for JPEG/PNG or lossless PNG conversion otherwise.
    """
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return path.read_bytes()
    output = BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


def _iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """
    Calculate intersection over union for two boxes.

    Parameters
    ----------
    first : tuple[float, float, float, float]
        First xyxy box.
    second : tuple[float, float, float, float]
        Second xyxy box.

    Returns
    -------
    float
        Intersection divided by union.
    """
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _match_class(
    predicted: list[tuple[float, float, float, float]],
    expected: list[tuple[float, float, float, float]],
    threshold: float,
) -> tuple[int, int, int]:
    """
    Greedily match same-class predictions to ground-truth boxes.

    Parameters
    ----------
    predicted : list[tuple[float, float, float, float]]
        Predicted boxes.
    expected : list[tuple[float, float, float, float]]
        Ground-truth boxes.
    threshold : float
        Minimum IoU for a true-positive match.

    Returns
    -------
    tuple[int, int, int]
        True positives, false positives, and false negatives.
    """
    unmatched = set(range(len(expected)))
    true_positives = 0
    for box in predicted:
        best_index = max(
            unmatched,
            key=lambda index: _iou(box, expected[index]),
            default=None,
        )
        if best_index is not None and _iou(box, expected[best_index]) >= threshold:
            unmatched.remove(best_index)
            true_positives += 1
    false_positives = len(predicted) - true_positives
    false_negatives = len(expected) - true_positives
    return true_positives, false_positives, false_negatives


def _ratio(numerator: int, denominator: int) -> float:
    """
    Divide metric counts safely.

    Parameters
    ----------
    numerator : int
        Metric numerator.
    denominator : int
        Metric denominator.

    Returns
    -------
    float
        Ratio or zero when the denominator is zero.
    """
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    """
    Return a nearest-rank latency percentile.

    Parameters
    ----------
    values : list[float]
        Latencies in milliseconds.
    fraction : float
        Quantile fraction from zero to one.

    Returns
    -------
    float
        Selected latency value.
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def benchmark(arguments: argparse.Namespace) -> dict[str, Any]:
    """
    Evaluate the complete inference pipeline on one held-out split.

    Parameters
    ----------
    arguments : argparse.Namespace
        Validated benchmark settings.

    Returns
    -------
    dict[str, Any]
        Operational metrics and exact model/configuration provenance.
    """
    detector = UltralyticsDetector(arguments.detector)
    detector.load()
    classifier = None
    if not arguments.disable_classifier:
        classifier = UltralyticsCropClassifier(arguments.classifier, image_size=224)
        classifier.load()
    service = DetectionService(
        detector=detector,
        confidence_threshold=min(
            arguments.bolt_detector_threshold,
            arguments.nut_detector_threshold,
        ),
        max_upload_size_bytes=50 * 1024 * 1024,
        crop_classifier=classifier,
        classifier_confidence_threshold=min(
            arguments.bolt_classifier_threshold,
            arguments.nut_classifier_threshold,
        ),
        classifier_image_size=224,
        min_box_area_ratio=arguments.min_box_area_ratio,
        detector_class_thresholds={
            "bolt": arguments.bolt_detector_threshold,
            "nut": arguments.nut_detector_threshold,
        },
        classifier_class_thresholds={
            "bolt": arguments.bolt_classifier_threshold,
            "nut": arguments.nut_classifier_threshold,
        },
        classifier_crop_contexts=tuple(arguments.contexts),
        final_nms_iou_threshold=arguments.nms_iou,
    )

    records: list[tuple[Path, list[tuple[str, tuple[float, float, float, float]]]]] = []
    seen: set[Path] = set()
    for image_root in _dataset_entries(arguments.data)[arguments.split]:
        for image_path in _image_paths(image_root):
            resolved = image_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            with Image.open(image_path) as image:
                width, height = image.size
            expected = _parse_yolo_boxes(
                _label_path(image_path, image_root), width, height
            )
            records.append((resolved, expected))
    if arguments.limit > 0:
        records = records[: arguments.limit]

    totals = {class_name: Counter() for class_name in TARGET_CLASSES}
    latencies: list[float] = []
    negative_images = 0
    negative_images_with_false_positive = 0
    exact_count_images = 0
    images_with_any_false_positive = 0

    for index, (image_path, expected) in enumerate(records, start=1):
        started = time.perf_counter()
        response = service.predict(_image_bytes(image_path))
        latencies.append((time.perf_counter() - started) * 1000.0)
        predicted_by_class = {
            class_name: [
                tuple(float(value) for value in item.bbox)
                for item in response.objects
                if item.class_name == class_name
            ]
            for class_name in TARGET_CLASSES
        }
        expected_by_class = {
            class_name: [box for label, box in expected if label == class_name]
            for class_name in TARGET_CLASSES
        }
        image_has_false_positive = False
        for class_name in TARGET_CLASSES:
            true_positive, false_positive, false_negative = _match_class(
                predicted_by_class[class_name],
                expected_by_class[class_name],
                arguments.match_iou,
            )
            totals[class_name]["tp"] += true_positive
            totals[class_name]["fp"] += false_positive
            totals[class_name]["fn"] += false_negative
            image_has_false_positive |= false_positive > 0
        images_with_any_false_positive += int(image_has_false_positive)
        is_negative = not expected
        negative_images += int(is_negative)
        negative_images_with_false_positive += int(is_negative and bool(response.objects))
        exact_count_images += int(
            all(
                len(predicted_by_class[class_name])
                == len(expected_by_class[class_name])
                for class_name in TARGET_CLASSES
            )
        )
        if index % 50 == 0 or index == len(records):
            print(f"Benchmarked {index}/{len(records)} images")

    class_metrics: dict[str, dict[str, float | int]] = {}
    for class_name, counts in totals.items():
        precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
        recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
        class_metrics[class_name] = {
            **dict(counts),
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
        }
    macro_f1 = statistics.fmean(
        float(class_metrics[class_name]["f1"]) for class_name in TARGET_CLASSES
    )
    return {
        "dataset": str(arguments.data.resolve()),
        "split": arguments.split,
        "images": len(records),
        "models": {
            "detector": {
                "path": str(arguments.detector.resolve()),
                "sha256": _sha256(arguments.detector),
            },
            "classifier": (
                {
                    "path": str(arguments.classifier.resolve()),
                    "sha256": _sha256(arguments.classifier),
                }
                if not arguments.disable_classifier
                else None
            ),
        },
        "configuration": {
            "bolt_detector_threshold": arguments.bolt_detector_threshold,
            "nut_detector_threshold": arguments.nut_detector_threshold,
            "bolt_classifier_threshold": arguments.bolt_classifier_threshold,
            "nut_classifier_threshold": arguments.nut_classifier_threshold,
            "contexts": list(arguments.contexts),
            "min_box_area_ratio": arguments.min_box_area_ratio,
            "nms_iou": arguments.nms_iou,
            "match_iou": arguments.match_iou,
            "classifier_enabled": not arguments.disable_classifier,
        },
        "classes": class_metrics,
        "macro_f1": macro_f1,
        "negative_images": negative_images,
        "negative_images_with_false_positive": negative_images_with_false_positive,
        "negative_image_false_positive_rate": _ratio(
            negative_images_with_false_positive, negative_images
        ),
        "images_with_any_false_positive": images_with_any_false_positive,
        "exact_count_image_accuracy": _ratio(exact_count_images, len(records)),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
    }


def main() -> None:
    """Validate inputs, execute the benchmark, and write its JSON report."""
    arguments = build_parser().parse_args()
    required_paths = [arguments.data, arguments.detector]
    if not arguments.disable_classifier:
        required_paths.append(arguments.classifier)
    for path in required_paths:
        if not path.is_file():
            raise ValueError(f"Required file was not found: {path}")
    threshold_values = (
        arguments.bolt_detector_threshold,
        arguments.nut_detector_threshold,
        arguments.bolt_classifier_threshold,
        arguments.nut_classifier_threshold,
        arguments.min_box_area_ratio,
        arguments.nms_iou,
        arguments.match_iou,
    )
    if any(not 0.0 <= value <= 1.0 for value in threshold_values):
        raise ValueError("All thresholds and IoU values must be between zero and one")
    if not arguments.contexts or any(value < 0.0 for value in arguments.contexts):
        raise ValueError("contexts must contain non-negative values")
    report = benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
