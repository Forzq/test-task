"""Choose an API confidence threshold using labelled positives and empty-label negatives."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


SUPPORTED_CLASSES = {"bolt", "nut"}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class Box:
    """
    Bounding box and class name used during threshold calibration.

    Parameters
    ----------
    class_name : str
        Canonical object class label.
    confidence : float
        Model confidence; ground-truth boxes use ``1.0``.
    xyxy : tuple[float, float, float, float]
        Bounding-box corners in image pixel coordinates.
    """

    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """
    Detection quality and background-rejection statistics at one threshold.

    Parameters
    ----------
    threshold : float
        Confidence threshold evaluated.
    precision : float
        Matched detections divided by all accepted detections.
    recall : float
        Matched ground-truth objects divided by all target objects.
    negative_image_fpr : float
        Fraction of empty-label images that produced any returned task object.
    negative_false_positive_images : int
        Number of negative images with at least one false returned object.
    """

    threshold: float
    precision: float
    recall: float
    negative_image_fpr: float
    negative_false_positive_images: int


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line options for repeatable confidence-threshold selection.

    Returns
    -------
    argparse.ArgumentParser
        Parser with validation split and conservative false-positive defaults.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate the API confidence threshold from a labelled YOLO split."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML path.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO weights path.")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="Dataset split used for calibration (default: val).",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Inference batch size used during calibration (default: 16).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="First threshold considered during the search (default: 0.25).",
    )
    parser.add_argument(
        "--max-confidence",
        type=float,
        default=0.95,
        help="Last threshold considered during the search (default: 0.95).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.01,
        help="Threshold grid step (default: 0.01).",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Minimum recall required for a recommended threshold (default: 0.95).",
    )
    parser.add_argument(
        "--max-negative-image-fpr",
        type=float,
        default=0.0,
        help="Maximum fraction of empty-label images allowed to produce a detection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/threshold_calibration.json"),
        help="JSON report path containing every evaluated threshold.",
    )
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    """
    Validate local files and numerical constraints before GPU inference begins.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed calibration options.

    Raises
    ------
    ValueError
        Raised when paths or threshold-search constraints are invalid.
    """
    if not arguments.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {arguments.data}")
    if not arguments.weights.is_file():
        raise ValueError(f"Model weights were not found: {arguments.weights}")
    if arguments.imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if arguments.batch <= 0:
        raise ValueError("batch must be positive")
    if not 0.0 <= arguments.min_confidence <= arguments.max_confidence <= 1.0:
        raise ValueError("Confidence range must satisfy 0 <= min <= max <= 1")
    if not 0.0 < arguments.step <= 1.0:
        raise ValueError("step must be between 0 and 1")
    if not 0.0 <= arguments.min_recall <= 1.0:
        raise ValueError("min-recall must be between 0 and 1")
    if not 0.0 <= arguments.max_negative_image_fpr <= 1.0:
        raise ValueError("max-negative-image-fpr must be between 0 and 1")


def _image_paths(image_sources: str | list[str]) -> list[Path]:
    """
    Collect image files from resolved YOLO image directories.

    Parameters
    ----------
    image_sources : str or list[str]
        One or more image directories returned by Ultralytics dataset parsing.

    Returns
    -------
    list[Path]
        Unique sorted source image paths.
    """
    sources = [image_sources] if isinstance(image_sources, str) else image_sources
    paths: set[Path] = set()
    for source in sources:
        directory = Path(source)
        paths.update(
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(paths)


def _label_path(image_path: Path) -> Path:
    """
    Derive the standard YOLO label path from an image under an ``images`` directory.

    Parameters
    ----------
    image_path : Path
        Source image path in a conventional YOLO split.

    Returns
    -------
    Path
        Matching expected text-label path.
    """
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def _parse_ground_truth(image_path: Path, names: dict[int, str]) -> list[Box]:
    """
    Convert YOLO bounding-box or polygon labels to pixel-coordinate boxes.

    Parameters
    ----------
    image_path : Path
        Image whose matching YOLO label file will be read.
    names : dict[int, str]
        Dataset class-name mapping.

    Returns
    -------
    list[Box]
        Target nut and bolt ground-truth boxes; empty labels create negatives.

    Raises
    ------
    ValueError
        Raised when a non-empty label line is malformed or has an unknown class ID.
    """
    label_path = _label_path(image_path)
    if not label_path.is_file():
        return []
    with Image.open(image_path) as image:
        width, height = image.size

    boxes: list[Box] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Malformed label at {label_path}:{line_number}")
        class_id = int(values[0])
        class_name = names.get(class_id, "").lower()
        if class_name not in SUPPORTED_CLASSES:
            raise ValueError(f"Unsupported ground-truth class at {label_path}:{line_number}")
        coordinates = [float(value) for value in values[1:]]
        if len(coordinates) == 4:
            center_x, center_y, box_width, box_height = coordinates
            x1 = (center_x - box_width / 2) * width
            y1 = (center_y - box_height / 2) * height
            x2 = (center_x + box_width / 2) * width
            y2 = (center_y + box_height / 2) * height
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            x_values = coordinates[0::2]
            y_values = coordinates[1::2]
            x1, x2 = min(x_values) * width, max(x_values) * width
            y1, y2 = min(y_values) * height, max(y_values) * height
        else:
            raise ValueError(f"Malformed box or polygon at {label_path}:{line_number}")
        boxes.append(Box(class_name, 1.0, (x1, y1, x2, y2)))
    return boxes


def _predict_boxes(
    image_paths: Iterable[Path],
    weights: Path,
    image_size: int,
    device: str,
    batch_size: int,
) -> dict[Path, list[Box]]:
    """
    Run one low-threshold inference pass and retain candidate task detections.

    Parameters
    ----------
    image_paths : Iterable[Path]
        Positive and negative images to evaluate.
    weights : Path
        Trained YOLO checkpoint.
    image_size : int
        YOLO inference image size.
    device : str
        Ultralytics device selector.
    batch_size : int
        Number of images processed together during inference.

    Returns
    -------
    dict[Path, list[Box]]
        Candidate bolt and nut predictions indexed by their source image path.
    """
    from ultralytics import YOLO

    paths = list(image_paths)
    model = YOLO(str(weights))
    predictions: dict[Path, list[Box]] = {path.resolve(): [] for path in paths}
    results = model.predict(
        source=[str(path) for path in paths],
        conf=0.001,
        imgsz=image_size,
        device=device,
        batch=batch_size,
        stream=True,
        verbose=False,
    )
    for image_path, result in zip(paths, results, strict=True):
        image_path = image_path.resolve()
        if result.boxes is None:
            continue
        for detection in result.boxes:
            class_name = str(result.names[int(detection.cls[0].item())]).lower()
            if class_name not in SUPPORTED_CLASSES:
                continue
            x1, y1, x2, y2 = (float(value) for value in detection.xyxy[0].tolist())
            predictions[image_path].append(
                Box(class_name, float(detection.conf[0].item()), (x1, y1, x2, y2))
            )
    return predictions


def _iou(first: Box, second: Box) -> float:
    """
    Calculate intersection-over-union for two pixel-coordinate bounding boxes.

    Parameters
    ----------
    first : Box
        First bounding box.
    second : Box
        Second bounding box.

    Returns
    -------
    float
        Intersection-over-union value in the inclusive range from zero to one.
    """
    left = max(first.xyxy[0], second.xyxy[0])
    top = max(first.xyxy[1], second.xyxy[1])
    right = min(first.xyxy[2], second.xyxy[2])
    bottom = min(first.xyxy[3], second.xyxy[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first.xyxy[2] - first.xyxy[0]) * max(0.0, first.xyxy[3] - first.xyxy[1])
    second_area = max(0.0, second.xyxy[2] - second.xyxy[0]) * max(0.0, second.xyxy[3] - second.xyxy[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _positive_counts(
    predictions: dict[Path, list[Box]], ground_truth: dict[Path, list[Box]], threshold: float
) -> tuple[int, int, int]:
    """
    Count true positives, accepted predictions, and target boxes at a threshold.

    Parameters
    ----------
    predictions : dict[Path, list[Box]]
        Low-threshold model detections per image.
    ground_truth : dict[Path, list[Box]]
        Target labels per positive image.
    threshold : float
        Confidence cutoff applied to stored candidate detections.

    Returns
    -------
    tuple[int, int, int]
        Matched detections, all accepted detections, and all target boxes.
    """
    true_positives = 0
    accepted_predictions = 0
    target_boxes = 0
    for image_path, targets in ground_truth.items():
        target_boxes += len(targets)
        unmatched = set(range(len(targets)))
        candidates = sorted(
            (box for box in predictions[image_path] if box.confidence >= threshold),
            key=lambda box: box.confidence,
            reverse=True,
        )
        accepted_predictions += len(candidates)
        for candidate in candidates:
            matching = [
                index
                for index in unmatched
                if targets[index].class_name == candidate.class_name and _iou(targets[index], candidate) >= 0.5
            ]
            if matching:
                best_index = max(matching, key=lambda index: _iou(targets[index], candidate))
                unmatched.remove(best_index)
                true_positives += 1
    return true_positives, accepted_predictions, target_boxes


def evaluate_threshold(
    predictions: dict[Path, list[Box]], ground_truth: dict[Path, list[Box]], threshold: float
) -> ThresholdMetrics:
    """
    Calculate positive detection quality and empty-image false-positive rate.

    Parameters
    ----------
    predictions : dict[Path, list[Box]]
        Candidate model outputs for all evaluated images.
    ground_truth : dict[Path, list[Box]]
        Parsed labels; images with an empty list are negatives.
    threshold : float
        Confidence cutoff used to accept detections.

    Returns
    -------
    ThresholdMetrics
        Precision, recall, and image-level negative false-positive statistics.
    """
    positive_truth = {path: boxes for path, boxes in ground_truth.items() if boxes}
    negative_paths = [path for path, boxes in ground_truth.items() if not boxes]
    true_positives, accepted_positive, targets = _positive_counts(
        predictions, positive_truth, threshold
    )
    accepted_negative = sum(
        sum(box.confidence >= threshold for box in predictions[path]) for path in negative_paths
    )
    false_positive_images = sum(
        any(box.confidence >= threshold for box in predictions[path]) for path in negative_paths
    )
    accepted = accepted_positive + accepted_negative
    return ThresholdMetrics(
        threshold=round(threshold, 4),
        precision=true_positives / accepted if accepted else 0.0,
        recall=true_positives / targets if targets else 0.0,
        negative_image_fpr=false_positive_images / len(negative_paths) if negative_paths else 0.0,
        negative_false_positive_images=false_positive_images,
    )


def _threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    """
    Generate an inclusive, rounded threshold grid.

    Parameters
    ----------
    minimum : float
        First confidence value.
    maximum : float
        Last confidence value.
    step : float
        Positive increment between candidates.

    Returns
    -------
    list[float]
        Ordered thresholds including the exact maximum value.
    """
    thresholds: list[float] = []
    value = minimum
    while value <= maximum + 1e-9:
        thresholds.append(round(value, 4))
        value += step
    if thresholds[-1] != round(maximum, 4):
        thresholds.append(round(maximum, 4))
    return thresholds


def _recommend(
    metrics: list[ThresholdMetrics], minimum_recall: float, maximum_fpr: float
) -> ThresholdMetrics | None:
    """
    Select the lowest threshold satisfying recall and negative-image constraints.

    Parameters
    ----------
    metrics : list[ThresholdMetrics]
        Results for every evaluated threshold.
    minimum_recall : float
        Required target-object recall.
    maximum_fpr : float
        Allowed empty-image false-positive rate.

    Returns
    -------
    ThresholdMetrics or None
        Most recall-friendly acceptable threshold, or None if constraints conflict.
    """
    candidates = [
        item
        for item in metrics
        if item.recall >= minimum_recall and item.negative_image_fpr <= maximum_fpr
    ]
    return min(candidates, key=lambda item: item.threshold) if candidates else None


def calibrate(arguments: argparse.Namespace) -> tuple[ThresholdMetrics | None, list[ThresholdMetrics], int, int]:
    """
    Run model inference once and evaluate all requested confidence cutoffs.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed and validated calibration options.

    Returns
    -------
    tuple[ThresholdMetrics or None, list[ThresholdMetrics], int, int]
        Recommended result, all grid metrics, positive-image count, and negative-image count.
    """
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(arguments.data), autodownload=False)
    split_images = _image_paths(data[arguments.split])
    names = {int(index): str(name) for index, name in data["names"].items()}
    ground_truth = {path: _parse_ground_truth(path, names) for path in split_images}
    predictions = _predict_boxes(
        split_images,
        arguments.weights,
        arguments.imgsz,
        arguments.device,
        arguments.batch,
    )
    metrics = [
        evaluate_threshold(predictions, ground_truth, threshold)
        for threshold in _threshold_grid(arguments.min_confidence, arguments.max_confidence, arguments.step)
    ]
    recommendation = _recommend(metrics, arguments.min_recall, arguments.max_negative_image_fpr)
    positive_images = sum(bool(boxes) for boxes in ground_truth.values())
    negative_images = len(ground_truth) - positive_images
    return recommendation, metrics, positive_images, negative_images


def _write_report(
    output_path: Path,
    recommendation: ThresholdMetrics | None,
    metrics: list[ThresholdMetrics],
    positive_images: int,
    negative_images: int,
) -> None:
    """
    Persist calibration evidence in a compact JSON report.

    Parameters
    ----------
    output_path : Path
        Report destination.
    recommendation : ThresholdMetrics, optional
        Selected threshold, if one satisfied the requested constraints.
    metrics : list[ThresholdMetrics]
        Complete threshold-grid results.
    positive_images : int
        Number of images containing target objects.
    negative_images : int
        Number of empty-label images used to measure false positives.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positive_images": positive_images,
        "negative_images": negative_images,
        "recommended": asdict(recommendation) if recommendation else None,
        "thresholds": [asdict(item) for item in metrics],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    """
    Calibrate the API confidence threshold and print the deployable environment value.

    Raises
    ------
    ValueError
        Raised when inputs or calibration constraints are invalid.
    """
    arguments = build_parser().parse_args()
    _validate_arguments(arguments)
    recommendation, metrics, positive_images, negative_images = calibrate(arguments)
    _write_report(arguments.output, recommendation, metrics, positive_images, negative_images)
    print(f"Calibration images: {positive_images} positive, {negative_images} negative")
    if recommendation is None:
        print("No threshold satisfies the requested recall and false-positive constraints.")
    else:
        print(
            "Recommended CONFIDENCE_THRESHOLD="
            f"{recommendation.threshold:.2f} "
            f"(recall={recommendation.recall:.3f}, "
            f"negative_image_fpr={recommendation.negative_image_fpr:.3f})"
        )
    print(f"Detailed report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
