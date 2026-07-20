"""Calibrate independent bolt and nut thresholds for detector or classifier weights."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_classifier_threshold import (
    PredictionRecord,
    _image_paths as classifier_image_paths,
    _predict as predict_classifier,
)
from scripts.calibrate_threshold import (
    Box,
    _image_paths as detector_image_paths,
    _iou,
    _parse_ground_truth,
    _predict_boxes,
)

TARGET_CLASSES = ("bolt", "nut")


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """
    Metrics for one target class at one confidence threshold.

    Parameters
    ----------
    class_name : str
        Target class evaluated.
    threshold : float
        Minimum confidence required for the target class.
    precision : float
        Correct accepted predictions divided by all accepted predictions.
    recall : float
        Correct accepted predictions divided by all target instances.
    negative_false_positive_rate : float
        Fraction of negative samples producing an accepted target prediction.
    """

    class_name: str
    threshold: float
    precision: float
    recall: float
    negative_false_positive_rate: float


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line arguments for class-aware threshold calibration.

    Returns
    -------
    argparse.ArgumentParser
        Parser supporting detector and crop-classifier calibration modes.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate independent bolt and nut confidence thresholds."
    )
    parser.add_argument("--task", choices=("detector", "classifier"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-negative-fpr", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/class_threshold_calibration.json"),
    )
    return parser


def _thresholds(arguments: argparse.Namespace) -> list[float]:
    """
    Build the inclusive confidence search grid.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed threshold range and step.

    Returns
    -------
    list[float]
        Ordered confidence thresholds.
    """
    values: list[float] = []
    value = arguments.threshold_min
    while value <= arguments.threshold_max + 1e-9:
        values.append(round(value, 4))
        value += arguments.threshold_step
    return values


def _select(
    candidates: list[ClassMetrics], minimum_recall: float, maximum_fpr: float
) -> tuple[ClassMetrics, bool]:
    """
    Select the most precise threshold that satisfies recall and FPR constraints.

    Parameters
    ----------
    candidates : list[ClassMetrics]
        Evaluated threshold candidates for one class.
    minimum_recall : float
        Required class recall.
    maximum_fpr : float
        Maximum allowed negative-sample false-positive rate.

    Returns
    -------
    tuple[ClassMetrics, bool]
        Selected metrics and whether both constraints were satisfied.
    """
    feasible = [
        item
        for item in candidates
        if item.recall >= minimum_recall
        and item.negative_false_positive_rate <= maximum_fpr
    ]
    if feasible:
        return max(feasible, key=lambda item: (item.precision, item.recall, item.threshold)), True
    return min(
        candidates,
        key=lambda item: (
            max(0.0, minimum_recall - item.recall)
            + max(0.0, item.negative_false_positive_rate - maximum_fpr),
            -item.precision,
        ),
    ), False


def _classifier_metrics(
    records: list[PredictionRecord], class_name: str, threshold: float
) -> ClassMetrics:
    """
    Calculate one classifier class metrics row.

    Parameters
    ----------
    records : list[PredictionRecord]
        Labelled crop predictions.
    class_name : str
        Class whose acceptance threshold is evaluated.
    threshold : float
        Candidate confidence threshold.

    Returns
    -------
    ClassMetrics
        Per-class precision, recall, and negative-crop false-positive rate.
    """
    positives = [record for record in records if record.true_class == class_name]
    negatives = [record for record in records if record.true_class != class_name]
    accepted = [
        record
        for record in records
        if record.predicted_class == class_name and record.confidence >= threshold
    ]
    correct = sum(record.true_class == class_name for record in accepted)
    false_positives = len(accepted) - correct
    return ClassMetrics(
        class_name=class_name,
        threshold=threshold,
        precision=correct / len(accepted) if accepted else 1.0,
        recall=correct / len(positives) if positives else 0.0,
        negative_false_positive_rate=(
            false_positives / len(negatives) if negatives else 0.0
        ),
    )


def _detector_metrics(
    predictions: dict[Path, list[Box]],
    ground_truth: dict[Path, list[Box]],
    class_name: str,
    threshold: float,
) -> ClassMetrics:
    """
    Calculate one detector class metrics row using IoU 0.5 matching.

    Parameters
    ----------
    predictions : dict[Path, list[Box]]
        Low-threshold detector predictions by image.
    ground_truth : dict[Path, list[Box]]
        Labelled target boxes by image.
    class_name : str
        Target class evaluated.
    threshold : float
        Candidate confidence threshold.

    Returns
    -------
    ClassMetrics
        Per-class precision, recall, and negative-image false-positive rate.
    """
    target_total = 0
    accepted_total = 0
    true_positives = 0
    negative_images = 0
    negative_false_positive_images = 0
    for path, boxes in ground_truth.items():
        targets = [box for box in boxes if box.class_name == class_name]
        candidates = sorted(
            (
                box
                for box in predictions[path]
                if box.class_name == class_name and box.confidence >= threshold
            ),
            key=lambda box: box.confidence,
            reverse=True,
        )
        target_total += len(targets)
        accepted_total += len(candidates)
        if not targets:
            negative_images += 1
            negative_false_positive_images += bool(candidates)
        unmatched = set(range(len(targets)))
        for candidate in candidates:
            matches = [index for index in unmatched if _iou(candidate, targets[index]) >= 0.5]
            if matches:
                best = max(matches, key=lambda index: _iou(candidate, targets[index]))
                unmatched.remove(best)
                true_positives += 1
    return ClassMetrics(
        class_name=class_name,
        threshold=threshold,
        precision=true_positives / accepted_total if accepted_total else 1.0,
        recall=true_positives / target_total if target_total else 0.0,
        negative_false_positive_rate=(
            negative_false_positive_images / negative_images if negative_images else 0.0
        ),
    )


def _classifier_candidates(
    arguments: argparse.Namespace, thresholds: list[float]
) -> dict[str, list[ClassMetrics]]:
    """
    Run classifier inference once and evaluate both class thresholds.

    Parameters
    ----------
    arguments : argparse.Namespace
        Classifier calibration configuration.
    thresholds : list[float]
        Confidence grid.

    Returns
    -------
    dict[str, list[ClassMetrics]]
        Metrics grid for bolt and nut.
    """
    arguments.imgsz = arguments.imgsz or 224
    paths = classifier_image_paths(arguments.data / arguments.split)
    records = predict_classifier(arguments, paths)
    return {
        class_name: [_classifier_metrics(records, class_name, value) for value in thresholds]
        for class_name in TARGET_CLASSES
    }


def _detector_candidates(
    arguments: argparse.Namespace, thresholds: list[float]
) -> dict[str, list[ClassMetrics]]:
    """
    Run detector inference once and evaluate both class thresholds.

    Parameters
    ----------
    arguments : argparse.Namespace
        Detector calibration configuration.
    thresholds : list[float]
        Confidence grid.

    Returns
    -------
    dict[str, list[ClassMetrics]]
        Metrics grid for bolt and nut.
    """
    from ultralytics.data.utils import check_det_dataset

    arguments.imgsz = arguments.imgsz or 640
    dataset = check_det_dataset(str(arguments.data), autodownload=False)
    paths = detector_image_paths(dataset[arguments.split])
    names = {int(index): str(name) for index, name in dataset["names"].items()}
    ground_truth = {path: _parse_ground_truth(path, names) for path in paths}
    predictions = _predict_boxes(
        paths, arguments.weights, arguments.imgsz, arguments.device, arguments.batch
    )
    return {
        class_name: [
            _detector_metrics(predictions, ground_truth, class_name, value)
            for value in thresholds
        ]
        for class_name in TARGET_CLASSES
    }


def main() -> None:
    """Calibrate both target classes and write deployable thresholds to JSON."""
    arguments = build_parser().parse_args()
    if not arguments.weights.is_file():
        raise ValueError(f"Weights were not found: {arguments.weights}")
    if not arguments.data.exists():
        raise ValueError(f"Dataset was not found: {arguments.data}")
    if arguments.batch <= 0 or (arguments.imgsz is not None and arguments.imgsz <= 0):
        raise ValueError("batch and imgsz must be positive")
    if not 0.0 <= arguments.threshold_min <= arguments.threshold_max <= 1.0:
        raise ValueError("threshold range must be between zero and one")
    if arguments.threshold_step <= 0.0:
        raise ValueError("threshold-step must be positive")
    thresholds = _thresholds(arguments)
    candidates = (
        _classifier_candidates(arguments, thresholds)
        if arguments.task == "classifier"
        else _detector_candidates(arguments, thresholds)
    )
    selected: dict[str, dict[str, object]] = {}
    for class_name, metrics in candidates.items():
        recommendation, constraints_met = _select(
            metrics, arguments.min_recall, arguments.max_negative_fpr
        )
        selected[class_name] = {
            "constraints_met": constraints_met,
            **asdict(recommendation),
        }
    payload = {
        "task": arguments.task,
        "weights": str(arguments.weights.resolve()),
        "split": arguments.split,
        "requirements": {
            "minimum_recall": arguments.min_recall,
            "maximum_negative_false_positive_rate": arguments.max_negative_fpr,
        },
        "selected": selected,
        "candidates": {
            name: [asdict(item) for item in metrics]
            for name, metrics in candidates.items()
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"Calibration report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
