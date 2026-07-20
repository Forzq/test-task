"""Calibrate the crop-verifier confidence threshold on validation data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier import configure_classifier_transforms


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """
    One labelled validation prediction used during calibration.

    Parameters
    ----------
    true_class : str
        Ground-truth directory name.
    predicted_class : str
        Highest-probability model class.
    confidence : float
        Confidence of the predicted class.
    """

    true_class: str
    predicted_class: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """
    Validation metrics calculated at one acceptance threshold.

    Parameters
    ----------
    threshold : float
        Minimum confidence for accepting a bolt or nut prediction.
    positive_recall : float
        Correct accepted bolt/nut predictions divided by all positive crops.
    other_false_positive_rate : float
        Other crops incorrectly accepted as bolt or nut.
    accepted_precision : float
        Correct predictions divided by all accepted target predictions.
    """

    threshold: float
    positive_recall: float
    other_false_positive_rate: float
    accepted_precision: float


def build_parser() -> argparse.ArgumentParser:
    """
    Create the classifier-threshold calibration parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with validation-oriented defaults.
    """
    parser = argparse.ArgumentParser(
        description="Select a crop-classifier threshold using the validation split."
    )
    parser.add_argument("--data", type=Path, default=Path("data/classifier"))
    parser.add_argument("--weights", type=Path, default=Path("models/classifier_best.pt"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-positive-recall", type=float, default=0.90)
    parser.add_argument("--max-other-fpr", type=float, default=0.02)
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/classifier_threshold_calibration.json"),
    )
    return parser


def _image_paths(split_root: Path) -> list[Path]:
    """
    Enumerate labelled validation images from class directories.

    Parameters
    ----------
    split_root : Path
        Classification split containing bolt, nut, and other directories.

    Returns
    -------
    list[Path]
        Sorted supported image paths.
    """
    suffixes = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        path
        for class_name in ("bolt", "nut", "other")
        for path in (split_root / class_name).rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not paths:
        raise ValueError(f"No classifier images were found below {split_root}")
    return paths


def _predict(arguments: argparse.Namespace, paths: list[Path]) -> list[PredictionRecord]:
    """
    Run the trained classifier over one labelled split.

    Parameters
    ----------
    arguments : argparse.Namespace
        Model and inference settings.
    paths : list[Path]
        Labelled crop paths.

    Returns
    -------
    list[PredictionRecord]
        Ground truth, top class, and confidence for every crop.
    """
    from ultralytics import YOLO

    model = YOLO(str(arguments.weights))
    configure_classifier_transforms(model, arguments.imgsz)
    class_names = {str(value).lower() for value in model.names.values()}
    if class_names != {"bolt", "nut", "other"}:
        raise ValueError(f"Unexpected classifier classes: {sorted(class_names)}")
    records: list[PredictionRecord] = []
    for start in range(0, len(paths), arguments.batch):
        batch_paths = paths[start : start + arguments.batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=arguments.imgsz,
            batch=len(batch_paths),
            device=arguments.device,
            stream=False,
            verbose=False,
        )
        for path, result in zip(batch_paths, results, strict=True):
            if result.probs is None:
                raise ValueError(f"Classifier returned no probabilities for {path}")
            class_id = int(result.probs.top1)
            records.append(
                PredictionRecord(
                    true_class=path.parent.name.lower(),
                    predicted_class=str(result.names[class_id]).lower(),
                    confidence=float(result.probs.top1conf.item()),
                )
            )
    return records


def _metrics(records: list[PredictionRecord], threshold: float) -> ThresholdMetrics:
    """
    Calculate acceptance metrics at one confidence threshold.

    Parameters
    ----------
    records : list[PredictionRecord]
        Labelled validation predictions.
    threshold : float
        Minimum confidence for a target-class acceptance.

    Returns
    -------
    ThresholdMetrics
        Recall, unrelated-object false-positive rate, and precision.
    """
    targets = {"bolt", "nut"}
    positive_total = sum(record.true_class in targets for record in records)
    other_total = sum(record.true_class == "other" for record in records)
    accepted = [
        record
        for record in records
        if record.predicted_class in targets and record.confidence >= threshold
    ]
    correct = sum(
        record.true_class == record.predicted_class for record in accepted
    )
    other_false_positives = sum(record.true_class == "other" for record in accepted)
    return ThresholdMetrics(
        threshold=round(threshold, 4),
        positive_recall=correct / positive_total if positive_total else 0.0,
        other_false_positive_rate=(
            other_false_positives / other_total if other_total else 0.0
        ),
        accepted_precision=correct / len(accepted) if accepted else 1.0,
    )


def _thresholds(arguments: argparse.Namespace) -> list[float]:
    """
    Generate the inclusive threshold search grid.

    Parameters
    ----------
    arguments : argparse.Namespace
        Threshold range and step settings.

    Returns
    -------
    list[float]
        Floating-point threshold candidates.
    """
    count = int(round((arguments.threshold_max - arguments.threshold_min) / arguments.threshold_step))
    return [arguments.threshold_min + index * arguments.threshold_step for index in range(count + 1)]


def _select(
    candidates: list[ThresholdMetrics],
    minimum_recall: float,
    maximum_other_fpr: float,
) -> tuple[ThresholdMetrics, bool]:
    """
    Select the highest-recall threshold satisfying false-positive constraints.

    Parameters
    ----------
    candidates : list[ThresholdMetrics]
        Metrics for every searched threshold.
    minimum_recall : float
        Required positive recall.
    maximum_other_fpr : float
        Maximum accepted false-positive fraction for other crops.

    Returns
    -------
    tuple[ThresholdMetrics, bool]
        Selected metrics and whether all requested constraints were met.
    """
    feasible = [
        item
        for item in candidates
        if item.positive_recall >= minimum_recall
        and item.other_false_positive_rate <= maximum_other_fpr
    ]
    if feasible:
        return max(
            feasible,
            key=lambda item: (
                item.positive_recall,
                -item.other_false_positive_rate,
                item.accepted_precision,
                item.threshold,
            ),
        ), True
    return min(
        candidates,
        key=lambda item: (
            max(0.0, item.other_false_positive_rate - maximum_other_fpr)
            + max(0.0, minimum_recall - item.positive_recall),
            -item.accepted_precision,
        ),
    ), False


def main() -> None:
    """Run validation inference, select a threshold, and write a JSON report."""
    arguments = build_parser().parse_args()
    if not arguments.weights.is_file():
        raise ValueError(f"Classifier weights were not found: {arguments.weights}")
    if min(arguments.imgsz, arguments.batch) <= 0:
        raise ValueError("imgsz and batch must be positive")
    for name in (
        "min_positive_recall",
        "max_other_fpr",
        "threshold_min",
        "threshold_max",
    ):
        if not 0.0 <= getattr(arguments, name) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be between 0 and 1")
    if arguments.threshold_step <= 0.0 or arguments.threshold_min > arguments.threshold_max:
        raise ValueError("threshold range and step are invalid")

    split_root = arguments.data / arguments.split
    paths = _image_paths(split_root)
    records = _predict(arguments, paths)
    candidates = [_metrics(records, threshold) for threshold in _thresholds(arguments)]
    selected, constraints_met = _select(
        candidates,
        arguments.min_positive_recall,
        arguments.max_other_fpr,
    )
    report = {
        "weights": str(arguments.weights.resolve()),
        "split": str(split_root.resolve()),
        "images": len(records),
        "constraints_met": constraints_met,
        "requirements": {
            "minimum_positive_recall": arguments.min_positive_recall,
            "maximum_other_false_positive_rate": arguments.max_other_fpr,
        },
        "selected": asdict(selected),
        "candidates": [asdict(item) for item in candidates],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"constraints_met": constraints_met, **asdict(selected)}, indent=2))
    print(f"Calibration report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
