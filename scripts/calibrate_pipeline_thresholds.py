"""Calibrate crop-classifier thresholds on the complete validation pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.classifier import configure_classifier_transforms
from app.detector import UltralyticsDetector
from app.image_utils import extract_square_crop
from scripts.benchmark_pipeline import _iou, _match_class
from scripts.calibrate_class_thresholds import ClassMetrics, _select
from scripts.prepare_classifier_dataset import (
    _dataset_entries,
    _image_paths,
    _label_path,
    _parse_yolo_boxes,
)

TARGET_CLASSES = ("bolt", "nut")
CLASS_ALIASES = {"bolt": "bolt", "bolts": "bolt", "nut": "nut", "nuts": "nut"}


@dataclass(slots=True)
class Candidate:
    """
    Store one detector proposal and accumulated multi-context probabilities.

    Parameters
    ----------
    image_index : int
        Index of the validation image owning the proposal.
    detector_confidence : float
        Confidence emitted by the detector.
    bbox : tuple[float, float, float, float]
        Proposal coordinates in xyxy pixel format.
    probability_sums : dict[str, float]
        Mutable probability sums for all evaluated contexts.
    contexts : int
        Number of successfully classified contexts.
    """

    image_index: int
    detector_confidence: float
    bbox: tuple[float, float, float, float]
    probability_sums: dict[str, float]
    contexts: int = 0

    @property
    def probabilities(self) -> dict[str, float]:
        """Return mean class probabilities across successful contexts."""
        if self.contexts == 0:
            return {}
        return {name: value / self.contexts for name, value in self.probability_sums.items()}


@dataclass(frozen=True, slots=True)
class VerifiedCandidate:
    """Represent one proposal after multi-context classifier aggregation."""

    image_index: int
    class_name: str
    classifier_confidence: float
    final_confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ValidationImage:
    """Store ground-truth boxes for one validation image."""

    path: Path
    expected: tuple[tuple[str, tuple[float, float, float, float]], ...]


def build_parser() -> argparse.ArgumentParser:
    """
    Create command-line arguments for end-to-end threshold calibration.

    Returns
    -------
    argparse.ArgumentParser
        Parser containing model, validation, and operating-point settings.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate classifier thresholds after multi-context aggregation."
    )
    parser.add_argument("--data", type=Path, default=Path("data/combined.yaml"))
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--bolt-detector-threshold", type=float, required=True)
    parser.add_argument("--nut-detector-threshold", type=float, required=True)
    parser.add_argument("--contexts", type=float, nargs="+", default=(0.2, 1.0))
    parser.add_argument("--classifier-imgsz", type=int, default=224)
    parser.add_argument("--classifier-batch", type=int, default=64)
    parser.add_argument("--crop-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--threshold-min", type=float, default=0.25)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-negative-fpr", type=float, default=0.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/calibration/pipeline_classifier.json"),
    )
    return parser


def _relative_area(
    bbox: tuple[float, float, float, float], image_size: tuple[int, int]
) -> float:
    """Calculate clipped proposal area divided by complete image area."""
    width, height = image_size
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
    y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height)


def _validation_images(data: Path, split: str) -> list[ValidationImage]:
    """Load unique validation image paths and their YOLO ground truth."""
    records: list[ValidationImage] = []
    seen: set[Path] = set()
    for image_root in _dataset_entries(data)[split]:
        for path in _image_paths(image_root):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            with Image.open(path) as image:
                width, height = image.size
            expected = tuple(_parse_yolo_boxes(_label_path(path, image_root), width, height))
            records.append(ValidationImage(resolved, expected))
    return records


def _classify_tasks(
    model: Any,
    tasks: list[tuple[int, Image.Image]],
    candidates: list[Candidate],
    image_size: int,
    batch: int,
    device: str,
) -> None:
    """Classify one crop chunk and add its probabilities to owning candidates."""
    if not tasks:
        return
    results = model.predict(
        source=[crop for _, crop in tasks],
        imgsz=image_size,
        batch=batch,
        device=device,
        verbose=False,
    )
    for (candidate_index, _), result in zip(tasks, results, strict=True):
        if result.probs is None:
            raise ValueError("Classifier returned no probabilities")
        candidate = candidates[candidate_index]
        for class_id, probability in enumerate(result.probs.data.tolist()):
            name = str(result.names[class_id]).lower()
            candidate.probability_sums[name] += float(probability)
        candidate.contexts += 1
    for _, crop in tasks:
        crop.close()
    tasks.clear()


def _collect(
    arguments: argparse.Namespace,
) -> tuple[list[ValidationImage], list[VerifiedCandidate]]:
    """Run detector and batched multi-context classification once on validation."""
    from ultralytics import YOLO

    images = _validation_images(arguments.data, arguments.split)
    detector = UltralyticsDetector(arguments.detector)
    detector.load()
    classifier = YOLO(str(arguments.classifier))
    configure_classifier_transforms(classifier, arguments.classifier_imgsz)
    candidates: list[Candidate] = []
    tasks: list[tuple[int, Image.Image]] = []
    detector_thresholds = {
        "bolt": arguments.bolt_detector_threshold,
        "nut": arguments.nut_detector_threshold,
    }

    for image_index, record in enumerate(images):
        with Image.open(record.path) as opened:
            image = opened.convert("RGB")
        detections = detector.predict(image, min(detector_thresholds.values()))
        for detection in detections:
            source_class = CLASS_ALIASES.get(detection.label.strip().lower())
            if source_class is None or detection.confidence < detector_thresholds[source_class]:
                continue
            if _relative_area(detection.bbox, image.size) < arguments.min_box_area_ratio:
                continue
            candidate_index = len(candidates)
            candidates.append(
                Candidate(
                    image_index=image_index,
                    detector_confidence=detection.confidence,
                    bbox=detection.bbox,
                    probability_sums={"bolt": 0.0, "nut": 0.0, "other": 0.0},
                )
            )
            for context in arguments.contexts:
                try:
                    crop = extract_square_crop(
                        image, detection.bbox, context, arguments.classifier_imgsz
                    )
                except ValueError:
                    continue
                tasks.append((candidate_index, crop))
                if len(tasks) >= arguments.crop_chunk_size:
                    _classify_tasks(
                        classifier,
                        tasks,
                        candidates,
                        arguments.classifier_imgsz,
                        arguments.classifier_batch,
                        arguments.device,
                    )
        image.close()
        if (image_index + 1) % 50 == 0:
            print(f"Collected {image_index + 1}/{len(images)} validation images")
    _classify_tasks(
        classifier,
        tasks,
        candidates,
        arguments.classifier_imgsz,
        arguments.classifier_batch,
        arguments.device,
    )

    verified: list[VerifiedCandidate] = []
    for candidate in candidates:
        probabilities = candidate.probabilities
        if not probabilities:
            continue
        class_name = max(probabilities, key=probabilities.__getitem__)
        if class_name not in TARGET_CLASSES:
            continue
        classifier_confidence = probabilities[class_name]
        verified.append(
            VerifiedCandidate(
                candidate.image_index,
                class_name,
                classifier_confidence,
                min(candidate.detector_confidence, classifier_confidence),
                candidate.bbox,
            )
        )
    return images, verified


def _nms(candidates: list[VerifiedCandidate], iou_threshold: float) -> list[VerifiedCandidate]:
    """Apply the API's confidence-sorted class-aware NMS to one class."""
    retained: list[VerifiedCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.final_confidence, reverse=True):
        if not any(_iou(candidate.bbox, accepted.bbox) > iou_threshold for accepted in retained):
            retained.append(candidate)
    return retained


def _metrics(
    images: list[ValidationImage],
    candidates: list[VerifiedCandidate],
    class_name: str,
    threshold: float,
    nms_iou: float,
) -> ClassMetrics:
    """Calculate end-to-end class metrics for one classifier threshold."""
    true_positive = false_positive = false_negative = 0
    negative_images = negative_false_positive_images = 0
    by_image: dict[int, list[VerifiedCandidate]] = {}
    for candidate in candidates:
        if candidate.class_name == class_name and candidate.classifier_confidence >= threshold:
            by_image.setdefault(candidate.image_index, []).append(candidate)
    for image_index, record in enumerate(images):
        selected = _nms(by_image.get(image_index, []), nms_iou)
        predicted = [candidate.bbox for candidate in selected]
        expected = [bbox for label, bbox in record.expected if label == class_name]
        tp, fp, fn = _match_class(predicted, expected, 0.5)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        if not expected:
            negative_images += 1
            negative_false_positive_images += int(bool(predicted))
    return ClassMetrics(
        class_name,
        threshold,
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0,
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        negative_false_positive_images / negative_images if negative_images else 0.0,
    )


def main() -> None:
    """Collect validation scores, select thresholds, and write calibration evidence."""
    arguments = build_parser().parse_args()
    if not arguments.detector.is_file() or not arguments.classifier.is_file():
        raise ValueError("Detector and classifier weights must exist")
    images, candidates = _collect(arguments)
    count = int(round((arguments.threshold_max - arguments.threshold_min) / arguments.threshold_step))
    thresholds = [round(arguments.threshold_min + index * arguments.threshold_step, 4) for index in range(count + 1)]
    grids = {
        class_name: [
            _metrics(images, candidates, class_name, threshold, arguments.nms_iou)
            for threshold in thresholds
        ]
        for class_name in TARGET_CLASSES
    }
    selected: dict[str, dict[str, object]] = {}
    for class_name, metrics in grids.items():
        recommendation, constraints_met = _select(
            metrics, arguments.min_recall, arguments.max_negative_fpr
        )
        selected[class_name] = {"constraints_met": constraints_met, **asdict(recommendation)}
    payload = {
        "task": "complete_pipeline_classifier_thresholds",
        "detector": str(arguments.detector.resolve()),
        "classifier": str(arguments.classifier.resolve()),
        "split": arguments.split,
        "contexts": list(arguments.contexts),
        "detector_thresholds": {
            "bolt": arguments.bolt_detector_threshold,
            "nut": arguments.nut_detector_threshold,
        },
        "validation_images": len(images),
        "verified_candidates": len(candidates),
        "requirements": {
            "minimum_recall": arguments.min_recall,
            "maximum_negative_false_positive_rate": arguments.max_negative_fpr,
        },
        "selected": selected,
        "candidates": {
            name: [asdict(item) for item in metrics] for name, metrics in grids.items()
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print(f"Calibration report: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
