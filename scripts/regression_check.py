"""Run exact user-reported regression cases through the deployed pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True, slots=True)
class RegressionCase:
    """
    Expected object counts for one regression image.

    Parameters
    ----------
    image : Path
        Image supplied to the detector.
    bolts : int
        Expected number of bolts.
    nuts : int
        Expected number of nuts.
    """

    image: Path
    bolts: int
    nuts: int


def build_parser() -> argparse.ArgumentParser:
    """
    Create the regression-check command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with deployable model defaults.
    """
    parser = argparse.ArgumentParser(
        description="Check deployed detector and classifier counts on fixed images."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/regression/cases.json"),
        help="JSON file containing image paths and expected counts.",
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("models/best.pt"), help="YOLO checkpoint."
    )
    parser.add_argument(
        "--classifier-weights",
        type=Path,
        default=Path("models/classifier_best.pt"),
        help="Crop-classifier checkpoint.",
    )
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--classifier-confidence", type=float, default=0.54)
    parser.add_argument("--bolt-detector-threshold", type=float, default=None)
    parser.add_argument("--nut-detector-threshold", type=float, default=None)
    parser.add_argument("--bolt-classifier-threshold", type=float, default=None)
    parser.add_argument("--nut-classifier-threshold", type=float, default=None)
    parser.add_argument("--contexts", type=float, nargs="+", default=[0.20])
    parser.add_argument("--min-box-area-ratio", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    return parser


def _read_cases(path: Path) -> list[RegressionCase]:
    """
    Parse regression cases and resolve paths relative to their JSON file.

    Parameters
    ----------
    path : Path
        Case manifest path.

    Returns
    -------
    list[RegressionCase]
        Validated expected-count cases.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[RegressionCase] = []
    for item in payload["cases"]:
        image = Path(item["image"])
        if not image.is_absolute():
            image = (path.parent / image).resolve()
        counts = item["counts"]
        cases.append(RegressionCase(image, int(counts["bolts"]), int(counts["nuts"])))
    return cases


def run(arguments: argparse.Namespace) -> bool:
    """
    Execute model inference and compare exact per-class counts.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed regression options.

    Returns
    -------
    bool
        ``True`` only when every case matches its expected counts.
    """
    from app.classifier import UltralyticsCropClassifier
    from app.detector import UltralyticsDetector
    from app.service import DetectionService

    cases = _read_cases(arguments.cases)
    missing = [str(case.image) for case in cases if not case.image.is_file()]
    if missing:
        raise ValueError(f"Regression images are missing: {', '.join(missing)}")
    detector = UltralyticsDetector(arguments.weights)
    detector.load()
    classifier = UltralyticsCropClassifier(arguments.classifier_weights, image_size=224)
    classifier.load()
    service = DetectionService(
        detector=detector,
        confidence_threshold=arguments.confidence,
        max_upload_size_bytes=20 * 1024 * 1024,
        crop_classifier=classifier,
        classifier_confidence_threshold=arguments.classifier_confidence,
        classifier_crop_context=0.20,
        classifier_image_size=224,
        min_box_area_ratio=arguments.min_box_area_ratio,
        detector_class_thresholds={
            "bolt": arguments.bolt_detector_threshold
            if arguments.bolt_detector_threshold is not None
            else arguments.confidence,
            "nut": arguments.nut_detector_threshold
            if arguments.nut_detector_threshold is not None
            else arguments.confidence,
        },
        classifier_class_thresholds={
            "bolt": arguments.bolt_classifier_threshold
            if arguments.bolt_classifier_threshold is not None
            else arguments.classifier_confidence,
            "nut": arguments.nut_classifier_threshold
            if arguments.nut_classifier_threshold is not None
            else arguments.classifier_confidence,
        },
        classifier_crop_contexts=tuple(arguments.contexts),
        final_nms_iou_threshold=arguments.nms_iou,
    )
    passed = True
    for case in cases:
        response = service.predict(case.image.read_bytes())
        counts = {"bolt": response.counts.bolts, "nut": response.counts.nuts}
        detections = [
            f"{item.class_name}:{item.confidence:.3f}" for item in response.objects
        ]
        expected = {"bolt": case.bolts, "nut": case.nuts}
        case_passed = counts == expected
        passed &= case_passed
        print(
            f"{'PASS' if case_passed else 'FAIL'} {case.image.name}: "
            f"expected={expected}, actual={counts}, detections={detections}"
        )
    return passed


def main() -> None:
    """
    Run regression cases and return a failing process code on mismatch.
    """
    arguments = build_parser().parse_args()
    if (
        not arguments.weights.is_file()
        or not arguments.classifier_weights.is_file()
        or not arguments.cases.is_file()
    ):
        raise ValueError("Detector, classifier, and regression case files must exist")
    for name in (
        "confidence",
        "classifier_confidence",
        "bolt_detector_threshold",
        "nut_detector_threshold",
        "bolt_classifier_threshold",
        "nut_classifier_threshold",
        "min_box_area_ratio",
        "nms_iou",
    ):
        value = getattr(arguments, name)
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be between zero and one")
    if not arguments.contexts or any(context < 0.0 for context in arguments.contexts):
        raise ValueError("contexts must contain non-negative values")
    if not run(arguments):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
