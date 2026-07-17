"""Select a YOLO checkpoint using positive recall and negative-image FPR."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from calibrate_threshold import ThresholdMetrics, calibrate


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    """
    Best threshold metrics retained for one checkpoint.

    Parameters
    ----------
    checkpoint : str
        Evaluated weights path.
    metrics : ThresholdMetrics
        Best validation operating point satisfying the recall floor.
    satisfies_fpr : bool
        Whether the operating point also satisfies the requested FPR ceiling.
    """

    checkpoint: str
    metrics: ThresholdMetrics
    satisfies_fpr: bool


def build_parser() -> argparse.ArgumentParser:
    """
    Create the checkpoint-selection command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for periodic checkpoint evaluation and deployment.
    """
    parser = argparse.ArgumentParser(
        description="Choose a checkpoint by recall-constrained negative-image FPR."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML path.")
    parser.add_argument(
        "--weights-dir", type=Path, required=True, help="Training run weights directory."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/best.pt"), help="Selected deployable weights."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("runs/checkpoint_selection.json"),
        help="JSON report destination.",
    )
    parser.add_argument("--min-recall", type=float, default=0.90, help="Required recall floor.")
    parser.add_argument(
        "--max-negative-image-fpr",
        type=float,
        default=0.03,
        help="Desired empty-image false-positive-rate ceiling.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--batch", type=int, default=8, help="Inference batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics inference device.")
    parser.add_argument("--min-confidence", type=float, default=0.20, help="Threshold-grid start.")
    parser.add_argument("--max-confidence", type=float, default=0.95, help="Threshold-grid end.")
    parser.add_argument("--step", type=float, default=0.02, help="Threshold-grid step.")
    return parser


def _checkpoint_paths(weights_directory: Path) -> list[Path]:
    """
    Enumerate unique best, last, and periodic checkpoints.

    Parameters
    ----------
    weights_directory : Path
        Ultralytics weights directory.

    Returns
    -------
    list[Path]
        Deterministically ordered checkpoint paths.
    """
    preferred = [weights_directory / "best.pt", weights_directory / "last.pt"]
    periodic = sorted(weights_directory.glob("epoch*.pt"))
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in [*preferred, *periodic]:
        resolved = path.resolve()
        if path.is_file() and resolved not in seen:
            paths.append(resolved)
            seen.add(resolved)
    return paths


def _operating_point(
    metrics: list[ThresholdMetrics], minimum_recall: float, maximum_fpr: float
) -> tuple[ThresholdMetrics, bool]:
    """
    Choose the lowest-FPR threshold that preserves the required recall.

    Parameters
    ----------
    metrics : list[ThresholdMetrics]
        Threshold calibration results.
    minimum_recall : float
        Required object recall.
    maximum_fpr : float
        Desired negative-image FPR ceiling.

    Returns
    -------
    tuple[ThresholdMetrics, bool]
        Selected operating point and FPR-constraint status.
    """
    candidates = [metric for metric in metrics if metric.recall >= minimum_recall]
    if not candidates:
        candidates = metrics
    selected = min(
        candidates,
        key=lambda item: (
            item.negative_image_fpr,
            -item.recall,
            -item.precision,
            item.threshold,
        ),
    )
    return selected, selected.recall >= minimum_recall and selected.negative_image_fpr <= maximum_fpr


def _evaluate_checkpoint(path: Path, arguments: argparse.Namespace) -> CheckpointEvaluation:
    """
    Calibrate one checkpoint and retain its recall-constrained operating point.

    Parameters
    ----------
    path : Path
        Checkpoint to evaluate.
    arguments : argparse.Namespace
        Shared dataset, threshold-grid, and device settings.

    Returns
    -------
    CheckpointEvaluation
        Comparable checkpoint evidence.
    """
    calibration_arguments = argparse.Namespace(
        data=arguments.data,
        weights=path,
        split="val",
        imgsz=arguments.imgsz,
        device=arguments.device,
        batch=arguments.batch,
        min_confidence=arguments.min_confidence,
        max_confidence=arguments.max_confidence,
        step=arguments.step,
        min_recall=arguments.min_recall,
        max_negative_image_fpr=arguments.max_negative_image_fpr,
        output=arguments.report,
    )
    _, metrics, _, _ = calibrate(calibration_arguments)
    selected, satisfies_fpr = _operating_point(
        metrics, arguments.min_recall, arguments.max_negative_image_fpr
    )
    return CheckpointEvaluation(str(path), selected, satisfies_fpr)


def _rank(evaluation: CheckpointEvaluation) -> tuple[float, ...]:
    """
    Return the lexicographic checkpoint ranking key.

    Parameters
    ----------
    evaluation : CheckpointEvaluation
        Checkpoint evidence to rank.

    Returns
    -------
    tuple[float, ...]
        Constraint status, FPR, negative recall, negative precision, and threshold.
    """
    metrics = evaluation.metrics
    return (
        0.0 if evaluation.satisfies_fpr else 1.0,
        metrics.negative_image_fpr,
        -metrics.recall,
        -metrics.precision,
        metrics.threshold,
    )


def main() -> None:
    """
    Evaluate periodic checkpoints, copy the winner, and persist evidence.
    """
    arguments = build_parser().parse_args()
    checkpoints = _checkpoint_paths(arguments.weights_dir)
    if not arguments.data.is_file() or not checkpoints:
        raise ValueError("Dataset YAML and at least one checkpoint must exist")
    if not 0.0 <= arguments.max_negative_image_fpr <= 1.0:
        raise ValueError("max-negative-image-fpr must be between zero and one")
    if not 0.0 <= arguments.min_recall <= 1.0:
        raise ValueError("min-recall must be between zero and one")

    evaluations: list[CheckpointEvaluation] = []
    for checkpoint in checkpoints:
        evaluation = _evaluate_checkpoint(checkpoint, arguments)
        evaluations.append(evaluation)
        print(
            f"{checkpoint.name}: threshold={evaluation.metrics.threshold:.2f}, "
            f"recall={evaluation.metrics.recall:.3f}, "
            f"precision={evaluation.metrics.precision:.3f}, "
            f"negative_fpr={evaluation.metrics.negative_image_fpr:.3f}"
        )
    selected = min(evaluations, key=_rank)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected.checkpoint, arguments.output)
    report = {
        "selection_policy": (
            "Prefer checkpoints satisfying the recall floor and FPR ceiling; then minimize "
            "negative-image FPR and maximize recall and precision."
        ),
        "minimum_recall": arguments.min_recall,
        "maximum_negative_image_fpr": arguments.max_negative_image_fpr,
        "selected": {
            "checkpoint": selected.checkpoint,
            "metrics": asdict(selected.metrics),
            "satisfies_fpr": selected.satisfies_fpr,
        },
        "evaluations": [
            {
                "checkpoint": item.checkpoint,
                "metrics": asdict(item.metrics),
                "satisfies_fpr": item.satisfies_fpr,
            }
            for item in evaluations
        ],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected checkpoint: {selected.checkpoint}")
    print(f"Deployed weights: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
