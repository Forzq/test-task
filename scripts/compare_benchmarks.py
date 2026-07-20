"""Compare a candidate pipeline benchmark against the deployed baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """
    Create the benchmark-comparison parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with explicit baseline, candidate, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Compare pipeline quality and false positives before promotion."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/benchmarks/comparison.json")
    )
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.005)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    return parser


def _read(path: Path) -> dict[str, Any]:
    """
    Read one benchmark JSON document.

    Parameters
    ----------
    path : Path
        Existing benchmark report.

    Returns
    -------
    dict[str, Any]
        Parsed report.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    max_macro_f1_drop: float,
    max_recall_drop: float,
) -> dict[str, Any]:
    """
    Calculate deltas and promotion constraints for two pipeline reports.

    Parameters
    ----------
    baseline : dict[str, Any]
        Deployed pipeline metrics.
    candidate : dict[str, Any]
        Candidate pipeline metrics on the same benchmark.
    max_macro_f1_drop : float
        Largest acceptable macro-F1 regression.
    max_recall_drop : float
        Largest acceptable per-class recall regression.

    Returns
    -------
    dict[str, Any]
        Metric deltas and an overall promotion recommendation.
    """
    if baseline["dataset"] != candidate["dataset"] or baseline["split"] != candidate["split"]:
        raise ValueError("Benchmark dataset and split must match")
    deltas = {
        "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "negative_image_false_positive_rate": (
            candidate["negative_image_false_positive_rate"]
            - baseline["negative_image_false_positive_rate"]
        ),
        "exact_count_image_accuracy": (
            candidate["exact_count_image_accuracy"]
            - baseline["exact_count_image_accuracy"]
        ),
        "latency_mean_ms": (
            candidate["latency_ms"]["mean"] - baseline["latency_ms"]["mean"]
        ),
        "bolt_recall": (
            candidate["classes"]["bolt"]["recall"]
            - baseline["classes"]["bolt"]["recall"]
        ),
        "nut_recall": (
            candidate["classes"]["nut"]["recall"]
            - baseline["classes"]["nut"]["recall"]
        ),
    }
    constraints = {
        "macro_f1": deltas["macro_f1"] >= -max_macro_f1_drop,
        "negative_image_fpr": deltas["negative_image_false_positive_rate"] <= 0.0,
        "bolt_recall": deltas["bolt_recall"] >= -max_recall_drop,
        "nut_recall": deltas["nut_recall"] >= -max_recall_drop,
    }
    return {
        "baseline_models": baseline["models"],
        "candidate_models": candidate["models"],
        "deltas": deltas,
        "constraints": constraints,
        "promotion_recommended": all(constraints.values()),
    }


def main() -> None:
    """Compare reports, write JSON, and print the promotion decision."""
    arguments = build_parser().parse_args()
    if not arguments.baseline.is_file() or not arguments.candidate.is_file():
        raise ValueError("Both benchmark reports must exist")
    if min(arguments.max_macro_f1_drop, arguments.max_recall_drop) < 0.0:
        raise ValueError("Allowed metric drops cannot be negative")
    report = compare(
        _read(arguments.baseline),
        _read(arguments.candidate),
        arguments.max_macro_f1_drop,
        arguments.max_recall_drop,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
