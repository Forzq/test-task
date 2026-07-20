"""Automate hard-negative mining and dependent dataset regeneration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """
    Create the hard-negative refresh command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser covering mining, combination, and crop generation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mine detector mistakes, rebuild combined.yaml, and regenerate "
            "the multi-scale crop-classifier dataset."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_negatives/caltech-101/101_ObjectCategories"),
    )
    parser.add_argument("--detector", type=Path, default=Path("models/best.pt"))
    parser.add_argument(
        "--strict-additional",
        type=Path,
        default=Path("data/arg_fixings_training"),
        help="Optional strict whole-object training source when prepared.",
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--scan-limit", type=int, default=3000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--skip-mining", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(command: list[str], dry_run: bool) -> None:
    """
    Print and optionally execute one pipeline command.

    Parameters
    ----------
    command : list[str]
        Argument-safe process command.
    dry_run : bool
        Whether execution should be skipped.
    """
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def refresh(arguments: argparse.Namespace) -> None:
    """
    Execute mining, dataset combination, and crop regeneration in order.

    Parameters
    ----------
    arguments : argparse.Namespace
        Validated refresh options.
    """
    python = sys.executable
    if not arguments.skip_mining:
        _run(
            [
                python,
                "scripts/mine_hard_negatives.py",
                "--source",
                str(arguments.source),
                "--weights",
                str(arguments.detector),
                "--output",
                "data/hard_negatives",
                "--limit",
                str(arguments.limit),
                "--scan-limit",
                str(arguments.scan_limit),
                "--device",
                arguments.device,
                "--batch",
                str(arguments.batch),
                "--chunk-size",
                str(arguments.chunk_size),
                "--replace",
            ],
            arguments.dry_run,
        )
    combine_command = [
            python,
            "scripts/combine_datasets.py",
            "--additional",
            "data/external_positives",
            "--negatives",
            "data/negatives",
            "--negative-additional",
            "data/hard_negatives",
            "--negative-additional",
            "data/manual_hard_negatives",
        ]
    if arguments.strict_additional.is_dir():
        combine_command.extend(["--additional", str(arguments.strict_additional)])
    _run(combine_command, arguments.dry_run)
    _run(
        [
            python,
            "scripts/prepare_classifier_dataset.py",
            "--data",
            "data/combined.yaml",
            "--weights",
            str(arguments.detector),
            "--output",
            "data/classifier",
            "--contexts",
            "0.2",
            "1.0",
            "--device",
            arguments.device,
            "--batch",
            str(arguments.batch),
            "--chunk-size",
            str(arguments.chunk_size),
            "--replace",
        ],
        arguments.dry_run,
    )


def main() -> None:
    """Validate refresh inputs and run the automation pipeline."""
    arguments = build_parser().parse_args()
    if not arguments.detector.is_file():
        raise ValueError(f"Detector weights were not found: {arguments.detector}")
    if not arguments.skip_mining and not arguments.source.is_dir():
        raise ValueError(f"Negative source was not found: {arguments.source}")
    if min(arguments.limit, arguments.scan_limit, arguments.batch, arguments.chunk_size) <= 0:
        raise ValueError("limits, batch, and chunk-size must be positive")
    refresh(arguments)


if __name__ == "__main__":
    main()
