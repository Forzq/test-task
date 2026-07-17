"""Command-line script for evaluating a trained YOLO checkpoint on a test split."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line parser for held-out test evaluation.

    Returns
    -------
    argparse.ArgumentParser
        Parser defining paths and inference options for model validation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained nut and bolt detector on the test split."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML path.")
    parser.add_argument("--weights", type=Path, required=True, help="Trained best.pt path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
    return parser


def main() -> None:
    """
    Evaluate a checkpoint on the dataset test split and print metrics.

    Raises
    ------
    ValueError
        Raised when the required data or weights path does not exist.
    """
    arguments = build_parser().parse_args()
    if not arguments.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {arguments.data}")
    if not arguments.weights.is_file():
        raise ValueError(f"Model weights were not found: {arguments.weights}")

    from ultralytics import YOLO

    model = YOLO(str(arguments.weights))
    metrics = model.val(
        data=str(arguments.data),
        split="test",
        imgsz=arguments.imgsz,
        device=arguments.device,
    )
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
