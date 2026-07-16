"""Command-line script for fine-tuning a YOLO model on nuts and bolts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """
    Parameters required for a reproducible Ultralytics training run.

    Parameters
    ----------
    data : Path
        Path to the YOLO dataset YAML file.
    model : str
        Pretrained Ultralytics checkpoint to fine-tune.
    epochs : int
        Number of full passes over the training dataset.
    image_size : int
        Input image edge size used by YOLO.
    batch_size : int
        Number of images processed per optimisation step.
    device : str
        Ultralytics device selector, for example ``0`` for the first GPU.
    project : Path
        Directory in which Ultralytics stores training outputs.
    run_name : str
        Name of the current training experiment.
    seed : int
        Random seed used to improve run reproducibility.
    workers : int
        Number of data-loader worker processes.
    """

    data: Path
    model: str
    epochs: int
    image_size: int
    batch_size: int
    device: str
    project: Path
    run_name: str
    seed: int
    workers: int


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser for fine-tuning.

    Returns
    -------
    argparse.ArgumentParser
        Parser with documented defaults suitable for an RTX 2060 Super.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on a nut and bolt detection dataset."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML path.")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained checkpoint to fine-tune (default: yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Output directory for run artefacts.",
    )
    parser.add_argument("--name", default="nut_bolt", help="Training run name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=4, help="Data-loader workers.")
    return parser


def parse_config() -> TrainingConfig:
    """
    Parse command-line options into an immutable training configuration.

    Returns
    -------
    TrainingConfig
        Validated command-line configuration.

    Raises
    ------
    SystemExit
        Raised by argparse when required arguments are invalid or missing.
    """
    arguments = build_parser().parse_args()
    config = TrainingConfig(
        data=arguments.data,
        model=arguments.model,
        epochs=arguments.epochs,
        image_size=arguments.imgsz,
        batch_size=arguments.batch,
        device=arguments.device,
        project=arguments.project,
        run_name=arguments.name,
        seed=arguments.seed,
        workers=arguments.workers,
    )
    _validate_config(config)
    return config


def _validate_config(config: TrainingConfig) -> None:
    """
    Check local paths and numeric training parameters before starting training.

    Parameters
    ----------
    config : TrainingConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        Raised when the dataset file is absent or a numeric option is invalid.
    """
    if not config.data.is_file():
        raise ValueError(f"Dataset YAML was not found: {config.data}")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.image_size <= 0:
        raise ValueError("imgsz must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch must be positive")
    if config.workers < 0:
        raise ValueError("workers cannot be negative")


def train(config: TrainingConfig) -> Path:
    """
    Fine-tune a pretrained YOLO checkpoint and return the produced best weights.

    Parameters
    ----------
    config : TrainingConfig
        Fully validated training parameters.

    Returns
    -------
    Path
        Expected location of the best validation checkpoint.

    Notes
    -----
    Ultralytics validates after every epoch using the validation split declared
    in the dataset YAML. The final run also writes metrics to results.csv.
    """
    from ultralytics import YOLO

    output_project = config.project.resolve()
    model = YOLO(config.model)
    model.train(
        data=str(config.data),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=config.device,
        project=str(output_project),
        name=config.run_name,
        seed=config.seed,
        workers=config.workers,
        pretrained=True,
    )
    best_weights = Path(model.trainer.best)
    print(f"Best checkpoint: {best_weights}")
    return best_weights


def main() -> None:
    """
    Run a fine-tuning job from command-line arguments.

    Raises
    ------
    ValueError
        Raised when configuration validation fails before model training.
    """
    train(parse_config())


if __name__ == "__main__":
    main()
