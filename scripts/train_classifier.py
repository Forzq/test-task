"""Fine-tune a three-class crop verifier without cropping long objects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    """
    Parameters for reproducible crop-classifier fine-tuning.

    Parameters
    ----------
    data : Path
        Classification root containing split and class directories.
    model : str
        Pretrained Ultralytics classification checkpoint.
    epochs : int
        Maximum number of training epochs.
    image_size : int
        Square model input size.
    batch_size : int
        Training images per optimisation step.
    device : str
        Ultralytics device selector such as ``0`` or ``cpu``.
    project : Path
        Training artefact directory.
    run_name : str
        Experiment name below the project directory.
    seed : int
        Reproducibility seed.
    workers : int
        Data-loader worker process count.
    patience : int
        Epochs without validation improvement before early stopping.
    save_period : int
        Periodic checkpoint interval in epochs.
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
    patience: int
    save_period: int


def build_parser() -> argparse.ArgumentParser:
    """
    Create the crop-classifier training argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with defaults suitable for an RTX 2060 Super.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n-cls on bolt, nut, and other crops."
    )
    parser.add_argument("--data", type=Path, default=Path("data/classifier"))
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/classify"))
    parser.add_argument("--name", default="crop_verifier_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--save-period", type=int, default=5)
    return parser


def parse_config() -> ClassifierTrainingConfig:
    """
    Parse and validate crop-classifier training options.

    Returns
    -------
    ClassifierTrainingConfig
        Validated immutable training configuration.
    """
    arguments = build_parser().parse_args()
    config = ClassifierTrainingConfig(
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
        patience=arguments.patience,
        save_period=arguments.save_period,
    )
    _validate_config(config)
    return config


def _validate_config(config: ClassifierTrainingConfig) -> None:
    """
    Validate classifier directory structure and numeric options.

    Parameters
    ----------
    config : ClassifierTrainingConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        Raised when input folders or numeric parameters are invalid.
    """
    required_directories = [
        config.data / split / class_name
        for split in ("train", "val", "test")
        for class_name in ("bolt", "nut", "other")
    ]
    missing = [path for path in required_directories if not path.is_dir()]
    if missing:
        raise ValueError(f"Classifier dataset directory was not found: {missing[0]}")
    if min(config.epochs, config.image_size, config.batch_size, config.save_period) <= 0:
        raise ValueError("epochs, imgsz, batch, and save-period must be positive")
    if config.workers < 0 or config.patience < 0:
        raise ValueError("workers and patience cannot be negative")


def _build_transforms(image_size: int, augment: bool) -> Any:
    """
    Build aspect-preserving full-frame classification transforms.

    Parameters
    ----------
    image_size : int
        Square model input size.
    augment : bool
        Whether stochastic train-time transformations are enabled.

    Returns
    -------
    torchvision.transforms.Compose
        Transform pipeline compatible with Ultralytics classification datasets.

    Notes
    -----
    The source crops are already square and letterboxed. Resize and affine
    transforms retain the entire candidate, unlike RandomResizedCrop, which can
    remove the head or shaft of a long bolt.
    """
    from torchvision import transforms as transforms

    operations: list[Any] = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        )
    ]
    if augment:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.10),
                    fill=(114, 114, 114),
                ),
                transforms.ColorJitter(
                    brightness=0.20,
                    contrast=0.20,
                    saturation=0.20,
                    hue=0.02,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    if augment:
        operations.append(transforms.RandomErasing(p=0.10, scale=(0.02, 0.10)))
    return transforms.Compose(operations)


def train(config: ClassifierTrainingConfig) -> Path:
    """
    Fine-tune the crop classifier and return its best checkpoint path.

    Parameters
    ----------
    config : ClassifierTrainingConfig
        Fully validated training configuration.

    Returns
    -------
    Path
        Best validation checkpoint generated by Ultralytics.

    Notes
    -----
    Validation-driven early stopping is enabled through ``patience``. The
    custom dataset class avoids the default RandomResizedCrop transformation.
    """
    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer

    class FullFrameClassificationTrainer(ClassificationTrainer):
        """Classification trainer that retains complete candidate geometry."""

        def build_dataset(
            self, image_path: str, mode: str = "train", batch: Any | None = None
        ) -> ClassificationDataset:
            """
            Construct one split with full-frame train or validation transforms.

            Parameters
            ----------
            image_path : str
                Classification split directory selected by Ultralytics.
            mode : str, optional
                Dataset mode; only ``train`` enables stochastic augmentation.
            batch : object, optional
                Unused compatibility parameter supplied by the trainer.

            Returns
            -------
            ClassificationDataset
                Ultralytics dataset with task-specific transforms.
            """
            del batch
            dataset = ClassificationDataset(
                root=image_path,
                args=self.args,
                augment=mode == "train",
                prefix=mode,
            )
            dataset.torch_transforms = _build_transforms(
                image_size=int(self.args.imgsz),
                augment=mode == "train",
            )
            return dataset

    model = YOLO(config.model)
    model.train(
        data=str(config.data.resolve()),
        epochs=config.epochs,
        imgsz=config.image_size,
        batch=config.batch_size,
        device=config.device,
        project=str(config.project.resolve()),
        name=config.run_name,
        seed=config.seed,
        workers=config.workers,
        patience=config.patience,
        save_period=config.save_period,
        pretrained=True,
        cache=False,
        trainer=FullFrameClassificationTrainer,
    )
    best_weights = Path(model.trainer.best)
    print(f"Best classifier checkpoint: {best_weights}")
    return best_weights


def main() -> None:
    """Start crop-classifier fine-tuning from command-line arguments."""
    train(parse_config())


if __name__ == "__main__":
    main()
