"""Ultralytics classification adapter used to verify detector crops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL.Image import Image

from app.domain import CropClassification

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class ClassifierPreprocessor:
    """
    Configure classifier inference transforms to match model training.

    Parameters
    ----------
    image_size : int
        Square input dimensions expected by the classifier.
    mean : tuple[float, float, float]
        Per-channel RGB normalisation mean.
    standard_deviation : tuple[float, float, float]
        Per-channel RGB normalisation standard deviation.
    """

    image_size: int
    mean: tuple[float, float, float] = IMAGENET_MEAN
    standard_deviation: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        """
        Validate preprocessing dimensions at construction time.

        Raises
        ------
        ValueError
            Raised when the classifier image size is not positive.
        """
        if self.image_size <= 0:
            raise ValueError("Classifier image size must be positive")

    def configure(self, model: Any) -> None:
        """
        Install deterministic inference transforms on an Ultralytics model.

        Parameters
        ----------
        model : object
            Loaded Ultralytics YOLO classification wrapper.

        Notes
        -----
        Ultralytics defaults use identity channel normalisation. The project
        trains with ImageNet statistics, so inference must install the same
        transform explicitly.
        """
        from ultralytics.data.augment import classify_transforms

        model.model.transforms = classify_transforms(
            size=self.image_size,
            mean=self.mean,
            std=self.standard_deviation,
        )


def configure_classifier_transforms(model: Any, image_size: int) -> None:
    """
    Configure inference preprocessing to match classifier training exactly.

    Parameters
    ----------
    model : object
        Loaded Ultralytics YOLO classification wrapper.
    image_size : int
        Square classifier input size.

    Notes
    -----
    Ultralytics classification defaults use identity channel normalisation.
    This project trains with ImageNet mean and standard deviation, so the same
    transform must be explicitly installed for standalone prediction.
    """
    ClassifierPreprocessor(image_size=image_size).configure(model)


class ClassifierUnavailableError(RuntimeError):
    """Raised when crop-classifier weights are missing or not loaded."""


class UltralyticsCropClassifier:
    """
    Adapter that converts Ultralytics classification output into domain data.

    Parameters
    ----------
    model_path : Path
        Location of the trained classification checkpoint.
    image_size : int
        Square inference image size expected by the classifier.
    """

    _EXPECTED_CLASSES = {"bolt", "nut", "other"}

    def __init__(self, model_path: Path, image_size: int = 224) -> None:
        """
        Store classifier settings without loading model resources yet.

        Parameters
        ----------
        model_path : Path
            Location of the trained classification checkpoint.
        image_size : int, optional
            Square inference image size expected by the classifier.
        """
        self._model_path = model_path
        self._image_size = image_size
        self._preprocessor = ClassifierPreprocessor(image_size)
        self._model: Any | None = None

    def load(self) -> None:
        """
        Load and validate the crop-classification checkpoint.

        Raises
        ------
        ClassifierUnavailableError
            Raised when weights are missing, invalid, or use wrong class names.
        """
        if not self._model_path.is_file():
            raise ClassifierUnavailableError(
                f"Classifier weights were not found at '{self._model_path}'. "
                "Train the crop classifier and set CLASSIFIER_MODEL_PATH."
            )

        try:
            from ultralytics import YOLO

            model = YOLO(str(self._model_path))
            self._preprocessor.configure(model)
            class_names = {
                str(name).strip().lower() for name in model.names.values()
            }
            if class_names != self._EXPECTED_CLASSES:
                raise ValueError(
                    "Classifier classes must be exactly bolt, nut, and other; "
                    f"received {sorted(class_names)}."
                )
            self._model = model
        except Exception as error:  # pragma: no cover - depends on local ML runtime
            raise ClassifierUnavailableError(
                f"Unable to load classifier weights from '{self._model_path}'."
            ) from error

    def predict(self, image: Image) -> CropClassification:
        """
        Classify one square candidate crop.

        Parameters
        ----------
        image : PIL.Image.Image
            RGB crop generated from a detector bounding box.

        Returns
        -------
        CropClassification
            Highest-probability class and its confidence.

        Raises
        ------
        ClassifierUnavailableError
            Raised when the classifier has not been loaded or returns no scores.
        """
        if self._model is None:
            raise ClassifierUnavailableError("The crop classifier has not been loaded.")

        results = self._model.predict(
            source=image,
            imgsz=self._image_size,
            verbose=False,
        )
        if not results or results[0].probs is None:
            raise ClassifierUnavailableError(
                "The crop classifier returned no probabilities."
            )

        result = results[0]
        class_id = int(result.probs.top1)
        probabilities = {
            str(result.names[index]).strip().lower(): float(probability)
            for index, probability in enumerate(result.probs.data.tolist())
        }
        return CropClassification(
            label=str(result.names[class_id]).strip().lower(),
            confidence=float(result.probs.top1conf.item()),
            probabilities=probabilities,
        )
