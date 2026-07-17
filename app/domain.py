"""Domain entities and interfaces for object detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class RawDetection:
    """
    A single detection produced by a model before API-specific filtering.

    Parameters
    ----------
    label : str
        Class label emitted by the model.
    confidence : float
        Model confidence score in the range from zero to one.
    bbox : tuple[float, float, float, float]
        Bounding box represented as x1, y1, x2, y2 pixel coordinates.
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CropClassification:
    """
    A classification result for one detector-generated image crop.

    Parameters
    ----------
    label : str
        Classifier label such as ``bolt``, ``nut``, or ``other``.
    confidence : float
        Classifier confidence score in the range from zero to one.
    """

    label: str
    confidence: float


class DetectorProtocol(Protocol):
    """Interface implemented by detection model adapters."""

    def predict(
        self, image: Image, confidence_threshold: float
    ) -> list[RawDetection]:
        """
        Run object detection for a decoded RGB image.

        Parameters
        ----------
        image : PIL.Image.Image
            Decoded input image in RGB mode.
        confidence_threshold : float
            Lowest confidence score that should be retained by the model.

        Returns
        -------
        list[RawDetection]
            Raw detections returned by the underlying model.
        """


class CropClassifierProtocol(Protocol):
    """Interface implemented by crop-classification model adapters."""

    def predict(self, image: Image) -> CropClassification:
        """
        Classify a detector-generated RGB crop.

        Parameters
        ----------
        image : PIL.Image.Image
            Square RGB crop containing a detector candidate and context.

        Returns
        -------
        CropClassification
            Predicted ``bolt``, ``nut``, or ``other`` class and confidence.
        """
