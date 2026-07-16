"""Image validation, class mapping, and prediction aggregation logic."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.domain import DetectorProtocol
from app.schemas import DetectedObject, DetectionCounts, PredictionResponse


class UnsupportedImageError(ValueError):
    """Raised when an uploaded file is not a supported JPEG or PNG image."""


class DetectionService:
    """
    Application service that prepares images and formats model predictions.

    Parameters
    ----------
    detector : DetectorProtocol
        Loaded model adapter responsible for low-level object detection.
    confidence_threshold : float
        Lowest confidence score allowed in the API response.
    max_upload_size_bytes : int
        Maximum accepted size of an uploaded image.
    """

    _CLASS_ALIASES = {
        "nut": "nut",
        "nuts": "nut",
        "bolt": "bolt",
        "bolts": "bolt",
    }
    _SUPPORTED_FORMATS = {"JPEG", "PNG"}

    def __init__(
        self,
        detector: DetectorProtocol,
        confidence_threshold: float,
        max_upload_size_bytes: int,
    ) -> None:
        """
        Initialise the service with a detector and upload constraints.

        Parameters
        ----------
        detector : DetectorProtocol
            Model adapter used for inference.
        confidence_threshold : float
            Minimum confidence retained in responses.
        max_upload_size_bytes : int
            Maximum number of bytes allowed in an uploaded image.
        """
        self._detector = detector
        self._confidence_threshold = confidence_threshold
        self._max_upload_size_bytes = max_upload_size_bytes

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """
        Validate an image, run inference, and calculate class counts.

        Parameters
        ----------
        image_bytes : bytes
            Raw bytes received from an HTTP image upload.

        Returns
        -------
        PredictionResponse
            Accepted nut and bolt detections with aggregated counts.

        Raises
        ------
        UnsupportedImageError
            Raised when the file is empty, too large, corrupt, or not JPEG/PNG.
        """
        image = self._decode_image(image_bytes)
        raw_detections = self._detector.predict(image, self._confidence_threshold)

        objects: list[DetectedObject] = []
        counts = {"nuts": 0, "bolts": 0}

        for detection in raw_detections:
            class_name = self._normalise_class_name(detection.label)
            if class_name is None or detection.confidence < self._confidence_threshold:
                continue

            bbox = [max(0, round(coordinate)) for coordinate in detection.bbox]
            objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=round(detection.confidence, 4),
                    bbox=bbox,
                )
            )
            counts[f"{class_name}s"] += 1

        return PredictionResponse(
            objects=objects,
            counts=DetectionCounts(**counts),
        )

    def _decode_image(self, image_bytes: bytes) -> Image.Image:
        """
        Decode a JPEG or PNG byte stream into a safe RGB image.

        Parameters
        ----------
        image_bytes : bytes
            Raw uploaded file content.

        Returns
        -------
        PIL.Image.Image
            Decoded image converted to RGB colour mode.

        Raises
        ------
        UnsupportedImageError
            Raised when size or image format validation fails.
        """
        if not image_bytes:
            raise UnsupportedImageError("The uploaded file is empty.")
        if len(image_bytes) > self._max_upload_size_bytes:
            raise UnsupportedImageError(
                f"The uploaded image exceeds {self._max_upload_size_bytes} bytes."
            )

        try:
            with Image.open(BytesIO(image_bytes)) as opened_image:
                if opened_image.format not in self._SUPPORTED_FORMATS:
                    raise UnsupportedImageError("Only JPEG and PNG images are supported.")
                return opened_image.convert("RGB")
        except UnidentifiedImageError as error:
            raise UnsupportedImageError("The uploaded file is not a valid image.") from error

    def _normalise_class_name(self, label: str) -> str | None:
        """
        Convert a model label to one of the public API classes.

        Parameters
        ----------
        label : str
            Raw label returned by the detection model.

        Returns
        -------
        str or None
            Canonical ``nut`` or ``bolt`` label, or None for unknown classes.

        Notes
        -----
        Unknown model classes are intentionally excluded from the API response.
        This prevents unsupported object types from being silently counted.
        """
        return self._CLASS_ALIASES.get(label.strip().lower())
