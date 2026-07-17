"""Image validation, class mapping, and prediction aggregation logic."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.domain import CropClassifierProtocol, DetectorProtocol
from app.image_utils import extract_square_crop
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
    crop_classifier : CropClassifierProtocol, optional
        Second-stage classifier that can reject candidates as ``other``.
    classifier_confidence_threshold : float
        Minimum classifier confidence required to retain a candidate.
    classifier_crop_context : float
        Context added around each detector bounding box.
    classifier_image_size : int
        Square crop size passed to the classifier.
    min_box_area_ratio : float
        Minimum candidate area divided by complete image area.
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
        crop_classifier: CropClassifierProtocol | None = None,
        classifier_confidence_threshold: float = 0.54,
        classifier_crop_context: float = 0.20,
        classifier_image_size: int = 224,
        min_box_area_ratio: float = 0.001,
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
        crop_classifier : CropClassifierProtocol, optional
            Optional second-stage candidate verifier.
        classifier_confidence_threshold : float, optional
            Minimum classification confidence required for acceptance.
        classifier_crop_context : float, optional
            Extra context added around detector boxes.
        classifier_image_size : int, optional
            Square crop width and height.
        min_box_area_ratio : float, optional
            Minimum relative area retained before crop classification.
        """
        self._detector = detector
        self._confidence_threshold = confidence_threshold
        self._max_upload_size_bytes = max_upload_size_bytes
        self._crop_classifier = crop_classifier
        self._classifier_confidence_threshold = classifier_confidence_threshold
        self._classifier_crop_context = classifier_crop_context
        self._classifier_image_size = classifier_image_size
        self._min_box_area_ratio = min_box_area_ratio

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
            if self._relative_box_area(detection.bbox, image.size) < self._min_box_area_ratio:
                continue

            final_confidence = detection.confidence
            if self._crop_classifier is not None:
                try:
                    crop = extract_square_crop(
                        image=image,
                        bbox=detection.bbox,
                        context_ratio=self._classifier_crop_context,
                        output_size=self._classifier_image_size,
                    )
                except ValueError:
                    continue
                classification = self._crop_classifier.predict(crop)
                verified_class = self._normalise_class_name(classification.label)
                if (
                    verified_class is None
                    or classification.confidence < self._classifier_confidence_threshold
                ):
                    continue
                class_name = verified_class
                final_confidence = min(detection.confidence, classification.confidence)

            bbox = [max(0, round(coordinate)) for coordinate in detection.bbox]
            objects.append(
                DetectedObject(
                    class_name=class_name,
                    confidence=round(final_confidence, 4),
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

    @staticmethod
    def _relative_box_area(
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> float:
        """
        Calculate a clipped bounding-box area relative to the full image.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Candidate box in ``x1, y1, x2, y2`` coordinates.
        image_size : tuple[int, int]
            Source image width and height.

        Returns
        -------
        float
            Clipped box area divided by image area, or zero for invalid boxes.
        """
        image_width, image_height = image_size
        x1, y1, x2, y2 = bbox
        clipped_x1 = max(0.0, min(float(image_width), x1))
        clipped_y1 = max(0.0, min(float(image_height), y1))
        clipped_x2 = max(0.0, min(float(image_width), x2))
        clipped_y2 = max(0.0, min(float(image_height), y2))
        box_width = max(0.0, clipped_x2 - clipped_x1)
        box_height = max(0.0, clipped_y2 - clipped_y1)
        image_area = image_width * image_height
        return (box_width * box_height / image_area) if image_area > 0 else 0.0
