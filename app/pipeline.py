"""Reusable components for the detector and crop-classifier pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from app.domain import CropClassification, CropClassifierProtocol, RawDetection
from app.image_utils import SquareCropExtractor
from app.schemas import DetectedObject


class UnsupportedImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an accepted image."""


@dataclass(frozen=True, slots=True)
class ImageDecoder:
    """
    Validate uploaded bytes and decode them into RGB images.

    Parameters
    ----------
    max_upload_size_bytes : int
        Maximum accepted encoded file size.
    supported_formats : frozenset[str]
        Pillow format names accepted by the API.
    """

    max_upload_size_bytes: int
    supported_formats: frozenset[str] = frozenset({"JPEG", "PNG"})

    def __post_init__(self) -> None:
        """
        Validate decoder configuration at application startup.

        Raises
        ------
        ValueError
            Raised when the upload limit is not positive.
        """
        if self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be positive")

    def decode(self, image_bytes: bytes) -> Image.Image:
        """
        Decode one validated JPEG or PNG upload.

        Parameters
        ----------
        image_bytes : bytes
            Encoded multipart upload content.

        Returns
        -------
        PIL.Image.Image
            Decoded image converted to RGB colour mode.

        Raises
        ------
        UnsupportedImageError
            Raised for empty, oversized, corrupt, or unsupported files.
        """
        if not image_bytes:
            raise UnsupportedImageError("The uploaded file is empty.")
        if len(image_bytes) > self.max_upload_size_bytes:
            raise UnsupportedImageError(
                f"The uploaded image exceeds {self.max_upload_size_bytes} bytes."
            )

        try:
            with Image.open(BytesIO(image_bytes)) as opened_image:
                if opened_image.format not in self.supported_formats:
                    raise UnsupportedImageError(
                        "Only JPEG and PNG images are supported."
                    )
                return opened_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise UnsupportedImageError(
                "The uploaded file is not a valid image."
            ) from error


@dataclass(frozen=True, slots=True)
class ClassNameMapper:
    """
    Map model-specific labels onto the two public API classes.

    Parameters
    ----------
    aliases : Mapping[str, str]
        Case-insensitive raw-to-canonical label mapping.
    """

    aliases: Mapping[str, str]

    @classmethod
    def nut_and_bolt(cls) -> "ClassNameMapper":
        """
        Create the canonical mapper used by the production pipeline.

        Returns
        -------
        ClassNameMapper
            Mapper accepting singular and plural nut and bolt labels.
        """
        return cls(
            aliases={
                "nut": "nut",
                "nuts": "nut",
                "bolt": "bolt",
                "bolts": "bolt",
            }
        )

    def normalise(self, label: str) -> str | None:
        """
        Convert a raw model label into a supported public class.

        Parameters
        ----------
        label : str
            Label emitted by a detector or classifier.

        Returns
        -------
        str or None
            Canonical ``bolt`` or ``nut``, or ``None`` for unsupported labels.
        """
        return self.aliases.get(label.strip().lower())


@dataclass(frozen=True, slots=True)
class ClassThresholds:
    """
    Store independent acceptance thresholds for supported classes.

    Parameters
    ----------
    values : Mapping[str, float]
        Canonical class names mapped to confidence thresholds.
    """

    values: Mapping[str, float]

    def __post_init__(self) -> None:
        """
        Validate class coverage and threshold ranges.

        Raises
        ------
        ValueError
            Raised when a supported class is missing or outside ``[0, 1]``.
        """
        missing = {"bolt", "nut"} - set(self.values)
        if missing:
            raise ValueError(f"Missing class thresholds: {sorted(missing)}")
        invalid = {
            name: value
            for name, value in self.values.items()
            if not 0.0 <= value <= 1.0
        }
        if invalid:
            raise ValueError(
                f"Class thresholds must be between zero and one: {invalid}"
            )

    @property
    def minimum(self) -> float:
        """
        Return the lowest threshold required for detector proposal generation.

        Returns
        -------
        float
            Minimum configured class threshold.
        """
        return min(self.values.values())

    def accepts(self, class_name: str, confidence: float) -> bool:
        """
        Decide whether a class-specific confidence is high enough.

        Parameters
        ----------
        class_name : str
            Canonical class name.
        confidence : float
            Model confidence score.

        Returns
        -------
        bool
            True when the score reaches the configured class threshold.
        """
        return confidence >= self.values[class_name]


class ClassificationAggregator:
    """Combine predictions from multiple crop context ratios."""

    _LABELS = ("bolt", "nut", "other")

    def aggregate(
        self, classifications: list[CropClassification]
    ) -> CropClassification:
        """
        Aggregate context predictions into one conservative classification.

        Parameters
        ----------
        classifications : list[CropClassification]
            Classifier outputs for the same candidate at different contexts.

        Returns
        -------
        CropClassification
            Mean-probability result when distributions are available; otherwise
            a consensus label or ``other`` for disagreement.

        Raises
        ------
        ValueError
            Raised when no context classifications are supplied.
        """
        if not classifications:
            raise ValueError("At least one crop classification is required")
        if all(item.probabilities for item in classifications):
            probabilities = {
                label: sum(
                    item.probabilities.get(label, 0.0) for item in classifications
                )
                / len(classifications)
                for label in self._LABELS
            }
            label = max(probabilities, key=probabilities.__getitem__)
            return CropClassification(label, probabilities[label], probabilities)

        labels = {item.label.strip().lower() for item in classifications}
        if len(labels) != 1:
            return CropClassification("other", 1.0, {"other": 1.0})
        label = labels.pop()
        confidence = sum(item.confidence for item in classifications) / len(
            classifications
        )
        return CropClassification(label, confidence)


@dataclass(frozen=True, slots=True)
class VerifiedClassification:
    """
    Represent a detector candidate accepted by the crop verifier.

    Parameters
    ----------
    class_name : str
        Canonical class selected by the crop classifier.
    confidence : float
        Aggregated classifier confidence.
    """

    class_name: str
    confidence: float


class CropCandidateVerifier:
    """
    Verify detector boxes using one or more context-aware crop classifications.

    Parameters
    ----------
    classifier : CropClassifierProtocol
        Loaded second-stage classifier adapter.
    crop_extractor : SquareCropExtractor
        Shared geometry component used to construct classifier inputs.
    contexts : tuple[float, ...]
        Context ratios evaluated around each detector box.
    thresholds : ClassThresholds
        Per-class classifier acceptance thresholds.
    class_mapper : ClassNameMapper
        Mapper that rejects unsupported classifier labels.
    aggregator : ClassificationAggregator, optional
        Strategy used to merge multi-context probabilities.
    """

    def __init__(
        self,
        classifier: CropClassifierProtocol,
        crop_extractor: SquareCropExtractor,
        contexts: tuple[float, ...],
        thresholds: ClassThresholds,
        class_mapper: ClassNameMapper,
        aggregator: ClassificationAggregator | None = None,
    ) -> None:
        """Initialise an immutable verifier policy around a classifier adapter."""
        if not contexts:
            raise ValueError("At least one classifier crop context is required")
        if any(context < 0.0 for context in contexts):
            raise ValueError("Classifier crop contexts cannot be negative")
        self._classifier = classifier
        self._crop_extractor = crop_extractor
        self._contexts = contexts
        self._thresholds = thresholds
        self._class_mapper = class_mapper
        self._aggregator = aggregator or ClassificationAggregator()

    def verify(
        self,
        image: Image.Image,
        detection: RawDetection,
    ) -> VerifiedClassification | None:
        """
        Classify all valid crops for one detector proposal.

        Parameters
        ----------
        image : PIL.Image.Image
            Complete RGB source image.
        detection : RawDetection
            Detector proposal whose box defines crop geometry.

        Returns
        -------
        VerifiedClassification or None
            Accepted canonical class and confidence, or ``None`` when the
            proposal is invalid, unsupported, or below its threshold.
        """
        classifications: list[CropClassification] = []
        for context_ratio in self._contexts:
            try:
                crop = self._crop_extractor.extract(
                    image=image,
                    bbox=detection.bbox,
                    context_ratio=context_ratio,
                )
            except ValueError:
                continue
            classifications.append(self._classifier.predict(crop))
        if not classifications:
            return None

        classification = self._aggregator.aggregate(classifications)
        class_name = self._class_mapper.normalise(classification.label)
        if class_name is None or not self._thresholds.accepts(
            class_name, classification.confidence
        ):
            return None
        return VerifiedClassification(class_name, classification.confidence)


class DetectionPostProcessor:
    """
    Apply geometry filters, response formatting, and class-aware NMS.

    Parameters
    ----------
    min_box_area_ratio : float
        Minimum clipped box area relative to the source image.
    nms_iou_threshold : float
        Maximum overlap retained between boxes of the same final class.
    """

    def __init__(self, min_box_area_ratio: float, nms_iou_threshold: float) -> None:
        """Validate and store postprocessing thresholds."""
        if not 0.0 <= min_box_area_ratio <= 1.0:
            raise ValueError("min_box_area_ratio must be between zero and one")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be between zero and one")
        self._min_box_area_ratio = min_box_area_ratio
        self._nms_iou_threshold = nms_iou_threshold

    def has_sufficient_area(
        self,
        detection: RawDetection,
        image_size: tuple[int, int],
    ) -> bool:
        """
        Check whether a proposal is large enough for reliable classification.

        Parameters
        ----------
        detection : RawDetection
            Detector proposal to measure.
        image_size : tuple[int, int]
            Source image width and height.

        Returns
        -------
        bool
            True when clipped relative area reaches the configured minimum.
        """
        return (
            self.relative_box_area(detection.bbox, image_size)
            >= self._min_box_area_ratio
        )

    def to_response_object(
        self,
        detection: RawDetection,
        class_name: str,
        confidence: float,
    ) -> DetectedObject:
        """
        Convert an internal detection into the public response schema.

        Parameters
        ----------
        detection : RawDetection
            Original detector proposal.
        class_name : str
            Final canonical class after optional crop verification.
        confidence : float
            Final combined confidence.

        Returns
        -------
        DetectedObject
            Rounded, non-negative response object.
        """
        return DetectedObject(
            class_name=class_name,
            confidence=round(confidence, 4),
            bbox=[max(0, round(value)) for value in detection.bbox],
        )

    def suppress_duplicates(
        self, objects: list[DetectedObject]
    ) -> list[DetectedObject]:
        """
        Remove lower-confidence overlaps only within the same final class.

        Parameters
        ----------
        objects : list[DetectedObject]
            Accepted objects before final duplicate suppression.

        Returns
        -------
        list[DetectedObject]
            Confidence-sorted objects with same-class duplicates removed.
        """
        retained: list[DetectedObject] = []
        for candidate in sorted(
            objects, key=lambda item: item.confidence, reverse=True
        ):
            duplicate = any(
                candidate.class_name == accepted.class_name
                and self.bbox_iou(candidate.bbox, accepted.bbox)
                > self._nms_iou_threshold
                for accepted in retained
            )
            if not duplicate:
                retained.append(candidate)
        return retained

    @staticmethod
    def bbox_iou(first: list[int], second: list[int]) -> float:
        """
        Calculate intersection over union for two axis-aligned boxes.

        Parameters
        ----------
        first : list[int]
            First box in ``x1, y1, x2, y2`` order.
        second : list[int]
            Second box in ``x1, y1, x2, y2`` order.

        Returns
        -------
        float
            Intersection area divided by union area.
        """
        intersection_width = max(
            0, min(first[2], second[2]) - max(first[0], second[0])
        )
        intersection_height = max(
            0, min(first[3], second[3]) - max(first[1], second[1])
        )
        intersection = intersection_width * intersection_height
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(
            0, second[3] - second[1]
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def relative_box_area(
        bbox: tuple[float, float, float, float],
        image_size: tuple[int, int],
    ) -> float:
        """
        Calculate clipped box area relative to the complete source image.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Candidate box in ``x1, y1, x2, y2`` coordinates.
        image_size : tuple[int, int]
            Source image width and height.

        Returns
        -------
        float
            Relative area, or zero for invalid image or box geometry.
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
        return box_width * box_height / image_area if image_area > 0 else 0.0
