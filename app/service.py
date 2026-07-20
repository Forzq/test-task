"""Application service orchestrating the complete inference pipeline."""

from __future__ import annotations

from typing import Mapping

from PIL import Image

from app.domain import CropClassifierProtocol, DetectorProtocol, RawDetection
from app.image_utils import SquareCropExtractor
from app.pipeline import (
    ClassNameMapper,
    ClassThresholds,
    CropCandidateVerifier,
    DetectionPostProcessor,
    ImageDecoder,
    UnsupportedImageError,
)
from app.schemas import DetectedObject, DetectionCounts, PredictionResponse

__all__ = ["DetectionService", "UnsupportedImageError"]


class DetectionService:
    """
    Coordinate image decoding, detection, verification, and response assembly.

    Parameters
    ----------
    detector : DetectorProtocol
        Loaded adapter that produces bounding-box proposals.
    confidence_threshold : float
        Legacy detector threshold used when class-specific values are absent.
    max_upload_size_bytes : int
        Maximum accepted encoded upload size.
    crop_classifier : CropClassifierProtocol, optional
        Optional second-stage classifier that can reject proposals as ``other``.
    classifier_confidence_threshold : float, optional
        Legacy classifier threshold used when class-specific values are absent.
    classifier_crop_context : float, optional
        Legacy single crop context retained for backward compatibility.
    classifier_image_size : int, optional
        Width and height of classifier crops.
    min_box_area_ratio : float, optional
        Minimum proposal area relative to the complete source image.
    detector_class_thresholds : Mapping[str, float], optional
        Independent detector thresholds for ``bolt`` and ``nut``.
    classifier_class_thresholds : Mapping[str, float], optional
        Independent classifier thresholds for ``bolt`` and ``nut``.
    classifier_crop_contexts : tuple[float, ...], optional
        Context ratios aggregated by the crop verifier.
    final_nms_iou_threshold : float, optional
        Same-class IoU above which the weaker duplicate is removed.

    Notes
    -----
    This class intentionally contains orchestration only. Validation, crop
    geometry, class mapping, score aggregation, and NMS are delegated to focused
    components from :mod:`app.pipeline`.
    """

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
        detector_class_thresholds: Mapping[str, float] | None = None,
        classifier_class_thresholds: Mapping[str, float] | None = None,
        classifier_crop_contexts: tuple[float, ...] | None = None,
        final_nms_iou_threshold: float = 0.45,
    ) -> None:
        """
        Build reusable pipeline components from runtime configuration.

        Parameters
        ----------
        detector : DetectorProtocol
            Detector adapter used for proposal generation.
        confidence_threshold : float
            Fallback detector confidence threshold.
        max_upload_size_bytes : int
            Maximum accepted upload size in bytes.
        crop_classifier : CropClassifierProtocol, optional
            Optional second-stage crop classifier.
        classifier_confidence_threshold : float, optional
            Fallback crop-classifier confidence threshold.
        classifier_crop_context : float, optional
            Backward-compatible single context ratio.
        classifier_image_size : int, optional
            Square crop output size.
        min_box_area_ratio : float, optional
            Minimum relative proposal area.
        detector_class_thresholds : Mapping[str, float], optional
            Class-specific detector thresholds.
        classifier_class_thresholds : Mapping[str, float], optional
            Class-specific classifier thresholds.
        classifier_crop_contexts : tuple[float, ...], optional
            Multi-context verifier configuration.
        final_nms_iou_threshold : float, optional
            Same-class duplicate suppression threshold.
        """
        self._detector = detector
        self._class_mapper = ClassNameMapper.nut_and_bolt()
        self._detector_thresholds = self._resolve_thresholds(
            fallback=confidence_threshold,
            overrides=detector_class_thresholds,
        )
        self._decoder = ImageDecoder(max_upload_size_bytes)
        self._postprocessor = DetectionPostProcessor(
            min_box_area_ratio=min_box_area_ratio,
            nms_iou_threshold=final_nms_iou_threshold,
        )
        self._verifier = self._build_verifier(
            classifier=crop_classifier,
            fallback_threshold=classifier_confidence_threshold,
            threshold_overrides=classifier_class_thresholds,
            contexts=classifier_crop_contexts or (classifier_crop_context,),
            image_size=classifier_image_size,
        )

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """
        Execute the production pipeline for one encoded image.

        Parameters
        ----------
        image_bytes : bytes
            JPEG or PNG bytes received from the HTTP layer.

        Returns
        -------
        PredictionResponse
            Accepted objects and final counts for both public classes.

        Raises
        ------
        UnsupportedImageError
            Raised when upload validation or decoding fails.
        """
        image = self._decoder.decode(image_bytes)
        detections = self._detector.predict(
            image,
            self._detector_thresholds.minimum,
        )
        objects = []
        for detection in detections:
            accepted = self._process_detection(image, detection)
            if accepted is not None:
                objects.append(accepted)

        objects = self._postprocessor.suppress_duplicates(objects)
        return PredictionResponse(
            objects=objects,
            counts=DetectionCounts(
                nuts=sum(item.class_name == "nut" for item in objects),
                bolts=sum(item.class_name == "bolt" for item in objects),
            ),
        )

    def _process_detection(
        self,
        image: Image.Image,
        detection: RawDetection,
    ) -> DetectedObject | None:
        """
        Apply class, confidence, geometry, and optional crop verification.

        Parameters
        ----------
        image : PIL.Image.Image
            Decoded complete source image.
        detection : RawDetection
            One detector proposal.

        Returns
        -------
        DetectedObject or None
            Public response object when accepted, otherwise ``None``.
        """
        class_name = self._class_mapper.normalise(detection.label)
        if class_name is None:
            return None
        if not self._detector_thresholds.accepts(
            class_name, detection.confidence
        ):
            return None
        if not self._postprocessor.has_sufficient_area(detection, image.size):
            return None

        final_confidence = detection.confidence
        if self._verifier is not None:
            verified = self._verifier.verify(image, detection)
            if verified is None:
                return None
            class_name = verified.class_name
            final_confidence = min(detection.confidence, verified.confidence)

        return self._postprocessor.to_response_object(
            detection=detection,
            class_name=class_name,
            confidence=final_confidence,
        )

    def _build_verifier(
        self,
        classifier: CropClassifierProtocol | None,
        fallback_threshold: float,
        threshold_overrides: Mapping[str, float] | None,
        contexts: tuple[float, ...],
        image_size: int,
    ) -> CropCandidateVerifier | None:
        """
        Create the second-stage verifier only when a classifier is enabled.

        Parameters
        ----------
        classifier : CropClassifierProtocol or None
            Optional classifier adapter.
        fallback_threshold : float
            Threshold used for classes without explicit overrides.
        threshold_overrides : Mapping[str, float] or None
            Per-class classifier thresholds.
        contexts : tuple[float, ...]
            Context ratios evaluated around every proposal.
        image_size : int
            Square crop dimensions.

        Returns
        -------
        CropCandidateVerifier or None
            Configured verifier, or ``None`` for detector-only operation.
        """
        if classifier is None:
            return None
        return CropCandidateVerifier(
            classifier=classifier,
            crop_extractor=SquareCropExtractor(output_size=image_size),
            contexts=contexts,
            thresholds=self._resolve_thresholds(
                fallback=fallback_threshold,
                overrides=threshold_overrides,
            ),
            class_mapper=self._class_mapper,
        )

    @staticmethod
    def _resolve_thresholds(
        fallback: float,
        overrides: Mapping[str, float] | None,
    ) -> ClassThresholds:
        """
        Merge a backward-compatible fallback with per-class overrides.

        Parameters
        ----------
        fallback : float
            Default value assigned to both supported classes.
        overrides : Mapping[str, float] or None
            Optional explicit class-specific values.

        Returns
        -------
        ClassThresholds
            Validated bolt and nut threshold policy.
        """
        return ClassThresholds(
            {
                "bolt": fallback,
                "nut": fallback,
                **(overrides or {}),
            }
        )
