"""Unit tests for focused inference-pipeline components."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.domain import CropClassification, RawDetection
from app.image_utils import SquareCropExtractor
from app.pipeline import (
    ClassNameMapper,
    ClassThresholds,
    ClassificationAggregator,
    DetectionPostProcessor,
    ImageDecoder,
    UnsupportedImageError,
)
from app.schemas import DetectedObject


def _encoded_image() -> bytes:
    """
    Create a small valid PNG for decoder unit tests.

    Returns
    -------
    bytes
        Encoded RGB image bytes.
    """
    output = BytesIO()
    Image.new("RGB", (20, 10), "white").save(output, format="PNG")
    return output.getvalue()


def test_image_decoder_validates_and_converts_uploads() -> None:
    """The decoder must return RGB and reject empty or corrupt uploads."""
    decoder = ImageDecoder(max_upload_size_bytes=10_000)

    image = decoder.decode(_encoded_image())

    assert image.mode == "RGB"
    assert image.size == (20, 10)
    with pytest.raises(UnsupportedImageError):
        decoder.decode(b"")
    with pytest.raises(UnsupportedImageError):
        decoder.decode(b"not an image")


def test_square_crop_extractor_preserves_square_output_at_image_edge() -> None:
    """Edge proposals must be padded instead of distorted or truncated."""
    extractor = SquareCropExtractor(output_size=64)
    image = Image.new("RGB", (40, 20), "white")

    crop = extractor.extract(image, (-5.0, 2.0, 20.0, 12.0), context_ratio=0.2)

    assert crop.size == (64, 64)
    assert crop.mode == "RGB"


def test_class_mapping_and_threshold_policy_are_explicit() -> None:
    """Aliases and independent class thresholds must remain deterministic."""
    mapper = ClassNameMapper.nut_and_bolt()
    thresholds = ClassThresholds({"bolt": 0.70, "nut": 0.50})

    assert mapper.normalise(" BOLTS ") == "bolt"
    assert mapper.normalise("washer") is None
    assert thresholds.minimum == 0.50
    assert not thresholds.accepts("bolt", 0.60)
    assert thresholds.accepts("nut", 0.60)


def test_classification_aggregator_averages_context_probabilities() -> None:
    """Multi-context classification must use mean probabilities by class."""
    aggregator = ClassificationAggregator()
    classifications = [
        CropClassification(
            "bolt",
            0.9,
            {"bolt": 0.9, "nut": 0.05, "other": 0.05},
        ),
        CropClassification(
            "other",
            0.6,
            {"bolt": 0.35, "nut": 0.05, "other": 0.6},
        ),
    ]

    result = aggregator.aggregate(classifications)

    assert result.label == "bolt"
    assert result.confidence == pytest.approx(0.625)


def test_postprocessor_filters_area_and_suppresses_same_class_only() -> None:
    """Geometry filtering and NMS must preserve overlapping different classes."""
    postprocessor = DetectionPostProcessor(
        min_box_area_ratio=0.01,
        nms_iou_threshold=0.45,
    )
    detection = RawDetection("bolt", 0.9, (0.0, 0.0, 20.0, 20.0))
    objects = [
        DetectedObject(class_name="bolt", confidence=0.9, bbox=[0, 0, 20, 20]),
        DetectedObject(class_name="bolt", confidence=0.8, bbox=[1, 1, 21, 21]),
        DetectedObject(class_name="nut", confidence=0.7, bbox=[1, 1, 21, 21]),
    ]

    retained = postprocessor.suppress_duplicates(objects)

    assert postprocessor.has_sufficient_area(detection, (100, 100))
    assert [item.class_name for item in retained] == ["bolt", "nut"]
