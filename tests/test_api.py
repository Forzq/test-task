"""HTTP-level tests for the nut and bolt prediction API."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.domain import CropClassification, RawDetection
from app.main import create_app


class FakeDetector:
    """Small deterministic detector used to test HTTP response formatting."""

    def predict(self, image: Image.Image, confidence_threshold: float) -> list[RawDetection]:
        """
        Return fixed detections without loading an ML model.

        Parameters
        ----------
        image : PIL.Image.Image
            Decoded image supplied by the prediction service.
        confidence_threshold : float
            Threshold passed through by the prediction service.

        Returns
        -------
        list[RawDetection]
            Fixture detections containing supported and unsupported classes.
        """
        return [
            RawDetection("nut", 0.91, (1.2, 2.3, 10.4, 12.6)),
            RawDetection("bolt", 0.72, (20.0, 30.0, 40.0, 60.0)),
            RawDetection("washer", 0.99, (0.0, 0.0, 3.0, 3.0)),
        ]


class FakeCropClassifier:
    """Deterministic verifier that rejects the first detector candidate."""

    def __init__(self) -> None:
        """Initialise a prediction counter for ordered fixture results."""
        self._prediction_index = 0

    def predict(self, image: Image.Image) -> CropClassification:
        """
        Return ordered ``other`` and ``bolt`` classification results.

        Parameters
        ----------
        image : PIL.Image.Image
            Square crop produced by the detection service.

        Returns
        -------
        CropClassification
            Fixture result used to verify second-stage filtering.
        """
        assert image.size == (224, 224)
        results = [
            CropClassification("other", 0.99),
            CropClassification("bolt", 0.88),
        ]
        result = results[self._prediction_index]
        self._prediction_index += 1
        return result


class SprinkleFalsePositiveDetector:
    """Detector fixture reproducing the two donut-sprinkle false positives."""

    def predict(self, image: Image.Image, confidence_threshold: float) -> list[RawDetection]:
        """
        Return the tiny bolt boxes observed on the donut regression image.

        Parameters
        ----------
        image : PIL.Image.Image
            Decoded 360 by 360 donut fixture.
        confidence_threshold : float
            Detector confidence threshold supplied by the service.

        Returns
        -------
        list[RawDetection]
            Two high-confidence but sub-threshold-area false detections.
        """
        assert image.size == (360, 360)
        return [
            RawDetection("bolt", 0.7559, (177.76, 276.31, 183.60, 285.37)),
            RawDetection("bolt", 0.5308, (86.17, 214.71, 97.80, 225.40)),
        ]


def _image_bytes(image_format: str = "PNG") -> bytes:
    """
    Create an in-memory image suitable for multipart request tests.

    Parameters
    ----------
    image_format : str
        Pillow image format to encode.

    Returns
    -------
    bytes
        Encoded image data.
    """
    output = BytesIO()
    Image.new("RGB", (32, 32), "white").save(output, format=image_format)
    return output.getvalue()


def _client() -> TestClient:
    """
    Create a test client with a deterministic injected detector.

    Returns
    -------
    TestClient
        Ready-to-use client that does not require local model weights.
    """
    app = create_app(
        settings=Settings(
            model_path=Path(__file__),
            enable_crop_classifier=False,
            confidence_threshold=0.5,
        ),
        detector=FakeDetector(),
    )
    return TestClient(app)


def test_predict_returns_objects_and_counts() -> None:
    """Verify that the endpoint returns canonical classes and correct counts."""
    response = _client().post(
        "/predict",
        files={"image": ("parts.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "objects": [
            {"class": "nut", "confidence": 0.91, "bbox": [1, 2, 10, 13]},
            {"class": "bolt", "confidence": 0.72, "bbox": [20, 30, 40, 60]},
        ],
        "counts": {"nuts": 1, "bolts": 1},
    }


def test_predict_rejects_an_unsupported_media_type() -> None:
    """Verify that non-image multipart uploads are rejected with HTTP 415."""
    response = _client().post(
        "/predict",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415


def test_crop_classifier_rejects_other_detector_candidates() -> None:
    """Verify that the optional verifier removes crops classified as other."""
    app = create_app(
        settings=Settings(model_path=Path(__file__), confidence_threshold=0.5),
        detector=FakeDetector(),
        classifier=FakeCropClassifier(),
    )
    response = TestClient(app).post(
        "/predict",
        files={"image": ("parts.png", _image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "objects": [
            {"class": "bolt", "confidence": 0.72, "bbox": [20, 30, 40, 60]},
        ],
        "counts": {"nuts": 0, "bolts": 1},
    }


def test_donut_sprinkles_are_rejected_by_relative_area() -> None:
    """Verify that the reported microscopic sprinkle boxes produce zero counts."""
    app = create_app(
        settings=Settings(
            model_path=Path(__file__),
            enable_crop_classifier=False,
            confidence_threshold=0.45,
            min_box_area_ratio=0.001,
        ),
        detector=SprinkleFalsePositiveDetector(),
    )
    fixture = Path(__file__).parent / "fixtures" / "donut_sprinkles.png"
    response = TestClient(app).post(
        "/predict",
        files={"image": (fixture.name, fixture.read_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "objects": [],
        "counts": {"nuts": 0, "bolts": 0},
    }
