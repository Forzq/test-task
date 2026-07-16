"""HTTP-level tests for the nut and bolt prediction API."""

from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.domain import RawDetection
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
        settings=Settings(model_path=__file__, confidence_threshold=0.5),
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
