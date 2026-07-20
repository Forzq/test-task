"""Ultralytics YOLO adapter used by the API service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL.Image import Image

from app.domain import RawDetection


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested before model weights are available."""


class UltralyticsDetector:
    """
    Adapter that converts Ultralytics YOLO results into domain detections.

    Parameters
    ----------
    model_path : Path
        Location of the trained YOLO weights file.
    image_size : int
        Square inference size passed to Ultralytics.
    iou_threshold : float
        Detector-level NMS intersection-over-union threshold.
    """

    def __init__(
        self,
        model_path: Path,
        image_size: int = 640,
        iou_threshold: float = 0.60,
    ) -> None:
        """
        Store the checkpoint location without loading GPU resources yet.

        Parameters
        ----------
        model_path : Path
            Location of the trained YOLO weights file.
        image_size : int, optional
            Square inference dimensions used for every request.
        iou_threshold : float, optional
            Detector-level NMS overlap threshold.

        Raises
        ------
        ValueError
            Raised when inference settings are outside valid ranges.
        """
        if image_size <= 0:
            raise ValueError("Detector image size must be positive")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("Detector IoU threshold must be between zero and one")
        self._model_path = model_path
        self._image_size = image_size
        self._iou_threshold = iou_threshold
        self._model: Any | None = None

    def load(self) -> None:
        """
        Load trained weights into an Ultralytics YOLO model instance.

        Raises
        ------
        ModelUnavailableError
            Raised when the weights file does not exist or cannot be loaded.
        """
        if not self._model_path.is_file():
            raise ModelUnavailableError(
                f"Model weights were not found at '{self._model_path}'. "
                "Train the model and set MODEL_PATH to the produced best.pt file."
            )

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self._model_path))
        except Exception as error:  # pragma: no cover - depends on local ML runtime
            raise ModelUnavailableError(
                f"Unable to load model weights from '{self._model_path}'."
            ) from error

    def predict(
        self, image: Image, confidence_threshold: float
    ) -> list[RawDetection]:
        """
        Detect objects in an RGB image using the loaded YOLO model.

        Parameters
        ----------
        image : PIL.Image.Image
            Decoded input image in RGB mode.
        confidence_threshold : float
            Minimum confidence forwarded to Ultralytics inference.

        Returns
        -------
        list[RawDetection]
            Model detections expressed using original class labels and xyxy boxes.

        Raises
        ------
        ModelUnavailableError
            Raised when the model was not successfully loaded.
        """
        if self._model is None:
            raise ModelUnavailableError("The detection model has not been loaded.")

        results = self._model.predict(
            source=image,
            conf=confidence_threshold,
            imgsz=self._image_size,
            iou=self._iou_threshold,
            verbose=False,
        )
        detections: list[RawDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                label = str(result.names[class_id])
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    RawDetection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return detections
