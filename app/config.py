"""Application configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Immutable runtime configuration for the detection service.

    Parameters
    ----------
    model_path : Path
        Path to the trained Ultralytics YOLO checkpoint.
    classifier_model_path : Path
        Path to the trained crop-classification checkpoint.
    enable_crop_classifier : bool
        Whether detector candidates must pass crop classification.
    classifier_confidence_threshold : float
        Minimum classifier confidence required to retain a candidate.
    classifier_crop_context : float
        Context added around every detector bounding box.
    classifier_image_size : int
        Square crop and classification input size.
    min_box_area_ratio : float
        Minimum bounding-box area relative to the complete image area.
    confidence_threshold : float
        Minimum confidence required for a detection to be returned.
    max_upload_size_bytes : int
        Maximum accepted image size in bytes.
    """

    model_path: Path
    classifier_model_path: Path = Path("models/classifier_best.pt")
    enable_crop_classifier: bool = True
    classifier_confidence_threshold: float = 0.54
    classifier_crop_context: float = 0.20
    classifier_image_size: int = 224
    min_box_area_ratio: float = 0.001
    confidence_threshold: float = 0.45
    max_upload_size_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "Settings":
        """
        Build settings from environment variables.

        Returns
        -------
        Settings
            Validated application settings.

        Raises
        ------
        ValueError
            Raised when a numeric environment variable has an invalid value.

        Notes
        -----
        Model paths, detector and classifier thresholds, crop settings, and
        upload limits can be overridden without changing application code.
        """
        model_path = Path(os.getenv("MODEL_PATH", "models/best.pt"))
        classifier_model_path = Path(
            os.getenv("CLASSIFIER_MODEL_PATH", "models/classifier_best.pt")
        )
        enable_crop_classifier = cls._parse_boolean(
            os.getenv("ENABLE_CROP_CLASSIFIER", "true")
        )
        classifier_confidence_threshold = float(
            os.getenv("CLASSIFIER_CONFIDENCE_THRESHOLD", "0.54")
        )
        classifier_crop_context = float(os.getenv("CLASSIFIER_CROP_CONTEXT", "0.20"))
        classifier_image_size = int(os.getenv("CLASSIFIER_IMAGE_SIZE", "224"))
        min_box_area_ratio = float(os.getenv("MIN_BOX_AREA_RATIO", "0.001"))
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
        max_upload_size_bytes = int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
        )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if not 0.0 <= classifier_confidence_threshold <= 1.0:
            raise ValueError(
                "CLASSIFIER_CONFIDENCE_THRESHOLD must be between 0 and 1."
            )
        if classifier_crop_context < 0.0:
            raise ValueError("CLASSIFIER_CROP_CONTEXT cannot be negative.")
        if classifier_image_size <= 0:
            raise ValueError("CLASSIFIER_IMAGE_SIZE must be positive.")
        if not 0.0 <= min_box_area_ratio <= 1.0:
            raise ValueError("MIN_BOX_AREA_RATIO must be between 0 and 1.")
        if max_upload_size_bytes <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be positive.")

        return cls(
            model_path=model_path,
            classifier_model_path=classifier_model_path,
            enable_crop_classifier=enable_crop_classifier,
            classifier_confidence_threshold=classifier_confidence_threshold,
            classifier_crop_context=classifier_crop_context,
            classifier_image_size=classifier_image_size,
            min_box_area_ratio=min_box_area_ratio,
            confidence_threshold=confidence_threshold,
            max_upload_size_bytes=max_upload_size_bytes,
        )

    @staticmethod
    def _parse_boolean(value: str) -> bool:
        """
        Parse a strict environment-style boolean value.

        Parameters
        ----------
        value : str
            Text such as ``true``, ``false``, ``1``, or ``0``.

        Returns
        -------
        bool
            Parsed boolean value.

        Raises
        ------
        ValueError
            Raised when the text does not represent a supported boolean.
        """
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {value!r}")
