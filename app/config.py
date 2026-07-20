"""Typed runtime configuration and environment-loading helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Store immutable runtime configuration for the detection application.

    Parameters
    ----------
    model_path : Path
        Detector checkpoint used by the API.
    classifier_model_path : Path
        Crop-classifier checkpoint used by the optional verifier.
    enable_crop_classifier : bool
        Whether detector proposals must pass second-stage classification.
    classifier_confidence_threshold : float
        Legacy classifier fallback threshold.
    bolt_confidence_threshold : float
        Detector threshold applied to bolt proposals.
    nut_confidence_threshold : float
        Detector threshold applied to nut proposals.
    bolt_classifier_threshold : float
        Crop-classifier threshold applied to bolt predictions.
    nut_classifier_threshold : float
        Crop-classifier threshold applied to nut predictions.
    classifier_crop_contexts : tuple[float, ...]
        Context ratios evaluated around every detector proposal.
    classifier_image_size : int
        Square classifier input dimensions.
    min_box_area_ratio : float
        Minimum proposal area relative to the complete image.
    final_nms_iou_threshold : float
        Same-class overlap threshold used by final NMS.
    confidence_threshold : float
        Legacy detector fallback threshold.
    max_upload_size_bytes : int
        Maximum accepted encoded upload size.
    """

    model_path: Path
    classifier_model_path: Path = Path("models/classifier_best.pt")
    enable_crop_classifier: bool = True
    classifier_confidence_threshold: float = 0.39
    bolt_confidence_threshold: float = 0.43
    nut_confidence_threshold: float = 0.56
    bolt_classifier_threshold: float = 0.39
    nut_classifier_threshold: float = 0.51
    classifier_crop_contexts: tuple[float, ...] = (0.20,)
    classifier_image_size: int = 224
    min_box_area_ratio: float = 0.001
    final_nms_iou_threshold: float = 0.45
    confidence_threshold: float = 0.43
    max_upload_size_bytes: int = 10 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> "Settings":
        """
        Load and validate settings from process environment variables.

        Returns
        -------
        Settings
            Fully validated immutable settings.

        Notes
        -----
        Parsing and validation are delegated to dedicated classes so the data
        object remains free of environment-specific control flow.
        """
        return EnvironmentSettingsLoader(os.environ).load()


class EnvironmentSettingsLoader:
    """
    Convert string environment variables into typed application settings.

    Parameters
    ----------
    values : Mapping[str, str]
        Environment-like key-value mapping, injectable for deterministic tests.
    validator : SettingsValidator, optional
        Validation policy applied after parsing.
    """

    def __init__(
        self,
        values: Mapping[str, str],
        validator: "SettingsValidator | None" = None,
    ) -> None:
        """Store the environment source and validation strategy."""
        self._values = values
        self._validator = validator or SettingsValidator()

    def load(self) -> Settings:
        """
        Parse every supported environment variable exactly once.

        Returns
        -------
        Settings
            Typed and validated runtime configuration.

        Raises
        ------
        ValueError
            Raised for malformed values or invalid configuration ranges.
        """
        defaults = Settings(model_path=Path("models/best.pt"))
        legacy_context = self._text("CLASSIFIER_CROP_CONTEXT", "0.20")
        settings = Settings(
            model_path=self._path("MODEL_PATH", defaults.model_path),
            classifier_model_path=self._path(
                "CLASSIFIER_MODEL_PATH", defaults.classifier_model_path
            ),
            enable_crop_classifier=self._boolean(
                "ENABLE_CROP_CLASSIFIER", defaults.enable_crop_classifier
            ),
            classifier_confidence_threshold=self._float(
                "CLASSIFIER_CONFIDENCE_THRESHOLD",
                defaults.classifier_confidence_threshold,
            ),
            bolt_confidence_threshold=self._float(
                "BOLT_CONFIDENCE_THRESHOLD", defaults.bolt_confidence_threshold
            ),
            nut_confidence_threshold=self._float(
                "NUT_CONFIDENCE_THRESHOLD", defaults.nut_confidence_threshold
            ),
            bolt_classifier_threshold=self._float(
                "BOLT_CLASSIFIER_THRESHOLD", defaults.bolt_classifier_threshold
            ),
            nut_classifier_threshold=self._float(
                "NUT_CLASSIFIER_THRESHOLD", defaults.nut_classifier_threshold
            ),
            classifier_crop_contexts=self._float_list(
                "CLASSIFIER_CROP_CONTEXTS",
                legacy_context,
            ),
            classifier_image_size=self._integer(
                "CLASSIFIER_IMAGE_SIZE", defaults.classifier_image_size
            ),
            min_box_area_ratio=self._float(
                "MIN_BOX_AREA_RATIO", defaults.min_box_area_ratio
            ),
            final_nms_iou_threshold=self._float(
                "FINAL_NMS_IOU_THRESHOLD", defaults.final_nms_iou_threshold
            ),
            confidence_threshold=self._float(
                "CONFIDENCE_THRESHOLD", defaults.confidence_threshold
            ),
            max_upload_size_bytes=self._integer(
                "MAX_UPLOAD_SIZE_BYTES", defaults.max_upload_size_bytes
            ),
        )
        self._validator.validate(settings)
        return settings

    def _text(self, name: str, default: str) -> str:
        """
        Read a raw environment value with a string fallback.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : str
            Value used when the variable is absent.

        Returns
        -------
        str
            Declared or fallback text.
        """
        return self._values.get(name, default)

    def _path(self, name: str, default: Path) -> Path:
        """
        Parse a filesystem path environment variable.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : Path
            Fallback path.

        Returns
        -------
        Path
            Parsed platform-native path.
        """
        return Path(self._text(name, str(default)))

    def _float(self, name: str, default: float) -> float:
        """
        Parse a floating-point environment variable.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : float
            Fallback numeric value.

        Returns
        -------
        float
            Parsed floating-point value.
        """
        return float(self._text(name, str(default)))

    def _integer(self, name: str, default: int) -> int:
        """
        Parse an integer environment variable.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : int
            Fallback integer value.

        Returns
        -------
        int
            Parsed integer value.
        """
        return int(self._text(name, str(default)))

    def _boolean(self, name: str, default: bool) -> bool:
        """
        Parse a strict boolean environment variable.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : bool
            Fallback boolean value.

        Returns
        -------
        bool
            Parsed boolean value.

        Raises
        ------
        ValueError
            Raised when text is not a supported true or false representation.
        """
        value = self._text(name, str(default)).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for {name}: {value!r}")

    def _float_list(self, name: str, default: str) -> tuple[float, ...]:
        """
        Parse a comma-separated ordered set of floating-point values.

        Parameters
        ----------
        name : str
            Environment variable name.
        default : str
            Fallback comma-separated text.

        Returns
        -------
        tuple[float, ...]
            Unique values in declaration order.

        Raises
        ------
        ValueError
            Raised when the variable contains no numeric entries.
        """
        values = tuple(
            dict.fromkeys(
                float(item.strip())
                for item in self._text(name, default).split(",")
                if item.strip()
            )
        )
        if not values:
            raise ValueError(f"{name} must contain at least one value")
        return values


class SettingsValidator:
    """Validate cross-field ranges after environment parsing."""

    _THRESHOLD_FIELDS = (
        "classifier_confidence_threshold",
        "bolt_confidence_threshold",
        "nut_confidence_threshold",
        "bolt_classifier_threshold",
        "nut_classifier_threshold",
        "final_nms_iou_threshold",
        "confidence_threshold",
    )

    def validate(self, settings: Settings) -> None:
        """
        Enforce numeric ranges and required non-empty values.

        Parameters
        ----------
        settings : Settings
            Parsed settings to validate.

        Raises
        ------
        ValueError
            Raised when any value cannot be used safely by the pipeline.
        """
        for field_name in self._THRESHOLD_FIELDS:
            value = getattr(settings, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between zero and one")
        if not settings.classifier_crop_contexts:
            raise ValueError("classifier_crop_contexts cannot be empty")
        if any(context < 0.0 for context in settings.classifier_crop_contexts):
            raise ValueError("classifier_crop_contexts cannot contain negatives")
        if settings.classifier_image_size <= 0:
            raise ValueError("classifier_image_size must be positive")
        if not 0.0 <= settings.min_box_area_ratio <= 1.0:
            raise ValueError("min_box_area_ratio must be between zero and one")
        if settings.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be positive")
