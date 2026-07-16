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
    confidence_threshold : float
        Minimum confidence required for a detection to be returned.
    max_upload_size_bytes : int
        Maximum accepted image size in bytes.
    """

    model_path: Path
    confidence_threshold: float = 0.50
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
        MODEL_PATH, CONFIDENCE_THRESHOLD, and MAX_UPLOAD_SIZE_BYTES can be
        overridden without changing application code.
        """
        model_path = Path(os.getenv("MODEL_PATH", "models/best.pt"))
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
        max_upload_size_bytes = int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
        )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if max_upload_size_bytes <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_BYTES must be positive.")

        return cls(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            max_upload_size_bytes=max_upload_size_bytes,
        )
