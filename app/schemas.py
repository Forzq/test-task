"""Pydantic schemas used by the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectedObject(BaseModel):
    """A nut or bolt detected in an input image."""

    model_config = ConfigDict(populate_by_name=True)

    class_name: Literal["nut", "bolt"] = Field(alias="class")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box in [x1, y1, x2, y2] pixel coordinates.",
    )


class DetectionCounts(BaseModel):
    """Aggregated number of detected nuts and bolts."""

    nuts: int = Field(ge=0)
    bolts: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Successful response returned by the POST /predict endpoint."""

    objects: list[DetectedObject]
    counts: DetectionCounts


class HealthResponse(BaseModel):
    """Readiness state returned by the GET /health endpoint."""

    status: Literal["ready", "model_unavailable"]
    model_path: str
    detail: str | None = None
