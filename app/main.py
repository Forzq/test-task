"""FastAPI entry point for nut and bolt object detection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from app.config import Settings
from app.detector import ModelUnavailableError, UltralyticsDetector
from app.domain import DetectorProtocol
from app.schemas import HealthResponse, PredictionResponse
from app.service import DetectionService, UnsupportedImageError

logger = logging.getLogger(__name__)
SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png"}


def create_app(
    settings: Settings | None = None,
    detector: DetectorProtocol | None = None,
) -> FastAPI:
    """
    Create and configure the HTTP API application.

    Parameters
    ----------
    settings : Settings, optional
        Explicit runtime settings, primarily useful for tests.
    detector : DetectorProtocol, optional
        Preconfigured detector, primarily useful for tests and dependency injection.

    Returns
    -------
    FastAPI
        Fully configured FastAPI application instance.

    Notes
    -----
    When no detector is injected, weights are loaded at application startup.
    The service remains available with a clear 503 response if weights are absent.
    """
    resolved_settings = settings or Settings.from_environment()
    resolved_detector = detector or UltralyticsDetector(resolved_settings.model_path)
    should_load_model = detector is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """
        Load model weights during startup without preventing API diagnostics.

        Parameters
        ----------
        app : FastAPI
            Application whose state stores detector readiness information.

        Yields
        ------
        None
            Control back to the FastAPI lifespan manager.
        """
        app.state.model_error = None
        if should_load_model:
            try:
                resolved_detector.load()  # type: ignore[attr-defined]
            except ModelUnavailableError as error:
                app.state.model_error = str(error)
                logger.error("Detection model is unavailable: %s", error)
        yield

    app = FastAPI(
        title="Nut and Bolt Detection API",
        version="1.0.0",
        description="Detects and counts nuts and bolts in JPG or PNG images.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.detection_service = DetectionService(
        detector=resolved_detector,
        confidence_threshold=resolved_settings.confidence_threshold,
        max_upload_size_bytes=resolved_settings.max_upload_size_bytes,
    )
    app.state.model_error = None

    @app.get("/health", response_model=HealthResponse, tags=["operational"])
    def health(request: Request) -> HealthResponse:
        """
        Return model readiness without performing an inference request.

        Parameters
        ----------
        request : Request
            FastAPI request used to access application state.

        Returns
        -------
        HealthResponse
            Readiness state and a diagnostic message when weights are unavailable.
        """
        error = request.app.state.model_error
        return HealthResponse(
            status="model_unavailable" if error else "ready",
            model_path=str(request.app.state.settings.model_path),
            detail=error,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        response_model_by_alias=True,
        tags=["prediction"],
    )
    async def predict(
        request: Request,
        image: Annotated[UploadFile, File(description="JPEG or PNG image to analyse")],
    ) -> PredictionResponse:
        """
        Detect, classify, and count nuts and bolts in one uploaded image.

        Parameters
        ----------
        request : Request
            FastAPI request used to access application services and readiness.
        image : UploadFile
            JPEG or PNG file sent in the multipart field named ``image``.

        Returns
        -------
        PredictionResponse
            Detected objects and counts for each supported class.

        Raises
        ------
        HTTPException
            Raised with 415 for invalid uploads, 503 for unavailable weights,
            or 500 for unexpected inference failures.
        """
        if image.content_type not in SUPPORTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only image/jpeg and image/png uploads are supported.",
            )
        if request.app.state.model_error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=request.app.state.model_error,
            )

        try:
            image_bytes = await image.read()
            service: DetectionService = request.app.state.detection_service
            return service.predict(image_bytes)
        except UnsupportedImageError as error:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(error),
            ) from error
        except ModelUnavailableError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception as error:  # pragma: no cover - last-resort API guard
            logger.exception("Unexpected prediction error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failed unexpectedly.",
            ) from error

    return app


app = create_app()
