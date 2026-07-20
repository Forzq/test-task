"""FastAPI composition root for nut and bolt detection."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from app.classifier import ClassifierUnavailableError, UltralyticsCropClassifier
from app.config import Settings
from app.detector import ModelUnavailableError, UltralyticsDetector
from app.domain import CropClassifierProtocol, DetectorProtocol
from app.schemas import HealthResponse, PredictionResponse
from app.service import DetectionService, UnsupportedImageError

logger = logging.getLogger(__name__)


class ModelRuntime:
    """
    Own model adapters, loading lifecycle, and the configured detection service.

    Parameters
    ----------
    settings : Settings
        Validated runtime configuration.
    detector : DetectorProtocol
        Detector adapter used by the pipeline.
    classifier : CropClassifierProtocol or None
        Optional crop-classifier adapter.
    load_detector : bool
        Whether this runtime owns and must load the detector adapter.
    load_classifier : bool
        Whether this runtime owns and must load the classifier adapter.
    """

    def __init__(
        self,
        settings: Settings,
        detector: DetectorProtocol,
        classifier: CropClassifierProtocol | None,
        load_detector: bool,
        load_classifier: bool,
    ) -> None:
        """Store adapters and construct the application inference service."""
        self.settings = settings
        self.detector = detector
        self.classifier = classifier
        self._load_detector = load_detector
        self._load_classifier = load_classifier
        self.model_error: str | None = None
        self.service = self._build_service()

    @classmethod
    def create(
        cls,
        settings: Settings,
        detector: DetectorProtocol | None = None,
        classifier: CropClassifierProtocol | None = None,
    ) -> "ModelRuntime":
        """
        Resolve injected test doubles or construct production model adapters.

        Parameters
        ----------
        settings : Settings
            Runtime model paths and pipeline thresholds.
        detector : DetectorProtocol, optional
            Injected detector that is already ready for inference.
        classifier : CropClassifierProtocol, optional
            Injected classifier that is already ready for inference.

        Returns
        -------
        ModelRuntime
            Runtime with explicit ownership flags for application startup.
        """
        resolved_detector = detector or UltralyticsDetector(settings.model_path)
        resolved_classifier = classifier
        if resolved_classifier is None and settings.enable_crop_classifier:
            resolved_classifier = UltralyticsCropClassifier(
                settings.classifier_model_path,
                settings.classifier_image_size,
            )
        return cls(
            settings=settings,
            detector=resolved_detector,
            classifier=resolved_classifier,
            load_detector=detector is None,
            load_classifier=(
                classifier is None and settings.enable_crop_classifier
            ),
        )

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        """
        Load owned model adapters while keeping diagnostics available on error.

        Parameters
        ----------
        app : FastAPI
            Application whose state exposes model readiness.

        Yields
        ------
        None
            Control to FastAPI after startup loading has completed.
        """
        self.model_error = self._load_models()
        app.state.model_error = self.model_error
        yield

    def attach(self, app: FastAPI) -> None:
        """
        Publish runtime dependencies through FastAPI application state.

        Parameters
        ----------
        app : FastAPI
            Application receiving settings, service, and readiness state.
        """
        app.state.settings = self.settings
        app.state.detection_service = self.service
        app.state.model_runtime = self
        app.state.model_error = self.model_error

    def _load_models(self) -> str | None:
        """
        Load adapters owned by the production runtime and collect diagnostics.

        Returns
        -------
        str or None
            Combined loading errors, or ``None`` when every adapter is ready.
        """
        errors: list[str] = []
        if self._load_detector:
            try:
                load = getattr(self.detector, "load")
                load()
            except ModelUnavailableError as error:
                errors.append(str(error))
                logger.error("Detection model is unavailable: %s", error)
        if self._load_classifier and self.classifier is not None:
            try:
                load = getattr(self.classifier, "load")
                load()
            except ClassifierUnavailableError as error:
                errors.append(str(error))
                logger.error("Crop classifier is unavailable: %s", error)
        return "; ".join(errors) or None

    def _build_service(self) -> DetectionService:
        """
        Translate settings into the reusable inference orchestration service.

        Returns
        -------
        DetectionService
            Fully configured detector and optional verifier pipeline.
        """
        return DetectionService(
            detector=self.detector,
            confidence_threshold=self.settings.confidence_threshold,
            max_upload_size_bytes=self.settings.max_upload_size_bytes,
            crop_classifier=self.classifier,
            classifier_confidence_threshold=(
                self.settings.classifier_confidence_threshold
            ),
            classifier_image_size=self.settings.classifier_image_size,
            min_box_area_ratio=self.settings.min_box_area_ratio,
            detector_class_thresholds={
                "bolt": self.settings.bolt_confidence_threshold,
                "nut": self.settings.nut_confidence_threshold,
            },
            classifier_class_thresholds={
                "bolt": self.settings.bolt_classifier_threshold,
                "nut": self.settings.nut_classifier_threshold,
            },
            classifier_crop_contexts=self.settings.classifier_crop_contexts,
            final_nms_iou_threshold=self.settings.final_nms_iou_threshold,
        )


class PredictionController:
    """
    Expose health and prediction use cases through FastAPI routes.

    Parameters
    ----------
    runtime : ModelRuntime
        Runtime used for readiness information and inference execution.
    """

    _SUPPORTED_MEDIA_TYPES = {"image/jpeg", "image/png"}

    def __init__(self, runtime: ModelRuntime) -> None:
        """Store the model runtime used by every request handler."""
        self._runtime = runtime

    def register(self, app: FastAPI) -> None:
        """
        Register controller methods without embedding logic in the app factory.

        Parameters
        ----------
        app : FastAPI
            Application receiving operational and prediction routes.
        """
        app.add_api_route(
            "/health",
            self.health,
            methods=["GET"],
            response_model=HealthResponse,
            tags=["operational"],
        )
        app.add_api_route(
            "/predict",
            self.predict,
            methods=["POST"],
            response_model=PredictionResponse,
            response_model_by_alias=True,
            tags=["prediction"],
        )

    async def health(self, request: Request) -> HealthResponse:
        """
        Return readiness without running model inference.

        Parameters
        ----------
        request : Request
            FastAPI request exposing current application state.

        Returns
        -------
        HealthResponse
            Model paths and an optional startup diagnostic.
        """
        error = request.app.state.model_error
        return HealthResponse(
            status="model_unavailable" if error else "ready",
            model_path=str(self._runtime.settings.model_path),
            classifier_enabled=self._runtime.classifier is not None,
            classifier_model_path=(
                str(self._runtime.settings.classifier_model_path)
                if self._runtime.classifier is not None
                else None
            ),
            detail=error,
        )

    async def predict(
        self,
        request: Request,
        image: Annotated[
            UploadFile,
            File(description="JPEG or PNG image to analyse"),
        ],
    ) -> PredictionResponse:
        """
        Validate an upload and execute the complete detection service.

        Parameters
        ----------
        request : Request
            FastAPI request exposing readiness and application services.
        image : UploadFile
            JPEG or PNG multipart field named ``image``.

        Returns
        -------
        PredictionResponse
            Detected objects and class counts.

        Raises
        ------
        HTTPException
            Raised with 415 for invalid uploads, 503 for unavailable models,
            or 500 for an unexpected inference failure.
        """
        self._ensure_supported_media_type(image.content_type)
        self._ensure_models_ready(request.app.state.model_error)
        try:
            return self._runtime.service.predict(await image.read())
        except UnsupportedImageError as error:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(error),
            ) from error
        except (ModelUnavailableError, ClassifierUnavailableError) as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        except Exception as error:  # pragma: no cover - final API safety boundary
            logger.exception("Unexpected prediction error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Inference failed unexpectedly.",
            ) from error

    def _ensure_supported_media_type(self, content_type: str | None) -> None:
        """
        Reject multipart content types that the decoder does not support.

        Parameters
        ----------
        content_type : str or None
            Declared upload MIME type.

        Raises
        ------
        HTTPException
            Raised with HTTP 415 for non-JPEG and non-PNG uploads.
        """
        if content_type not in self._SUPPORTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only image/jpeg and image/png uploads are supported.",
            )

    @staticmethod
    def _ensure_models_ready(error: str | None) -> None:
        """
        Prevent inference when startup model loading failed.

        Parameters
        ----------
        error : str or None
            Startup diagnostic stored in application state.

        Raises
        ------
        HTTPException
            Raised with HTTP 503 when model adapters are unavailable.
        """
        if error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=error,
            )


class ApplicationFactory:
    """Build a fully wired FastAPI application from settings and adapters."""

    def create(
        self,
        settings: Settings | None = None,
        detector: DetectorProtocol | None = None,
        classifier: CropClassifierProtocol | None = None,
    ) -> FastAPI:
        """
        Compose runtime dependencies, controller routes, and model lifecycle.

        Parameters
        ----------
        settings : Settings, optional
            Explicit settings, primarily used by tests.
        detector : DetectorProtocol, optional
            Injected ready-to-use detector.
        classifier : CropClassifierProtocol, optional
            Injected ready-to-use crop classifier.

        Returns
        -------
        FastAPI
            Configured application preserving the public API contract.
        """
        resolved_settings = settings or Settings.from_environment()
        runtime = ModelRuntime.create(
            settings=resolved_settings,
            detector=detector,
            classifier=classifier,
        )
        app = FastAPI(
            title="Nut and Bolt Detection API",
            version="1.0.0",
            description="Detects and counts nuts and bolts in JPG or PNG images.",
            lifespan=runtime.lifespan,
        )
        runtime.attach(app)
        PredictionController(runtime).register(app)
        return app


def create_app(
    settings: Settings | None = None,
    detector: DetectorProtocol | None = None,
    classifier: CropClassifierProtocol | None = None,
) -> FastAPI:
    """
    Preserve the original functional entry point around the application factory.

    Parameters
    ----------
    settings : Settings, optional
        Explicit runtime settings.
    detector : DetectorProtocol, optional
        Injected detector for testing or custom composition.
    classifier : CropClassifierProtocol, optional
        Injected crop classifier for testing or custom composition.

    Returns
    -------
    FastAPI
        Fully configured API application.
    """
    return ApplicationFactory().create(settings, detector, classifier)


app = create_app()
