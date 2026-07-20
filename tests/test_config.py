"""Tests for typed environment configuration loading."""

from pathlib import Path

import pytest

from app.config import EnvironmentSettingsLoader


def test_environment_loader_preserves_deployment_defaults() -> None:
    """Ensure an empty environment reproduces the calibrated deployment."""
    settings = EnvironmentSettingsLoader({}).load()

    assert settings.model_path == Path("models/best.pt")
    assert settings.classifier_model_path == Path("models/classifier_best.pt")
    assert settings.bolt_confidence_threshold == 0.43
    assert settings.nut_confidence_threshold == 0.56
    assert settings.bolt_classifier_threshold == 0.39
    assert settings.nut_classifier_threshold == 0.51
    assert settings.classifier_crop_contexts == (0.20,)


def test_environment_loader_parses_explicit_overrides() -> None:
    """Convert supported string values into their strongly typed forms."""
    settings = EnvironmentSettingsLoader(
        {
            "MODEL_PATH": "models/candidate.pt",
            "ENABLE_CROP_CLASSIFIER": "false",
            "CLASSIFIER_CROP_CONTEXTS": "0.1, 0.3, 0.1",
            "CLASSIFIER_IMAGE_SIZE": "320",
            "MAX_UPLOAD_SIZE_BYTES": "2048",
        }
    ).load()

    assert settings.model_path == Path("models/candidate.pt")
    assert settings.enable_crop_classifier is False
    assert settings.classifier_crop_contexts == (0.1, 0.3)
    assert settings.classifier_image_size == 320
    assert settings.max_upload_size_bytes == 2048


def test_environment_loader_supports_legacy_crop_context() -> None:
    """Keep old single-context deployments compatible after refactoring."""
    settings = EnvironmentSettingsLoader(
        {"CLASSIFIER_CROP_CONTEXT": "0.35"}
    ).load()

    assert settings.classifier_crop_contexts == (0.35,)


@pytest.mark.parametrize(
    ("values", "expected_message"),
    [
        ({"ENABLE_CROP_CLASSIFIER": "sometimes"}, "Invalid boolean"),
        ({"BOLT_CONFIDENCE_THRESHOLD": "1.5"}, "between zero and one"),
        ({"CLASSIFIER_IMAGE_SIZE": "0"}, "must be positive"),
    ],
)
def test_environment_loader_rejects_invalid_values(
    values: dict[str, str],
    expected_message: str,
) -> None:
    """Reject unsafe configuration before model initialization starts."""
    with pytest.raises(ValueError, match=expected_message):
        EnvironmentSettingsLoader(values).load()
