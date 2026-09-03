"""Tests for environment-backed configuration."""

from pathlib import Path

import pytest
from autovalue_api.core.config import Settings
from pydantic import ValidationError


def test_default_development_origins_are_explicit() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert settings.trusted_models_root == Path("models")
    assert settings.model_bundle_dir == Path("models/retail-rf05-v1")
    assert settings.trusted_calibration_root == Path("docs/experiments")
    assert settings.prediction_history_path == Path("data/local/prediction-history.sqlite3")


def test_environment_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTOVALUE_ENVIRONMENT", "test")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.environment == "test"


def test_production_rejects_development_cors_defaults() -> None:
    with pytest.raises(ValidationError, match="explicit HTTPS origins"):
        Settings(environment="production", _env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "origin",
    [
        "http://portfolio.example",
        "https://localhost:5173",
        "https://*.example.com",
        "https://portfolio.example/app",
    ],
)
def test_production_rejects_unsafe_cors_origin(origin: str) -> None:
    with pytest.raises(ValidationError, match="explicit HTTPS origins"):
        Settings(  # type: ignore[call-arg]
            environment="production",
            cors_origins=[origin],
            _env_file=None,
        )


def test_production_accepts_explicit_https_cors_origin() -> None:
    settings = Settings(  # type: ignore[call-arg]
        environment="production",
        cors_origins=["https://autovalue.example"],
        _env_file=None,
    )

    assert settings.cors_origins == ["https://autovalue.example"]
