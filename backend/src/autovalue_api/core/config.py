"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by application startup and route handlers."""

    model_config = SettingsConfigDict(
        env_file="backend/.env",
        env_prefix="AUTOVALUE_",
        extra="ignore",
    )

    app_name: str = "AutoValue AI API"
    environment: Literal["development", "test", "production"] = "development"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    trusted_models_root: Path = Path("models")
    model_bundle_dir: Path = Path("models/retail-rf05-v1")
    trusted_calibration_root: Path = Path("docs/experiments")
    calibration_artifact_path: Path = Path(
        "docs/experiments/retail-rf05-calibration-v1.artifact.json"
    )
    prediction_history_path: Path = Path("data/local/prediction-history.sqlite3")

    @model_validator(mode="after")
    def validate_production_origins(self) -> "Settings":
        """Require explicit public HTTPS browser origins in production."""
        if self.environment != "production":
            return self
        if not self.cors_origins:
            raise ValueError("production requires at least one explicit CORS origin")
        for origin in self.cors_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "https"
                or not parsed.netloc
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                or "*" in origin
            ):
                raise ValueError(
                    "production CORS origins must be explicit HTTPS origins without paths"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings object per process."""
    return Settings()
