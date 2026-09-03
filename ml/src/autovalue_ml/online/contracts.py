"""Target-free prediction and delayed-outcome contracts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from autovalue_ml.online.errors import ShadowValidationError

FEATURE_CONTRACT_VERSION = "shadow-vehicle-features-v1"
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,127}$")


@dataclass(frozen=True, slots=True)
class ShadowVehicleFeatures:
    """Predictors available before price outcome observation."""

    year: int
    make: str
    model: str
    mileage: int
    condition: str | None = None
    engine: str | None = None
    transmission: str | None = None
    drivetrain: str | None = None
    accident_count: int | None = None
    owner_count: int | None = None
    vehicle_type: str | None = None

    def validate(self, *, observed_at: datetime) -> None:
        """Validate a U.S. vehicle predictor record without accepting a target."""
        _validate_utc(observed_at, field="observed_at")
        if type(self.year) is not int or not 1886 <= self.year <= observed_at.year + 2:
            raise ShadowValidationError("year is outside the accepted vehicle range")
        if not isinstance(self.make, str) or not self.make.strip():
            raise ShadowValidationError("make is required")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ShadowValidationError("model is required")
        if type(self.mileage) is not int or not 0 <= self.mileage <= 1_500_000:
            raise ShadowValidationError("mileage must be an integer from 0 to 1,500,000 miles")
        for name, text_value in (
            ("condition", self.condition),
            ("engine", self.engine),
            ("transmission", self.transmission),
            ("drivetrain", self.drivetrain),
            ("vehicle_type", self.vehicle_type),
        ):
            if text_value is not None and (
                not isinstance(text_value, str) or not text_value.strip()
            ):
                raise ShadowValidationError(f"{name} must be non-empty when present")
        for name, integer_value, maximum in (
            ("accident_count", self.accident_count, 100),
            ("owner_count", self.owner_count, 100),
        ):
            if integer_value is not None and (
                type(integer_value) is not int or not 0 <= integer_value <= maximum
            ):
                raise ShadowValidationError(f"{name} is outside the accepted range")

    def model_features(self, *, observed_at: datetime) -> dict[str, str | float]:
        """Return the versioned target-free feature map consumed by River."""
        self.validate(observed_at=observed_at)
        age = max(0, observed_at.year - self.year)
        age_denominator = max(1, age)
        return {
            "year": float(self.year),
            "vehicle_age": float(age),
            "mileage": float(self.mileage),
            "mileage_per_year": float(self.mileage / age_denominator),
            "accident_count": float(self.accident_count or 0),
            "owner_count": float(self.owner_count or 0),
            "accident_count_missing": str(self.accident_count is None).lower(),
            "owner_count_missing": str(self.owner_count is None).lower(),
            "make": self.make.strip(),
            "model": self.model.strip(),
            "condition": _category(self.condition),
            "engine": _category(self.engine),
            "transmission": _category(self.transmission),
            "drivetrain": _category(self.drivetrain),
            "vehicle_type": _category(self.vehicle_type),
        }


@dataclass(frozen=True, slots=True)
class ShadowPredictionRequest:
    """Request for an experimental prediction that cannot update the model."""

    event_id: str
    source_id: str
    observed_at: datetime
    features: ShadowVehicleFeatures
    reference_prediction_usd: float

    def validate(self) -> None:
        _validate_identifier(self.event_id, field="event_id")
        _validate_identifier(self.source_id, field="source_id")
        _validate_utc(self.observed_at, field="observed_at")
        self.features.validate(observed_at=self.observed_at)
        if isinstance(self.reference_prediction_usd, bool) or not isinstance(
            self.reference_prediction_usd, (int, float)
        ):
            raise ShadowValidationError("reference prediction must be numeric")
        if not math.isfinite(float(self.reference_prediction_usd)):
            raise ShadowValidationError("reference prediction must be finite")


@dataclass(frozen=True, slots=True)
class ShadowOutcomeSubmission:
    """A later outcome; validation occurs inside the quarantine boundary."""

    outcome_id: str
    event_id: str
    source_id: str
    occurred_at: datetime
    target_price_usd: int | float | str | None


def validate_identifier(value: str, *, field: str) -> None:
    """Public identifier validator used by the service boundary."""
    _validate_identifier(value, field=field)


def validate_utc(value: datetime, *, field: str) -> None:
    """Public UTC timestamp validator used by the service boundary."""
    _validate_utc(value, field=field)


def _validate_identifier(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ShadowValidationError(f"{field} is invalid")


def _validate_utc(value: datetime, *, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ShadowValidationError(f"{field} must be a timezone-aware datetime")
    if value.utcoffset() != timedelta(0):
        raise ShadowValidationError(f"{field} must use UTC")


def _category(value: str | None) -> str:
    return value.strip() if value is not None else "__missing__"
