"""Public request and response contracts for calibrated retail valuations."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ConfidenceLabel = Literal["High confidence", "Moderate confidence", "Low confidence"]


class VehicleValuationRequest(BaseModel):
    """Exact public input contract supported by the frozen retail model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    year: int = Field(ge=1900, le=2023)
    make: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    vehicle_status: Literal["certified", "new", "used"]
    mileage: float | None = Field(default=None, ge=0.0, le=500_000.0)
    interval_coverage: float = 0.9

    @field_validator("make", "model")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("vehicle text fields must not contain control characters")
        return value

    @field_validator("interval_coverage")
    @classmethod
    def require_supported_interval(cls, value: float) -> float:
        if value not in {0.8, 0.9, 0.95}:
            raise ValueError("interval coverage must be 0.8, 0.9, or 0.95")
        return value


class ModelInformation(BaseModel):
    """Non-sensitive model identity and target semantics returned to clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1, max_length=100)
    model_version: Literal["retail-rf05-v1"] = "retail-rf05-v1"
    model_family: Literal["random_forest"] = "random_forest"
    feature_contract_version: str = Field(min_length=1, max_length=100)
    target_context: Literal["historical_us_advertised_asking_price_usd_2023"] = (
        "historical_us_advertised_asking_price_usd_2023"
    )


class PredictionResponse(BaseModel):
    """Point estimate with optional all-or-none calibrated interval fields.

    A response containing only ``predicted_value`` remains valid for compatibility
    while clients and the serving layer adopt the calibrated contract.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_value: float = Field(ge=0.0)
    currency: Literal["USD"] = "USD"
    interval_lower: float | None = Field(default=None, ge=0.0)
    interval_upper: float | None = Field(default=None, ge=0.0)
    interval_coverage: float | None = None
    interval_width: float | None = Field(default=None, ge=0.0)
    confidence_label: ConfidenceLabel | None = Field(
        default=None,
        description=(
            "Backward-compatible support/relative-width label; not an empirical "
            "probability of correctness and not shown as confidence in the primary UI."
        ),
    )
    calibration_version: str | None = Field(default=None, min_length=1, max_length=100)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    model_information: ModelInformation | None = None

    @model_validator(mode="after")
    def validate_calibrated_interval(self) -> Self:
        if len(self.warnings) != len(set(self.warnings)):
            raise ValueError("warnings must be unique")
        values = (
            self.interval_lower,
            self.interval_upper,
            self.interval_coverage,
            self.interval_width,
            self.confidence_label,
            self.calibration_version,
        )
        present = sum(value is not None for value in values)
        if present not in {0, len(values)}:
            raise ValueError("calibrated interval fields must be supplied together")
        if present == 0:
            return self
        if (
            self.interval_lower is None
            or self.interval_upper is None
            or self.interval_width is None
            or self.interval_coverage is None
        ):
            raise ValueError("calibrated interval fields must be supplied together")
        lower = float(self.interval_lower)
        upper = float(self.interval_upper)
        width = float(self.interval_width)
        if not all(math.isfinite(value) for value in (self.predicted_value, lower, upper, width)):
            raise ValueError("valuation response values must be finite")
        if not lower <= self.predicted_value <= upper:
            raise ValueError("predicted value must fall inside its calibrated interval")
        if self.interval_coverage not in {0.8, 0.9, 0.95}:
            raise ValueError("interval coverage must be 0.8, 0.9, or 0.95")
        if not math.isclose(width, upper - lower, rel_tol=1e-9, abs_tol=0.01):
            raise ValueError("interval width must equal upper minus lower")
        return self


__all__ = [
    "ConfidenceLabel",
    "ModelInformation",
    "PredictionResponse",
    "VehicleValuationRequest",
]
