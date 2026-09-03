"""Aggregate-only regression metrics for model comparison and error slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Three standard aggregate regression metrics and their sample count."""

    sample_count: int
    mae: float
    rmse: float
    r2: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a JSON-safe object with a stable field order."""

        return {
            "sample_count": self.sample_count,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
        }


@dataclass(frozen=True, slots=True)
class StatusSliceMetrics:
    """Aggregate evaluation for one retail vehicle-status segment."""

    status: str
    metrics: RegressionMetrics

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "metrics": self.metrics.to_dict()}


@dataclass(frozen=True, slots=True)
class RetailEvaluation:
    """Overall retail metrics plus deterministically ordered status slices."""

    overall: RegressionMetrics
    status_slices: tuple[StatusSliceMetrics, ...]


def regression_metrics(y_true: object, y_predicted: object) -> RegressionMetrics:
    """Compute MAE, RMSE, and R-squared without forcing undefined R-squared."""

    actual = _finite_vector(y_true, label="y_true")
    predicted = _finite_vector(y_predicted, label="y_predicted")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_predicted must have the same number of values")
    if len(actual) == 0:
        raise ValueError("metrics require at least one observation")

    residual = actual - predicted
    mae = float(np.mean(np.abs(residual), dtype=np.float64))
    rmse = float(np.sqrt(np.mean(np.square(residual), dtype=np.float64)))
    centered = actual - float(np.mean(actual, dtype=np.float64))
    denominator = float(np.sum(np.square(centered), dtype=np.float64))
    if len(actual) < 2 or denominator == 0.0:
        r2 = None
    else:
        numerator = float(np.sum(np.square(residual), dtype=np.float64))
        r2 = float(1.0 - numerator / denominator)

    if not np.isfinite((mae, rmse)).all() or (r2 is not None and not np.isfinite(r2)):
        raise ValueError("metric calculation produced a non-finite result")
    return RegressionMetrics(sample_count=len(actual), mae=mae, rmse=rmse, r2=r2)


def retail_status_metrics(
    y_true: object,
    y_predicted: object,
    vehicle_status: object,
) -> RetailEvaluation:
    """Compute overall and status-specific aggregates for the retail track."""

    actual = _finite_vector(y_true, label="y_true")
    predicted = _finite_vector(y_predicted, label="y_predicted")
    statuses = np.asarray(vehicle_status, dtype=object)
    if statuses.ndim != 1:
        raise ValueError("vehicle_status must be one-dimensional")
    if len(actual) != len(predicted) or len(actual) != len(statuses):
        raise ValueError("targets, predictions, and vehicle_status must have equal lengths")

    normalized: list[str] = []
    for status in statuses:
        if not isinstance(status, str) or not status.strip():
            raise ValueError("vehicle_status values must be non-empty strings")
        normalized.append(status.strip().lower())
    normalized_array = np.asarray(normalized, dtype=np.str_)

    slices = tuple(
        StatusSliceMetrics(
            status=status,
            metrics=regression_metrics(
                actual[normalized_array == status],
                predicted[normalized_array == status],
            ),
        )
        for status in sorted(set(normalized))
    )
    return RetailEvaluation(
        overall=regression_metrics(actual, predicted),
        status_slices=slices,
    )


def _finite_vector(values: object, *, label: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    inspect_values = (
        np.asarray(values, dtype=object)
        if not isinstance(values, np.ndarray) or array.dtype == object
        else array
    )
    if np.issubdtype(array.dtype, np.bool_) or any(
        isinstance(value, (bool, np.bool_)) for value in inspect_values.flat
    ):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        numeric = array.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only numeric values") from error
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain only finite values")
    return numeric
