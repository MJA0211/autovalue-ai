"""Leakage-safe calibration partitions and split-conformal price ranges.

The fitted calibration contract stores only aggregate counts and quantiles. Raw
targets, predictions, and residuals are intentionally kept out of returned
calibration objects so callers cannot accidentally persist row-level evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import RETAIL_TRACK, validate_feature_frame
from .cv import retail_predictor_groups
from .feature_engineering import VehicleFeatureEngineer

RETAIL_VEHICLE_STATUSES: Final[tuple[str, ...]] = ("certified", "new", "used")
WHOLESALE_DEVELOPMENT_BUCKETS: Final[tuple[str, ...]] = (
    "warmup",
    "2015_01",
    "2015_02",
    "2015_03_04",
)
WHOLESALE_CALIBRATION_BUCKET: Final = "2015_05"

_RETAIL_CALIBRATION_DENOMINATOR: Final = 10
_RETAIL_HASH_DOMAIN: Final = b"autovalue-retail-calibration-v1\x00"
_MAX_UINT32: Final = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class CalibrationPartition:
    """Disjoint, positional development and calibration row indices."""

    development_indices: NDArray[np.int64]
    calibration_indices: NDArray[np.int64]

    def __post_init__(self) -> None:
        development = _index_vector(self.development_indices, label="development_indices")
        calibration = _index_vector(self.calibration_indices, label="calibration_indices")
        if development.size == 0:
            raise ValueError("development partition must not be empty")
        if calibration.size == 0:
            raise ValueError("calibration partition must not be empty")
        if np.intersect1d(development, calibration, assume_unique=True).size:
            raise ValueError("development and calibration indices must not overlap")
        object.__setattr__(self, "development_indices", development)
        object.__setattr__(self, "calibration_indices", calibration)

    @property
    def sample_count(self) -> int:
        """Return the total number of represented rows."""

        return int(self.development_indices.size + self.calibration_indices.size)

    def validate_full_coverage(self, expected_rows: int) -> None:
        """Fail unless every position from zero through ``expected_rows - 1`` appears."""

        if isinstance(expected_rows, (bool, np.bool_)) or not isinstance(expected_rows, Integral):
            raise ValueError("expected_rows must be an integer")
        if expected_rows < 1:
            raise ValueError("expected_rows must be positive")
        combined = np.sort(np.concatenate((self.development_indices, self.calibration_indices)))
        expected = np.arange(int(expected_rows), dtype=np.int64)
        if not np.array_equal(combined, expected):
            raise ValueError("partition indices must provide full positional coverage")


@dataclass(frozen=True, slots=True)
class PredictionRanges:
    """Ephemeral lower and upper price bounds for one prediction batch."""

    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    def __post_init__(self) -> None:
        lower = _finite_vector(self.lower_bounds, label="lower_bounds", allow_empty=False)
        upper = _finite_vector(self.upper_bounds, label="upper_bounds", allow_empty=False)
        if len(lower) != len(upper):
            raise ValueError("lower_bounds and upper_bounds must have equal lengths")
        if (lower < 0.0).any():
            raise ValueError("prediction range lower bounds must be nonnegative")
        if (upper < lower).any():
            raise ValueError("prediction range upper bounds must not be below lower bounds")
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)


@dataclass(frozen=True, slots=True)
class RetailConformalCalibration:
    """Aggregate-only global and vehicle-status conformal quantiles."""

    alpha: float
    sample_count: int
    global_quantile: float
    status_sample_counts: tuple[tuple[str, int], ...]
    status_quantiles: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        alpha = _validated_alpha(self.alpha)
        if isinstance(self.sample_count, (bool, np.bool_)) or not isinstance(
            self.sample_count, Integral
        ):
            raise ValueError("sample_count must be an integer")
        if self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if _finite_sample_order(int(self.sample_count), alpha) > self.sample_count:
            raise ValueError("sample_count is too small for alpha")
        global_quantile = _validated_quantile(
            self.global_quantile,
            label="global_quantile",
        )

        count_names = tuple(name for name, _ in self.status_sample_counts)
        if count_names != RETAIL_VEHICLE_STATUSES:
            raise ValueError("status_sample_counts must contain each retail status in order")
        counts: list[tuple[str, int]] = []
        for status, count in self.status_sample_counts:
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
                raise ValueError("status sample counts must be integers")
            if count < 0:
                raise ValueError("status sample counts must be nonnegative")
            counts.append((status, int(count)))
        if sum(count for _, count in counts) != self.sample_count:
            raise ValueError("status sample counts must sum to sample_count")

        quantile_names = tuple(name for name, _ in self.status_quantiles)
        expected_quantile_names = tuple(
            status
            for status, count in counts
            if count and _finite_sample_order(count, alpha) <= count
        )
        if quantile_names != expected_quantile_names:
            raise ValueError(
                "status_quantiles must exactly cover sufficiently large strata in status order"
            )
        quantiles = tuple(
            (status, _validated_quantile(value, label=f"{status} quantile"))
            for status, value in self.status_quantiles
        )

        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "global_quantile", global_quantile)
        object.__setattr__(self, "status_sample_counts", tuple(counts))
        object.__setattr__(self, "status_quantiles", quantiles)

    def quantile_for(self, vehicle_status: object) -> float:
        """Return a status quantile or the global fallback for unknown/missing status."""

        if isinstance(vehicle_status, str):
            normalized = vehicle_status.strip().lower()
            for status, quantile in self.status_quantiles:
                if normalized == status:
                    return quantile
        return self.global_quantile


def retail_calibration_partition(
    features: object,
    *,
    seed: int,
) -> CalibrationPartition:
    """Reserve approximately one tenth of each retail status without splitting groups.

    Allocation is target-free. Within each status, predictor-group digests are
    ordered by a domain-separated seeded SHA-256 digest. The selected prefix is
    the one whose row count is closest to one tenth of that status; equal
    distances retain the smaller prefix.
    """

    resolved_seed = _validated_seed(seed)
    frame = validate_feature_frame(features, RETAIL_TRACK)
    if len(frame) == 0:
        raise ValueError("retail calibration features must not be empty")

    engineered = VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(frame)
    statuses = _validated_calibration_statuses(engineered["vehicle_status"])
    missing_statuses = [
        status for status in RETAIL_VEHICLE_STATUSES if status not in set(statuses.tolist())
    ]
    if missing_statuses:
        raise ValueError(
            "retail calibration input has empty status strata: " + ", ".join(missing_statuses)
        )

    groups = retail_predictor_groups(frame, RETAIL_TRACK)
    selected_groups: set[str] = set()
    for status in RETAIL_VEHICLE_STATUSES:
        stratum_groups = groups[statuses == status]
        counts = Counter(str(group) for group in stratum_groups)
        ordered_groups = sorted(
            counts,
            key=lambda group: (_seeded_group_digest(group, resolved_seed), group),
        )
        prefix_length = _closest_tenth_prefix(
            [counts[group] for group in ordered_groups],
            total_rows=len(stratum_groups),
        )
        selected_groups.update(ordered_groups[:prefix_length])

    calibration_mask = np.fromiter(
        (str(group) in selected_groups for group in groups),
        dtype=np.bool_,
        count=len(groups),
    )
    partition = _complete_partition(calibration_mask)
    _validate_retail_group_partition(groups, statuses, partition)
    return partition


def wholesale_calibration_partition(
    cv_buckets: Sequence[str] | pd.Series,
) -> CalibrationPartition:
    """Use the four approved earlier buckets for development and May for calibration."""

    values = np.asarray(cv_buckets, dtype=object)
    if values.ndim != 1:
        raise ValueError("cv_buckets must be one-dimensional")
    if len(values) == 0:
        raise ValueError("cv_buckets must not be empty")

    approved = (*WHOLESALE_DEVELOPMENT_BUCKETS, WHOLESALE_CALIBRATION_BUCKET)
    observed: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("every wholesale row must have a CV bucket")
        if value != value.strip():
            raise ValueError("wholesale CV bucket values must use exact canonical spelling")
        observed.append(value)

    unknown = sorted(set(observed) - set(approved))
    if unknown:
        raise ValueError("cv_buckets contain unapproved values: " + ", ".join(unknown))
    empty = [bucket for bucket in approved if bucket not in observed]
    if empty:
        raise ValueError("approved wholesale buckets must not be empty: " + ", ".join(empty))

    calibration_mask = np.asarray(
        [bucket == WHOLESALE_CALIBRATION_BUCKET for bucket in observed],
        dtype=np.bool_,
    )
    return _complete_partition(calibration_mask)


def conformal_absolute_residual_quantile(
    y_true: object,
    y_predicted: object,
    *,
    alpha: float = 0.1,
) -> float:
    """Return the exact finite-sample absolute-residual order statistic."""

    resolved_alpha = _validated_alpha(alpha)
    residuals = _absolute_residuals(y_true, y_predicted)
    return _finite_sample_quantile(residuals, alpha=resolved_alpha)


def conformal_prediction_ranges(
    predictions: object,
    quantile: float,
) -> PredictionRanges:
    """Apply one nonnegative conformal radius and clip price bounds at zero."""

    predicted = _finite_vector(predictions, label="predictions", allow_empty=False)
    radius = _validated_quantile(quantile, label="quantile")
    radii = np.full(len(predicted), radius, dtype=np.float64)
    return _prediction_ranges(predicted, radii)


def fit_retail_status_conformal(
    y_true: object,
    y_predicted: object,
    vehicle_status: object,
    *,
    alpha: float = 0.1,
) -> RetailConformalCalibration:
    """Fit aggregate status radii, omitting undersized strata in favor of fallback."""

    resolved_alpha = _validated_alpha(alpha)
    residuals = _absolute_residuals(y_true, y_predicted)
    statuses = _validated_calibration_statuses(vehicle_status)
    if len(statuses) != len(residuals):
        raise ValueError("targets, predictions, and vehicle_status must have equal lengths")

    global_quantile = _finite_sample_quantile(residuals, alpha=resolved_alpha)
    counts: list[tuple[str, int]] = []
    quantiles: list[tuple[str, float]] = []
    for status in RETAIL_VEHICLE_STATUSES:
        status_residuals = residuals[statuses == status]
        count = len(status_residuals)
        counts.append((status, count))
        if count and _finite_sample_order(count, resolved_alpha) <= count:
            quantiles.append(
                (
                    status,
                    _finite_sample_quantile(status_residuals, alpha=resolved_alpha),
                )
            )

    return RetailConformalCalibration(
        alpha=resolved_alpha,
        sample_count=len(residuals),
        global_quantile=global_quantile,
        status_sample_counts=tuple(counts),
        status_quantiles=tuple(quantiles),
    )


def retail_conformal_prediction_ranges(
    predictions: object,
    vehicle_status: object,
    calibration: RetailConformalCalibration,
) -> PredictionRanges:
    """Apply status radii row-wise, using the frozen global fallback when needed."""

    if not isinstance(calibration, RetailConformalCalibration):
        raise TypeError("calibration must be a RetailConformalCalibration")
    predicted = _finite_vector(predictions, label="predictions", allow_empty=False)
    statuses = np.asarray(vehicle_status, dtype=object)
    if statuses.ndim != 1:
        raise ValueError("vehicle_status must be one-dimensional")
    if len(predicted) != len(statuses):
        raise ValueError("predictions and vehicle_status must have equal lengths")
    radii = np.asarray(
        [calibration.quantile_for(status) for status in statuses],
        dtype=np.float64,
    )
    return _prediction_ranges(predicted, radii)


def _complete_partition(calibration_mask: NDArray[np.bool_]) -> CalibrationPartition:
    development = np.flatnonzero(~calibration_mask).astype(np.int64, copy=False)
    calibration = np.flatnonzero(calibration_mask).astype(np.int64, copy=False)
    partition = CalibrationPartition(development, calibration)
    partition.validate_full_coverage(len(calibration_mask))
    return partition


def _validate_retail_group_partition(
    groups: NDArray[np.str_],
    statuses: NDArray[np.str_],
    partition: CalibrationPartition,
) -> None:
    calibration_positions = np.zeros(len(groups), dtype=np.bool_)
    calibration_positions[partition.calibration_indices] = True
    membership_by_group: dict[str, bool] = {}
    for group_value, is_calibration_value in zip(groups, calibration_positions, strict=True):
        group = str(group_value)
        is_calibration = bool(is_calibration_value)
        previous = membership_by_group.setdefault(group, is_calibration)
        if previous != is_calibration:
            raise RuntimeError("retail predictor group crossed the calibration boundary")
    for status in RETAIL_VEHICLE_STATUSES:
        status_positions = statuses == status
        if not status_positions.any():
            raise RuntimeError("retail calibration partition lost a status stratum")


def _closest_tenth_prefix(group_sizes: Sequence[int], *, total_rows: int) -> int:
    if total_rows < 1 or not group_sizes or sum(group_sizes) != total_rows:
        raise ValueError("retail status group sizes must cover a non-empty stratum")
    best_prefix = 0
    best_distance = total_rows
    cumulative = 0
    for prefix, size in enumerate(group_sizes, start=1):
        if size < 1:
            raise ValueError("retail predictor groups must not be empty")
        cumulative += size
        distance = abs(cumulative * _RETAIL_CALIBRATION_DENOMINATOR - total_rows)
        if distance < best_distance:
            best_prefix = prefix
            best_distance = distance
    return best_prefix


def _seeded_group_digest(group: str, seed: int) -> bytes:
    payload = _RETAIL_HASH_DOMAIN + seed.to_bytes(4, "big") + group.encode("ascii")
    return hashlib.sha256(payload).digest()


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    resolved = int(seed)
    if not 0 <= resolved <= _MAX_UINT32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    return resolved


def _validated_calibration_statuses(values: object) -> NDArray[np.str_]:
    statuses = np.asarray(values, dtype=object)
    if statuses.ndim != 1:
        raise ValueError("vehicle_status must be one-dimensional")
    normalized: list[str] = []
    for value in statuses:
        if not isinstance(value, str) or value not in RETAIL_VEHICLE_STATUSES:
            raise ValueError(
                "vehicle_status must contain only exact certified, new, or used values"
            )
        normalized.append(value)
    return np.asarray(normalized, dtype=np.str_)


def _absolute_residuals(y_true: object, y_predicted: object) -> NDArray[np.float64]:
    actual = _finite_vector(y_true, label="y_true", allow_empty=False)
    predicted = _finite_vector(y_predicted, label="y_predicted", allow_empty=False)
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_predicted must have equal lengths")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = np.abs(actual - predicted)
    if not np.isfinite(residuals).all():
        raise ValueError("absolute residual calculation produced a non-finite value")
    return residuals


def _finite_sample_quantile(
    residuals: NDArray[np.float64],
    *,
    alpha: float,
) -> float:
    order = _finite_sample_order(len(residuals), alpha)
    if order > len(residuals):
        raise ValueError("calibration sample is too small for the requested alpha")
    value = float(np.partition(residuals.copy(), order - 1)[order - 1])
    return _validated_quantile(value, label="conformal quantile")


def _finite_sample_order(sample_count: int, alpha: float) -> int:
    return math.ceil((sample_count + 1) * (1.0 - alpha))


def _prediction_ranges(
    predictions: NDArray[np.float64],
    radii: NDArray[np.float64],
) -> PredictionRanges:
    with np.errstate(over="ignore", invalid="ignore"):
        lower = np.maximum(0.0, predictions - radii)
        upper = np.maximum(0.0, predictions + radii)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("prediction range calculation produced a non-finite value")
    return PredictionRanges(lower_bounds=lower, upper_bounds=upper)


def _validated_alpha(alpha: object) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a real number")
    resolved = float(alpha)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    return resolved


def _validated_quantile(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a real number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _finite_vector(
    values: object,
    *,
    label: str,
    allow_empty: bool,
) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    inspected: NDArray[Any] = np.asarray(values, dtype=object)
    if np.issubdtype(array.dtype, np.bool_) or any(
        isinstance(value, (bool, np.bool_)) for value in inspected.flat
    ):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        numeric = array.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only numeric values") from error
    if not allow_empty and numeric.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} must contain only finite values")
    return numeric


def _index_vector(values: object, *, label: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    inspected = np.asarray(values, dtype=object)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in inspected.flat
    ):
        raise ValueError(f"{label} must contain only integer positions")
    indices = array.astype(np.int64, copy=True)
    if (indices < 0).any():
        raise ValueError(f"{label} must contain only nonnegative positions")
    if indices.size > 1 and not (indices[1:] > indices[:-1]).all():
        raise ValueError(f"{label} must be strictly increasing and unique")
    indices.setflags(write=False)
    return indices
