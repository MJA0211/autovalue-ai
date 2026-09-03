"""Exact target-free screening samples for the frozen Phase 4 protocol."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .contracts import RETAIL_TRACK, validate_feature_frame
from .cv import retail_predictor_groups
from .feature_engineering import VehicleFeatureEngineer

RETAIL_SCREENING_STATUSES: Final[tuple[str, ...]] = ("certified", "new", "used")
WHOLESALE_SCREENING_BUCKETS: Final[tuple[str, ...]] = (
    "warmup",
    "2015_01",
    "2015_02",
    "2015_03_04",
)

_RETAIL_HASH_DOMAIN: Final = b"autovalue-retail-screening-v1\x00"
_WHOLESALE_HASH_DOMAIN: Final = b"autovalue-wholesale-screening-v1\x00"
_RETAIL_FRACTION: Final = (3, 10)
_WHOLESALE_FRACTION: Final = (1, 4)
_MAX_UINT32: Final = (1 << 32) - 1
_MAX_UINT64: Final = (1 << 64) - 1


@dataclass(frozen=True, slots=True)
class ScreeningSelection:
    """Immutable sorted positional indices into one development input."""

    sample_indices: NDArray[np.int64]
    population_count: int

    def __post_init__(self) -> None:
        if isinstance(self.population_count, (bool, np.bool_)) or not isinstance(
            self.population_count, Integral
        ):
            raise ValueError("population_count must be an integer")
        population_count = int(self.population_count)
        if population_count < 1:
            raise ValueError("population_count must be positive")

        indices = _sample_index_vector(self.sample_indices)
        if indices.size == 0:
            raise ValueError("screening sample must not be empty")
        if indices[-1] >= population_count:
            raise ValueError("screening sample index is outside the development population")
        indices.setflags(write=False)
        object.__setattr__(self, "sample_indices", indices)
        object.__setattr__(self, "population_count", population_count)

    @property
    def sample_count(self) -> int:
        """Return the number of selected development rows."""

        return int(self.sample_indices.size)


def retail_screening_sample(
    development_features: object,
    *,
    seed: int,
) -> ScreeningSelection:
    """Select the exact group-safe three-tenths retail screening sample."""

    resolved_seed = _validated_seed(seed)
    frame = validate_feature_frame(development_features, RETAIL_TRACK)
    if len(frame) == 0:
        raise ValueError("retail development features must not be empty")

    engineered = VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(frame)
    statuses = _retail_status_vector(engineered["vehicle_status"])
    observed_statuses = set(statuses.tolist())
    empty_statuses = [
        status for status in RETAIL_SCREENING_STATUSES if status not in observed_statuses
    ]
    if empty_statuses:
        raise ValueError(
            "retail development status strata must not be empty: " + ", ".join(empty_statuses)
        )

    groups = retail_predictor_groups(frame, RETAIL_TRACK)
    selected_groups: set[str] = set()
    for status in RETAIL_SCREENING_STATUSES:
        stratum_groups = groups[statuses == status]
        counts = Counter(str(group) for group in stratum_groups)
        ranked_groups = sorted(
            counts,
            key=lambda group: (_retail_group_rank(group, resolved_seed), group),
        )
        prefix_length = _closest_prefix_length(
            [counts[group] for group in ranked_groups],
            total_rows=len(stratum_groups),
            numerator=_RETAIL_FRACTION[0],
            denominator=_RETAIL_FRACTION[1],
        )
        if prefix_length == 0:
            raise ValueError(f"retail screening sample would leave {status} empty")
        selected_groups.update(ranked_groups[:prefix_length])

    sampled_mask = np.fromiter(
        (str(group) in selected_groups for group in groups),
        count=len(groups),
        dtype=np.bool_,
    )
    selection = ScreeningSelection(
        sample_indices=np.flatnonzero(sampled_mask).astype(np.int64, copy=False),
        population_count=len(frame),
    )
    _validate_retail_selection(groups, statuses, selection)
    return selection


def wholesale_screening_sample(
    development_cv_buckets: Sequence[str] | pd.Series,
    phase3_outer_train_positions: object,
    *,
    seed: int,
) -> ScreeningSelection:
    """Select one quarter of every development bucket using stable outer positions."""

    resolved_seed = _validated_seed(seed)
    buckets = _wholesale_bucket_vector(development_cv_buckets)
    positions = _outer_position_vector(phase3_outer_train_positions)
    if len(buckets) != len(positions):
        raise ValueError("development CV buckets and outer-train positions must have equal lengths")
    if len(buckets) == 0:
        raise ValueError("wholesale development input must not be empty")

    observed_buckets = set(buckets.tolist())
    empty_buckets = [
        bucket for bucket in WHOLESALE_SCREENING_BUCKETS if bucket not in observed_buckets
    ]
    if empty_buckets:
        raise ValueError(
            "wholesale development buckets must not be empty: " + ", ".join(empty_buckets)
        )

    selected_local_positions: list[int] = []
    for bucket in WHOLESALE_SCREENING_BUCKETS:
        local_positions = np.flatnonzero(buckets == bucket).astype(np.int64, copy=False)
        ranked = sorted(
            (
                (
                    _wholesale_row_rank(
                        bucket,
                        int(positions[local_position]),
                        resolved_seed,
                    ),
                    int(positions[local_position]),
                    int(local_position),
                )
                for local_position in local_positions
            ),
            key=lambda item: (item[0], item[1]),
        )
        prefix_length = _closest_prefix_length(
            [1] * len(ranked),
            total_rows=len(ranked),
            numerator=_WHOLESALE_FRACTION[0],
            denominator=_WHOLESALE_FRACTION[1],
        )
        if prefix_length == 0:
            raise ValueError(f"wholesale screening sample would leave {bucket} empty")
        selected_local_positions.extend(item[2] for item in ranked[:prefix_length])

    selection = ScreeningSelection(
        sample_indices=np.asarray(sorted(selected_local_positions), dtype=np.int64),
        population_count=len(buckets),
    )
    _validate_wholesale_selection(buckets, selection)
    return selection


def _closest_prefix_length(
    group_sizes: Sequence[int],
    *,
    total_rows: int,
    numerator: int,
    denominator: int,
) -> int:
    """Choose an exact rational closest prefix while retaining the smaller tie."""

    if (
        isinstance(total_rows, (bool, np.bool_))
        or not isinstance(total_rows, Integral)
        or total_rows < 1
    ):
        raise ValueError("total_rows must be a positive integer")
    if (
        isinstance(numerator, (bool, np.bool_))
        or not isinstance(numerator, Integral)
        or isinstance(denominator, (bool, np.bool_))
        or not isinstance(denominator, Integral)
        or not 0 < numerator < denominator
    ):
        raise ValueError("screening fraction must be proper positive integers")

    inspected_sizes: list[int] = []
    for size in group_sizes:
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
            raise ValueError("screening group sizes must be positive integers")
        inspected_sizes.append(int(size))
    if not inspected_sizes or sum(inspected_sizes) != total_rows:
        raise ValueError("screening group sizes must exactly cover total_rows")

    best_prefix = 0
    best_distance = int(total_rows) * int(numerator)
    cumulative = 0
    for prefix, size in enumerate(inspected_sizes, start=1):
        cumulative += size
        distance = abs(cumulative * int(denominator) - int(total_rows) * int(numerator))
        if distance < best_distance:
            best_prefix = prefix
            best_distance = distance
    return best_prefix


def _retail_group_rank(group_id: str, seed: int) -> bytes:
    payload = _RETAIL_HASH_DOMAIN + seed.to_bytes(4, "big") + group_id.encode("ascii")
    return hashlib.sha256(payload).digest()


def _wholesale_row_rank(bucket: str, outer_train_position: int, seed: int) -> bytes:
    payload = (
        _WHOLESALE_HASH_DOMAIN
        + seed.to_bytes(4, "big")
        + bucket.encode("utf-8")
        + b"\x00"
        + outer_train_position.to_bytes(8, "big")
    )
    return hashlib.sha256(payload).digest()


def _validate_retail_selection(
    groups: NDArray[np.str_],
    statuses: NDArray[np.str_],
    selection: ScreeningSelection,
) -> None:
    sampled = np.zeros(len(groups), dtype=np.bool_)
    sampled[selection.sample_indices] = True
    membership_by_group: dict[str, bool] = {}
    for group_value, is_sampled_value in zip(groups, sampled, strict=True):
        group = str(group_value)
        is_sampled = bool(is_sampled_value)
        previous = membership_by_group.setdefault(group, is_sampled)
        if previous != is_sampled:
            raise RuntimeError("retail screening split a predictor group")
    for status in RETAIL_SCREENING_STATUSES:
        if not sampled[statuses == status].any():
            raise RuntimeError("retail screening sample lost a status stratum")


def _validate_wholesale_selection(
    buckets: NDArray[np.str_],
    selection: ScreeningSelection,
) -> None:
    sampled_buckets = set(buckets[selection.sample_indices].tolist())
    if sampled_buckets != set(WHOLESALE_SCREENING_BUCKETS):
        raise RuntimeError("wholesale screening sample must preserve every development bucket")


def _retail_status_vector(values: object) -> NDArray[np.str_]:
    statuses = np.asarray(values, dtype=object)
    if statuses.ndim != 1:
        raise ValueError("vehicle_status must be one-dimensional")
    normalized: list[str] = []
    for value in statuses:
        if not isinstance(value, str) or value not in RETAIL_SCREENING_STATUSES:
            raise ValueError(
                "vehicle_status must contain only exact certified, new, or used values"
            )
        normalized.append(value)
    return np.asarray(normalized, dtype=np.str_)


def _wholesale_bucket_vector(values: object) -> NDArray[np.str_]:
    buckets = np.asarray(values, dtype=object)
    if buckets.ndim != 1:
        raise ValueError("development_cv_buckets must be one-dimensional")
    normalized: list[str] = []
    for value in buckets:
        if not isinstance(value, str) or value not in WHOLESALE_SCREENING_BUCKETS:
            raise ValueError(
                "development_cv_buckets must contain only exact approved bucket values"
            )
        normalized.append(value)
    return np.asarray(normalized, dtype=np.str_)


def _outer_position_vector(values: object) -> NDArray[np.uint64]:
    positions = np.asarray(values, dtype=object)
    if positions.ndim != 1:
        raise ValueError("phase3_outer_train_positions must be one-dimensional")
    inspected: list[int] = []
    for value in positions:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("outer-train positions must be integers, not booleans")
        resolved = int(value)
        if not 0 <= resolved <= _MAX_UINT64:
            raise ValueError("outer-train positions must fit unsigned 64-bit integers")
        inspected.append(resolved)
    if len(inspected) != len(set(inspected)):
        raise ValueError("outer-train positions must be unique")
    return np.asarray(inspected, dtype=np.uint64)


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer, not boolean")
    resolved = int(seed)
    if not 0 <= resolved <= _MAX_UINT32:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    return resolved


def _sample_index_vector(values: object) -> NDArray[np.int64]:
    indices = np.asarray(values, dtype=object)
    if indices.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional")
    inspected: list[int] = []
    for value in indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError("sample_indices must contain only integer positions")
        resolved = int(value)
        if resolved < 0:
            raise ValueError("sample_indices must contain only nonnegative positions")
        inspected.append(resolved)
    result = np.asarray(inspected, dtype=np.int64)
    if result.size > 1 and not (result[1:] > result[:-1]).all():
        raise ValueError("sample_indices must be strictly increasing and unique")
    return result
