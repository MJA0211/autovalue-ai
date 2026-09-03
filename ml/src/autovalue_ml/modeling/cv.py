"""Cross-validation splitters that preserve each track's leakage boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from .contracts import RETAIL_TRACK, FeatureContractError, TrackConfig, get_track_config
from .feature_engineering import VehicleFeatureEngineer

CVSplit = tuple[NDArray[np.int64], NDArray[np.int64]]


def retail_predictor_groups(
    features: object,
    config: TrackConfig = RETAIL_TRACK,
) -> NDArray[np.str_]:
    """Recompute privacy-safe groups from the retail predictor tuple only.

    The target never enters this function. Equal
    ``(year, make, model, mileage-or-null, vehicle_status)`` tuples receive the
    same stable digest, including equal missing-mileage tuples.
    """

    resolved = get_track_config(config)
    if resolved.name != "retail":
        raise FeatureContractError("retail predictor grouping requires the retail track")

    engineer = VehicleFeatureEngineer(resolved)
    engineered = engineer.fit_transform(features)

    digests: list[str] = []
    for position in range(len(engineered)):
        mileage_value = engineered["mileage"].iloc[position]
        payload = (
            _canonical_number(engineered["model_year"].iloc[position]),
            _canonical_category(engineered["make"].iloc[position]),
            _canonical_category(engineered["model"].iloc[position]),
            _canonical_number(mileage_value),
            _canonical_category(engineered["vehicle_status"].iloc[position]),
        )
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        digests.append(hashlib.sha256(encoded).hexdigest())
    return np.asarray(digests, dtype=np.str_)


def retail_group_cv_splits(
    features: object,
    *,
    n_splits: int = 5,
    config: TrackConfig = RETAIL_TRACK,
) -> tuple[CVSplit, ...]:
    """Return deterministic ``GroupKFold`` indices for the retail train partition."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    groups = retail_predictor_groups(features, config)
    unique_group_count = len(np.unique(groups))
    if unique_group_count < n_splits:
        raise ValueError(
            f"retail CV needs at least {n_splits} predictor groups; got {unique_group_count}"
        )
    row_indices = np.arange(len(groups), dtype=np.int64)
    splitter = GroupKFold(n_splits=n_splits, shuffle=False)
    return tuple(
        (
            train_indices.astype(np.int64, copy=False),
            validation_indices.astype(np.int64, copy=False),
        )
        for train_indices, validation_indices in splitter.split(row_indices, groups=groups)
    )


def wholesale_forward_cv_splits(
    cv_buckets: Sequence[str] | pd.Series,
    *,
    bucket_order: Sequence[str],
) -> tuple[CVSplit, ...]:
    """Train on prior buckets and validate on exactly the next ordered bucket.

    Callers must pass only the already-established outer training partition. The
    outer test partition has no CV bucket and therefore fails this contract.
    """

    order = tuple(bucket_order)
    if len(order) < 2:
        raise ValueError("bucket_order must contain at least two buckets")
    if any(not isinstance(bucket, str) or not bucket.strip() for bucket in order):
        raise ValueError("bucket_order values must be non-empty strings")
    if len(order) != len(set(order)):
        raise ValueError("bucket_order values must be unique")

    values = np.asarray(cv_buckets, dtype=object)
    if values.ndim != 1:
        raise ValueError("cv_buckets must be one-dimensional")
    if len(values) == 0:
        raise ValueError("cv_buckets must not be empty")

    observed: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("every wholesale training row must have a CV bucket")
        observed.append(value)
    unknown = sorted(set(observed) - set(order))
    if unknown:
        raise ValueError(f"cv_buckets contain values outside bucket_order: {', '.join(unknown)}")
    missing = [bucket for bucket in order if bucket not in observed]
    if missing:
        raise ValueError(f"bucket_order contains empty buckets: {', '.join(missing)}")

    bucket_positions = {bucket: position for position, bucket in enumerate(order)}
    positions = np.asarray([bucket_positions[value] for value in observed], dtype=np.int64)
    folds: list[CVSplit] = []
    for validation_position in range(1, len(order)):
        train_indices = np.flatnonzero(positions < validation_position).astype(np.int64, copy=False)
        validation_indices = np.flatnonzero(positions == validation_position).astype(
            np.int64, copy=False
        )
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError("each forward CV fold must have non-empty train and validation sets")
        folds.append((train_indices, validation_indices))
    return tuple(folds)


def _canonical_number(value: object) -> int | float | None:
    if pd.isna(value):
        return None
    number = float(str(value))
    return int(number) if number.is_integer() else number


def _canonical_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)
