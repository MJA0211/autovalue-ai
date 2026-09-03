from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration import (
    RETAIL_VEHICLE_STATUSES,
    CalibrationPartition,
    PredictionRanges,
    RetailConformalCalibration,
    _closest_tenth_prefix,
    _validate_retail_group_partition,
    conformal_absolute_residual_quantile,
    conformal_prediction_ranges,
    fit_retail_status_conformal,
    retail_calibration_partition,
    retail_conformal_prediction_ranges,
    wholesale_calibration_partition,
)
from autovalue_ml.modeling.cv import retail_predictor_groups


def _retail_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status_number, status in enumerate(RETAIL_VEHICLE_STATUSES):
        for group_number in range(20):
            row = {
                "year": 2000 + group_number,
                "make": f"Make {status_number}",
                "model": f"Model {group_number}",
                "mileage": None if group_number == 0 else group_number * 1_000,
                "vehicle_status": status,
            }
            rows.extend(row.copy() for _ in range(1 + group_number % 3))
    return pd.DataFrame(rows)


def test_retail_partition_is_deterministic_stratified_and_group_indivisible() -> None:
    features = _retail_features()
    first = retail_calibration_partition(features, seed=1_416_582_761)
    second = retail_calibration_partition(features, seed=1_416_582_761)
    groups = retail_predictor_groups(features)

    assert np.array_equal(first.development_indices, second.development_indices)
    assert np.array_equal(first.calibration_indices, second.calibration_indices)
    assert first.sample_count == len(features)
    first.validate_full_coverage(len(features))
    assert set(first.development_indices).isdisjoint(first.calibration_indices)
    assert first.development_indices.flags.writeable is False
    assert first.calibration_indices.flags.writeable is False

    calibration_groups = set(groups[first.calibration_indices])
    assert calibration_groups
    for group in set(groups):
        positions = np.flatnonzero(groups == group)
        assert (group in calibration_groups) == all(
            position in set(first.calibration_indices) for position in positions
        )

    for status in RETAIL_VEHICLE_STATUSES:
        status_mask = features["vehicle_status"].to_numpy() == status
        calibration_count = int(status_mask[first.calibration_indices].sum())
        target_count = int(status_mask.sum()) / 10
        largest_group = max(
            int(((groups == group) & status_mask).sum()) for group in set(groups[status_mask])
        )
        assert calibration_count > 0
        assert abs(calibration_count - target_count) <= largest_group


def test_retail_partition_selects_the_same_groups_after_row_reordering() -> None:
    features = _retail_features()
    shuffled = features.sample(frac=1.0, random_state=42).reset_index(drop=True)
    original_partition = retail_calibration_partition(features, seed=123)
    shuffled_partition = retail_calibration_partition(shuffled, seed=123)

    original_groups = retail_predictor_groups(features)[original_partition.calibration_indices]
    shuffled_groups = retail_predictor_groups(shuffled)[shuffled_partition.calibration_indices]
    assert set(original_groups) == set(shuffled_groups)


def test_retail_group_validation_scales_linearly_with_high_cardinality() -> None:
    group_count = 20_000
    groups = np.asarray([f"{position:064x}" for position in range(group_count)])
    statuses = np.resize(np.asarray(RETAIL_VEHICLE_STATUSES), group_count)
    calibration = np.arange(0, group_count, 10, dtype=np.int64)
    development = np.setdiff1d(np.arange(group_count, dtype=np.int64), calibration)
    partition = CalibrationPartition(development, calibration)

    _validate_retail_group_partition(groups, statuses, partition)


def test_retail_group_validation_rejects_conflicting_membership() -> None:
    groups = np.asarray(["a", "a", "b", "c"])
    statuses = np.asarray(["certified", "certified", "new", "used"])
    partition = CalibrationPartition(np.asarray([1, 2]), np.asarray([0, 3]))

    with pytest.raises(RuntimeError, match="crossed"):
        _validate_retail_group_partition(groups, statuses, partition)


def test_retail_prefix_ties_choose_the_smaller_prefix() -> None:
    assert _closest_tenth_prefix([2, 8], total_rows=10) == 0
    assert _closest_tenth_prefix([1, 9], total_rows=10) == 1


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_retail_partition_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        retail_calibration_partition(_retail_features(), seed=seed)  # type: ignore[arg-type]


def test_retail_partition_fails_closed_on_schema_and_status_drift() -> None:
    contaminated = _retail_features().assign(price_usd=20_000)
    with pytest.raises(ValueError, match="forbidden feature"):
        retail_calibration_partition(contaminated, seed=1)

    unknown = _retail_features()
    unknown.loc[0, "vehicle_status"] = "leased"
    with pytest.raises(ValueError, match="certified, new, or used"):
        retail_calibration_partition(unknown, seed=1)

    missing_stratum = _retail_features().query("vehicle_status != 'certified'")
    with pytest.raises(ValueError, match="empty status strata: certified"):
        retail_calibration_partition(missing_stratum, seed=1)


def test_wholesale_partition_uses_only_may_for_calibration() -> None:
    buckets = pd.Series(["2015_05", "warmup", "2015_02", "2015_01", "2015_03_04", "2015_05"])
    partition = wholesale_calibration_partition(buckets)

    assert partition.development_indices.tolist() == [1, 2, 3, 4]
    assert partition.calibration_indices.tolist() == [0, 5]
    partition.validate_full_coverage(len(buckets))


@pytest.mark.parametrize(
    ("buckets", "message"),
    [
        ([], "must not be empty"),
        (["warmup", None], "every wholesale"),
        (
            ["warmup", "2015_01", "2015_02", "2015_03_04", "future"],
            "unapproved values: future",
        ),
        (["warmup", "2015_01", "2015_02", "2015_03_04"], "must not be empty: 2015_05"),
        (
            ["warmup", "2015_01", "2015_02", "2015_03_04", " 2015_05"],
            "canonical spelling",
        ),
        ([11, 12], "every wholesale"),
        ([["warmup"], ["2015_05"]], "one-dimensional"),
    ],
)
def test_wholesale_partition_fails_closed(buckets: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wholesale_calibration_partition(buckets)


def test_conformal_quantile_uses_exact_finite_sample_higher_order_statistic() -> None:
    predicted = np.zeros(10)
    actual = np.arange(1.0, 11.0)

    quantile = conformal_absolute_residual_quantile(actual, predicted, alpha=0.2)

    assert quantile == 9.0


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, float("nan"), True, "0.1"])
def test_conformal_quantile_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_absolute_residual_quantile([1.0] * 10, [0.0] * 10, alpha=alpha)  # type: ignore[arg-type]


def test_conformal_quantile_rejects_too_small_or_invalid_samples() -> None:
    with pytest.raises(ValueError, match="too small"):
        conformal_absolute_residual_quantile([1.0] * 8, [0.0] * 8, alpha=0.1)
    with pytest.raises(ValueError, match="equal lengths"):
        conformal_absolute_residual_quantile([1.0] * 10, [0.0] * 9)
    with pytest.raises(ValueError, match="not boolean"):
        conformal_absolute_residual_quantile([True] * 10, [0.0] * 10)
    with pytest.raises(ValueError, match="finite"):
        conformal_absolute_residual_quantile([float("inf")] * 10, [0.0] * 10)
    with pytest.raises(ValueError, match="one-dimensional"):
        conformal_absolute_residual_quantile([[1.0] * 10], [[0.0] * 10])


def test_prediction_ranges_are_zero_clipped_and_read_only() -> None:
    ranges = conformal_prediction_ranges([5.0, 100.0, -20.0], 10.0)

    assert ranges.lower_bounds.tolist() == [0.0, 90.0, 0.0]
    assert ranges.upper_bounds.tolist() == [15.0, 110.0, 0.0]
    assert ranges.lower_bounds.flags.writeable is False
    assert ranges.upper_bounds.flags.writeable is False

    with pytest.raises(ValueError, match="nonnegative"):
        conformal_prediction_ranges([10.0], -1.0)
    with pytest.raises(ValueError, match="non-finite"):
        conformal_prediction_ranges([np.finfo(np.float64).max], np.finfo(np.float64).max)


def test_retail_status_conformal_quantiles_and_global_fallback() -> None:
    residuals = np.arange(1.0, 31.0)
    statuses = ["certified"] * 10 + ["new"] * 10 + ["used"] * 10
    calibration = fit_retail_status_conformal(
        residuals,
        np.zeros(30),
        statuses,
        alpha=0.2,
    )

    assert calibration.global_quantile == 25.0
    assert calibration.status_quantiles == (
        ("certified", 9.0),
        ("new", 19.0),
        ("used", 29.0),
    )
    assert calibration.status_sample_counts == (
        ("certified", 10),
        ("new", 10),
        ("used", 10),
    )
    assert set(field.name for field in fields(calibration)) == {
        "alpha",
        "sample_count",
        "global_quantile",
        "status_sample_counts",
        "status_quantiles",
    }

    ranges = retail_conformal_prediction_ranges(
        [30.0] * 5,
        ["certified", "New", "used", "leased", None],
        calibration,
    )
    assert ranges.lower_bounds.tolist() == [21.0, 11.0, 1.0, 5.0, 5.0]
    assert ranges.upper_bounds.tolist() == [39.0, 49.0, 59.0, 55.0, 55.0]


def test_retail_status_conformal_uses_global_for_undersized_strata() -> None:
    residuals = np.arange(1.0, 21.0)
    statuses = ["certified"] * 2 + ["new"] * 2 + ["used"] * 16
    calibration = fit_retail_status_conformal(
        residuals,
        np.zeros(20),
        statuses,
        alpha=0.1,
    )

    assert calibration.global_quantile == 19.0
    assert calibration.status_quantiles == (("used", 20.0),)
    assert calibration.quantile_for("certified") == 19.0
    assert calibration.quantile_for("unknown") == 19.0


def test_retail_status_conformal_rejects_status_and_length_drift() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        fit_retail_status_conformal([1.0] * 10, [0.0] * 10, ["used"] * 9)
    with pytest.raises(ValueError, match="certified, new, or used"):
        fit_retail_status_conformal([1.0] * 10, [0.0] * 10, ["leased"] * 10)
    with pytest.raises(ValueError, match="equal lengths"):
        retail_conformal_prediction_ranges(
            [10.0, 20.0],
            ["used"],
            fit_retail_status_conformal(
                np.arange(1.0, 31.0),
                np.zeros(30),
                ["certified"] * 10 + ["new"] * 10 + ["used"] * 10,
                alpha=0.2,
            ),
        )


def test_public_value_objects_reject_inconsistent_manual_construction() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        CalibrationPartition(np.array([0, 1]), np.array([1, 2]))
    with pytest.raises(ValueError, match="upper bounds"):
        PredictionRanges(np.array([10.0]), np.array([9.0]))
    with pytest.raises(ValueError, match="sum to sample_count"):
        RetailConformalCalibration(
            alpha=0.1,
            sample_count=10,
            global_quantile=1.0,
            status_sample_counts=(("certified", 1), ("new", 1), ("used", 1)),
            status_quantiles=(),
        )
    with pytest.raises(ValueError, match="too small for alpha"):
        RetailConformalCalibration(
            alpha=0.1,
            sample_count=8,
            global_quantile=1.0,
            status_sample_counts=(("certified", 2), ("new", 2), ("used", 4)),
            status_quantiles=(),
        )
    with pytest.raises(ValueError, match="sufficiently large strata"):
        RetailConformalCalibration(
            alpha=0.2,
            sample_count=15,
            global_quantile=1.0,
            status_sample_counts=(("certified", 5), ("new", 5), ("used", 5)),
            status_quantiles=(),
        )
