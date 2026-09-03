from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.cv import retail_predictor_groups
from autovalue_ml.modeling.screening import (
    RETAIL_SCREENING_STATUSES,
    WHOLESALE_SCREENING_BUCKETS,
    ScreeningSelection,
    _closest_prefix_length,
    _validate_retail_selection,
    retail_screening_sample,
    wholesale_screening_sample,
)

_RETAIL_DOMAIN = b"autovalue-retail-screening-v1\x00"
_WHOLESALE_DOMAIN = b"autovalue-wholesale-screening-v1\x00"


def _retail_features(*, duplicate_groups: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for status_number, status in enumerate(RETAIL_SCREENING_STATUSES):
        for group_number in range(10):
            row = {
                "year": 2014 + group_number,
                "make": f"Make {status_number}",
                "model": f"Model {group_number}",
                "mileage": group_number * 2_000,
                "vehicle_status": status,
            }
            repeats = group_number % 4 + 1 if duplicate_groups else 1
            rows.extend(row.copy() for _ in range(repeats))
    return pd.DataFrame(rows)


def _retail_rank(group_id: str, seed: int) -> bytes:
    return hashlib.sha256(
        _RETAIL_DOMAIN + seed.to_bytes(4, "big") + group_id.encode("ascii")
    ).digest()


def _wholesale_rank(bucket: str, position: int, seed: int) -> bytes:
    return hashlib.sha256(
        _WHOLESALE_DOMAIN
        + seed.to_bytes(4, "big")
        + bucket.encode("utf-8")
        + b"\x00"
        + position.to_bytes(8, "big")
    ).digest()


def _wholesale_inputs(rows_per_bucket: int = 8) -> tuple[list[str], list[int]]:
    pairs: list[tuple[str, int]] = []
    for bucket_number, bucket in enumerate(WHOLESALE_SCREENING_BUCKETS):
        pairs.extend(
            (bucket, 1_000 + bucket_number * 100 + offset) for offset in range(rows_per_bucket)
        )
    permutation = np.random.default_rng(42).permutation(len(pairs))
    shuffled = [pairs[int(position)] for position in permutation]
    return [bucket for bucket, _ in shuffled], [position for _, position in shuffled]


def test_retail_selection_matches_independently_recomputed_hash_ranks() -> None:
    seed = 1_707_037_927
    features = _retail_features()
    groups = retail_predictor_groups(features)
    selection = retail_screening_sample(features, seed=seed)

    expected_groups: set[str] = set()
    for status in RETAIL_SCREENING_STATUSES:
        status_groups = groups[features["vehicle_status"].to_numpy() == status]
        ranked = sorted(set(status_groups), key=lambda group: (_retail_rank(group, seed), group))
        expected_groups.update(ranked[:3])

    assert set(groups[selection.sample_indices]) == expected_groups
    assert selection.sample_count == 9
    assert selection.population_count == 30
    assert selection.sample_indices.tolist() == sorted(selection.sample_indices.tolist())
    assert selection.sample_indices.flags.writeable is False


def test_retail_selection_is_deterministic_and_row_order_invariant_by_group() -> None:
    seed = 91
    features = _retail_features(duplicate_groups=True)
    first = retail_screening_sample(features, seed=seed)
    second = retail_screening_sample(features, seed=seed)
    shuffled = features.sample(frac=1.0, random_state=9).reset_index(drop=True)
    shuffled_selection = retail_screening_sample(shuffled, seed=seed)
    groups = retail_predictor_groups(features)
    shuffled_groups = retail_predictor_groups(shuffled)

    assert np.array_equal(first.sample_indices, second.sample_indices)
    assert set(groups[first.sample_indices]) == set(
        shuffled_groups[shuffled_selection.sample_indices]
    )

    sampled = np.zeros(len(features), dtype=np.bool_)
    sampled[first.sample_indices] = True
    for group in set(groups):
        membership = sampled[groups == group]
        assert membership.all() or not membership.any()
    for status in RETAIL_SCREENING_STATUSES:
        assert sampled[features["vehicle_status"].to_numpy() == status].any()


def test_retail_selection_validation_scales_linearly_with_high_cardinality() -> None:
    group_count = 20_000
    groups = np.asarray([f"{position:064x}" for position in range(group_count)])
    statuses = np.resize(np.asarray(RETAIL_SCREENING_STATUSES), group_count)
    selection = ScreeningSelection(
        sample_indices=np.arange(0, group_count, 10, dtype=np.int64),
        population_count=group_count,
    )

    _validate_retail_selection(groups, statuses, selection)


def test_retail_selection_validation_rejects_conflicting_membership() -> None:
    selection = ScreeningSelection(np.asarray([0, 2, 3]), population_count=4)

    with pytest.raises(RuntimeError, match="split a predictor group"):
        _validate_retail_selection(
            np.asarray(["a", "a", "b", "c"]),
            np.asarray(["certified", "certified", "new", "used"]),
            selection,
        )


def test_closest_prefix_uses_exact_rational_distance_and_smaller_ties() -> None:
    assert _closest_prefix_length([1, 2, 7], total_rows=10, numerator=3, denominator=10) == 2
    assert _closest_prefix_length([6, 4], total_rows=10, numerator=3, denominator=10) == 0
    assert _closest_prefix_length([1, 1], total_rows=2, numerator=1, denominator=4) == 0


def test_wholesale_selection_matches_independently_recomputed_hash_ranks() -> None:
    seed = 759_966_512
    buckets, positions = _wholesale_inputs()
    selection = wholesale_screening_sample(buckets, positions, seed=seed)

    expected_local_positions: set[int] = set()
    for bucket in WHOLESALE_SCREENING_BUCKETS:
        ranked = sorted(
            (
                (_wholesale_rank(bucket, position, seed), position, local_position)
                for local_position, (observed_bucket, position) in enumerate(
                    zip(buckets, positions, strict=True)
                )
                if observed_bucket == bucket
            ),
            key=lambda item: (item[0], item[1]),
        )
        expected_local_positions.update(item[2] for item in ranked[:2])

    assert set(selection.sample_indices) == expected_local_positions
    assert selection.sample_count == 8
    sampled_buckets = np.asarray(buckets)[selection.sample_indices]
    assert {bucket: int((sampled_buckets == bucket).sum()) for bucket in set(buckets)} == {
        bucket: 2 for bucket in WHOLESALE_SCREENING_BUCKETS
    }


def test_wholesale_selection_uses_explicit_positions_not_local_row_order() -> None:
    seed = 55
    buckets, positions = _wholesale_inputs()
    first = wholesale_screening_sample(buckets, positions, seed=seed)
    first_outer_positions = {positions[index] for index in first.sample_indices}

    permutation = np.random.default_rng(8).permutation(len(buckets))
    reordered_buckets = [buckets[int(index)] for index in permutation]
    reordered_positions = [positions[int(index)] for index in permutation]
    reordered = wholesale_screening_sample(reordered_buckets, reordered_positions, seed=seed)
    reordered_outer_positions = {reordered_positions[index] for index in reordered.sample_indices}

    assert reordered_outer_positions == first_outer_positions


def test_screening_functions_have_no_target_parameter() -> None:
    assert "target" not in inspect.signature(retail_screening_sample).parameters
    assert "target" not in inspect.signature(wholesale_screening_sample).parameters
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        retail_screening_sample(_retail_features(), seed=1, target=[1])  # type: ignore[call-arg]


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_screening_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        retail_screening_sample(_retail_features(), seed=seed)  # type: ignore[arg-type]
    buckets, positions = _wholesale_inputs()
    with pytest.raises(ValueError, match="seed"):
        wholesale_screening_sample(buckets, positions, seed=seed)  # type: ignore[arg-type]


def test_retail_screening_fails_closed_on_schema_status_and_tiny_strata() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        retail_screening_sample(
            pd.DataFrame(columns=["year", "make", "model", "mileage", "vehicle_status"]),
            seed=1,
        )
    with pytest.raises(ValueError, match="forbidden feature"):
        retail_screening_sample(_retail_features().assign(price_usd=1), seed=1)

    unknown = _retail_features()
    unknown.loc[0, "vehicle_status"] = "leased"
    with pytest.raises(ValueError, match="certified, new, or used"):
        retail_screening_sample(unknown, seed=1)

    missing = _retail_features().query("vehicle_status != 'new'")
    with pytest.raises(ValueError, match="must not be empty: new"):
        retail_screening_sample(missing, seed=1)

    tiny = _retail_features().groupby("vehicle_status", sort=False).head(1)
    with pytest.raises(ValueError, match="would leave certified empty"):
        retail_screening_sample(tiny, seed=1)


@pytest.mark.parametrize(
    ("buckets", "positions", "message"),
    [
        ([], [], "must not be empty"),
        (
            ["warmup", "2015_01", "2015_02", "2015_03_04"],
            [0, 1, 2],
            "equal lengths",
        ),
        (
            ["warmup", "2015_01", "2015_02", "future"],
            [0, 1, 2, 3],
            "exact approved",
        ),
        (
            ["warmup", "2015_01", "2015_02"],
            [0, 1, 2],
            "must not be empty: 2015_03_04",
        ),
        ([["warmup"]], [0], "one-dimensional"),
        (["warmup"], [[0]], "one-dimensional"),
        (["warmup"], [True], "not booleans"),
        (["warmup"], [1.0], "not booleans"),
        (["warmup"], [-1], "unsigned 64-bit"),
        (["warmup"], [2**64], "unsigned 64-bit"),
        (["warmup", "2015_01"], [1, 1], "unique"),
    ],
)
def test_wholesale_screening_rejects_invalid_inputs(
    buckets: object,
    positions: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        wholesale_screening_sample(buckets, positions, seed=1)


def test_wholesale_screening_rejects_bucket_whose_exact_prefix_is_empty() -> None:
    buckets, positions = _wholesale_inputs(rows_per_bucket=2)
    with pytest.raises(ValueError, match="would leave warmup empty"):
        wholesale_screening_sample(buckets, positions, seed=1)


@pytest.mark.parametrize(
    ("sizes", "total", "numerator", "denominator", "message"),
    [
        ([], 1, 1, 4, "exactly cover"),
        ([1], 0, 1, 4, "positive integer"),
        ([True], 1, 1, 4, "positive integers"),
        ([1], 1, True, 4, "proper positive integers"),
        ([1], 1, 4, 4, "proper positive integers"),
        ([2], 1, 1, 4, "exactly cover"),
    ],
)
def test_closest_prefix_rejects_invalid_contracts(
    sizes: object,
    total: object,
    numerator: object,
    denominator: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _closest_prefix_length(
            sizes,  # type: ignore[arg-type]
            total_rows=total,  # type: ignore[arg-type]
            numerator=numerator,  # type: ignore[arg-type]
            denominator=denominator,  # type: ignore[arg-type]
        )


def test_screening_selection_rejects_invalid_manual_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        ScreeningSelection(np.array([0]), population_count=0)
    with pytest.raises(ValueError, match="must not be empty"):
        ScreeningSelection(np.array([], dtype=np.int64), population_count=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        ScreeningSelection(np.array([1, 0]), population_count=2)
    with pytest.raises(ValueError, match="outside"):
        ScreeningSelection(np.array([2]), population_count=2)
