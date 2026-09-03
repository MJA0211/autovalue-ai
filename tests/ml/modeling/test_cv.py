from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import (
    WHOLESALE_TRACK,
    FeatureContractError,
    retail_group_cv_splits,
    retail_predictor_groups,
    wholesale_forward_cv_splits,
)


def _grouped_retail_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_number in range(5):
        row = {
            "year": 2018 + group_number,
            "make": f"Make {group_number}",
            "model": f"Model {group_number}",
            "mileage": None if group_number == 0 else group_number * 10_000,
            "vehicle_status": "used" if group_number % 2 else "certified",
        }
        rows.extend((row.copy(), row.copy()))
    return pd.DataFrame(rows)


def test_retail_group_folds_keep_predictor_duplicates_disjoint() -> None:
    features = _grouped_retail_frame()
    groups = retail_predictor_groups(features)
    folds = retail_group_cv_splits(features)

    assert len(folds) == 5
    assert len(set(groups)) == 5
    assert groups[0] == groups[1]
    for train_indices, validation_indices in folds:
        assert set(groups[train_indices]).isdisjoint(groups[validation_indices])
        assert set(train_indices).isdisjoint(validation_indices)
    validation_rows = np.concatenate([validation for _, validation in folds])
    assert sorted(validation_rows.tolist()) == list(range(len(features)))


def test_retail_groups_do_not_depend_on_target_or_row_index() -> None:
    features = _grouped_retail_frame()
    reordered_index = features.copy()
    reordered_index.index = np.arange(100, 100 + len(features))
    assert np.array_equal(
        retail_predictor_groups(features),
        retail_predictor_groups(reordered_index),
    )


def test_retail_groups_distinguish_adjacent_future_model_years() -> None:
    features = pd.DataFrame(
        {
            "year": [2023, 2024],
            "make": ["Ford", "Ford"],
            "model": ["F-150", "F-150"],
            "mileage": [0, 0],
            "vehicle_status": ["new", "new"],
        }
    )

    groups = retail_predictor_groups(features)

    assert len(set(groups)) == 2


def test_retail_cv_rejects_wrong_track_and_too_few_groups() -> None:
    with pytest.raises(FeatureContractError, match="retail track"):
        retail_predictor_groups(_grouped_retail_frame(), WHOLESALE_TRACK)
    with pytest.raises(ValueError, match="at least 6 predictor groups"):
        retail_group_cv_splits(_grouped_retail_frame(), n_splits=6)
    with pytest.raises(ValueError, match="at least two"):
        retail_group_cv_splits(_grouped_retail_frame(), n_splits=1)


def test_wholesale_forward_cv_trains_only_on_prior_buckets() -> None:
    order = ("warmup", "month_1", "month_2", "month_3")
    buckets = pd.Series(["month_2", "warmup", "month_1", "month_3", "warmup", "month_2"])
    folds = wholesale_forward_cv_splits(buckets, bucket_order=order)
    bucket_position = {bucket: position for position, bucket in enumerate(order)}

    assert len(folds) == 3
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_positions = [bucket_position[buckets.iloc[index]] for index in train_indices]
        validation_positions = [
            bucket_position[buckets.iloc[index]] for index in validation_indices
        ]
        assert max(train_positions) < fold_number
        assert set(validation_positions) == {fold_number}
        assert set(train_indices).isdisjoint(validation_indices)


@pytest.mark.parametrize(
    ("buckets", "order", "message"),
    [
        (["warmup"], ["warmup"], "at least two"),
        (["warmup", "later"], ["warmup", "warmup"], "unique"),
        (["warmup", None], ["warmup", "later"], "every wholesale"),
        (["warmup", "outside"], ["warmup", "later"], "outside bucket_order"),
        (["warmup", "later"], ["warmup", "middle", "later"], "empty buckets"),
    ],
)
def test_wholesale_forward_cv_fails_closed(buckets: object, order: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wholesale_forward_cv_splits(buckets, bucket_order=order)  # type: ignore[arg-type]
