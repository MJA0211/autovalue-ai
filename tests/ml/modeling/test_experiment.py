from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.baselines import BaselineName
from autovalue_ml.modeling.contracts import FeatureContractError, TrackConfig
from autovalue_ml.modeling.experiment import (
    BaselineExperimentResult,
    ExperimentValidationError,
    FoldAggregate,
    HoldoutResult,
    ModelCrossValidationResult,
    ModelSelection,
    canonical_experiment_json,
    parse_experiment_json,
    run_retail_baseline_experiment,
    run_wholesale_baseline_experiment,
    validate_experiment_result,
)
from autovalue_ml.modeling.metrics import RegressionMetrics, StatusSliceMetrics
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


def _retail_partitions() -> tuple[pd.DataFrame, pd.Series[float], pd.DataFrame, pd.Series[float]]:
    statuses = ("certified", "new", "used")
    train_rows: list[dict[str, object]] = []
    train_prices: list[float] = []
    for row_number in range(12):
        year = 2014 + row_number % 9
        mileage = None if row_number in {2, 8} else float(8_000 + row_number * 3_100)
        train_rows.append(
            {
                "year": year,
                "make": f"Train Make {row_number % 4}",
                "model": f"Train Model {row_number}",
                "mileage": mileage,
                "vehicle_status": statuses[row_number % 3],
            }
        )
        train_prices.append(54_000.0 - (2023 - year) * 2_000.0 - row_number * 250.0)

    test_rows: list[dict[str, object]] = []
    test_prices: list[float] = []
    for row_number in range(6):
        year = 2017 + row_number
        test_rows.append(
            {
                "year": year,
                "make": f"Test Make {row_number % 2}",
                "model": f"Test Model {row_number}",
                "mileage": float(5_000 + row_number * 4_000),
                "vehicle_status": statuses[row_number % 3],
            }
        )
        test_prices.append(55_000.0 - (2023 - year) * 2_100.0)

    train = pd.DataFrame(train_rows, index=np.arange(100, 112))
    test = pd.DataFrame(test_rows, index=np.arange(500, 506))
    return (
        train,
        pd.Series(train_prices, index=train.index, name="price_usd"),
        test,
        pd.Series(test_prices, index=test.index, name="price_usd"),
    )


def _wholesale_partitions() -> tuple[
    pd.DataFrame,
    pd.Series[float],
    pd.DataFrame,
    pd.Series[float],
    pd.Series[str],
    tuple[str, ...],
]:
    order = ("warmup", "january", "february", "march")
    rows: list[dict[str, object]] = []
    prices: list[float] = []
    buckets: list[str] = []
    for bucket_number, bucket in enumerate(order):
        for item in range(3):
            row_number = bucket_number * 3 + item
            year = 2008 + row_number % 7
            rows.append(
                {
                    "year": year,
                    "make": f"Auction Make {row_number % 3}",
                    "model": f"Auction Model {row_number % 5}",
                    "trim": None if item == 0 else f"Trim {item}",
                    "mileage": None if row_number == 4 else float(45_000 + row_number * 5_000),
                    "condition": None if row_number == 7 else float(20 + row_number),
                    "vehicle_type": "sedan" if item % 2 == 0 else "suv",
                }
            )
            prices.append(25_000.0 - (2015 - year) * 1_400.0 + bucket_number * 300.0)
            buckets.append(bucket)

    test = pd.DataFrame(
        {
            "year": [2010, 2011, 2012, 2013],
            "make": ["Holdout A", "Holdout B", "Holdout C", "Holdout D"],
            "model": ["One", "Two", "Three", "Four"],
            "trim": ["Base", None, "Sport", "Base"],
            "mileage": [90_000.0, 75_000.0, None, 42_000.0],
            "condition": [22.0, 28.0, 31.0, None],
            "vehicle_type": ["sedan", "suv", "truck", "sedan"],
        },
        index=np.arange(900, 904),
    )
    train = pd.DataFrame(rows, index=np.arange(200, 212))
    return (
        train,
        pd.Series(prices, index=train.index, name="price_usd"),
        test,
        pd.Series([13_000.0, 15_000.0, 17_000.0, 19_000.0], index=test.index, name="price_usd"),
        pd.Series(buckets, index=train.index, name="cv_bucket"),
        order,
    )


@pytest.fixture(scope="module")
def retail_result() -> BaselineExperimentResult:
    train, y_train, test, y_test = _retail_partitions()
    return run_retail_baseline_experiment(
        outer_train_features=train,
        outer_train_target=y_train,
        outer_test_features=test,
        outer_test_target=y_test,
        n_splits=3,
    )


@pytest.fixture(scope="module")
def wholesale_result() -> BaselineExperimentResult:
    train, y_train, test, y_test, buckets, order = _wholesale_partitions()
    return run_wholesale_baseline_experiment(
        outer_train_features=train,
        outer_train_target=y_train,
        outer_test_features=test,
        outer_test_target=y_test,
        train_cv_buckets=buckets,
        bucket_order=order,
    )


def test_retail_experiment_runs_both_real_fold_local_pipelines(
    retail_result: BaselineExperimentResult,
) -> None:
    assert retail_result.track == "retail"
    assert retail_result.cv_scheme == "retail_predictor_group_kfold_v1"
    assert retail_result.bucket_order == ()
    assert retail_result.outer_train_sample_count == 12
    assert retail_result.outer_test_sample_count == 6
    assert tuple(model.model_name for model in retail_result.models) == (
        "dummy_median",
        "linear_regression",
    )
    assert all(len(model.folds) == 3 for model in retail_result.models)
    assert all(model.overall.sample_count == 12 for model in retail_result.models)
    assert all(
        tuple(item.status for item in model.status_slices) == ("certified", "new", "used")
        for model in retail_result.models
    )
    assert retail_result.selection.selected_cv_mae == min(
        model.overall.mae for model in retail_result.models
    )
    assert retail_result.holdout.model_name == retail_result.selection.selected_model
    assert retail_result.holdout.overall.sample_count == 6
    assert tuple(item.status for item in retail_result.holdout.status_slices) == (
        "certified",
        "new",
        "used",
    )


def test_wholesale_experiment_is_forward_only_and_excludes_warmup_from_oof(
    wholesale_result: BaselineExperimentResult,
) -> None:
    assert wholesale_result.track == "wholesale"
    assert wholesale_result.cv_scheme == "wholesale_forward_chaining_cv_bucket_v1"
    assert wholesale_result.bucket_order == ("warmup", "january", "february", "march")
    assert wholesale_result.outer_train_sample_count == 12
    assert wholesale_result.outer_test_sample_count == 4
    for model in wholesale_result.models:
        assert model.overall.sample_count == 9
        assert model.status_slices == ()
        assert tuple(fold.training_sample_count for fold in model.folds) == (3, 6, 9)
        assert tuple(fold.validation_sample_count for fold in model.folds) == (3, 3, 3)
        assert tuple(fold.validation_bucket for fold in model.folds) == (
            "january",
            "february",
            "march",
        )
    assert wholesale_result.holdout.status_slices == ()
    assert wholesale_result.holdout.overall.sample_count == 4


def test_canonical_report_round_trip_is_deterministic_and_aggregate_only(
    retail_result: BaselineExperimentResult,
) -> None:
    serialized = canonical_experiment_json(retail_result)
    reconstructed = parse_experiment_json(serialized.encode("utf-8"))

    assert reconstructed == retail_result
    assert canonical_experiment_json(reconstructed) == serialized
    assert serialized.endswith("\n")
    assert (
        serialized
        == json.dumps(
            retail_result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    forbidden_keys = {
        "coefficient",
        "feature_vocabulary",
        "path",
        "prediction",
        "raw_row",
        "timestamp",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(reconstructed.to_dict())


class _RecordingRegressor(BaseEstimator):  # type: ignore[misc]
    events: ClassVar[list[tuple[str, BaselineName, int, bool, int]]] = []
    instances: ClassVar[list[_RecordingRegressor]] = []
    next_serial: ClassVar[int] = 0
    predictions: ClassVar[dict[BaselineName, float]] = {
        "dummy_median": 0.0,
        "linear_regression": 10.0,
    }

    def __init__(self, model_name: BaselineName) -> None:
        self.model_name = model_name
        self.serial = type(self).next_serial
        type(self).next_serial += 1

    @classmethod
    def reset(cls, *, dummy: float, linear: float) -> None:
        cls.events = []
        cls.instances = []
        cls.next_serial = 0
        cls.predictions = {"dummy_median": dummy, "linear_regression": linear}

    def fit(self, X: pd.DataFrame, y: object) -> Self:
        del y
        is_holdout = bool(X["make"].str.startswith("Test ").any())
        type(self).events.append(("fit", self.model_name, self.serial, is_holdout, len(X)))
        type(self).instances.append(self)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        is_holdout = bool(X["make"].str.startswith("Test ").any())
        type(self).events.append(("predict", self.model_name, self.serial, is_holdout, len(X)))
        return np.full(len(X), type(self).predictions[self.model_name], dtype=np.float64)


def _recording_factory(name: BaselineName, config: TrackConfig) -> Pipeline:
    del config
    return Pipeline((("regressor", _RecordingRegressor(name)),))


def test_holdout_is_predicted_only_by_selected_model_after_cv_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autovalue_ml.modeling import experiment

    train, _, test, _ = _retail_partitions()
    y_train = pd.Series(np.full(len(train), 10.0), index=train.index, name="price_usd")
    y_test = pd.Series(np.full(len(test), 100.0), index=test.index, name="price_usd")
    _RecordingRegressor.reset(dummy=0.0, linear=10.0)
    monkeypatch.setattr(experiment, "make_baseline_pipeline", _recording_factory)

    result = run_retail_baseline_experiment(
        outer_train_features=train,
        outer_train_target=y_train,
        outer_test_features=test,
        outer_test_target=y_test,
        n_splits=3,
    )

    assert result.selection.selected_model == "linear_regression"
    assert result.selection.selected_cv_mae == 0.0
    holdout_events = [event for event in _RecordingRegressor.events if event[3]]
    assert holdout_events == [("predict", "linear_regression", holdout_events[0][2], True, 6)]
    first_holdout_position = _RecordingRegressor.events.index(holdout_events[0])
    assert all(not event[3] for event in _RecordingRegressor.events[:first_holdout_position])

    fit_events = [event for event in _RecordingRegressor.events if event[0] == "fit"]
    assert len(fit_events) == 7
    assert len({event[2] for event in fit_events}) == 7
    assert sum(event[1] == "dummy_median" for event in fit_events) == 3
    assert sum(event[1] == "linear_regression" for event in fit_events) == 4
    assert result.holdout.overall.mae == 90.0


def test_model_selection_uses_stable_name_tie_break(monkeypatch: pytest.MonkeyPatch) -> None:
    from autovalue_ml.modeling import experiment

    train, _, test, _ = _retail_partitions()
    y_train = pd.Series(np.full(len(train), 10.0), index=train.index, name="price_usd")
    y_test = pd.Series(np.full(len(test), 10.0), index=test.index, name="price_usd")
    _RecordingRegressor.reset(dummy=10.0, linear=10.0)
    monkeypatch.setattr(experiment, "make_baseline_pipeline", _recording_factory)

    result = run_retail_baseline_experiment(
        outer_train_features=train,
        outer_train_target=y_train,
        outer_test_features=test,
        outer_test_target=y_test,
        n_splits=3,
    )

    assert result.models[0].overall.mae == result.models[1].overall.mae == 0.0
    assert result.selection.selected_model == "dummy_median"
    assert all(
        model_name != "linear_regression"
        for action, model_name, _, is_holdout, _ in _RecordingRegressor.events
        if action == "predict" and is_holdout
    )


def test_retail_rejects_outer_predictor_group_overlap() -> None:
    train, y_train, test, y_test = _retail_partitions()
    test.iloc[0] = train.iloc[0]
    with pytest.raises(ExperimentValidationError, match="isolated across outer partitions"):
        run_retail_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            n_splits=3,
        )


@pytest.mark.parametrize("bad_status", ["Used", " used", "lease"])
def test_retail_rejects_noncanonical_or_unsupported_statuses(bad_status: str) -> None:
    train, y_train, test, y_test = _retail_partitions()
    train.loc[train.index[0], "vehicle_status"] = bad_status
    message = "canonical lowercase" if bad_status != "lease" else "unsupported"
    with pytest.raises(ExperimentValidationError, match=message):
        run_retail_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            n_splits=3,
        )


def test_retail_requires_all_status_slices_in_each_outer_partition() -> None:
    train, y_train, test, y_test = _retail_partitions()
    test.loc[test["vehicle_status"] == "certified", "vehicle_status"] = "used"
    with pytest.raises(ExperimentValidationError, match="every retail status slice: certified"):
        run_retail_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            n_splits=3,
        )


def test_outer_feature_target_indexes_and_columns_must_align() -> None:
    train, y_train, test, y_test = _retail_partitions()
    misindexed = y_train.copy()
    misindexed.index = np.arange(len(misindexed))
    with pytest.raises(FeatureContractError, match="indexes must align"):
        run_retail_baseline_experiment(
            outer_train_features=train,
            outer_train_target=misindexed,
            outer_test_features=test,
            outer_test_target=y_test,
            n_splits=3,
        )

    with pytest.raises(FeatureContractError, match="same feature columns"):
        run_retail_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test.drop(columns="mileage"),
            outer_test_target=y_test,
            n_splits=3,
        )


def test_wholesale_bucket_contract_fails_closed() -> None:
    train, y_train, test, y_test, buckets, order = _wholesale_partitions()
    misindexed = buckets.copy()
    misindexed.index = np.arange(len(misindexed))
    with pytest.raises(ExperimentValidationError, match="index must align"):
        run_wholesale_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            train_cv_buckets=misindexed,
            bucket_order=order,
        )

    with pytest.raises(ExperimentValidationError, match="same number of rows"):
        run_wholesale_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            train_cv_buckets=[*buckets.tolist(), "march"],
            bucket_order=order,
        )

    with pytest.raises(ValueError, match="empty buckets"):
        run_wholesale_baseline_experiment(
            outer_train_features=train,
            outer_train_target=y_train,
            outer_test_features=test,
            outer_test_target=y_test,
            train_cv_buckets=buckets,
            bucket_order=(*order, "april"),
        )


def _mutated_payload(
    result: BaselineExperimentResult,
    mutation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    payload = copy.deepcopy(result.to_dict())
    mutation(payload)
    return payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("raw_rows", []), "unexpected raw_rows"),
        (lambda value: value.__setitem__("schema_version", 2), "schema_version"),
        (lambda value: value.__setitem__("report_type", "other"), "report_type"),
        (lambda value: value.__setitem__("track", "combined"), "track"),
        (
            lambda value: value.__setitem__("feature_contract_version", "changed"),
            "feature_contract_version",
        ),
        (
            lambda value: value.__setitem__("target_semantics", "combined_prices"),
            "target_semantics",
        ),
        (
            lambda value: value["outer_partition"].__setitem__("train_sample_count", True),
            "train_sample_count must be an integer",
        ),
        (
            lambda value: value["cross_validation"].__setitem__("scheme", "random_kfold"),
            "CV scheme",
        ),
        (
            lambda value: value["cross_validation"].__setitem__("bucket_order", "none"),
            "bucket_order must be an array",
        ),
        (
            lambda value: value["cross_validation"].__setitem__("models", {}),
            "models must be an array",
        ),
        (
            lambda value: value["selection"].__setitem__("selected_model", "forest"),
            "selected_model is unsupported",
        ),
        (
            lambda value: value["selection"].__setitem__("selected_cv_mae", float("nan")),
            "selected_cv_mae must be finite",
        ),
        (
            lambda value: value["holdout"].__setitem__("model_name", "forest"),
            "holdout model is unsupported",
        ),
        (
            lambda value: value["holdout"]["overall"].__setitem__("prediction", []),
            "unexpected prediction",
        ),
        (
            lambda value: value["holdout"]["overall"].__setitem__("mae", -1),
            "MAE must be finite and nonnegative",
        ),
    ],
)
def test_report_parser_rejects_malformed_or_expansive_payloads(
    retail_result: BaselineExperimentResult,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    with pytest.raises(ExperimentValidationError, match=message):
        validate_experiment_result(_mutated_payload(retail_result, mutation))


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ('{"schema_version":1,"schema_version":1}', "duplicate JSON key"),
        ("not-json", "not valid JSON"),
        (b"\xff", "must be UTF-8"),
        (123, "must be text or bytes"),
    ],
)
def test_json_parser_rejects_ambiguous_invalid_or_oversized_input(
    serialized: object, message: str
) -> None:
    with pytest.raises(ExperimentValidationError, match=message):
        parse_experiment_json(serialized)  # type: ignore[arg-type]


def test_json_parser_rejects_oversized_text_and_bytes() -> None:
    for serialized in (" " * 250_001, b" " * 250_001):
        with pytest.raises(ExperimentValidationError, match="exceeds the maximum size"):
            parse_experiment_json(serialized)


def test_strict_value_objects_reject_invalid_counts_metrics_and_labels() -> None:
    valid_metrics = RegressionMetrics(sample_count=1, mae=0.0, rmse=0.0, r2=None)
    with pytest.raises(ExperimentValidationError, match="fold_number"):
        FoldAggregate(1.5, 1, 1, None, valid_metrics)  # type: ignore[arg-type]
    with pytest.raises(ExperimentValidationError, match="training count"):
        FoldAggregate(1, True, 1, None, valid_metrics)
    with pytest.raises(ExperimentValidationError, match="validation count"):
        FoldAggregate(1, 1, 0, None, valid_metrics)
    with pytest.raises(ExperimentValidationError, match="metric count"):
        FoldAggregate(1, 1, 2, None, valid_metrics)
    with pytest.raises(ExperimentValidationError, match="canonical non-empty"):
        FoldAggregate(1, 1, 1, " january ", valid_metrics)
    with pytest.raises(ExperimentValidationError, match="selected_cv_mae"):
        ModelSelection("dummy_median", True)
    with pytest.raises(ExperimentValidationError, match="metric sample_count"):
        HoldoutResult(
            "dummy_median",
            RegressionMetrics(sample_count=1.5, mae=0.0, rmse=0.0, r2=None),  # type: ignore[arg-type]
            (),
        )
    with pytest.raises(ExperimentValidationError, match="metric MAE"):
        HoldoutResult(
            "dummy_median",
            RegressionMetrics(sample_count=1, mae=True, rmse=0.0, r2=None),
            (),
        )
    with pytest.raises(ExperimentValidationError, match="at most one"):
        HoldoutResult(
            "dummy_median",
            RegressionMetrics(sample_count=1, mae=0.0, rmse=0.0, r2=1.01),
            (),
        )


def test_result_validation_rejects_inconsistent_selection_and_track_shapes(
    retail_result: BaselineExperimentResult,
    wholesale_result: BaselineExperimentResult,
) -> None:
    with pytest.raises(ExperimentValidationError, match="does not minimize"):
        BaselineExperimentResult(
            track=retail_result.track,
            outer_train_sample_count=retail_result.outer_train_sample_count,
            outer_test_sample_count=retail_result.outer_test_sample_count,
            cv_scheme=retail_result.cv_scheme,
            bucket_order=retail_result.bucket_order,
            models=retail_result.models,
            selection=ModelSelection(
                selected_model=(
                    "linear_regression"
                    if retail_result.selection.selected_model == "dummy_median"
                    else "dummy_median"
                ),
                selected_cv_mae=retail_result.selection.selected_cv_mae,
            ),
            holdout=retail_result.holdout,
        )

    with pytest.raises(ExperimentValidationError, match="canonical non-empty"):
        BaselineExperimentResult(
            track=wholesale_result.track,
            outer_train_sample_count=wholesale_result.outer_train_sample_count,
            outer_test_sample_count=wholesale_result.outer_test_sample_count,
            cv_scheme=wholesale_result.cv_scheme,
            bucket_order=("warmup", " march "),
            models=wholesale_result.models,
            selection=wholesale_result.selection,
            holdout=wholesale_result.holdout,
        )

    with pytest.raises(ExperimentValidationError, match="must be a BaselineExperimentResult"):
        canonical_experiment_json({})  # type: ignore[arg-type]


def test_model_and_slice_value_objects_reject_unsupported_or_unsorted_values() -> None:
    metrics = RegressionMetrics(sample_count=1, mae=0.0, rmse=0.0, r2=None)
    fold = FoldAggregate(1, 1, 1, None, metrics)
    with pytest.raises(ExperimentValidationError, match="model is unsupported"):
        ModelCrossValidationResult("forest", metrics, (), (fold,))  # type: ignore[arg-type]
    with pytest.raises(ExperimentValidationError, match="at least one fold"):
        ModelCrossValidationResult("dummy_median", metrics, (), ())
    with pytest.raises(ExperimentValidationError, match="unique and sorted"):
        HoldoutResult(
            "dummy_median",
            RegressionMetrics(sample_count=2, mae=0.0, rmse=0.0, r2=None),
            (
                StatusSliceMetrics("used", metrics),
                StatusSliceMetrics("new", metrics),
            ),
        )
