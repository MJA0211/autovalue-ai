"""Reproducible, leakage-safe orchestration for AutoValue baseline experiments.

This module consumes outer train/test partitions that have already passed the
source-specific split gate.  It compares the two transparent Phase 3 baselines
inside the outer training partition, selects by true out-of-fold MAE, and only
then evaluates the selected model on the untouched outer holdout.  It never
persists a fitted estimator or emits row-level data.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .baselines import BaselineName, make_baseline_pipeline
from .contracts import (
    RETAIL_TRACK,
    TRACKS,
    WHOLESALE_TRACK,
    FeatureContractError,
    TrackConfig,
    TrackName,
    validate_feature_frame,
    validate_target,
)
from .cv import (
    CVSplit,
    retail_group_cv_splits,
    retail_predictor_groups,
    wholesale_forward_cv_splits,
)
from .metrics import (
    RegressionMetrics,
    StatusSliceMetrics,
    regression_metrics,
    retail_status_metrics,
)

CVScheme: TypeAlias = Literal[
    "retail_predictor_group_kfold_v1",
    "wholesale_forward_chaining_cv_bucket_v1",
]

_SCHEMA_VERSION: Final = 1
_REPORT_TYPE: Final = "baseline_experiment"
_MAX_REPORT_BYTES: Final = 250_000
_BASELINES: Final[tuple[BaselineName, ...]] = (
    "dummy_median",
    "linear_regression",
)
_RETAIL_STATUSES: Final = ("certified", "new", "used")
_SELECTION_CRITERION: Final = "lowest_cv_mae_then_model_name"
_RETAIL_CV_SCHEME: Final[CVScheme] = "retail_predictor_group_kfold_v1"
_WHOLESALE_CV_SCHEME: Final[CVScheme] = "wholesale_forward_chaining_cv_bucket_v1"

_ROOT_KEYS: Final = {
    "schema_version",
    "report_type",
    "track",
    "feature_contract_version",
    "target_semantics",
    "outer_partition",
    "cross_validation",
    "selection",
    "holdout",
}
_PARTITION_KEYS: Final = {"train_sample_count", "test_sample_count"}
_CV_KEYS: Final = {"scheme", "bucket_order", "models"}
_MODEL_KEYS: Final = {"model_name", "overall", "status_slices", "folds"}
_FOLD_KEYS: Final = {
    "fold_number",
    "training_sample_count",
    "validation_sample_count",
    "validation_bucket",
    "metrics",
}
_SELECTION_KEYS: Final = {"criterion", "selected_model", "selected_cv_mae"}
_HOLDOUT_KEYS: Final = {"model_name", "overall", "status_slices"}
_METRIC_KEYS: Final = {"sample_count", "mae", "rmse", "r2"}
_SLICE_KEYS: Final = {"status", "metrics"}


class ExperimentValidationError(ValueError):
    """Raised when experiment inputs or an aggregate report violate the contract."""


@dataclass(frozen=True, slots=True)
class FoldAggregate:
    """Aggregate validation result for one fold-local fitted estimator."""

    fold_number: int
    training_sample_count: int
    validation_sample_count: int
    validation_bucket: str | None
    metrics: RegressionMetrics

    def __post_init__(self) -> None:
        if (
            isinstance(self.fold_number, bool)
            or not isinstance(self.fold_number, int)
            or self.fold_number < 1
        ):
            raise ExperimentValidationError("fold_number must be a positive integer")
        if (
            isinstance(self.training_sample_count, bool)
            or not isinstance(self.training_sample_count, int)
            or self.training_sample_count < 1
        ):
            raise ExperimentValidationError("fold training count must be positive")
        if (
            isinstance(self.validation_sample_count, bool)
            or not isinstance(self.validation_sample_count, int)
            or self.validation_sample_count < 1
        ):
            raise ExperimentValidationError("fold validation count must be positive")
        if self.metrics.sample_count != self.validation_sample_count:
            raise ExperimentValidationError("fold metric count must equal validation count")
        if self.validation_bucket is not None and (
            not isinstance(self.validation_bucket, str)
            or not self.validation_bucket.strip()
            or self.validation_bucket != self.validation_bucket.strip()
        ):
            raise ExperimentValidationError(
                "validation_bucket must be null or canonical non-empty text"
            )
        _validate_metrics(self.metrics)

    def to_dict(self) -> dict[str, object]:
        return {
            "fold_number": self.fold_number,
            "training_sample_count": self.training_sample_count,
            "validation_sample_count": self.validation_sample_count,
            "validation_bucket": self.validation_bucket,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ModelCrossValidationResult:
    """True out-of-fold aggregates for one candidate baseline."""

    model_name: BaselineName
    overall: RegressionMetrics
    status_slices: tuple[StatusSliceMetrics, ...]
    folds: tuple[FoldAggregate, ...]

    def __post_init__(self) -> None:
        if self.model_name not in _BASELINES:
            raise ExperimentValidationError("cross-validation model is unsupported")
        if not self.folds:
            raise ExperimentValidationError("a model result must contain at least one fold")
        _validate_metrics(self.overall)
        _validate_status_slices(self.status_slices, self.overall)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "overall": self.overall.to_dict(),
            "status_slices": [item.to_dict() for item in self.status_slices],
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Deterministic baseline selection decision made from CV only."""

    selected_model: BaselineName
    selected_cv_mae: float
    criterion: str = _SELECTION_CRITERION

    def __post_init__(self) -> None:
        if self.selected_model not in _BASELINES:
            raise ExperimentValidationError("selected_model is unsupported")
        if self.criterion != _SELECTION_CRITERION:
            raise ExperimentValidationError("selection criterion is invalid")
        if (
            isinstance(self.selected_cv_mae, bool)
            or not isinstance(self.selected_cv_mae, (int, float))
            or not math.isfinite(self.selected_cv_mae)
            or self.selected_cv_mae < 0.0
        ):
            raise ExperimentValidationError("selected_cv_mae must be finite and nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "criterion": self.criterion,
            "selected_model": self.selected_model,
            "selected_cv_mae": self.selected_cv_mae,
        }


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Aggregate result from the single selected-model outer-holdout evaluation."""

    model_name: BaselineName
    overall: RegressionMetrics
    status_slices: tuple[StatusSliceMetrics, ...]

    def __post_init__(self) -> None:
        if self.model_name not in _BASELINES:
            raise ExperimentValidationError("holdout model is unsupported")
        _validate_metrics(self.overall)
        _validate_status_slices(self.status_slices, self.overall)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "overall": self.overall.to_dict(),
            "status_slices": [item.to_dict() for item in self.status_slices],
        }


@dataclass(frozen=True, slots=True)
class BaselineExperimentResult:
    """Strict, deterministic, aggregate-only Phase 3 experiment result."""

    track: TrackName
    outer_train_sample_count: int
    outer_test_sample_count: int
    cv_scheme: CVScheme
    bucket_order: tuple[str, ...]
    models: tuple[ModelCrossValidationResult, ...]
    selection: ModelSelection
    holdout: HoldoutResult

    def __post_init__(self) -> None:
        _validate_experiment_result(self)

    @property
    def feature_contract_version(self) -> str:
        return TRACKS[self.track].contract_version

    @property
    def target_semantics(self) -> str:
        return TRACKS[self.track].target_semantics

    def to_dict(self) -> dict[str, object]:
        """Return the exact row-free public schema in stable field order."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "report_type": _REPORT_TYPE,
            "track": self.track,
            "feature_contract_version": self.feature_contract_version,
            "target_semantics": self.target_semantics,
            "outer_partition": {
                "train_sample_count": self.outer_train_sample_count,
                "test_sample_count": self.outer_test_sample_count,
            },
            "cross_validation": {
                "scheme": self.cv_scheme,
                "bucket_order": list(self.bucket_order),
                "models": [model.to_dict() for model in self.models],
            },
            "selection": self.selection.to_dict(),
            "holdout": self.holdout.to_dict(),
        }


def run_retail_baseline_experiment(
    *,
    outer_train_features: object,
    outer_train_target: object,
    outer_test_features: object,
    outer_test_target: object,
    n_splits: int = 5,
) -> BaselineExperimentResult:
    """Compare retail baselines with predictor-group CV and one outer holdout.

    Predictor groups are recomputed solely from the approved feature tuple.  A
    group appearing in both outer partitions is rejected before any fitting.
    Both outer partitions must contain the three approved status slices so the
    aggregate report remains comparable across runs.
    """

    train, y_train, test, y_test = _validated_outer_partitions(
        outer_train_features,
        outer_train_target,
        outer_test_features,
        outer_test_target,
        config=RETAIL_TRACK,
    )
    _validate_retail_status_contract(train, label="outer train")
    _validate_retail_status_contract(test, label="outer test")

    train_groups = retail_predictor_groups(train, RETAIL_TRACK)
    test_groups = retail_predictor_groups(test, RETAIL_TRACK)
    if set(train_groups).intersection(test_groups):
        raise ExperimentValidationError(
            "retail predictor groups must be isolated across outer partitions"
        )

    splits = retail_group_cv_splits(train, n_splits=n_splits, config=RETAIL_TRACK)
    return _run_experiment(
        config=RETAIL_TRACK,
        train=train,
        y_train=y_train,
        test=test,
        y_test=y_test,
        splits=splits,
        cv_scheme=_RETAIL_CV_SCHEME,
        bucket_order=(),
        validation_buckets=(None,) * len(splits),
        expected_oof_mask=np.ones(len(train), dtype=np.bool_),
    )


def run_wholesale_baseline_experiment(
    *,
    outer_train_features: object,
    outer_train_target: object,
    outer_test_features: object,
    outer_test_target: object,
    train_cv_buckets: Sequence[str] | pd.Series,
    bucket_order: Sequence[str],
) -> BaselineExperimentResult:
    """Compare wholesale baselines with forward-only CV and one outer holdout.

    Only the already-separated outer-training partition accepts CV bucket
    labels.  The first ordered bucket is warm-up training data; each later
    bucket is predicted exactly once from strictly earlier buckets.
    """

    train, y_train, test, y_test = _validated_outer_partitions(
        outer_train_features,
        outer_train_target,
        outer_test_features,
        outer_test_target,
        config=WHOLESALE_TRACK,
    )
    if isinstance(train_cv_buckets, pd.Series) and not train_cv_buckets.index.equals(train.index):
        raise ExperimentValidationError("train_cv_buckets index must align with outer train")

    order = tuple(bucket_order)
    splits = wholesale_forward_cv_splits(train_cv_buckets, bucket_order=order)
    bucket_values = np.asarray(train_cv_buckets, dtype=object)
    if len(bucket_values) != len(train):
        raise ExperimentValidationError(
            "train_cv_buckets and outer train must have the same number of rows"
        )
    expected_oof_mask = np.asarray(bucket_values != order[0], dtype=np.bool_)
    return _run_experiment(
        config=WHOLESALE_TRACK,
        train=train,
        y_train=y_train,
        test=test,
        y_test=y_test,
        splits=splits,
        cv_scheme=_WHOLESALE_CV_SCHEME,
        bucket_order=order,
        validation_buckets=tuple(order[1:]),
        expected_oof_mask=expected_oof_mask,
    )


def canonical_experiment_json(result: BaselineExperimentResult) -> str:
    """Serialize a validated experiment result as deterministic canonical JSON."""

    if not isinstance(result, BaselineExperimentResult):
        raise ExperimentValidationError("result must be a BaselineExperimentResult")
    _validate_experiment_result(result)
    return (
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def validate_experiment_result(payload: object) -> BaselineExperimentResult:
    """Validate the exact aggregate-only mapping schema and reconstruct the result."""

    root = _object(payload, label="experiment")
    _exact_keys(root, _ROOT_KEYS, label="experiment")
    if _integer(root["schema_version"], label="schema_version") != _SCHEMA_VERSION:
        raise ExperimentValidationError("unsupported experiment schema_version")
    if root["report_type"] != _REPORT_TYPE:
        raise ExperimentValidationError("experiment report_type is invalid")

    track_text = _text(root["track"], label="track")
    if track_text not in TRACKS:
        raise ExperimentValidationError("experiment track is invalid")
    track: TrackName = track_text
    config = TRACKS[track]
    if root["feature_contract_version"] != config.contract_version:
        raise ExperimentValidationError("feature_contract_version does not match track")
    if root["target_semantics"] != config.target_semantics:
        raise ExperimentValidationError("target_semantics does not match track")

    partition = _object(root["outer_partition"], label="outer_partition")
    _exact_keys(partition, _PARTITION_KEYS, label="outer_partition")
    train_count = _integer(partition["train_sample_count"], label="train_sample_count")
    test_count = _integer(partition["test_sample_count"], label="test_sample_count")

    cv = _object(root["cross_validation"], label="cross_validation")
    _exact_keys(cv, _CV_KEYS, label="cross_validation")
    scheme_text = _text(cv["scheme"], label="CV scheme")
    if scheme_text not in {_RETAIL_CV_SCHEME, _WHOLESALE_CV_SCHEME}:
        raise ExperimentValidationError("CV scheme is invalid")
    scheme: CVScheme = scheme_text
    raw_bucket_order = cv["bucket_order"]
    if not isinstance(raw_bucket_order, list):
        raise ExperimentValidationError("bucket_order must be an array")
    parsed_bucket_order = tuple(_text(item, label="CV bucket") for item in raw_bucket_order)

    raw_models = cv["models"]
    if not isinstance(raw_models, list):
        raise ExperimentValidationError("cross_validation.models must be an array")
    models = tuple(_parse_model(item, index=index) for index, item in enumerate(raw_models))

    selection_object = _object(root["selection"], label="selection")
    _exact_keys(selection_object, _SELECTION_KEYS, label="selection")
    selected_name = _baseline_name(selection_object["selected_model"], label="selected_model")
    selection = ModelSelection(
        criterion=_text(selection_object["criterion"], label="selection criterion"),
        selected_model=selected_name,
        selected_cv_mae=_number(selection_object["selected_cv_mae"], label="selected_cv_mae"),
    )

    holdout_object = _object(root["holdout"], label="holdout")
    _exact_keys(holdout_object, _HOLDOUT_KEYS, label="holdout")
    holdout = HoldoutResult(
        model_name=_baseline_name(holdout_object["model_name"], label="holdout model"),
        overall=_parse_metrics(holdout_object["overall"], label="holdout overall"),
        status_slices=_parse_status_slices(
            holdout_object["status_slices"], label="holdout status_slices"
        ),
    )
    return BaselineExperimentResult(
        track=track,
        outer_train_sample_count=train_count,
        outer_test_sample_count=test_count,
        cv_scheme=scheme,
        bucket_order=parsed_bucket_order,
        models=models,
        selection=selection,
        holdout=holdout,
    )


def parse_experiment_json(serialized: str | bytes) -> BaselineExperimentResult:
    """Parse strict UTF-8 JSON while rejecting duplicate keys and oversized payloads."""

    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_REPORT_BYTES:
            raise ExperimentValidationError("experiment report exceeds the maximum size")
        try:
            text = serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ExperimentValidationError("experiment report must be UTF-8") from error
    elif isinstance(serialized, str):
        text = serialized
        if len(text.encode("utf-8")) > _MAX_REPORT_BYTES:
            raise ExperimentValidationError("experiment report exceeds the maximum size")
    else:
        raise ExperimentValidationError("serialized experiment report must be text or bytes")

    try:
        payload = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ExperimentValidationError("experiment report is not valid JSON") from error
    return validate_experiment_result(payload)


def _run_experiment(
    *,
    config: TrackConfig,
    train: pd.DataFrame,
    y_train: NDArray[np.float64],
    test: pd.DataFrame,
    y_test: NDArray[np.float64],
    splits: tuple[CVSplit, ...],
    cv_scheme: CVScheme,
    bucket_order: tuple[str, ...],
    validation_buckets: tuple[str | None, ...],
    expected_oof_mask: NDArray[np.bool_],
) -> BaselineExperimentResult:
    if len(splits) != len(validation_buckets):
        raise ExperimentValidationError("every CV fold must have one validation bucket label")
    _validate_splits(splits, expected_oof_mask=expected_oof_mask, row_count=len(train))

    prototypes: dict[BaselineName, Pipeline] = {
        name: make_baseline_pipeline(name, config) for name in _BASELINES
    }
    model_results: list[ModelCrossValidationResult] = []
    for model_name in _BASELINES:
        oof_predictions = np.full(len(train), np.nan, dtype=np.float64)
        fold_results: list[FoldAggregate] = []
        for fold_number, ((train_indices, validation_indices), validation_bucket) in enumerate(
            zip(splits, validation_buckets, strict=True), start=1
        ):
            estimator = clone(prototypes[model_name])
            estimator.fit(train.iloc[train_indices], y_train[train_indices])
            predictions = _prediction_vector(
                estimator.predict(train.iloc[validation_indices]),
                expected_rows=len(validation_indices),
            )
            oof_predictions[validation_indices] = predictions
            fold_results.append(
                FoldAggregate(
                    fold_number=fold_number,
                    training_sample_count=len(train_indices),
                    validation_sample_count=len(validation_indices),
                    validation_bucket=validation_bucket,
                    metrics=regression_metrics(y_train[validation_indices], predictions),
                )
            )

        actual_oof_mask = np.isfinite(oof_predictions)
        if not np.array_equal(actual_oof_mask, expected_oof_mask):
            raise ExperimentValidationError("CV predictions did not cover the expected rows once")
        overall, slices = _evaluation_parts(
            config,
            y_train[actual_oof_mask],
            oof_predictions[actual_oof_mask],
            train.loc[actual_oof_mask],
        )
        model_results.append(
            ModelCrossValidationResult(
                model_name=model_name,
                overall=overall,
                status_slices=slices,
                folds=tuple(fold_results),
            )
        )

    ordered_results = tuple(model_results)
    selected_cv = min(ordered_results, key=lambda item: (item.overall.mae, item.model_name))
    selection = ModelSelection(
        selected_model=selected_cv.model_name,
        selected_cv_mae=selected_cv.overall.mae,
    )

    selected_estimator = clone(prototypes[selection.selected_model])
    selected_estimator.fit(train, y_train)
    holdout_predictions = _prediction_vector(
        selected_estimator.predict(test), expected_rows=len(test)
    )
    holdout_overall, holdout_slices = _evaluation_parts(config, y_test, holdout_predictions, test)

    return BaselineExperimentResult(
        track=config.name,
        outer_train_sample_count=len(train),
        outer_test_sample_count=len(test),
        cv_scheme=cv_scheme,
        bucket_order=bucket_order,
        models=ordered_results,
        selection=selection,
        holdout=HoldoutResult(
            model_name=selection.selected_model,
            overall=holdout_overall,
            status_slices=holdout_slices,
        ),
    )


def _validated_outer_partitions(
    train_features: object,
    train_target: object,
    test_features: object,
    test_target: object,
    *,
    config: TrackConfig,
) -> tuple[pd.DataFrame, NDArray[np.float64], pd.DataFrame, NDArray[np.float64]]:
    train = validate_feature_frame(train_features, config)
    test = validate_feature_frame(test_features, config)
    if tuple(train.columns) != tuple(test.columns):
        raise FeatureContractError("outer train and test must provide the same feature columns")
    _validate_index_alignment(train, train_target, label="outer train")
    _validate_index_alignment(test, test_target, label="outer test")
    y_train = validate_target(train_target, expected_rows=len(train), config=config)
    y_test = validate_target(test_target, expected_rows=len(test), config=config)
    return train, y_train, test, y_test


def _validate_index_alignment(frame: pd.DataFrame, target: object, *, label: str) -> None:
    if isinstance(target, (pd.Series, pd.DataFrame)) and not target.index.equals(frame.index):
        raise FeatureContractError(f"{label} feature and target indexes must align")


def _validate_retail_status_contract(frame: pd.DataFrame, *, label: str) -> None:
    values = frame["vehicle_status"].tolist()
    statuses: list[str] = []
    for value in values:
        if not isinstance(value, str) or value != value.strip().lower():
            raise ExperimentValidationError(
                f"{label} vehicle_status values must be canonical lowercase text"
            )
        if value not in _RETAIL_STATUSES:
            raise ExperimentValidationError(f"{label} contains an unsupported vehicle_status")
        statuses.append(value)
    missing = sorted(set(_RETAIL_STATUSES) - set(statuses))
    if missing:
        raise ExperimentValidationError(
            f"{label} must contain every retail status slice: {', '.join(missing)}"
        )


def _validate_splits(
    splits: tuple[CVSplit, ...],
    *,
    expected_oof_mask: NDArray[np.bool_],
    row_count: int,
) -> None:
    if not splits:
        raise ExperimentValidationError("cross-validation must contain at least one fold")
    if expected_oof_mask.shape != (row_count,):
        raise ExperimentValidationError("expected OOF mask has an invalid shape")
    validation_counts = np.zeros(row_count, dtype=np.int64)
    for train_indices, validation_indices in splits:
        for indices, label in (
            (train_indices, "training"),
            (validation_indices, "validation"),
        ):
            if indices.ndim != 1 or len(indices) == 0:
                raise ExperimentValidationError(f"each fold needs non-empty {label} indices")
            if (indices < 0).any() or (indices >= row_count).any():
                raise ExperimentValidationError("CV index is outside the outer train partition")
            if len(np.unique(indices)) != len(indices):
                raise ExperimentValidationError(f"fold {label} indices must be unique")
        if np.intersect1d(train_indices, validation_indices).size:
            raise ExperimentValidationError("fold training and validation rows must be disjoint")
        validation_counts[validation_indices] += 1
    if (validation_counts > 1).any():
        raise ExperimentValidationError("a row must not be predicted in multiple CV folds")
    if not np.array_equal(validation_counts == 1, expected_oof_mask):
        raise ExperimentValidationError("CV validation rows do not match the approved OOF rows")


def _prediction_vector(predictions: object, *, expected_rows: int) -> NDArray[np.float64]:
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) != expected_rows:
        raise ExperimentValidationError("model predictions must be a one-dimensional row match")
    try:
        numeric = values.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise ExperimentValidationError("model predictions must be numeric") from error
    if not np.isfinite(numeric).all():
        raise ExperimentValidationError("model predictions must be finite")
    return numeric


def _evaluation_parts(
    config: TrackConfig,
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    features: pd.DataFrame,
) -> tuple[RegressionMetrics, tuple[StatusSliceMetrics, ...]]:
    if config.name == "retail":
        evaluation = retail_status_metrics(actual, predicted, features["vehicle_status"])
        return evaluation.overall, evaluation.status_slices
    return regression_metrics(actual, predicted), ()


def _validate_experiment_result(result: BaselineExperimentResult) -> None:
    if not isinstance(result.track, str) or result.track not in TRACKS:
        raise ExperimentValidationError("experiment track is invalid")
    if (
        isinstance(result.outer_train_sample_count, bool)
        or not isinstance(result.outer_train_sample_count, int)
        or result.outer_train_sample_count < 1
    ):
        raise ExperimentValidationError("outer train sample count must be positive")
    if (
        isinstance(result.outer_test_sample_count, bool)
        or not isinstance(result.outer_test_sample_count, int)
        or result.outer_test_sample_count < 1
    ):
        raise ExperimentValidationError("outer test sample count must be positive")
    if tuple(model.model_name for model in result.models) != _BASELINES:
        raise ExperimentValidationError("CV models must be the two ordered Phase 3 baselines")

    expected_scheme = _RETAIL_CV_SCHEME if result.track == "retail" else _WHOLESALE_CV_SCHEME
    if result.cv_scheme != expected_scheme:
        raise ExperimentValidationError("CV scheme does not match the experiment track")
    if result.track == "retail":
        if result.bucket_order:
            raise ExperimentValidationError("retail experiments must not report CV buckets")
    else:
        if len(result.bucket_order) < 2:
            raise ExperimentValidationError("wholesale bucket_order needs at least two buckets")
        if any(
            not isinstance(bucket, str) or not bucket.strip() or bucket != bucket.strip()
            for bucket in result.bucket_order
        ):
            raise ExperimentValidationError(
                "wholesale bucket_order must be canonical non-empty text"
            )
        if len(result.bucket_order) != len(set(result.bucket_order)):
            raise ExperimentValidationError("wholesale bucket_order must be unique")

    reference_signature: tuple[tuple[int, int, str | None], ...] | None = None
    for model in result.models:
        _validate_track_slices(result.track, model.status_slices)
        if model.overall.sample_count != sum(fold.validation_sample_count for fold in model.folds):
            raise ExperimentValidationError("CV overall count must equal all fold validations")
        numbers = tuple(fold.fold_number for fold in model.folds)
        if numbers != tuple(range(1, len(model.folds) + 1)):
            raise ExperimentValidationError("fold numbers must be consecutive from one")
        signature = tuple(
            (
                fold.training_sample_count,
                fold.validation_sample_count,
                fold.validation_bucket,
            )
            for fold in model.folds
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ExperimentValidationError("all models must use identical CV folds")

        if result.track == "retail":
            if model.overall.sample_count != result.outer_train_sample_count:
                raise ExperimentValidationError("retail OOF predictions must cover outer train")
            if any(fold.validation_bucket is not None for fold in model.folds):
                raise ExperimentValidationError("retail folds must not have validation buckets")
            if len(model.folds) < 2:
                raise ExperimentValidationError("retail CV must have at least two folds")
        else:
            if len(model.folds) != len(result.bucket_order) - 1:
                raise ExperimentValidationError("wholesale folds must match ordered CV buckets")
            for position, fold in enumerate(model.folds, start=1):
                if fold.validation_bucket != result.bucket_order[position]:
                    raise ExperimentValidationError(
                        "wholesale validation buckets must follow bucket_order"
                    )
                if position > 1:
                    previous = model.folds[position - 2]
                    expected_training = (
                        previous.training_sample_count + previous.validation_sample_count
                    )
                    if fold.training_sample_count != expected_training:
                        raise ExperimentValidationError(
                            "wholesale fold training counts must grow forward only"
                        )
            last_fold = model.folds[-1]
            if (
                last_fold.training_sample_count + last_fold.validation_sample_count
                != result.outer_train_sample_count
            ):
                raise ExperimentValidationError(
                    "final wholesale fold must account for the outer train partition"
                )

    expected_selection = min(result.models, key=lambda item: (item.overall.mae, item.model_name))
    if result.selection.selected_model != expected_selection.model_name:
        raise ExperimentValidationError("selected model does not minimize CV MAE")
    if result.selection.selected_cv_mae != expected_selection.overall.mae:
        raise ExperimentValidationError("selected_cv_mae does not match selected model")
    if result.holdout.model_name != result.selection.selected_model:
        raise ExperimentValidationError("holdout must evaluate only the selected model")
    if result.holdout.overall.sample_count != result.outer_test_sample_count:
        raise ExperimentValidationError("holdout metric count must equal outer test count")
    _validate_track_slices(result.track, result.holdout.status_slices)


def _validate_track_slices(track: TrackName, slices: tuple[StatusSliceMetrics, ...]) -> None:
    statuses = tuple(item.status for item in slices)
    if track == "retail" and statuses != _RETAIL_STATUSES:
        raise ExperimentValidationError(
            "retail evaluations must contain certified, new, and used status slices"
        )
    if track == "wholesale" and slices:
        raise ExperimentValidationError("wholesale evaluations must not contain status slices")


def _validate_status_slices(
    slices: tuple[StatusSliceMetrics, ...], overall: RegressionMetrics
) -> None:
    statuses = tuple(item.status for item in slices)
    if statuses != tuple(sorted(statuses)) or len(statuses) != len(set(statuses)):
        raise ExperimentValidationError("status slices must be unique and sorted")
    for item in slices:
        if not isinstance(item.status, str) or item.status != item.status.strip().lower():
            raise ExperimentValidationError("status names must be canonical lowercase text")
        _validate_metrics(item.metrics)
    if slices and sum(item.metrics.sample_count for item in slices) != overall.sample_count:
        raise ExperimentValidationError("status slice counts must sum to overall count")


def _validate_metrics(metrics: RegressionMetrics) -> None:
    if not isinstance(metrics, RegressionMetrics):
        raise ExperimentValidationError("metrics must be RegressionMetrics")
    if type(metrics.sample_count) is not int or metrics.sample_count < 1:
        raise ExperimentValidationError("metric sample_count must be positive")
    if type(metrics.mae) not in {int, float} or not math.isfinite(metrics.mae) or metrics.mae < 0.0:
        raise ExperimentValidationError("metric MAE must be finite and nonnegative")
    if (
        type(metrics.rmse) not in {int, float}
        or not math.isfinite(metrics.rmse)
        or metrics.rmse < 0.0
    ):
        raise ExperimentValidationError("metric RMSE must be finite and nonnegative")
    if metrics.r2 is not None and (
        type(metrics.r2) not in {int, float} or not math.isfinite(metrics.r2) or metrics.r2 > 1.0
    ):
        raise ExperimentValidationError(
            "metric R-squared must be finite numeric, at most one, or null"
        )


def _parse_model(value: object, *, index: int) -> ModelCrossValidationResult:
    model = _object(value, label=f"models[{index}]")
    _exact_keys(model, _MODEL_KEYS, label=f"models[{index}]")
    raw_folds = model["folds"]
    if not isinstance(raw_folds, list):
        raise ExperimentValidationError("model folds must be an array")
    folds = tuple(_parse_fold(item, index=position) for position, item in enumerate(raw_folds))
    return ModelCrossValidationResult(
        model_name=_baseline_name(model["model_name"], label="model_name"),
        overall=_parse_metrics(model["overall"], label="model overall"),
        status_slices=_parse_status_slices(model["status_slices"], label="model status_slices"),
        folds=folds,
    )


def _parse_fold(value: object, *, index: int) -> FoldAggregate:
    fold = _object(value, label=f"folds[{index}]")
    _exact_keys(fold, _FOLD_KEYS, label=f"folds[{index}]")
    bucket = fold["validation_bucket"]
    if bucket is not None:
        bucket = _text(bucket, label="validation_bucket")
    return FoldAggregate(
        fold_number=_integer(fold["fold_number"], label="fold_number"),
        training_sample_count=_integer(
            fold["training_sample_count"], label="training_sample_count"
        ),
        validation_sample_count=_integer(
            fold["validation_sample_count"], label="validation_sample_count"
        ),
        validation_bucket=bucket,
        metrics=_parse_metrics(fold["metrics"], label="fold metrics"),
    )


def _parse_status_slices(value: object, *, label: str) -> tuple[StatusSliceMetrics, ...]:
    if not isinstance(value, list):
        raise ExperimentValidationError(f"{label} must be an array")
    parsed: list[StatusSliceMetrics] = []
    for index, raw_slice in enumerate(value):
        status_slice = _object(raw_slice, label=f"{label}[{index}]")
        _exact_keys(status_slice, _SLICE_KEYS, label=f"{label}[{index}]")
        parsed.append(
            StatusSliceMetrics(
                status=_text(status_slice["status"], label="status"),
                metrics=_parse_metrics(status_slice["metrics"], label="slice metrics"),
            )
        )
    return tuple(parsed)


def _parse_metrics(value: object, *, label: str) -> RegressionMetrics:
    metrics = _object(value, label=label)
    _exact_keys(metrics, _METRIC_KEYS, label=label)
    raw_r2 = metrics["r2"]
    return RegressionMetrics(
        sample_count=_integer(metrics["sample_count"], label="sample_count"),
        mae=_number(metrics["mae"], label="mae"),
        rmse=_number(metrics["rmse"], label="rmse"),
        r2=None if raw_r2 is None else _number(raw_r2, label="r2"),
    )


def _baseline_name(value: object, *, label: str) -> BaselineName:
    text = _text(value, label=label)
    if text not in _BASELINES:
        raise ExperimentValidationError(f"{label} is unsupported")
    return text


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ExperimentValidationError(f"{label} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unexpected {', '.join(extra)}")
        raise ExperimentValidationError(f"{label} has invalid fields: {'; '.join(details)}")


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentValidationError(f"{label} must be an integer")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ExperimentValidationError(f"{label} must be finite")
    return number


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentValidationError(f"{label} must be non-empty text")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
