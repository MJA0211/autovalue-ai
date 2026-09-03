from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from typing import Any, ClassVar, Self

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.candidates import CandidateSpec, candidate_specs, get_candidate_spec
from autovalue_ml.modeling.cv import wholesale_forward_cv_splits
from autovalue_ml.modeling.experiment import FoldAggregate
from autovalue_ml.modeling.metrics import RegressionMetrics, StatusSliceMetrics
from autovalue_ml.modeling.phase4_evaluation import (
    Phase4CandidateCVResult,
    Phase4EvaluationError,
    Phase4Shortlist,
    evaluate_phase4_candidate_cv,
    parse_phase4_candidate_cv_result,
    shortlist_phase4_candidates,
)
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class _RecordingRegressor(BaseEstimator):  # type: ignore[misc]
    fit_indexes: ClassVar[list[tuple[int, ...]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.fit_indexes = []

    def fit(self, X: pd.DataFrame, y: object) -> Self:
        values = np.asarray(y, dtype=np.float64)
        if len(values) != len(X):
            raise AssertionError("fit row mismatch")
        type(self).fit_indexes.append(tuple(int(value) for value in X.index))
        self.mean_ = float(np.mean(values))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.full(len(X), self.mean_, dtype=np.float64)


class _RecordingFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def reset(self) -> None:
        self.calls = []

    def __call__(
        self,
        track: object,
        family: object,
        index: int = 0,
        **kwargs: object,
    ) -> Pipeline:
        del track, family, index
        self.calls.append(dict(kwargs))
        return Pipeline((("regressor", _RecordingRegressor()),))


_recording_factory = _RecordingFactory()


def _retail_frame() -> tuple[
    pd.DataFrame, pd.Series[float], tuple[tuple[np.ndarray, np.ndarray], ...]
]:
    statuses = ("certified", "new", "used")
    rows: list[dict[str, object]] = []
    for fold_number in range(5):
        for status_number, status in enumerate(statuses):
            position = fold_number * 3 + status_number
            rows.append(
                {
                    "year": 2010 + position,
                    "make": f"Make {position}",
                    "model": f"Model {position}",
                    "mileage": float(position * 5_000),
                    "vehicle_status": status,
                }
            )
    frame = pd.DataFrame(rows, index=np.arange(100, 115))
    target = pd.Series(
        np.linspace(10_000.0, 31_000.0, len(frame)),
        index=frame.index,
        name="price_usd",
    )
    splits = []
    positions = np.arange(len(frame), dtype=np.int64)
    for fold_number in range(5):
        validation = np.arange(fold_number * 3, fold_number * 3 + 3, dtype=np.int64)
        training = positions[~np.isin(positions, validation)]
        splits.append((training, validation))
    return frame, target, tuple(splits)


def _wholesale_frame() -> tuple[pd.DataFrame, pd.Series[float], pd.Series[str]]:
    buckets = ("warmup", "2015_01", "2015_02", "2015_03_04")
    rows: list[dict[str, object]] = []
    labels: list[str] = []
    for bucket_number, bucket in enumerate(buckets):
        for item in range(3):
            position = bucket_number * 3 + item
            rows.append(
                {
                    "year": 2000 + position,
                    "make": f"Make {item}",
                    "model": f"Model {position}",
                    "trim": None,
                    "mileage": float(40_000 + position * 2_000),
                    "condition": 3.0,
                    "vehicle_type": "sedan",
                }
            )
            labels.append(bucket)
    frame = pd.DataFrame(rows, index=np.arange(200, 212))
    target = pd.Series(
        np.linspace(5_000.0, 20_000.0, len(frame)),
        index=frame.index,
        name="price_usd",
    )
    return frame, target, pd.Series(labels, index=frame.index, name="cv_bucket")


def test_retail_candidate_evaluation_is_fold_local_and_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autovalue_ml.modeling import phase4_evaluation

    frame, target, splits = _retail_frame()
    _RecordingRegressor.reset()
    _recording_factory.reset()
    monkeypatch.setattr(phase4_evaluation, "make_candidate_pipeline", _recording_factory)

    result = evaluate_phase4_candidate_cv(
        features=frame,
        target=target,
        spec=get_candidate_spec("retail", "linear_regression_incumbent", 0),
        splits=splits,
        expected_oof_mask=np.ones(len(frame), dtype=np.bool_),
        validation_buckets=(None,) * 5,
    )

    assert result.overall.sample_count == 15
    assert tuple(item.status for item in result.status_slices) == (
        "certified",
        "new",
        "used",
    )
    assert tuple(item.metrics.sample_count for item in result.status_slices) == (5, 5, 5)
    assert len(result.folds) == 5
    assert len(_RecordingRegressor.fit_indexes) == 5
    assert all(len(indexes) == 12 for indexes in _RecordingRegressor.fit_indexes)
    assert _recording_factory.calls == [{}] * 5
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert all(name not in serialized for name in ("prediction", "residual", "raw_row"))


def test_wholesale_candidate_evaluation_preserves_forward_oof_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autovalue_ml.modeling import phase4_evaluation

    frame, target, buckets = _wholesale_frame()
    order = ("warmup", "2015_01", "2015_02", "2015_03_04")
    splits = wholesale_forward_cv_splits(buckets, bucket_order=order)
    expected_mask = np.asarray(buckets != "warmup", dtype=np.bool_)
    _RecordingRegressor.reset()
    _recording_factory.reset()
    monkeypatch.setattr(phase4_evaluation, "make_candidate_pipeline", _recording_factory)

    result = evaluate_phase4_candidate_cv(
        features=frame,
        target=target,
        spec=get_candidate_spec("wholesale", "random_forest", 0),
        splits=splits,
        expected_oof_mask=expected_mask,
        validation_buckets=order[1:],
    )

    assert result.overall.sample_count == 9
    assert result.status_slices == ()
    assert tuple(fold.training_sample_count for fold in result.folds) == (3, 6, 9)
    assert tuple(fold.validation_bucket for fold in result.folds) == order[1:]
    assert result.latest_fold.validation_bucket == "2015_03_04"
    assert _recording_factory.calls == [{"random_forest_n_jobs": 4}] * 3


def _aggregate_result(
    spec: CandidateSpec,
    mae: float,
    *,
    training_count: int = 20,
) -> Phase4CandidateCVResult:
    rmse = mae + 100.0
    metrics = RegressionMetrics(sample_count=5, mae=mae, rmse=rmse, r2=0.0)
    folds = tuple(
        FoldAggregate(
            fold_number=number,
            training_sample_count=training_count,
            validation_sample_count=1,
            validation_bucket=None,
            metrics=RegressionMetrics(sample_count=1, mae=mae, rmse=rmse, r2=None),
        )
        for number in range(1, 6)
    )
    slices = tuple(
        StatusSliceMetrics(
            status=status,
            metrics=RegressionMetrics(sample_count=count, mae=mae, rmse=rmse, r2=0.0),
        )
        for status, count in (("certified", 1), ("new", 2), ("used", 2))
    )
    return Phase4CandidateCVResult(spec=spec, overall=metrics, status_slices=slices, folds=folds)


def test_shortlist_uses_exact_mae_then_stable_id_per_family() -> None:
    specs = candidate_specs("retail")
    family_maes = {
        "linear_regression_incumbent": (200.0,),
        "random_forest": (100.0, 90.0, 90.0, 110.0, 120.0, 130.0),
        "gradient_boosting": (85.0, 86.0, 84.0, 120.0, 130.0, 140.0),
    }
    seen: dict[str, int] = {family: 0 for family in family_maes}
    results: list[Phase4CandidateCVResult] = []
    for spec in specs:
        family_index = seen[spec.family]
        seen[spec.family] += 1
        results.append(_aggregate_result(spec, family_maes[spec.family][family_index]))

    shortlist = shortlist_phase4_candidates("retail", tuple(results))

    assert shortlist.random_forest_candidate_ids == (
        "phase4-retail-random_forest-01",
        "phase4-retail-random_forest-02",
    )
    assert shortlist.gradient_boosting_candidate_ids == (
        "phase4-retail-gradient_boosting-02",
        "phase4-retail-gradient_boosting-00",
    )
    assert shortlist.full_development_candidate_ids == (
        "phase4-retail-linear_regression_incumbent-00",
        *shortlist.random_forest_candidate_ids,
        *shortlist.gradient_boosting_candidate_ids,
    )


def test_shortlist_requires_complete_same_fold_screening_evidence() -> None:
    results = tuple(
        _aggregate_result(spec, 100.0 + spec.index) for spec in candidate_specs("retail")
    )
    with pytest.raises(Phase4EvaluationError, match="every approved candidate"):
        shortlist_phase4_candidates("retail", results[:-1])
    with pytest.raises(Phase4EvaluationError, match="stable order"):
        shortlist_phase4_candidates("retail", (results[1], results[0], *results[2:]))

    changed = _aggregate_result(results[1].spec, results[1].overall.mae, training_count=21)
    with pytest.raises(Phase4EvaluationError, match="identical CV folds"):
        shortlist_phase4_candidates("retail", (results[0], changed, *results[2:]))


def test_candidate_result_rejects_forged_or_inconsistent_aggregates() -> None:
    valid = _aggregate_result(get_candidate_spec("retail", "random_forest", 0), 100.0)
    with pytest.raises(Phase4EvaluationError, match="frozen policy"):
        replace(valid, spec=replace(valid.spec, random_state=1))
    with pytest.raises(Phase4EvaluationError, match="weighted fold MAEs"):
        replace(valid, overall=replace(valid.overall, mae=101.0))
    with pytest.raises(Phase4EvaluationError, match="weighted status MAEs"):
        changed_slice = replace(
            valid.status_slices[0],
            metrics=replace(valid.status_slices[0].metrics, mae=101.0),
        )
        replace(valid, status_slices=(changed_slice, *valid.status_slices[1:]))
    with pytest.raises(FrozenInstanceError):
        valid.spec = get_candidate_spec("retail", "random_forest", 1)  # type: ignore[misc]


def test_evaluator_rejects_wrong_masks_splits_labels_statuses_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autovalue_ml.modeling import phase4_evaluation

    frame, target, splits = _retail_frame()
    spec = get_candidate_spec("retail", "linear_regression_incumbent", 0)
    monkeypatch.setattr(phase4_evaluation, "make_candidate_pipeline", _recording_factory)

    with pytest.raises(Phase4EvaluationError, match="boolean row match"):
        evaluate_phase4_candidate_cv(
            features=frame,
            target=target,
            spec=spec,
            splits=splits,
            expected_oof_mask=np.ones(len(frame), dtype=np.int64),
            validation_buckets=(None,) * 5,
        )
    with pytest.raises(Phase4EvaluationError, match="do not match"):
        evaluate_phase4_candidate_cv(
            features=frame,
            target=target,
            spec=spec,
            splits=splits,
            expected_oof_mask=np.zeros(len(frame), dtype=np.bool_),
            validation_buckets=(None,) * 5,
        )
    with pytest.raises(Phase4EvaluationError, match="bucket labels must be null"):
        evaluate_phase4_candidate_cv(
            features=frame,
            target=target,
            spec=spec,
            splits=splits,
            expected_oof_mask=np.ones(len(frame), dtype=np.bool_),
            validation_buckets=("wrong",) * 5,
        )

    invalid_status = frame.copy()
    invalid_status.loc[invalid_status.index[0], "vehicle_status"] = "Used"
    with pytest.raises(Phase4EvaluationError, match="exact certified"):
        evaluate_phase4_candidate_cv(
            features=invalid_status,
            target=target,
            spec=spec,
            splits=splits,
            expected_oof_mask=np.ones(len(frame), dtype=np.bool_),
            validation_buckets=(None,) * 5,
        )


def test_shortlist_value_object_rejects_invalid_family_ids() -> None:
    valid = Phase4Shortlist(
        track="retail",
        linear_reference_id="phase4-retail-linear_regression_incumbent-00",
        random_forest_candidate_ids=(
            "phase4-retail-random_forest-00",
            "phase4-retail-random_forest-01",
        ),
        gradient_boosting_candidate_ids=(
            "phase4-retail-gradient_boosting-00",
            "phase4-retail-gradient_boosting-01",
        ),
    )
    with pytest.raises(Phase4EvaluationError, match="exactly two"):
        replace(valid, random_forest_candidate_ids=("phase4-retail-random_forest-00",))  # type: ignore[arg-type]
    with pytest.raises(Phase4EvaluationError, match="unique"):
        replace(
            valid,
            gradient_boosting_candidate_ids=(
                "phase4-retail-gradient_boosting-00",
                "phase4-retail-gradient_boosting-00",
            ),
        )
    with pytest.raises(Phase4EvaluationError, match="invalid candidate ID"):
        replace(
            valid,
            random_forest_candidate_ids=(
                "phase4-retail-gradient_boosting-00",
                "phase4-retail-random_forest-01",
            ),
        )


def test_candidate_result_parser_round_trips_and_rejects_policy_drift() -> None:
    result = _aggregate_result(get_candidate_spec("retail", "random_forest", 0), 100.0)
    payload = result.to_dict()

    assert parse_phase4_candidate_cv_result(payload) == result

    changed = dict(payload)
    changed["random_state"] = 1
    with pytest.raises(Phase4EvaluationError, match="metadata"):
        parse_phase4_candidate_cv_result(changed)
    changed_index = dict(payload)
    changed_index["index"] = False
    with pytest.raises(Phase4EvaluationError, match="metadata"):
        parse_phase4_candidate_cv_result(changed_index)
    changed_parameters = dict(payload)
    parameters = list(result.spec.parameters)
    parameters[0] = True
    changed_parameters["parameters"] = parameters
    with pytest.raises(Phase4EvaluationError, match="metadata"):
        parse_phase4_candidate_cv_result(changed_parameters)
    extra = dict(payload)
    extra["raw_rows"] = []
    with pytest.raises(Phase4EvaluationError, match="fields"):
        parse_phase4_candidate_cv_result(extra)
