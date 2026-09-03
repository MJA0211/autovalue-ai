from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import retail_uncertainty_diagnostics as diagnostics
from autovalue_ml.modeling.calibration_artifact import PHASE4_RETAIL_CONFIRMATION_SHA256
from autovalue_ml.modeling.cv import CVSplit
from autovalue_ml.modeling.metrics import RegressionMetrics, regression_metrics
from autovalue_ml.modeling.phase4_confirmation import Phase4ConfirmationReport
from numpy.typing import NDArray


@pytest.fixture(scope="module")
def exact_development() -> tuple[pd.DataFrame, NDArray[np.float64]]:
    positions = np.arange(diagnostics.DEVELOPMENT_SAMPLE_COUNT, dtype=np.int64)
    features = pd.DataFrame(
        {
            "year": 2000 + positions % 24,
            "make": np.asarray([f"make-{value % 25:02d}" for value in positions]),
            "model": np.asarray([f"model-{value % 4}" for value in positions]),
            "vehicle_status": np.asarray(
                [("certified", "new", "used")[value % 3] for value in positions]
            ),
            # A unique mileage makes every row a distinct predictor group.
            "mileage": positions.astype(np.float64),
        }
    )
    target = np.asarray(20_000.0 + positions * 0.05, dtype=np.float64)
    return features, target


@dataclass
class _EstimatorTrace:
    fit_ids: NDArray[np.int64] | None = None
    predict_ids: NDArray[np.int64] | None = None


class _DeterministicEstimator:
    def __init__(self, trace: _EstimatorTrace) -> None:
        self._trace = trace

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
    ) -> _DeterministicEstimator:
        del target
        self._trace.fit_ids = features["mileage"].to_numpy(dtype=np.int64, copy=True)
        return self

    def predict(self, features: pd.DataFrame) -> object:
        ids = features["mileage"].to_numpy(dtype=np.int64, copy=True)
        self._trace.predict_ids = ids
        return 20_000.0 + ids.astype(np.float64) * 0.05


def test_reconstruct_oof_keeps_fits_separate_and_scores_every_row_once(
    exact_development: tuple[pd.DataFrame, NDArray[np.float64]],
) -> None:
    features, target = exact_development
    traces: list[_EstimatorTrace] = []
    progress: list[tuple[int, int]] = []

    def factory() -> _DeterministicEstimator:
        trace = _EstimatorTrace()
        traces.append(trace)
        return _DeterministicEstimator(trace)

    predictions, splits = diagnostics.reconstruct_rf05_development_oof(
        development_features=features,
        development_target=target,
        estimator_factory=factory,
        progress=lambda fold, total: progress.append((fold, total)),
    )

    np.testing.assert_allclose(predictions, target, rtol=0.0, atol=0.0)
    assert predictions.flags.writeable is False
    assert len(splits) == 5
    assert len(traces) == 5
    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    validation_counts = np.zeros(len(features), dtype=np.int8)
    for trace, (training, validation) in zip(traces, splits, strict=True):
        assert trace.fit_ids is not None
        assert trace.predict_ids is not None
        assert len(trace.fit_ids) == len(training)
        assert len(trace.predict_ids) == len(validation)
        assert np.intersect1d(trace.fit_ids, trace.predict_ids).size == 0
        validation_counts[validation] += 1
    assert np.all(validation_counts == 1)


def test_reconstruct_rejects_incomplete_or_repeated_validation_assignments(
    exact_development: tuple[pd.DataFrame, NDArray[np.float64]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, target = exact_development
    validation_chunks = [part.copy() for part in np.array_split(np.arange(len(features)), 5)]
    validation_chunks[1][0] = validation_chunks[0][0]
    splits = tuple(
        (np.asarray([len(features) - 1], dtype=np.int64), validation.astype(np.int64))
        for validation in validation_chunks
    )

    def invalid_splits(
        frame: object,
        *,
        n_splits: int,
    ) -> tuple[CVSplit, ...]:
        del frame
        assert n_splits == 5
        return splits

    monkeypatch.setattr(diagnostics, "retail_group_cv_splits", invalid_splits)

    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="every row once"):
        diagnostics.reconstruct_rf05_development_oof(
            development_features=features,
            development_target=target,
            estimator_factory=lambda: _DeterministicEstimator(_EstimatorTrace()),
        )


def test_reconstruct_enforces_exact_boundary_and_callable_progress(
    exact_development: tuple[pd.DataFrame, NDArray[np.float64]],
) -> None:
    features, target = exact_development
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="exact development set"):
        diagnostics.reconstruct_rf05_development_oof(
            development_features=features.iloc[:-1],
            development_target=target[:-1],
            estimator_factory=lambda: _DeterministicEstimator(_EstimatorTrace()),
        )

    invalid_progress = cast(Callable[[int, int], None], object())
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="progress must be callable"):
        diagnostics.reconstruct_rf05_development_oof(
            development_features=features,
            development_target=target,
            estimator_factory=lambda: _DeterministicEstimator(_EstimatorTrace()),
            progress=invalid_progress,
        )


@pytest.mark.parametrize(
    ("values", "rows", "message"),
    [
        ([1.0, 2.0], 1, "one-dimensional row match"),
        ([[1.0]], 1, "one-dimensional row match"),
        ([True], 1, "numeric, not boolean"),
        (["not-a-number"], 1, "must be numeric"),
        ([-1.0], 1, "finite and nonnegative"),
        ([np.nan], 1, "finite and nonnegative"),
        ([np.inf], 1, "finite and nonnegative"),
    ],
)
def test_prediction_vector_rejects_invalid_output(
    values: object,
    rows: int,
    message: str,
) -> None:
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match=message):
        diagnostics._prediction_vector(values, expected_rows=rows)


def test_confirmation_guards_model_and_checksum() -> None:
    valid = _confirmation_stub()
    diagnostics._validate_confirmation(valid, PHASE4_RETAIL_CONFIRMATION_SHA256)

    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="checksum differs"):
        diagnostics._validate_confirmation(valid, "0" * 64)

    wrong_track = cast(
        Phase4ConfirmationReport,
        SimpleNamespace(
            track="wholesale",
            metric_ranking=("phase4-retail-random_forest-05",),
            candidates=(),
        ),
    )
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="does not freeze retail RF05"):
        diagnostics._validate_confirmation(
            wrong_track,
            PHASE4_RETAIL_CONFIRMATION_SHA256,
        )


def test_reconstruction_metric_guard_accepts_match_and_rejects_drift() -> None:
    target = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    predictions = np.asarray([11.0, 18.0, 33.0, 36.0], dtype=np.float64)
    confirmation = _confirmation_with_metrics(regression_metrics(target, predictions))

    diagnostics._validate_reconstruction_metrics(target, predictions, confirmation)
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="differs from frozen"):
        diagnostics._validate_reconstruction_metrics(target, predictions + 10.0, confirmation)


def test_build_diagnostics_is_aggregate_complete_and_deterministic(
    exact_development: tuple[pd.DataFrame, NDArray[np.float64]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_features, _ = exact_development
    features = base_features.copy(deep=True)
    positions = np.arange(len(features), dtype=np.int64)
    mileage = positions.astype(np.float64)
    mileage[positions % 10 == 0] = np.nan
    features["mileage"] = mileage
    predictions = np.asarray(8_000.0 + positions * 0.5, dtype=np.float64)
    residuals = np.asarray(100.0 + (positions % 100) * 5.0 + predictions * 0.02)
    direction = np.where(positions % 2 == 0, 1.0, -1.0)
    target = np.asarray(predictions + direction * residuals, dtype=np.float64)
    splits = tuple(
        (
            np.asarray([0], dtype=np.int64),
            np.asarray([fold], dtype=np.int64),
        )
        for fold in range(5)
    )

    def fake_reconstruction(
        *,
        development_features: object,
        development_target: object,
        estimator_factory: diagnostics.EstimatorFactory | None = None,
        progress: diagnostics.ProgressCallback | None = None,
    ) -> tuple[NDArray[np.float64], tuple[CVSplit, ...]]:
        del estimator_factory, progress
        assert isinstance(development_features, pd.DataFrame)
        assert len(development_features) == len(features)
        assert len(np.asarray(development_target)) == len(target)
        result = predictions.copy()
        result.setflags(write=False)
        return result, splits

    def accept_reconstruction_metrics(
        actual: NDArray[np.float64],
        estimated: NDArray[np.float64],
        confirmation: Phase4ConfirmationReport,
    ) -> None:
        del confirmation
        assert len(actual) == len(estimated) == len(features)

    monkeypatch.setattr(
        diagnostics,
        "reconstruct_rf05_development_oof",
        fake_reconstruction,
    )
    monkeypatch.setattr(
        diagnostics,
        "_validate_reconstruction_metrics",
        accept_reconstruction_metrics,
    )

    report = diagnostics.build_development_residual_diagnostics(
        development_features=features,
        development_target=target,
        confirmation=_confirmation_stub(),
        confirmation_sha256=PHASE4_RETAIL_CONFIRMATION_SHA256,
        generated_at="2026-09-02T19:00:00+00:00",
    )

    assert report["report_type"] == diagnostics.DIAGNOSTIC_REPORT_TYPE
    assert report["classification"] == (
        "development_only_diagnostic_not_calibration_or_model_selection"
    )
    boundaries = cast(dict[str, object], report["data_boundaries"])
    assert boundaries == {
        "development_rows": diagnostics.DEVELOPMENT_SAMPLE_COUNT,
        "development_oof_rows": diagnostics.DEVELOPMENT_SAMPLE_COUNT,
        "development_group_folds": 5,
        "calibration_rows_accessed": False,
        "calibration_targets_accessed": False,
        "legacy_holdout_accessed": False,
        "yoad_accessed": False,
        "river_accessed": False,
        "autotrader_accessed": False,
        "carson_shively_accessed": False,
        "raw_rows_predictions_or_residuals_persisted": False,
    }
    overall = cast(dict[str, object], report["overall_residual_distribution"])
    assert overall["support"] == diagnostics.DEVELOPMENT_SAMPLE_COUNT
    slices = cast(dict[str, list[dict[str, object]]], report["slices"])
    assert set(slices) == {
        "predicted_value_band",
        "actual_price_band_evaluation_only",
        "vehicle_age_band",
        "mileage_band",
        "mileage_per_year_band",
        "vehicle_status",
        "missing_mileage",
        "manufacturer",
        "model",
        "vehicle_status_by_predicted_value_band",
    }
    assert len(slices["manufacturer"]) == diagnostics.MAXIMUM_REPORTED_CATEGORIES
    assert {item["label"] for item in slices["missing_mileage"]} == {
        "mileage_missing",
        "mileage_present",
    }
    implication = cast(dict[str, object], report["design_implication"])
    assert implication["actual_price_is_evaluation_only"] is True
    assert implication["calibration_outcomes_used_to_choose_diagnostics"] is False

    serialized = diagnostics.canonical_diagnostics_json(report)
    assert serialized == diagnostics.canonical_diagnostics_json(report)
    assert (
        diagnostics.diagnostics_sha256(report)
        == hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    )


def test_diagnostic_helpers_enforce_support_and_numeric_boundaries() -> None:
    residuals = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    target = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    statistics = diagnostics._residual_statistics(residuals, target)
    assert statistics["support"] == 4
    assert statistics["median_absolute_residual_usd"] == 2.5
    assert statistics["mean_absolute_residual_usd"] == 2.5
    assert statistics["residual_variance_usd2"] == 1.25

    labels = np.asarray(["b", "b", "a", "a", "c"], dtype=np.str_)
    supported = diagnostics._dimension_report(
        labels,
        residuals=np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        target=np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
        minimum_support=2,
        maximum_categories=1,
    )
    assert [item["label"] for item in supported] == ["a"]
    assert diagnostics._normalized_text([" Ford ", None]).tolist() == ["ford", "__missing__"]
    assert diagnostics._strict_quartiles(np.asarray([1.0, 2.0, 3.0, 4.0]), label="values") == (
        1.75,
        2.5,
        3.25,
    )

    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="aligned non-empty"):
        diagnostics._residual_statistics(np.asarray([]), np.asarray([]))
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="quartiles must be distinct"):
        diagnostics._strict_quartiles(np.ones(8), label="constant")
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="non-empty finite"):
        diagnostics._strict_quartiles(np.asarray([1.0, np.nan]), label="invalid")


def test_predicted_value_relationship_reports_monotonic_heteroscedasticity() -> None:
    predictions = np.arange(1.0, 9.0)
    residuals = predictions * 2.0
    bands = np.asarray(
        [
            "predicted_value_1",
            "predicted_value_1",
            "predicted_value_2",
            "predicted_value_2",
            "predicted_value_3",
            "predicted_value_3",
            "predicted_value_4",
            "predicted_value_4",
        ],
        dtype=np.str_,
    )

    result = diagnostics._predicted_value_relationship(predictions, residuals, bands)

    assert cast(float, result["log_prediction_log_residual_pearson"]) > 0.99
    assert result["prediction_residual_spearman"] == pytest.approx(1.0)
    assert result["highest_to_lowest_quartile_mean_residual_ratio"] == pytest.approx(5.0)


def test_canonical_serialization_sorts_keys_and_rejects_non_finite_values() -> None:
    assert diagnostics.canonical_diagnostics_json({"z": 1, "a": 2}) == '{"a":2,"z":1}\n'
    with pytest.raises(diagnostics.ResidualDiagnosticsError, match="not JSON-safe"):
        diagnostics.canonical_diagnostics_json({"invalid": np.nan})


def _confirmation_stub() -> Phase4ConfirmationReport:
    return cast(
        Phase4ConfirmationReport,
        SimpleNamespace(
            track="retail",
            metric_ranking=("phase4-retail-random_forest-05",),
            candidates=(),
        ),
    )


def _confirmation_with_metrics(metrics: RegressionMetrics) -> Phase4ConfirmationReport:
    candidate = SimpleNamespace(
        spec=SimpleNamespace(candidate_id="phase4-retail-random_forest-05"),
        overall=metrics,
    )
    return cast(
        Phase4ConfirmationReport,
        SimpleNamespace(
            track="retail",
            metric_ranking=("phase4-retail-random_forest-05",),
            candidates=(candidate,),
        ),
    )
