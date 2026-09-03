from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import (
    COVERAGE_LEVELS,
    ConfidenceThresholds,
)
from autovalue_ml.modeling.retail_calibration_experiment import (
    METHODS,
    CalibrationExperimentError,
    _apply_intervals,
    _conditional_gate_results,
    _confidence_diagnostics,
    _crossfit_diagnostics,
    _finite_sample_radius,
    _fit_coverage_calibration,
    _interval_metrics,
    _normalized_statuses,
    _numeric_bands,
    _public_crossfit,
    _selected_method_validated,
    _selected_slice_diagnostics,
    canonical_calibration_report_json,
    report_sha256,
)
from numpy.typing import NDArray


def _synthetic_calibration_population() -> tuple[
    pd.DataFrame,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.str_],
    NDArray[np.str_],
]:
    row_count = 6_000
    positions = np.arange(row_count)
    status_values = np.asarray(("certified", "new", "used"), dtype=np.str_)
    prediction_values = np.asarray((5_000.0, 15_000.0, 25_000.0, 50_000.0), dtype=np.float64)
    statuses = status_values[positions % len(status_values)]
    predictions = np.asarray(
        prediction_values[(positions // len(status_values)) % 4], dtype=np.float64
    )
    residuals = np.asarray(100.0 + (positions % 40) * 100.0, dtype=np.float64)
    directions = np.asarray(np.where(positions % 2 == 0, 1.0, -1.0), dtype=np.float64)
    target = np.asarray(predictions + directions * residuals, dtype=np.float64)
    mileage = (positions % 200_000).astype(np.float64)
    mileage[positions % 10 == 0] = np.nan
    makes = np.asarray(("Ford", "Honda", "Toyota", "BMW", "GMC"), dtype=np.str_)
    features = pd.DataFrame(
        {
            "year": 2008 + positions % 16,
            "make": makes[positions % len(makes)],
            "model": [f"model-{position}" for position in positions],
            "mileage": mileage,
            "vehicle_status": statuses,
        }
    )
    value_bands = _numeric_bands(predictions, (10_000.0, 20_000.0, 30_000.0), prefix="band")
    return features, target, predictions, residuals, statuses, value_bands


def test_crossfit_calibration_and_slice_diagnostics_are_complete_and_row_free() -> None:
    features, target, predictions, residuals, statuses, value_bands = (
        _synthetic_calibration_population()
    )

    crossfit = _crossfit_diagnostics(
        features=features,
        target=target,
        predictions=predictions,
        residuals=residuals,
        statuses=statuses,
        value_bands=value_bands,
    )
    assert len(cast(list[object], crossfit["folds"])) == 5
    methods = cast(dict[str, object], crossfit["methods"])
    assert set(methods) == set(METHODS)
    for method in METHODS:
        coverages = cast(dict[str, object], cast(dict[str, object], methods[method])["coverages"])
        assert set(coverages) == {str(level) for level in COVERAGE_LEVELS}
        for item in coverages.values():
            metrics = cast(dict[str, object], item)
            assert metrics["sample_count"] == len(features)
            assert len(cast(list[object], metrics["folds"])) == 5
            assert np.isfinite(cast(NDArray[np.float64], metrics["_lower"])).all()

    gates = _conditional_gate_results(crossfit)
    assert [gate["gate"] for gate in gates] == [
        "minimum_overall_coverage_gap",
        "minimum_status_coverage_gap",
        "maximum_average_width_ratio_vs_status",
        "maximum_90pct_high_price_coverage_regression_vs_status",
    ]

    selected = cast(
        dict[str, object],
        cast(dict[str, object], methods["vehicle_status"])["coverages"],
    )
    selected_intervals = {key: cast(dict[str, object], value) for key, value in selected.items()}
    slices = _selected_slice_diagnostics(
        features=features,
        target=target,
        predictions=predictions,
        intervals=selected_intervals,
    )
    assert set(slices) == {
        "actual_price_band",
        "mileage_band",
        "vehicle_age_band",
        "manufacturer",
    }
    assert all(cast(list[object], values) for values in slices.values())

    public = _public_crossfit(crossfit)
    serialized = canonical_calibration_report_json(public)
    assert '"_lower"' not in serialized
    assert len(report_sha256(public)) == 64


def test_interval_fitting_hierarchy_confidence_and_validation_guards() -> None:
    _, target, predictions, residuals, statuses, value_bands = _synthetic_calibration_population()
    fitted = _fit_coverage_calibration(
        residuals,
        statuses,
        value_bands,
        coverage=0.9,
    )
    assert fitted.global_radius_usd == _finite_sample_radius(residuals, 0.9)
    assert all(item.radius_usd is not None for item in fitted.status_radii)
    assert all(item.radius_usd is not None for item in fitted.status_value_band_radii)

    applied = _apply_intervals(
        predictions,
        statuses,
        value_bands,
        fitted,
        method="vehicle_status_and_predicted_value_band_hierarchy",
        global_support=len(residuals),
    )
    assert np.all(applied.lower >= 0.0)
    assert np.all(applied.upper >= predictions)
    metrics = _interval_metrics(target, applied.lower, applied.upper, 0.9)
    assert metrics["sample_count"] == len(target)

    relative_widths = (applied.upper - applied.lower) / np.maximum(predictions, 1.0)
    confidence = _confidence_diagnostics(
        relative_widths,
        applied.supports,
        ConfidenceThresholds(
            coverage=0.9,
            high_max_relative_width=float(np.quantile(relative_widths, 0.33)),
            moderate_max_relative_width=float(np.quantile(relative_widths, 0.67)),
        ),
    )
    counts = cast(dict[str, int], confidence["counts"])
    assert sum(counts.values()) == len(predictions)

    with pytest.raises(CalibrationExperimentError, match="unknown calibration method"):
        _apply_intervals(
            predictions[:1],
            statuses[:1],
            value_bands[:1],
            fitted,
            method="unknown",
            global_support=len(residuals),
        )
    with pytest.raises(CalibrationExperimentError, match="non-empty finite vector"):
        _finite_sample_radius(np.asarray([], dtype=np.float64), 0.9)
    with pytest.raises(CalibrationExperimentError, match="aligned non-empty vectors"):
        _interval_metrics(target[:1], applied.lower[:0], applied.upper[:1], 0.9)
    with pytest.raises(CalibrationExperimentError, match="JSON-safe"):
        canonical_calibration_report_json({"invalid": np.nan})


def test_status_normalization_and_selected_method_coverage_gates_fail_closed() -> None:
    normalized = _normalized_statuses(pd.Series([" Used ", "NEW", "certified"]))
    assert normalized.tolist() == ["used", "new", "certified"]
    with pytest.raises(CalibrationExperimentError, match="one-dimensional"):
        _normalized_statuses(np.asarray([["used"]], dtype=object))
    with pytest.raises(CalibrationExperimentError, match="unsupported vehicle status"):
        _normalized_statuses(["salvage"])

    passing = {
        str(level): {
            "coverage_gap": 0.0,
            "status": {status: {"coverage_gap": 0.0} for status in ("certified", "new", "used")},
        }
        for level in COVERAGE_LEVELS
    }
    assert _selected_method_validated(passing)
    passing["0.9"]["coverage_gap"] = -0.03
    assert not _selected_method_validated(passing)
    passing["0.9"]["coverage_gap"] = 0.0
    status_metrics = cast(dict[str, dict[str, float]], passing["0.8"]["status"])
    status_metrics["used"]["coverage_gap"] = -0.06
    assert not _selected_method_validated(passing)
