from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import cast

import autovalue_ml.modeling.retail_uncertainty_sharpness as sharpness
import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import COVERAGE_LEVELS
from autovalue_ml.modeling.metrics import regression_metrics
from autovalue_ml.modeling.uncertainty_sharpness_policy import (
    UncertaintySharpnessPolicy,
    load_uncertainty_sharpness_policy_file,
)
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-policy-v1.json"
)


@pytest.fixture(scope="module")
def frozen_policy() -> UncertaintySharpnessPolicy:
    return load_uncertainty_sharpness_policy_file(POLICY_PATH)


class RecordingScaleEstimator:
    def __init__(self, prediction: float = 2_000.0) -> None:
        self.prediction = prediction
        self.fit_features: pd.DataFrame | None = None
        self.fit_target: NDArray[np.float64] | None = None
        self.predict_row_counts: list[int] = []

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
    ) -> RecordingScaleEstimator:
        self.fit_features = features.copy(deep=True)
        self.fit_target = target.copy()
        return self

    def predict(self, features: pd.DataFrame) -> object:
        self.predict_row_counts.append(len(features))
        return np.full(len(features), self.prediction, dtype=np.float64)


def _features(row_count: int) -> pd.DataFrame:
    makes = ("gmc", "genesis", "bmw", "audi", "mercedes")
    statuses = ("certified", "new", "used")
    years = (2023, 2018, 2012, 2000)
    return pd.DataFrame(
        {
            "year": [years[position % len(years)] for position in range(row_count)],
            "make": [makes[position % len(makes)] for position in range(row_count)],
            "model": [f"model-{position % 12}" for position in range(row_count)],
            "vehicle_status": [statuses[position % len(statuses)] for position in range(row_count)],
            "mileage": np.arange(row_count, dtype=np.float64) * 250.0,
        }
    )


def _point_and_target(row_count: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    point = np.linspace(20_000.0, 80_000.0, row_count, dtype=np.float64)
    residual = 400.0 + (np.arange(row_count, dtype=np.float64) % 17.0) * 75.0
    direction = np.where(np.arange(row_count) % 2 == 0, 1.0, -1.0)
    return point, np.asarray(point + residual * direction, dtype=np.float64)


def _intervals(
    predictions: NDArray[np.float64],
    radius: float | NDArray[np.float64],
    *,
    support: int = 1_000,
    fallback: bool = False,
) -> sharpness.IntervalArrays:
    radii = np.broadcast_to(np.asarray(radius, dtype=np.float64), predictions.shape).copy()
    unbounded = predictions - radii
    return sharpness.IntervalArrays(
        lower=np.maximum(0.0, unbounded),
        unbounded_lower=unbounded,
        upper=predictions + radii,
        radius=radii,
        support=np.full(len(predictions), support, dtype=np.int64),
        fallback=np.full(len(predictions), fallback, dtype=np.bool_),
    )


def _coverage_item(level: float, *, width: float, coverage: float) -> dict[str, object]:
    distribution = {
        "mean": width,
        "median": width,
        "p10": width,
        "p25": width,
        "p75": width,
        "p90": width,
        "p95": width,
        "maximum": width,
    }
    status = {name: {"empirical_coverage": coverage} for name in ("certified", "new", "used")}
    folds = [{"fold_number": fold, "empirical_coverage": coverage} for fold in range(1, 6)]
    return {
        "empirical_coverage": coverage,
        "coverage_gap": coverage - level,
        "displayed_width_usd": distribution,
        "unclipped_symmetric_width_usd": distribution,
        "invalid_or_nonfinite_interval_count": 0,
        "negative_displayed_lower_bound_count": 0,
        "point_exclusion_or_reversed_count": 0,
        "clipped_and_unclipped_coverage_equal": True,
        "status": status,
        "folds": folds,
        "fold_coverage_standard_deviation": 0.0,
        "fallback_rate": 0.0,
    }


def _slice_item(label: str, coverage: float = 0.9) -> dict[str, object]:
    return {
        "label": label,
        "coverages": {str(level): {"empirical_coverage": coverage} for level in COVERAGE_LEVELS},
    }


def _fake_evaluation(
    method: sharpness.MethodId,
    *,
    width: float,
    coverage: float = 0.9,
) -> sharpness.MethodEvaluation:
    reports = {
        str(level): _coverage_item(level, width=width, coverage=level) for level in COVERAGE_LEVELS
    }
    broad = {
        "actual_price_band": [_slice_item("price_4", coverage)],
        "predicted_value_band": [_slice_item("predicted_value_4", coverage)],
        "mileage_band": [_slice_item("mileage_1", coverage)],
        "vehicle_age_band": [_slice_item("age_1", coverage)],
        "manufacturer": [
            _slice_item(name, coverage) for name in ("gmc", "genesis", "bmw", "audi", "mercedes")
        ],
    }
    arrays = {
        str(level): _intervals(np.asarray([50_000.0]), width / 2.0) for level in COVERAGE_LEVELS
    }
    return sharpness.MethodEvaluation(
        method_id=method,
        coverages=arrays,
        report={"coverages": reports, "slices": broad, "confidence": {}},
    )


def _passing_bootstrap(width_ratio: float = 0.8) -> dict[str, object]:
    return {
        "coverage": {
            str(level): {"coverage_delta_vs_baseline_95pct_ci": {"lower": 0.0, "upper": 0.0}}
            for level in COVERAGE_LEVELS
        },
        "mean_unclipped_width_ratio_90pct_95pct_ci": {
            "lower": width_ratio,
            "upper": width_ratio,
        },
    }


def _v1_report(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    baseline: sharpness.MethodEvaluation,
    cutpoints: tuple[float, float, float],
) -> dict[str, object]:
    coverages = cast(Mapping[str, object], baseline.report["coverages"])
    expected: dict[str, object] = {}
    for level in COVERAGE_LEVELS:
        item = cast(Mapping[str, object], coverages[str(level)])
        status = cast(Mapping[str, object], item["status"])
        folds = cast(list[Mapping[str, object]], item["folds"])
        expected[str(level)] = {
            **_v1_metric_block(item),
            "fold_coverage_standard_deviation": item["fold_coverage_standard_deviation"],
            "status": {
                name: _v1_metric_block(cast(Mapping[str, object], status[name]))
                for name in ("certified", "new", "used")
            },
            "folds": [
                {"fold_number": fold["fold_number"], **_v1_metric_block(fold)} for fold in folds
            ],
        }
    return {
        "report_type": "retail_rf05_split_conformal_calibration_report",
        "classification": "validated_for_calibrated_prediction_intervals",
        "decision": {"selected_method": "vehicle_status"},
        "point_prediction_metrics_on_calibration": {
            "overall": regression_metrics(target, predictions).to_dict()
        },
        "cross_calibration": {"methods": {"vehicle_status": {"coverages": expected}}},
        "target_free_predicted_value_cutpoints_usd": list(cutpoints),
    }


def _v1_metric_block(item: Mapping[str, object]) -> dict[str, object]:
    displayed = cast(Mapping[str, float], item["displayed_width_usd"])
    return {
        "sample_count": item["sample_count"],
        "nominal_coverage": item["nominal_coverage"],
        "empirical_coverage": item["empirical_coverage"],
        "coverage_gap": item["coverage_gap"],
        "undercoverage_rate": item["undercoverage_rate"],
        "overcoverage_rate": item["overcoverage_rate"],
        "average_width_usd": displayed["mean"],
        "median_width_usd": displayed["median"],
        "width_percentiles_usd": {
            percentile: displayed[f"p{percentile}"] for percentile in ("10", "25", "75", "90", "95")
        },
    }


def test_finite_sample_quantile_is_deterministic_and_uses_ceiling_order() -> None:
    scores = np.asarray([9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0])

    first = sharpness._finite_sample_quantile(scores, 0.8)
    second = sharpness._finite_sample_quantile(scores[::-1], 0.8)

    assert first == second == 9.0  # ceil((10 + 1) * .8) = ninth order statistic
    with pytest.raises(sharpness.UncertaintySharpnessError, match="nonnegative"):
        sharpness._finite_sample_quantile(np.asarray([-1.0, 2.0]), 0.5)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="too small"):
        sharpness._finite_sample_quantile(np.asarray([1.0]), 0.95)


@pytest.mark.parametrize("coverage", [0.0, -0.1, 1.0, 1.1, float("nan"), float("inf"), True])
def test_finite_sample_quantile_rejects_invalid_coverage(coverage: float) -> None:
    with pytest.raises(sharpness.UncertaintySharpnessError, match="strictly between"):
        sharpness._finite_sample_quantile(np.asarray([1.0, 2.0]), coverage)


def test_quantiles_and_intervals_use_status_then_global_fallback_and_clip_zero() -> None:
    supported = np.full(450, "used", dtype="<U12")
    unsupported = np.full(20, "new", dtype="<U12")
    statuses = np.concatenate((supported, unsupported))
    scores = np.arange(1, 471, dtype=np.float64)
    fitted = sharpness._fit_quantiles(scores, statuses, 0.8)
    predictions = np.asarray([100.0, 1_000.0])
    scales = np.asarray([2.0, 1.0])

    intervals = sharpness._apply_scaled_intervals(
        predictions,
        scales,
        np.asarray(["new", "used"], dtype=np.str_),
        fitted,
    )

    status = cast(Mapping[str, object], fitted["status"])
    assert cast(Mapping[str, object], status["new"])["quantile"] is None
    assert cast(Mapping[str, object], status["used"])["quantile"] is not None
    assert intervals.fallback.tolist() == [True, False]
    assert intervals.support.tolist() == [470, 450]
    assert intervals.lower[0] == 0.0
    assert intervals.unbounded_lower[0] < 0.0
    assert np.all(intervals.upper >= predictions)


def test_interval_metrics_report_clipping_widths_fallback_and_validity() -> None:
    predictions = np.asarray([100.0, 1_000.0, 2_000.0])
    target = np.asarray([50.0, 950.0, 2_050.0])
    intervals = _intervals(
        predictions,
        np.asarray([200.0, 100.0, 100.0]),
        fallback=True,
    )

    metrics = sharpness._interval_metrics(target, predictions, intervals, 0.9)

    displayed = cast(Mapping[str, float], metrics["displayed_width_usd"])
    symmetric = cast(Mapping[str, float], metrics["unclipped_symmetric_width_usd"])
    assert metrics["empirical_coverage"] == 1.0
    assert metrics["zero_lower_bound_clipping_count"] == 1
    assert metrics["clipped_and_unclipped_coverage_equal"] is True
    assert metrics["fallback_count"] == 3
    assert displayed["mean"] < symmetric["mean"]
    assert metrics["invalid_or_nonfinite_interval_count"] == 0
    assert metrics["negative_displayed_lower_bound_count"] == 0
    assert metrics["point_exclusion_or_reversed_count"] == 0


def test_scale_fit_uses_only_development_oof_residual_target() -> None:
    features = _features(30)
    predictions, target = _point_and_target(30)
    recorder = RecordingScaleEstimator()

    fitted = sharpness.fit_gamma_residual_scale(
        development_features=features,
        development_target=target,
        development_oof_predictions=predictions,
        estimator_factory=lambda: recorder,
    )

    assert fitted is recorder
    assert recorder.fit_features is not None
    assert recorder.fit_target is not None
    assert tuple(recorder.fit_features.columns) == (
        "rf05_log_value",
        "model_year",
        "mileage",
        "mileage_per_year",
        "mileage_missing",
        "make",
        "model",
        "vehicle_status",
    )
    np.testing.assert_array_equal(recorder.fit_target, np.maximum(np.abs(target - predictions), 1))
    assert recorder.predict_row_counts == [30]
    assert not any("price" in column for column in recorder.fit_features.columns)


def test_real_gamma_and_smooth_scales_are_positive_and_deterministic() -> None:
    features = _features(90)
    predictions, target = _point_and_target(90)
    model = sharpness.fit_gamma_residual_scale(
        development_features=features,
        development_target=target,
        development_oof_predictions=predictions,
    )

    first = sharpness._predict_gamma_scale(model, features, predictions)
    second = sharpness._predict_gamma_scale(model, features, predictions)
    smooth_first = sharpness.smooth_value_scale(predictions)
    smooth_second = sharpness.smooth_value_scale(predictions.copy())

    assert np.all(first.raw > 0.0)
    assert np.all(first.clipped >= sharpness.GAMMA_SCALE_FLOOR_USD)
    assert np.all(first.clipped <= sharpness.GAMMA_SCALE_CAP_USD)
    np.testing.assert_array_equal(first.raw, second.raw)
    np.testing.assert_array_equal(first.clipped, second.clipped)
    np.testing.assert_array_equal(smooth_first, smooth_second)
    assert smooth_first[0] < smooth_first[-1]


def test_invalid_prediction_and_scale_vectors_fail_closed() -> None:
    with pytest.raises(sharpness.UncertaintySharpnessError, match="one-dimensional"):
        sharpness.smooth_value_scale(1.0)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="not boolean"):
        sharpness.smooth_value_scale([True])
    with pytest.raises(sharpness.UncertaintySharpnessError, match="nonnegative"):
        sharpness.smooth_value_scale([-1.0])
    with pytest.raises(sharpness.UncertaintySharpnessError, match="strictly positive"):
        sharpness._positive_scale_vector([0.0], expected_rows=1)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="row match"):
        sharpness._prediction_vector([1.0, 2.0], expected_rows=1, label="test")


def test_slice_and_confidence_diagnostics_respect_support_and_width() -> None:
    row_count = 400
    features = _features(row_count)
    features.loc[:198, "make"] = "under-supported"
    predictions, target = _point_and_target(row_count)
    intervals = {str(level): _intervals(predictions, 4_000.0) for level in COVERAGE_LEVELS}
    slices = sharpness._slice_diagnostics(
        features,
        target,
        predictions,
        intervals,
        (35_000.0, 50_000.0, 65_000.0),
    )
    manufacturers = cast(list[Mapping[str, object]], slices["manufacturer"])
    assert "under-supported" not in {item["label"] for item in manufacturers}

    confidence_predictions = np.asarray([10_000.0, 10_000.0, 10_000.0])
    confidence_intervals = _intervals(
        confidence_predictions,
        np.asarray([2_000.0, 4_500.0, 7_000.0]),
    )
    confidence_intervals.support[:] = np.asarray([1_000, 400, 399])
    confidence = sharpness._confidence_diagnostics(
        np.asarray([10_000.0, 10_000.0, 10_000.0]),
        confidence_predictions,
        confidence_intervals,
    )
    labels = cast(list[Mapping[str, object]], confidence["labels"])
    assert [item["label"] for item in labels] == [
        "High confidence",
        "Moderate confidence",
        "Low confidence",
    ]
    assert confidence["data_quality_warnings_are_separate"] is True


def test_cluster_bootstrap_and_seed_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sharpness, "BOOTSTRAP_REPLICATES", 50)
    values = np.asarray([1.0, 2.0, -1.0, 4.0])
    groups = np.asarray([0, 0, 1, 2], dtype=np.int64)

    first = sharpness._cluster_bootstrap_mean(values, groups, seed=91)
    second = sharpness._cluster_bootstrap_mean(values, groups, seed=91)
    ratio = sharpness._cluster_bootstrap_ratio(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([2.0, 4.0, 6.0, 8.0]),
        groups,
        seed=91,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(ratio, 0.5)
    assert sharpness._confidence_interval(first) == sharpness._confidence_interval(second)
    assert sharpness._method_seed(sharpness.GAMMA_METHOD, 0.9, "coverage") == (
        sharpness._method_seed(sharpness.GAMMA_METHOD, 0.9, "coverage")
    )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="remain positive"):
        sharpness._cluster_bootstrap_ratio(
            np.ones(4),
            np.zeros(4),
            groups,
            seed=91,
        )


def test_candidate_gates_and_selection_apply_preregistered_all_gate_rule() -> None:
    baseline = _fake_evaluation(sharpness.BASELINE_METHOD, width=100.0)
    gamma = _fake_evaluation(sharpness.GAMMA_METHOD, width=70.0)
    smooth = _fake_evaluation(sharpness.SMOOTH_METHOD, width=75.0)

    gamma_gates = sharpness._candidate_gates(
        method=sharpness.GAMMA_METHOD,
        evaluation=gamma,
        baseline=baseline,
        bootstrap=_passing_bootstrap(0.7),
        gamma_scale=sharpness.ScalePrediction(
            raw=np.asarray([500.0, 250_000.0]),
            clipped=np.asarray([500.0, 250_000.0]),
        ),
    )
    smooth_gates = sharpness._candidate_gates(
        method=sharpness.SMOOTH_METHOD,
        evaluation=smooth,
        baseline=baseline,
        bootstrap=_passing_bootstrap(0.75),
        gamma_scale=None,
    )
    evaluations = {
        sharpness.BASELINE_METHOD: baseline,
        sharpness.GAMMA_METHOD: gamma,
        sharpness.SMOOTH_METHOD: smooth,
    }
    passing = {
        sharpness.GAMMA_METHOD: gamma_gates,
        sharpness.SMOOTH_METHOD: smooth_gates,
    }

    assert gamma_gates["passed_all"] is True
    assert smooth_gates["passed_all"] is True
    gamma_outcomes = cast(list[Mapping[str, object]], gamma_gates["outcomes"])
    negative_lower_gates = [
        outcome
        for outcome in gamma_outcomes
        if str(outcome["gate"]).startswith("negative_displayed_lower_bound_count_")
    ]
    assert len(negative_lower_gates) == len(COVERAGE_LEVELS)
    assert all(
        outcome["observed"] == 0 and outcome["passed"] is True for outcome in negative_lower_gates
    )
    assert sharpness._select_method(evaluations, passing) == sharpness.GAMMA_METHOD
    passing[sharpness.GAMMA_METHOD] = {"passed_all": False}
    assert sharpness._select_method(evaluations, passing) == sharpness.SMOOTH_METHOD
    passing[sharpness.SMOOTH_METHOD] = {"passed_all": False}
    assert sharpness._select_method(evaluations, passing) == sharpness.BASELINE_METHOD


def test_selection_prefers_simple_method_when_gamma_gain_is_under_three_points() -> None:
    baseline = _fake_evaluation(sharpness.BASELINE_METHOD, width=100.0)
    gamma = _fake_evaluation(sharpness.GAMMA_METHOD, width=73.0)
    smooth = _fake_evaluation(sharpness.SMOOTH_METHOD, width=75.0)
    evaluations = {
        sharpness.BASELINE_METHOD: baseline,
        sharpness.GAMMA_METHOD: gamma,
        sharpness.SMOOTH_METHOD: smooth,
    }
    gates = {
        sharpness.GAMMA_METHOD: {"passed_all": True},
        sharpness.SMOOTH_METHOD: {"passed_all": True},
    }

    assert sharpness._select_method(evaluations, gates) == sharpness.SMOOTH_METHOD


def test_both_pass_tie_break_requires_gamma_to_have_no_worse_common_gate() -> None:
    baseline = _fake_evaluation(sharpness.BASELINE_METHOD, width=100.0)
    gamma = _fake_evaluation(sharpness.GAMMA_METHOD, width=72.0)
    smooth = _fake_evaluation(sharpness.SMOOTH_METHOD, width=75.0)
    gamma_gates = sharpness._candidate_gates(
        method=sharpness.GAMMA_METHOD,
        evaluation=gamma,
        baseline=baseline,
        bootstrap=_passing_bootstrap(0.72),
        gamma_scale=sharpness.ScalePrediction(
            raw=np.asarray([500.0, 250_000.0]),
            clipped=np.asarray([500.0, 250_000.0]),
        ),
    )
    smooth_gates = sharpness._candidate_gates(
        method=sharpness.SMOOTH_METHOD,
        evaluation=smooth,
        baseline=baseline,
        bootstrap=_passing_bootstrap(0.75),
        gamma_scale=None,
    )
    evaluations = {
        sharpness.BASELINE_METHOD: baseline,
        sharpness.GAMMA_METHOD: gamma,
        sharpness.SMOOTH_METHOD: smooth,
    }
    gates = {
        sharpness.GAMMA_METHOD: gamma_gates,
        sharpness.SMOOTH_METHOD: smooth_gates,
    }

    assert sharpness._select_method(evaluations, gates) == sharpness.GAMMA_METHOD

    gamma_outcomes = cast(list[dict[str, object]], gamma_gates["outcomes"])
    coverage_gate = next(
        outcome for outcome in gamma_outcomes if outcome["gate"] == "overall_coverage_gap_0.8"
    )
    coverage_gate["observed"] = -0.005
    assert sharpness._select_method(evaluations, gates) == sharpness.SMOOTH_METHOD


def test_runtime_policy_binding_rejects_typed_policy_and_runtime_drift(
    frozen_policy: UncertaintySharpnessPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sharpness._validate_runtime_policy(frozen_policy)

    drifted_method = replace(
        frozen_policy,
        baseline_method=replace(frozen_policy.baseline_method, method_id="drifted"),
    )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="candidate methods"):
        sharpness._validate_runtime_policy(drifted_method)

    drifted_bootstrap = replace(
        frozen_policy,
        calibration_comparison=replace(
            frozen_policy.calibration_comparison,
            bootstrap=replace(
                frozen_policy.calibration_comparison.bootstrap,
                random_state=frozen_policy.calibration_comparison.bootstrap.random_state + 1,
            ),
        ),
    )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="bootstrap seed"):
        sharpness._validate_runtime_policy(drifted_bootstrap)

    drifted_gate = replace(
        frozen_policy,
        acceptance_gates=replace(
            frozen_policy.acceptance_gates,
            stability=replace(
                frozen_policy.acceptance_gates.stability,
                maximum_interval_width_usd=749_999.0,
            ),
        ),
    )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="maximum width gate"):
        sharpness._validate_runtime_policy(drifted_gate)

    drifted_confidence = replace(
        frozen_policy,
        confidence_policy=replace(frozen_policy.confidence_policy, coverage_level=0.8),
    )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="confidence coverage"):
        sharpness._validate_runtime_policy(drifted_confidence)

    monkeypatch.setattr(sharpness, "BOOTSTRAP_SEED", sharpness.BOOTSTRAP_SEED + 1)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="bootstrap seed"):
        sharpness._validate_runtime_policy(frozen_policy)


def test_baseline_reproduction_rejects_overall_fold_and_status_drift() -> None:
    row_count = 600
    features = _features(row_count)
    predictions, target = _point_and_target(row_count)
    cutpoints = (35_000.0, 50_000.0, 65_000.0)
    scales: dict[sharpness.MethodId, NDArray[np.float64]] = {
        method: np.ones(row_count, dtype=np.float64) for method in sharpness.METHODS
    }
    baseline = sharpness._crossfit_methods(
        features=features,
        target=target,
        predictions=predictions,
        scales=scales,
        predicted_value_cutpoints=cutpoints,
    )[sharpness.BASELINE_METHOD]
    report = _v1_report(target, predictions, baseline, cutpoints)
    sharpness._validate_baseline_reproduction(baseline, report)

    overall_drift = copy.deepcopy(report)
    overall = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(dict[str, object], overall_drift["cross_calibration"])["methods"],
        )["vehicle_status"],
    )
    overall_coverage = cast(dict[str, object], cast(dict[str, object], overall["coverages"])["0.9"])
    percentiles = cast(dict[str, object], overall_coverage["width_percentiles_usd"])
    percentiles["90"] = cast(float, percentiles["90"]) + 1.0
    with pytest.raises(sharpness.UncertaintySharpnessError, match="90 differs at overall 0.9"):
        sharpness._validate_baseline_reproduction(baseline, overall_drift)

    fold_drift = copy.deepcopy(report)
    fold_coverages = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(
                dict[str, object],
                cast(dict[str, object], fold_drift["cross_calibration"])["methods"],
            )["vehicle_status"],
        )["coverages"],
    )
    folds = cast(list[dict[str, object]], cast(dict[str, object], fold_coverages["0.8"])["folds"])
    folds[0]["sample_count"] = cast(int, folds[0]["sample_count"]) + 1
    with pytest.raises(sharpness.UncertaintySharpnessError, match="sample_count differs at fold"):
        sharpness._validate_baseline_reproduction(baseline, fold_drift)

    status_drift = copy.deepcopy(report)
    status_coverages = cast(
        dict[str, object],
        cast(
            dict[str, object],
            cast(
                dict[str, object],
                cast(dict[str, object], status_drift["cross_calibration"])["methods"],
            )["vehicle_status"],
        )["coverages"],
    )
    statuses = cast(dict[str, object], cast(dict[str, object], status_coverages["0.95"])["status"])
    used = cast(dict[str, object], statuses["used"])
    used["undercoverage_rate"] = cast(float, used["undercoverage_rate"]) + 0.01
    with pytest.raises(sharpness.UncertaintySharpnessError, match="undercoverage_rate differs"):
        sharpness._validate_baseline_reproduction(baseline, status_drift)


def test_comparison_runs_paired_and_emits_aggregate_row_free_report(
    frozen_policy: UncertaintySharpnessPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development_count = 120
    calibration_count = 600
    monkeypatch.setattr(sharpness, "DEVELOPMENT_SAMPLE_COUNT", development_count)
    monkeypatch.setattr(sharpness, "CALIBRATION_SAMPLE_COUNT", calibration_count)
    monkeypatch.setattr(sharpness, "MINIMUM_BUCKET_SUPPORT", 50)
    monkeypatch.setattr(sharpness, "MINIMUM_SLICE_SUPPORT", 50)
    monkeypatch.setattr(sharpness, "BOOTSTRAP_REPLICATES", 20)
    monkeypatch.setattr(sharpness, "_validate_runtime_policy", lambda policy: None)
    development = _features(development_count)
    development_predictions, development_target = _point_and_target(development_count)
    calibration = _features(calibration_count)
    calibration_predictions, calibration_target = _point_and_target(calibration_count)
    cutpoints = (35_000.0, 50_000.0, 65_000.0)
    scales: dict[sharpness.MethodId, NDArray[np.float64]] = {
        sharpness.BASELINE_METHOD: np.ones(calibration_count),
        sharpness.GAMMA_METHOD: np.full(calibration_count, 2_000.0),
        sharpness.SMOOTH_METHOD: sharpness.smooth_value_scale(calibration_predictions),
    }
    baseline = sharpness._crossfit_methods(
        features=calibration,
        target=calibration_target,
        predictions=calibration_predictions,
        scales=scales,
        predicted_value_cutpoints=cutpoints,
    )[sharpness.BASELINE_METHOD]
    v1 = _v1_report(calibration_target, calibration_predictions, baseline, cutpoints)
    recorder = RecordingScaleEstimator()

    result = sharpness.compare_uncertainty_methods(
        policy=frozen_policy,
        development_features=development,
        development_target=development_target,
        development_oof_predictions=development_predictions,
        calibration_features=calibration,
        calibration_target=calibration_target,
        calibration_predictions=calibration_predictions,
        calibration_v1_report=v1,
        gamma_estimator_factory=lambda: recorder,
    )

    assert recorder.fit_target is not None
    np.testing.assert_array_equal(
        recorder.fit_target,
        np.maximum(np.abs(development_target - development_predictions), 1.0),
    )
    assert recorder.predict_row_counts == [development_count, calibration_count]
    assert tuple(cast(Mapping[str, object], result.report["methods"])) == sharpness.METHODS
    boundaries = cast(Mapping[str, object], result.report["data_boundaries"])
    assert boundaries["calibration_targets_used_to_fit_gamma_scale"] is False
    assert boundaries["legacy_holdout_accessed_by_modeling_code"] is False
    publication = cast(Mapping[str, object], result.report["publication"])
    assert publication["aggregate_only"] is True
    assert publication["raw_rows_predictions_residuals_or_category_vocabularies_in_report"] is (
        False
    )
    serialized = sharpness.canonical_sharpness_report_json(result.report)
    assert '"calibration_target"' not in serialized
    assert '"development_target"' not in serialized
    assert sharpness.sharpness_report_sha256(result.report) == (
        sharpness.sharpness_report_sha256(result.report)
    )


def test_comparison_and_frozen_baseline_validation_fail_closed(
    frozen_policy: UncertaintySharpnessPolicy,
) -> None:
    features = _features(3)
    predictions, target = _point_and_target(3)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="frozen row boundaries"):
        sharpness.compare_uncertainty_methods(
            policy=frozen_policy,
            development_features=features,
            development_target=target,
            development_oof_predictions=predictions,
            calibration_features=features,
            calibration_target=target,
            calibration_predictions=predictions,
            calibration_v1_report={},
            gamma_estimator_factory=RecordingScaleEstimator,
        )
    with pytest.raises(sharpness.UncertaintySharpnessError, match="report type"):
        sharpness._validate_calibration_v1_report({"report_type": "wrong"})
    with pytest.raises(sharpness.UncertaintySharpnessError, match="decision is invalid"):
        sharpness._validate_calibration_v1_report(
            {
                "report_type": "retail_rf05_split_conformal_calibration_report",
                "classification": "validated_for_calibrated_prediction_intervals",
            }
        )
    good_metrics = {
        "point_prediction_metrics_on_calibration": {
            "overall": regression_metrics(target, predictions + 1_000.0).to_dict()
        }
    }
    with pytest.raises(sharpness.UncertaintySharpnessError, match="differs from v1"):
        sharpness._validate_point_metrics(target, predictions, good_metrics)
    with pytest.raises(sharpness.UncertaintySharpnessError, match="cutpoints"):
        sharpness._v1_prediction_cutpoints(
            {"target_free_predicted_value_cutpoints_usd": [3.0, 2.0, 1.0]}
        )


def test_canonical_report_rejects_non_json_numbers() -> None:
    first = sharpness.canonical_sharpness_report_json({"z": 1, "a": [2, 3]})
    second = sharpness.canonical_sharpness_report_json({"a": [2, 3], "z": 1})

    assert first == second == '{"a":[2,3],"z":1}\n'
    with pytest.raises(sharpness.UncertaintySharpnessError, match="not JSON-safe"):
        sharpness.canonical_sharpness_report_json({"bad": float("nan")})
