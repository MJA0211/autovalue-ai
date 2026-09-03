"""Preregistered heteroscedastic conformal comparison for frozen retail RF05."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import RETAIL_VEHICLE_STATUSES
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_SAMPLE_COUNT,
    COVERAGE_LEVELS,
    DEVELOPMENT_SAMPLE_COUNT,
    MINIMUM_BUCKET_SUPPORT,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
    ConfidenceThresholds,
    active_rf05_identity,
)
from .contracts import RETAIL_TRACK, validate_feature_frame, validate_target
from .cv import retail_group_cv_splits, retail_predictor_groups
from .feature_engineering import VehicleFeatureEngineer
from .metrics import regression_metrics
from .retail_calibration_experiment import (
    AGE_CUTPOINTS,
    MILEAGE_CUTPOINTS,
    PRICE_CUTPOINTS,
    _mileage_bands,
    _normalized_statuses,
    _numeric_bands,
)
from .uncertainty_sharpness_policy import UncertaintySharpnessPolicy

MethodId: TypeAlias = Literal[
    "vehicle_status_absolute_residual_v1",
    "normalized_gamma_scale_v1",
    "normalized_smooth_value_scale_v1",
]

METHODS: Final[tuple[MethodId, ...]] = (
    "vehicle_status_absolute_residual_v1",
    "normalized_gamma_scale_v1",
    "normalized_smooth_value_scale_v1",
)
BASELINE_METHOD: Final[MethodId] = "vehicle_status_absolute_residual_v1"
GAMMA_METHOD: Final[MethodId] = "normalized_gamma_scale_v1"
SMOOTH_METHOD: Final[MethodId] = "normalized_smooth_value_scale_v1"
SHARPNESS_POLICY_SHA256: Final = "ec1787be963a907bbae2d1d521aeaef4239b8a5bf7816ced844dcd16902f1058"
DEVELOPMENT_DIAGNOSTICS_SHA256: Final = (
    "8f79ac027a72fff2512ab0b168d91a3a7b46677d72374dc00571a4646aac925d"
)
CALIBRATION_V1_POLICY_SHA256: Final = (
    "1398519c699bd129ef4fbb552813c064839c6c1e1c4ecd35c7f5d42bcf8e1ca2"
)
CALIBRATION_V1_ARTIFACT_SHA256: Final = (
    "b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0"
)
CALIBRATION_V1_REPORT_SHA256: Final = (
    "e7fafff505603669e73cfbff2fe1cf5e04f9c5d896666470fe212411aa1b3084"
)
REPORT_TYPE: Final = "retail_rf05_uncertainty_sharpness_comparison"
GENERATED_AT: Final = "2026-09-02T20:00:00+00:00"
SCALE_VERSION: Final = "retail-rf05-gamma-residual-scale-v1"
GAMMA_SCALE_FLOOR_USD: Final = 500.0
GAMMA_SCALE_CAP_USD: Final = 250_000.0
GAMMA_ALPHA: Final = 1.0
GAMMA_MAX_ITER: Final = 2_000
GAMMA_TOLERANCE: Final = 1e-7
PREDICTION_SCALE_REFERENCE_USD: Final = 10_000.0
MINIMUM_SLICE_SUPPORT: Final = 200
CALIBRATION_FOLD_COUNT: Final = 5
BOOTSTRAP_REPLICATES: Final = 2_000
BOOTSTRAP_SEED: Final = 2_841_907_531
BOOTSTRAP_CONFIDENCE_LEVEL: Final = 0.95
OVERALL_MINIMUM_COVERAGE_GAP: Final = -0.02
OVERALL_MAXIMUM_COVERAGE_REGRESSION: Final = 0.01
BOOTSTRAP_MINIMUM_COVERAGE_DELTA_LOWER: Final = -0.015
MEAN_WIDTH_REDUCTION_THRESHOLDS: Final[Mapping[str, float]] = {
    "0.8": 0.05,
    "0.9": 0.10,
    "0.95": 0.05,
}
MINIMUM_MEDIAN_WIDTH_REDUCTION: Final = 0.05
MAXIMUM_BOOTSTRAP_WIDTH_RATIO: Final = 0.95
MAXIMUM_P95_WIDTH_RATIO: Final = 1.75
MAXIMUM_P95_TO_MEDIAN_WIDTH_RATIO: Final = 6.0
MAXIMUM_INTERVAL_WIDTH_USD: Final = 750_000.0
MINIMUM_STATUS_COVERAGE_GAP: Final = -0.05
MAXIMUM_STATUS_COVERAGE_REGRESSION: Final = 0.02
MAXIMUM_FOLD_COVERAGE_REGRESSION: Final = 0.03
MAXIMUM_FOLD_COVERAGE_SD_INCREASE: Final = 0.01
MAXIMUM_FALLBACK_RATE_INCREASE: Final = 0.005
MAXIMUM_BROAD_SLICE_REGRESSION: Final = 0.03
BROAD_SLICE_UNDERCOVERAGE_BOUNDARY: Final = -0.10
MAXIMUM_MANUFACTURER_REGRESSION: Final = 0.05
MANUFACTURER_LOW_COVERAGE_BOUNDARY: Final = 0.80
MAXIMUM_FOCUS_SLICE_REGRESSION: Final = 0.01
GAMMA_SCALE_FLOOR_MAXIMUM_RATE: Final = 0.10
GAMMA_SCALE_CAP_MAXIMUM_RATE: Final = 0.01
REQUIRED_INVALID_COUNT: Final = 0
REQUIRE_CLIPPED_COVERAGE_MATCH: Final = True
SELECTION_GAMMA_EXTRA_WIDTH_REDUCTION: Final = 0.03
SELECTION_COVERAGE_LEVEL: Final = 0.9
FOCUS_SLICES: Final = ("highest_price_band", "gmc", "genesis", "bmw", "audi", "mercedes")
CONFIDENCE_THRESHOLDS: Final = ConfidenceThresholds(
    coverage=0.9,
    high_max_relative_width=0.686682300031913,
    moderate_max_relative_width=1.094478855653873,
)


class UncertaintySharpnessError(ValueError):
    """The preregistered comparison or a protected boundary was invalid."""


class ScaleEstimator(Protocol):
    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> ScaleEstimator: ...

    def predict(self, features: pd.DataFrame) -> object: ...


ScaleEstimatorFactory = Callable[[], ScaleEstimator]


@dataclass(frozen=True, slots=True)
class ScalePrediction:
    raw: NDArray[np.float64]
    clipped: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class IntervalArrays:
    lower: NDArray[np.float64]
    unbounded_lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    radius: NDArray[np.float64]
    support: NDArray[np.int64]
    fallback: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class MethodEvaluation:
    method_id: MethodId
    coverages: Mapping[str, IntervalArrays]
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SharpnessExperimentResult:
    report: Mapping[str, object]
    selected_method: MethodId
    gamma_scale_model: ScaleEstimator
    full_quantiles: Mapping[str, object]


def make_gamma_scale_pipeline() -> Pipeline:
    """Build the one frozen positive residual-scale candidate."""

    numeric = (
        "rf05_log_value",
        "model_year",
        "mileage",
        "mileage_per_year",
        "mileage_missing",
    )
    categorical = RETAIL_TRACK.categorical_features
    numeric_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True, copy=True),
            ),
            ("scaler", StandardScaler(with_mean=False, copy=True)),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                    copy=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=RETAIL_TRACK.one_hot_min_frequency,
                    max_categories=RETAIL_TRACK.one_hot_max_categories,
                    sparse_output=True,
                    dtype=np.float64,
                ),
            ),
        )
    )
    preprocessor = ColumnTransformer(
        transformers=(
            ("numeric", numeric_pipeline, list(numeric)),
            ("categorical", categorical_pipeline, list(categorical)),
        ),
        remainder="drop",
        sparse_threshold=1.0,
    )
    return Pipeline(
        steps=(
            ("preprocessor", preprocessor),
            (
                "regressor",
                GammaRegressor(
                    alpha=GAMMA_ALPHA,
                    fit_intercept=True,
                    solver="lbfgs",
                    max_iter=GAMMA_MAX_ITER,
                    tol=GAMMA_TOLERANCE,
                    warm_start=False,
                    verbose=0,
                ),
            ),
        )
    )


def fit_gamma_residual_scale(
    *,
    development_features: object,
    development_target: object,
    development_oof_predictions: object,
    estimator_factory: ScaleEstimatorFactory | None = None,
) -> ScaleEstimator:
    """Fit only development OOF residuals; calibration targets cannot enter this path."""

    features = validate_feature_frame(development_features, RETAIL_TRACK)
    target = validate_target(
        development_target,
        expected_rows=len(features),
        config=RETAIL_TRACK,
    )
    predictions = _prediction_vector(
        development_oof_predictions,
        expected_rows=len(features),
        label="development OOF predictions",
    )
    scale_features = build_scale_features(features, predictions)
    scale_target = np.maximum(np.abs(target - predictions), 1.0)
    factory = estimator_factory or _gamma_factory
    estimator = factory()
    estimator.fit(scale_features, scale_target)
    _predict_gamma_scale(estimator, features, predictions)
    return estimator


def build_scale_features(
    features: object,
    rf05_predictions: object,
) -> pd.DataFrame:
    """Create predictor-time-only features for the residual-scale model."""

    frame = validate_feature_frame(features, RETAIL_TRACK)
    predictions = _prediction_vector(
        rf05_predictions,
        expected_rows=len(frame),
        label="RF05 scale inputs",
    )
    engineered = VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(frame)
    result = engineered.copy(deep=True)
    result.insert(
        0,
        "rf05_log_value",
        np.log1p(predictions / PREDICTION_SCALE_REFERENCE_USD),
    )
    return result


def compare_uncertainty_methods(
    *,
    policy: UncertaintySharpnessPolicy,
    development_features: object,
    development_target: object,
    development_oof_predictions: object,
    calibration_features: object,
    calibration_target: object,
    calibration_predictions: object,
    calibration_v1_report: Mapping[str, object],
    gamma_estimator_factory: ScaleEstimatorFactory | None = None,
    generated_at: str = GENERATED_AT,
) -> SharpnessExperimentResult:
    """Execute the frozen three-arm comparison on the authorized boundaries."""

    _validate_runtime_policy(policy)
    development = validate_feature_frame(development_features, RETAIL_TRACK)
    development_y = validate_target(
        development_target,
        expected_rows=len(development),
        config=RETAIL_TRACK,
    )
    development_predictions = _prediction_vector(
        development_oof_predictions,
        expected_rows=len(development),
        label="development OOF predictions",
    )
    calibration = validate_feature_frame(calibration_features, RETAIL_TRACK)
    calibration_y = validate_target(
        calibration_target,
        expected_rows=len(calibration),
        config=RETAIL_TRACK,
    )
    point_predictions = _prediction_vector(
        calibration_predictions,
        expected_rows=len(calibration),
        label="calibration predictions",
    )
    if len(development) != DEVELOPMENT_SAMPLE_COUNT or len(calibration) != CALIBRATION_SAMPLE_COUNT:
        raise UncertaintySharpnessError("sharpness comparison requires the frozen row boundaries")
    _validate_calibration_v1_report(calibration_v1_report)
    _validate_point_metrics(calibration_y, point_predictions, calibration_v1_report)

    gamma_model = fit_gamma_residual_scale(
        development_features=development,
        development_target=development_y,
        development_oof_predictions=development_predictions,
        estimator_factory=gamma_estimator_factory,
    )
    gamma_scale = _predict_gamma_scale(gamma_model, calibration, point_predictions)
    scales: dict[MethodId, NDArray[np.float64]] = {
        BASELINE_METHOD: np.ones(len(calibration), dtype=np.float64),
        GAMMA_METHOD: gamma_scale.clipped,
        SMOOTH_METHOD: smooth_value_scale(point_predictions),
    }
    evaluations = _crossfit_methods(
        features=calibration,
        target=calibration_y,
        predictions=point_predictions,
        scales=scales,
        predicted_value_cutpoints=_v1_prediction_cutpoints(calibration_v1_report),
    )
    _validate_baseline_reproduction(evaluations[BASELINE_METHOD], calibration_v1_report)
    bootstraps = _bootstrap_comparisons(calibration, calibration_y, evaluations)
    gates = {
        method: _candidate_gates(
            method=method,
            evaluation=evaluations[method],
            baseline=evaluations[BASELINE_METHOD],
            bootstrap=bootstraps[method],
            gamma_scale=gamma_scale if method == GAMMA_METHOD else None,
        )
        for method in (GAMMA_METHOD, SMOOTH_METHOD)
    }
    selected_method = _select_method(evaluations, gates)
    full_quantiles = _fit_full_quantiles(
        calibration_y,
        point_predictions,
        _normalized_statuses(calibration["vehicle_status"]),
        scales[selected_method],
    )
    report = _build_report(
        development_target=development_y,
        development_predictions=development_predictions,
        calibration_target=calibration_y,
        calibration_predictions=point_predictions,
        evaluations=evaluations,
        bootstraps=bootstraps,
        gates=gates,
        selected_method=selected_method,
        gamma_model=gamma_model,
        gamma_scale=gamma_scale,
        full_quantiles=full_quantiles,
        generated_at=generated_at,
    )
    return SharpnessExperimentResult(
        report=report,
        selected_method=selected_method,
        gamma_scale_model=gamma_model,
        full_quantiles=full_quantiles,
    )


def smooth_value_scale(predictions: object) -> NDArray[np.float64]:
    inspected = np.asarray(predictions, dtype=object)
    if inspected.ndim != 1:
        raise UncertaintySharpnessError("point must be a one-dimensional row match")
    values = _prediction_vector(
        inspected,
        expected_rows=len(inspected),
        label="point",
    )
    scale = 1.0 + np.log1p(values / PREDICTION_SCALE_REFERENCE_USD)
    if not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise UncertaintySharpnessError("smooth predicted-value scale must be positive finite")
    return scale


def canonical_sharpness_report_json(report: Mapping[str, object]) -> str:
    try:
        return (
            json.dumps(
                report,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise UncertaintySharpnessError("sharpness report is not JSON-safe") from error


def sharpness_report_sha256(report: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_sharpness_report_json(report).encode("utf-8")).hexdigest()


def _gamma_factory() -> ScaleEstimator:
    return cast(ScaleEstimator, make_gamma_scale_pipeline())


def _predict_gamma_scale(
    estimator: ScaleEstimator,
    features: pd.DataFrame,
    predictions: NDArray[np.float64],
) -> ScalePrediction:
    raw = _positive_scale_vector(
        estimator.predict(build_scale_features(features, predictions)),
        expected_rows=len(features),
    )
    clipped = np.clip(raw, GAMMA_SCALE_FLOOR_USD, GAMMA_SCALE_CAP_USD)
    return ScalePrediction(raw=raw, clipped=clipped)


def _prediction_vector(values: object, *, expected_rows: int, label: str) -> NDArray[np.float64]:
    inspected = np.asarray(values, dtype=object)
    if inspected.ndim != 1 or len(inspected) != expected_rows:
        raise UncertaintySharpnessError(f"{label} must be a one-dimensional row match")
    if any(isinstance(value, (bool, np.bool_)) for value in inspected.tolist()):
        raise UncertaintySharpnessError(f"{label} must be numeric, not boolean")
    try:
        result = inspected.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise UncertaintySharpnessError(f"{label} must be numeric") from error
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise UncertaintySharpnessError(f"{label} must be finite and nonnegative")
    return result


def _positive_scale_vector(values: object, *, expected_rows: int) -> NDArray[np.float64]:
    inspected = np.asarray(values, dtype=object)
    if inspected.ndim != 1 or len(inspected) != expected_rows:
        raise UncertaintySharpnessError("Gamma scale predictions must be a one-dimensional match")
    try:
        result = inspected.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise UncertaintySharpnessError("Gamma scale predictions must be numeric") from error
    if not np.isfinite(result).all() or (result <= 0.0).any():
        raise UncertaintySharpnessError("Gamma scale predictions must be strictly positive finite")
    return result


def _crossfit_methods(
    *,
    features: pd.DataFrame,
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    scales: Mapping[MethodId, NDArray[np.float64]],
    predicted_value_cutpoints: tuple[float, float, float],
) -> dict[MethodId, MethodEvaluation]:
    statuses = _normalized_statuses(features["vehicle_status"])
    splits = retail_group_cv_splits(features, n_splits=CALIBRATION_FOLD_COUNT)
    stored: dict[MethodId, dict[str, IntervalArrays]] = {
        method: {
            str(level): IntervalArrays(
                lower=np.full(len(features), np.nan, dtype=np.float64),
                unbounded_lower=np.full(len(features), np.nan, dtype=np.float64),
                upper=np.full(len(features), np.nan, dtype=np.float64),
                radius=np.full(len(features), np.nan, dtype=np.float64),
                support=np.zeros(len(features), dtype=np.int64),
                fallback=np.zeros(len(features), dtype=np.bool_),
            )
            for level in COVERAGE_LEVELS
        }
        for method in METHODS
    }
    fold_reports: dict[MethodId, dict[str, list[dict[str, object]]]] = {
        method: {str(level): [] for level in COVERAGE_LEVELS} for method in METHODS
    }
    for fold_number, (training, validation) in enumerate(splits, start=1):
        for method in METHODS:
            method_scale = scales[method]
            _validate_scale(method_scale, expected_rows=len(features), method=method)
            scores = np.abs(target - predictions) / method_scale
            for level in COVERAGE_LEVELS:
                quantiles = _fit_quantiles(scores[training], statuses[training], level)
                applied = _apply_scaled_intervals(
                    predictions[validation],
                    method_scale[validation],
                    statuses[validation],
                    quantiles,
                )
                destination = stored[method][str(level)]
                _assign_interval_arrays(destination, validation, applied)
                fold_reports[method][str(level)].append(
                    {
                        "fold_number": fold_number,
                        **_interval_metrics(
                            target[validation],
                            predictions[validation],
                            applied,
                            level,
                        ),
                    }
                )
    evaluations: dict[MethodId, MethodEvaluation] = {}
    for method in METHODS:
        coverage_report: dict[str, object] = {}
        for level in COVERAGE_LEVELS:
            arrays = stored[method][str(level)]
            _validate_completed_intervals(arrays, predictions)
            overall = _interval_metrics(target, predictions, arrays, level)
            status_metrics = {
                status: _interval_metrics(
                    target[statuses == status],
                    predictions[statuses == status],
                    _masked_intervals(arrays, statuses == status),
                    level,
                )
                for status in RETAIL_VEHICLE_STATUSES
            }
            folds = fold_reports[method][str(level)]
            coverage_report[str(level)] = {
                **overall,
                "status": status_metrics,
                "folds": folds,
                "fold_coverage_standard_deviation": float(
                    np.std([cast(float, item["empirical_coverage"]) for item in folds])
                ),
            }
        slices = _slice_diagnostics(
            features,
            target,
            predictions,
            stored[method],
            predicted_value_cutpoints,
        )
        confidence = _confidence_diagnostics(
            target,
            predictions,
            stored[method][str(SELECTION_COVERAGE_LEVEL)],
        )
        evaluations[method] = MethodEvaluation(
            method_id=method,
            coverages=stored[method],
            report={
                "coverages": coverage_report,
                "slices": slices,
                "confidence": confidence,
            },
        )
    return evaluations


def _fit_quantiles(
    scores: NDArray[np.float64],
    statuses: NDArray[np.str_],
    coverage: float,
) -> dict[str, object]:
    global_quantile = _finite_sample_quantile(scores, coverage)
    status_quantiles: dict[str, dict[str, object]] = {}
    for status in RETAIL_VEHICLE_STATUSES:
        subset = scores[statuses == status]
        supported = len(subset) >= MINIMUM_BUCKET_SUPPORT and math.ceil(
            (len(subset) + 1) * coverage
        ) <= len(subset)
        status_quantiles[status] = {
            "support": len(subset),
            "quantile": _finite_sample_quantile(subset, coverage) if supported else None,
        }
    return {
        "coverage": coverage,
        "global_support": len(scores),
        "global_quantile": global_quantile,
        "status": status_quantiles,
    }


def _finite_sample_quantile(scores: NDArray[np.float64], coverage: float) -> float:
    if isinstance(coverage, bool) or not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise UncertaintySharpnessError("coverage must be finite and strictly between zero and one")
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise UncertaintySharpnessError("conformal scores must be a non-empty finite vector")
    if (scores < 0.0).any():
        raise UncertaintySharpnessError("conformal scores must be nonnegative")
    order = math.ceil((len(scores) + 1) * coverage)
    if order > len(scores):
        raise UncertaintySharpnessError("score bucket is too small for requested coverage")
    return float(np.partition(scores.copy(), order - 1)[order - 1])


def _apply_scaled_intervals(
    predictions: NDArray[np.float64],
    scales: NDArray[np.float64],
    statuses: NDArray[np.str_],
    quantiles: Mapping[str, object],
) -> IntervalArrays:
    global_quantile = cast(float, quantiles["global_quantile"])
    global_support = cast(int, quantiles["global_support"])
    status_quantiles = cast(Mapping[str, object], quantiles["status"])
    radius = np.empty(len(predictions), dtype=np.float64)
    support = np.empty(len(predictions), dtype=np.int64)
    fallback = np.zeros(len(predictions), dtype=np.bool_)
    for position, status in enumerate(statuses):
        entry = cast(Mapping[str, object], status_quantiles[str(status)])
        value = entry["quantile"]
        if value is None:
            quantile = global_quantile
            support[position] = global_support
            fallback[position] = True
        else:
            quantile = cast(float, value)
            support[position] = cast(int, entry["support"])
        radius[position] = quantile * scales[position]
    unbounded_lower = predictions - radius
    lower = np.maximum(0.0, unbounded_lower)
    upper = predictions + radius
    return IntervalArrays(
        lower=lower,
        unbounded_lower=unbounded_lower,
        upper=upper,
        radius=radius,
        support=support,
        fallback=fallback,
    )


def _assign_interval_arrays(
    destination: IntervalArrays,
    indices: NDArray[np.int64],
    values: IntervalArrays,
) -> None:
    destination.lower[indices] = values.lower
    destination.unbounded_lower[indices] = values.unbounded_lower
    destination.upper[indices] = values.upper
    destination.radius[indices] = values.radius
    destination.support[indices] = values.support
    destination.fallback[indices] = values.fallback


def _masked_intervals(arrays: IntervalArrays, mask: NDArray[np.bool_]) -> IntervalArrays:
    return IntervalArrays(
        lower=arrays.lower[mask],
        unbounded_lower=arrays.unbounded_lower[mask],
        upper=arrays.upper[mask],
        radius=arrays.radius[mask],
        support=arrays.support[mask],
        fallback=arrays.fallback[mask],
    )


def _interval_metrics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: IntervalArrays,
    coverage: float,
) -> dict[str, object]:
    if not len(target) or len(target) != len(predictions) or len(target) != len(intervals.lower):
        raise UncertaintySharpnessError("interval metrics require aligned non-empty vectors")
    covered = (target >= intervals.lower) & (target <= intervals.upper)
    unbounded_covered = (target >= intervals.unbounded_lower) & (target <= intervals.upper)
    displayed_width = intervals.upper - intervals.lower
    symmetric_width = 2.0 * intervals.radius
    relative_width = displayed_width / np.maximum(predictions, 1.0)
    empirical = float(np.mean(covered))
    return {
        "sample_count": len(target),
        "nominal_coverage": coverage,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - coverage,
        "undercoverage_rate": float(np.mean(target < intervals.lower)),
        "overcoverage_rate": float(np.mean(target > intervals.upper)),
        "displayed_width_usd": _distribution(displayed_width),
        "unclipped_symmetric_width_usd": _distribution(symmetric_width),
        "relative_displayed_width": _distribution(relative_width),
        "fallback_count": int(np.count_nonzero(intervals.fallback)),
        "fallback_rate": float(np.mean(intervals.fallback)),
        "zero_lower_bound_clipping_count": int(np.count_nonzero(intervals.unbounded_lower < 0.0)),
        "zero_lower_bound_clipping_rate": float(np.mean(intervals.unbounded_lower < 0.0)),
        "clipped_and_unclipped_coverage_equal": bool(np.array_equal(covered, unbounded_covered)),
        "invalid_or_nonfinite_interval_count": int(
            np.count_nonzero(
                ~np.isfinite(intervals.lower)
                | ~np.isfinite(intervals.upper)
                | ~np.isfinite(intervals.radius)
            )
        ),
        "negative_displayed_lower_bound_count": int(np.count_nonzero(intervals.lower < 0.0)),
        "point_exclusion_or_reversed_count": int(
            np.count_nonzero((intervals.lower > predictions) | (intervals.upper < predictions))
        ),
    }


def _distribution(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.1, method="linear")),
        "p25": float(np.quantile(values, 0.25, method="linear")),
        "p75": float(np.quantile(values, 0.75, method="linear")),
        "p90": float(np.quantile(values, 0.9, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "maximum": float(np.max(values)),
    }


def _slice_diagnostics(
    features: pd.DataFrame,
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: Mapping[str, IntervalArrays],
    predicted_value_cutpoints: tuple[float, float, float],
) -> dict[str, object]:
    mileage = pd.to_numeric(features["mileage"], errors="coerce").to_numpy(dtype=np.float64)
    year = pd.to_numeric(features["year"], errors="raise").to_numpy(dtype=np.float64)
    age = np.maximum(0.0, float(RETAIL_TRACK.reference_year) - year)
    dimensions: dict[str, NDArray[np.str_]] = {
        "actual_price_band": _numeric_bands(target, PRICE_CUTPOINTS, prefix="price"),
        "predicted_value_band": _numeric_bands(
            predictions,
            predicted_value_cutpoints,
            prefix="predicted_value",
        ),
        "mileage_band": _mileage_bands(mileage),
        "vehicle_age_band": _numeric_bands(age, AGE_CUTPOINTS, prefix="age"),
        "manufacturer": np.asarray(
            [str(value).strip().lower() for value in features["make"]],
            dtype=np.str_,
        ),
    }
    report: dict[str, object] = {}
    for dimension, labels in dimensions.items():
        slices: list[dict[str, object]] = []
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            support = int(np.count_nonzero(mask))
            if support < MINIMUM_SLICE_SUPPORT:
                continue
            slices.append(
                {
                    "label": label,
                    "sample_count": support,
                    "point_metrics": regression_metrics(target[mask], predictions[mask]).to_dict(),
                    "coverages": {
                        str(level): _interval_metrics(
                            target[mask],
                            predictions[mask],
                            _masked_intervals(intervals[str(level)], mask),
                            level,
                        )
                        for level in COVERAGE_LEVELS
                    },
                }
            )
        report[dimension] = slices
    return report


def _confidence_diagnostics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: IntervalArrays,
) -> dict[str, object]:
    relative = (intervals.upper - intervals.lower) / np.maximum(predictions, 1.0)
    high = (intervals.support >= CONFIDENCE_THRESHOLDS.high_minimum_support) & (
        relative <= CONFIDENCE_THRESHOLDS.high_max_relative_width
    )
    moderate = (
        (~high)
        & (intervals.support >= CONFIDENCE_THRESHOLDS.moderate_minimum_support)
        & (relative <= CONFIDENCE_THRESHOLDS.moderate_max_relative_width)
    )
    labels = np.full(len(target), "Low confidence", dtype="<U20")
    labels[moderate] = "Moderate confidence"
    labels[high] = "High confidence"
    return {
        "semantics": "precision_and_support_label_not_probability",
        "thresholds": CONFIDENCE_THRESHOLDS.to_dict(),
        "data_quality_warnings_are_separate": True,
        "labels": [
            {
                "label": label,
                **_interval_metrics(
                    target[labels == label],
                    predictions[labels == label],
                    _masked_intervals(intervals, labels == label),
                    SELECTION_COVERAGE_LEVEL,
                ),
            }
            for label in ("High confidence", "Moderate confidence", "Low confidence")
            if np.any(labels == label)
        ],
    }


def _bootstrap_comparisons(
    features: pd.DataFrame,
    target: NDArray[np.float64],
    evaluations: Mapping[MethodId, MethodEvaluation],
) -> dict[MethodId, dict[str, object]]:
    groups = retail_predictor_groups(features)
    _, inverse = np.unique(groups, return_inverse=True)
    baseline = evaluations[BASELINE_METHOD]
    output: dict[MethodId, dict[str, object]] = {}
    for method in (GAMMA_METHOD, SMOOTH_METHOD):
        coverage_results: dict[str, object] = {}
        for level in COVERAGE_LEVELS:
            baseline_arrays = baseline.coverages[str(level)]
            candidate_arrays = evaluations[method].coverages[str(level)]
            baseline_covered = (
                (target >= baseline_arrays.lower) & (target <= baseline_arrays.upper)
            ).astype(np.float64)
            candidate_covered = (
                (target >= candidate_arrays.lower) & (target <= candidate_arrays.upper)
            ).astype(np.float64)
            coverage_delta = _cluster_bootstrap_mean(
                candidate_covered - baseline_covered,
                inverse,
                seed=_method_seed(method, level, "coverage"),
            )
            coverage_results[str(level)] = {
                "coverage_delta_vs_baseline_95pct_ci": _confidence_interval(coverage_delta),
            }
        selection_key = str(SELECTION_COVERAGE_LEVEL)
        baseline_width = 2.0 * baseline.coverages[selection_key].radius
        candidate_width = 2.0 * evaluations[method].coverages[selection_key].radius
        width_ratios = _cluster_bootstrap_ratio(
            candidate_width,
            baseline_width,
            inverse,
            seed=_method_seed(method, SELECTION_COVERAGE_LEVEL, "width"),
        )
        output[method] = {
            "replicates": BOOTSTRAP_REPLICATES,
            "unit": "retail_predictor_group",
            "coverage": coverage_results,
            "mean_unclipped_width_ratio_90pct_95pct_ci": _confidence_interval(width_ratios),
        }
    return output


def _cluster_bootstrap_mean(
    row_values: NDArray[np.float64],
    group_inverse: NDArray[np.int64],
    *,
    seed: int,
) -> NDArray[np.float64]:
    group_count = int(np.max(group_inverse)) + 1
    group_sums = np.bincount(group_inverse, weights=row_values, minlength=group_count)
    group_sizes = np.bincount(group_inverse, minlength=group_count)
    random = np.random.default_rng(seed)
    results = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = random.integers(0, group_count, size=group_count)
        results[index] = float(np.sum(group_sums[sampled]) / np.sum(group_sizes[sampled]))
    return results


def _cluster_bootstrap_ratio(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    group_inverse: NDArray[np.int64],
    *,
    seed: int,
) -> NDArray[np.float64]:
    group_count = int(np.max(group_inverse)) + 1
    numerator_sums = np.bincount(group_inverse, weights=numerator, minlength=group_count)
    denominator_sums = np.bincount(group_inverse, weights=denominator, minlength=group_count)
    random = np.random.default_rng(seed)
    results = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = random.integers(0, group_count, size=group_count)
        denominator_total = float(np.sum(denominator_sums[sampled]))
        if denominator_total <= 0.0:
            raise UncertaintySharpnessError("bootstrap baseline width must remain positive")
        results[index] = float(np.sum(numerator_sums[sampled]) / denominator_total)
    return results


def _confidence_interval(values: NDArray[np.float64]) -> dict[str, float]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return {
        "lower": float(np.quantile(values, tail, method="linear")),
        "upper": float(np.quantile(values, 1.0 - tail, method="linear")),
    }


def _method_seed(method: MethodId, coverage: float, purpose: str) -> int:
    payload = f"{BOOTSTRAP_SEED}|{method}|{coverage}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _candidate_gates(
    *,
    method: MethodId,
    evaluation: MethodEvaluation,
    baseline: MethodEvaluation,
    bootstrap: Mapping[str, object],
    gamma_scale: ScalePrediction | None,
) -> dict[str, object]:
    outcomes: list[dict[str, object]] = []
    candidate_coverages = cast(Mapping[str, object], evaluation.report["coverages"])
    baseline_coverages = cast(Mapping[str, object], baseline.report["coverages"])
    bootstrap_coverages = cast(Mapping[str, object], bootstrap["coverage"])
    for level in COVERAGE_LEVELS:
        key = str(level)
        candidate_item = cast(Mapping[str, object], candidate_coverages[key])
        baseline_item = cast(Mapping[str, object], baseline_coverages[key])
        coverage_gap = cast(float, candidate_item["coverage_gap"])
        coverage_regression = cast(float, baseline_item["empirical_coverage"]) - cast(
            float, candidate_item["empirical_coverage"]
        )
        coverage_ci = cast(
            Mapping[str, float],
            cast(Mapping[str, object], bootstrap_coverages[key])[
                "coverage_delta_vs_baseline_95pct_ci"
            ],
        )
        candidate_unclipped = cast(
            Mapping[str, float], candidate_item["unclipped_symmetric_width_usd"]
        )
        baseline_unclipped = cast(
            Mapping[str, float], baseline_item["unclipped_symmetric_width_usd"]
        )
        candidate_displayed = cast(Mapping[str, float], candidate_item["displayed_width_usd"])
        baseline_displayed = cast(Mapping[str, float], baseline_item["displayed_width_usd"])
        mean_reduction = 1.0 - candidate_unclipped["mean"] / baseline_unclipped["mean"]
        median_reduction = 1.0 - candidate_displayed["median"] / baseline_displayed["median"]
        _gate(
            outcomes,
            f"overall_coverage_gap_{key}",
            coverage_gap,
            OVERALL_MINIMUM_COVERAGE_GAP,
            ">=",
        )
        _gate(
            outcomes,
            f"overall_coverage_regression_{key}",
            coverage_regression,
            OVERALL_MAXIMUM_COVERAGE_REGRESSION,
            "<=",
        )
        _gate(
            outcomes,
            f"bootstrap_coverage_delta_lower_{key}",
            coverage_ci["lower"],
            BOOTSTRAP_MINIMUM_COVERAGE_DELTA_LOWER,
            ">=",
        )
        _gate(
            outcomes,
            f"mean_unclipped_width_reduction_{key}",
            mean_reduction,
            MEAN_WIDTH_REDUCTION_THRESHOLDS[key],
            ">=",
        )
        _gate(
            outcomes,
            f"median_displayed_width_reduction_{key}",
            median_reduction,
            MINIMUM_MEDIAN_WIDTH_REDUCTION,
            ">=",
        )
        _gate(
            outcomes,
            f"p95_width_ratio_{key}",
            candidate_displayed["p95"] / baseline_displayed["p95"],
            MAXIMUM_P95_WIDTH_RATIO,
            "<=",
        )
        _gate(
            outcomes,
            f"p95_to_median_width_ratio_{key}",
            candidate_displayed["p95"] / max(candidate_displayed["median"], 1.0),
            MAXIMUM_P95_TO_MEDIAN_WIDTH_RATIO,
            "<=",
        )
        _gate(
            outcomes,
            f"maximum_interval_width_{key}",
            candidate_displayed["maximum"],
            MAXIMUM_INTERVAL_WIDTH_USD,
            "<=",
        )
        _validity_gates(outcomes, key, candidate_item)
        _status_gates(outcomes, key, level, candidate_item, baseline_item)
        _fold_gates(outcomes, key, candidate_item, baseline_item)
    width_ci = cast(Mapping[str, float], bootstrap["mean_unclipped_width_ratio_90pct_95pct_ci"])
    _gate(
        outcomes,
        "bootstrap_90pct_mean_width_ratio_upper",
        width_ci["upper"],
        MAXIMUM_BOOTSTRAP_WIDTH_RATIO,
        "<",
    )
    _slice_gates(outcomes, evaluation.report, baseline.report)
    if method == GAMMA_METHOD:
        if gamma_scale is None:
            raise UncertaintySharpnessError("Gamma gates require scale diagnostics")
        _gate(
            outcomes,
            "gamma_scale_floor_hit_rate",
            float(np.mean(gamma_scale.raw < GAMMA_SCALE_FLOOR_USD)),
            GAMMA_SCALE_FLOOR_MAXIMUM_RATE,
            "<=",
        )
        _gate(
            outcomes,
            "gamma_scale_cap_hit_rate",
            float(np.mean(gamma_scale.raw > GAMMA_SCALE_CAP_USD)),
            GAMMA_SCALE_CAP_MAXIMUM_RATE,
            "<=",
        )
    passed = all(cast(bool, outcome["passed"]) for outcome in outcomes)
    return {
        "passed_all": passed,
        "failed_gate_count": sum(not cast(bool, outcome["passed"]) for outcome in outcomes),
        "outcomes": outcomes,
    }


def _gate(
    outcomes: list[dict[str, object]],
    name: str,
    observed: float | int,
    threshold: float | int,
    operator: Literal[">=", "<=", "<"],
) -> None:
    passed = (
        observed >= threshold
        if operator == ">="
        else observed <= threshold
        if operator == "<="
        else observed < threshold
    )
    outcomes.append(
        {
            "gate": name,
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
    )


def _validity_gates(
    outcomes: list[dict[str, object]],
    key: str,
    item: Mapping[str, object],
) -> None:
    _gate(
        outcomes,
        f"invalid_interval_count_{key}",
        cast(int, item["invalid_or_nonfinite_interval_count"]),
        REQUIRED_INVALID_COUNT,
        "<=",
    )
    _gate(
        outcomes,
        f"point_exclusion_or_reversed_count_{key}",
        cast(int, item["point_exclusion_or_reversed_count"]),
        REQUIRED_INVALID_COUNT,
        "<=",
    )
    _gate(
        outcomes,
        f"negative_displayed_lower_bound_count_{key}",
        cast(int, item["negative_displayed_lower_bound_count"]),
        REQUIRED_INVALID_COUNT,
        "<=",
    )
    _gate(
        outcomes,
        f"clipped_unclipped_coverage_mismatch_{key}",
        0
        if cast(bool, item["clipped_and_unclipped_coverage_equal"])
        == REQUIRE_CLIPPED_COVERAGE_MATCH
        else 1,
        REQUIRED_INVALID_COUNT,
        "<=",
    )


def _status_gates(
    outcomes: list[dict[str, object]],
    key: str,
    level: float,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> None:
    candidate_status = cast(Mapping[str, object], candidate["status"])
    baseline_status = cast(Mapping[str, object], baseline["status"])
    gaps = []
    regressions = []
    for status in RETAIL_VEHICLE_STATUSES:
        candidate_item = cast(Mapping[str, object], candidate_status[status])
        baseline_item = cast(Mapping[str, object], baseline_status[status])
        gaps.append(cast(float, candidate_item["empirical_coverage"]) - level)
        regressions.append(
            cast(float, baseline_item["empirical_coverage"])
            - cast(float, candidate_item["empirical_coverage"])
        )
    _gate(
        outcomes, f"worst_status_coverage_gap_{key}", min(gaps), MINIMUM_STATUS_COVERAGE_GAP, ">="
    )
    _gate(
        outcomes,
        f"worst_status_coverage_regression_{key}",
        max(regressions),
        MAXIMUM_STATUS_COVERAGE_REGRESSION,
        "<=",
    )


def _fold_gates(
    outcomes: list[dict[str, object]],
    key: str,
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> None:
    candidate_folds = cast(list[Mapping[str, object]], candidate["folds"])
    baseline_folds = cast(list[Mapping[str, object]], baseline["folds"])
    regressions = [
        cast(float, baseline_fold["empirical_coverage"])
        - cast(float, candidate_fold["empirical_coverage"])
        for baseline_fold, candidate_fold in zip(baseline_folds, candidate_folds, strict=True)
    ]
    _gate(
        outcomes,
        f"worst_fold_coverage_regression_{key}",
        max(regressions),
        MAXIMUM_FOLD_COVERAGE_REGRESSION,
        "<=",
    )
    sd_increase = cast(float, candidate["fold_coverage_standard_deviation"]) - cast(
        float, baseline["fold_coverage_standard_deviation"]
    )
    _gate(
        outcomes,
        f"fold_coverage_sd_increase_{key}",
        sd_increase,
        MAXIMUM_FOLD_COVERAGE_SD_INCREASE,
        "<=",
    )
    fallback_increase = cast(float, candidate["fallback_rate"]) - cast(
        float, baseline["fallback_rate"]
    )
    _gate(
        outcomes,
        f"fallback_rate_increase_{key}",
        fallback_increase,
        MAXIMUM_FALLBACK_RATE_INCREASE,
        "<=",
    )


def _slice_gates(
    outcomes: list[dict[str, object]],
    candidate_report: Mapping[str, object],
    baseline_report: Mapping[str, object],
) -> None:
    candidate_slices = cast(Mapping[str, object], candidate_report["slices"])
    baseline_slices = cast(Mapping[str, object], baseline_report["slices"])
    broad_dimensions = (
        "actual_price_band",
        "predicted_value_band",
        "mileage_band",
        "vehicle_age_band",
    )
    for level in COVERAGE_LEVELS:
        key = str(level)
        regressions: list[float] = []
        new_undercoverage_violations = 0
        for dimension in broad_dimensions:
            candidate_items = _slice_map(candidate_slices, dimension)
            baseline_items = _slice_map(baseline_slices, dimension)
            for label, candidate_item in candidate_items.items():
                baseline_item = baseline_items[label]
                candidate_metrics = _slice_coverage(candidate_item, key)
                baseline_metrics = _slice_coverage(baseline_item, key)
                candidate_coverage = cast(float, candidate_metrics["empirical_coverage"])
                baseline_coverage = cast(float, baseline_metrics["empirical_coverage"])
                regressions.append(baseline_coverage - candidate_coverage)
                boundary = level + BROAD_SLICE_UNDERCOVERAGE_BOUNDARY
                if baseline_coverage >= boundary and candidate_coverage < boundary:
                    new_undercoverage_violations += 1
        _gate(
            outcomes,
            f"worst_broad_slice_regression_{key}",
            max(regressions),
            MAXIMUM_BROAD_SLICE_REGRESSION,
            "<=",
        )
        _gate(
            outcomes,
            f"new_broad_slice_undercoverage_count_{key}",
            new_undercoverage_violations,
            0,
            "<=",
        )
    candidate_manufacturers = _slice_map(candidate_slices, "manufacturer")
    baseline_manufacturers = _slice_map(baseline_slices, "manufacturer")
    manufacturer_regressions = [
        cast(float, _slice_coverage(baseline_manufacturers[label], "0.9")["empirical_coverage"])
        - cast(float, _slice_coverage(item, "0.9")["empirical_coverage"])
        for label, item in candidate_manufacturers.items()
    ]
    baseline_low = sum(
        cast(float, _slice_coverage(item, "0.9")["empirical_coverage"])
        < MANUFACTURER_LOW_COVERAGE_BOUNDARY
        for item in baseline_manufacturers.values()
    )
    candidate_low = sum(
        cast(float, _slice_coverage(item, "0.9")["empirical_coverage"])
        < MANUFACTURER_LOW_COVERAGE_BOUNDARY
        for item in candidate_manufacturers.values()
    )
    _gate(
        outcomes,
        "worst_manufacturer_regression_0.9",
        max(manufacturer_regressions),
        MAXIMUM_MANUFACTURER_REGRESSION,
        "<=",
    )
    _gate(
        outcomes,
        "manufacturer_below_0.8_count_increase_0.9",
        candidate_low - baseline_low,
        0,
        "<=",
    )
    focus_regressions: list[float] = []
    candidate_price = _slice_map(candidate_slices, "actual_price_band")
    baseline_price = _slice_map(baseline_slices, "actual_price_band")
    focus_regressions.append(
        cast(float, _slice_coverage(baseline_price["price_4"], "0.9")["empirical_coverage"])
        - cast(float, _slice_coverage(candidate_price["price_4"], "0.9")["empirical_coverage"])
    )
    for manufacturer in FOCUS_SLICES[1:]:
        focus_regressions.append(
            cast(
                float,
                _slice_coverage(baseline_manufacturers[manufacturer], "0.9")["empirical_coverage"],
            )
            - cast(
                float,
                _slice_coverage(candidate_manufacturers[manufacturer], "0.9")["empirical_coverage"],
            )
        )
    _gate(
        outcomes,
        "worst_focus_slice_regression_0.9",
        max(focus_regressions),
        MAXIMUM_FOCUS_SLICE_REGRESSION,
        "<=",
    )


def _slice_map(slices: Mapping[str, object], dimension: str) -> dict[str, Mapping[str, object]]:
    return {
        cast(str, item["label"]): item
        for item in cast(list[Mapping[str, object]], slices[dimension])
    }


def _slice_coverage(item: Mapping[str, object], level: str) -> Mapping[str, object]:
    return cast(Mapping[str, object], cast(Mapping[str, object], item["coverages"])[level])


def _select_method(
    evaluations: Mapping[MethodId, MethodEvaluation],
    gates: Mapping[MethodId, Mapping[str, object]],
) -> MethodId:
    gamma_passed = cast(bool, gates[GAMMA_METHOD]["passed_all"])
    smooth_passed = cast(bool, gates[SMOOTH_METHOD]["passed_all"])
    if not gamma_passed and not smooth_passed:
        return BASELINE_METHOD
    if gamma_passed and not smooth_passed:
        return GAMMA_METHOD
    if smooth_passed and not gamma_passed:
        return SMOOTH_METHOD
    selection_key = str(SELECTION_COVERAGE_LEVEL)
    baseline_mean = _mean_unclipped_width(evaluations[BASELINE_METHOD], selection_key)
    gamma_reduction = (
        1.0 - _mean_unclipped_width(evaluations[GAMMA_METHOD], selection_key) / baseline_mean
    )
    smooth_reduction = (
        1.0 - _mean_unclipped_width(evaluations[SMOOTH_METHOD], selection_key) / baseline_mean
    )
    gamma_has_width_advantage = (
        gamma_reduction - smooth_reduction >= SELECTION_GAMMA_EXTRA_WIDTH_REDUCTION
    )
    if not gamma_has_width_advantage:
        return SMOOTH_METHOD
    gamma_has_no_worse_gate = _has_no_worse_common_gate_result(
        cast(list[Mapping[str, object]], gates[GAMMA_METHOD]["outcomes"]),
        cast(list[Mapping[str, object]], gates[SMOOTH_METHOD]["outcomes"]),
    )
    return GAMMA_METHOD if gamma_has_no_worse_gate else SMOOTH_METHOD


def _has_no_worse_common_gate_result(
    gamma_outcomes: list[Mapping[str, object]],
    smooth_outcomes: list[Mapping[str, object]],
) -> bool:
    smooth_by_name = {cast(str, item["gate"]): item for item in smooth_outcomes}
    compared = 0
    for gamma in gamma_outcomes:
        name = cast(str, gamma["gate"])
        smooth = smooth_by_name.get(name)
        if smooth is None:
            continue
        if gamma["operator"] != smooth["operator"] or gamma["threshold"] != smooth["threshold"]:
            raise UncertaintySharpnessError("candidate gate definitions differ")
        operator = cast(str, gamma["operator"])
        gamma_observed = float(cast(float | int, gamma["observed"]))
        smooth_observed = float(cast(float | int, smooth["observed"]))
        tolerance = 1e-12
        gamma_is_worse = (
            gamma_observed + tolerance < smooth_observed
            if operator == ">="
            else gamma_observed - tolerance > smooth_observed
        )
        if gamma_is_worse:
            return False
        compared += 1
    if compared == 0:
        raise UncertaintySharpnessError("candidate gates have no common outcomes")
    return True


def _mean_unclipped_width(evaluation: MethodEvaluation, level: str) -> float:
    coverage = cast(Mapping[str, object], evaluation.report["coverages"])
    item = cast(Mapping[str, object], coverage[level])
    return cast(Mapping[str, float], item["unclipped_symmetric_width_usd"])["mean"]


def _fit_full_quantiles(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    statuses: NDArray[np.str_],
    scale: NDArray[np.float64],
) -> dict[str, object]:
    scores = np.abs(target - predictions) / scale
    return {str(level): _fit_quantiles(scores, statuses, level) for level in COVERAGE_LEVELS}


def _build_report(
    *,
    development_target: NDArray[np.float64],
    development_predictions: NDArray[np.float64],
    calibration_target: NDArray[np.float64],
    calibration_predictions: NDArray[np.float64],
    evaluations: Mapping[MethodId, MethodEvaluation],
    bootstraps: Mapping[MethodId, Mapping[str, object]],
    gates: Mapping[MethodId, Mapping[str, object]],
    selected_method: MethodId,
    gamma_model: ScaleEstimator,
    gamma_scale: ScalePrediction,
    full_quantiles: Mapping[str, object],
    generated_at: str,
) -> dict[str, object]:
    gamma_regressor = (
        gamma_model.named_steps["regressor"] if isinstance(gamma_model, Pipeline) else None
    )
    candidates_passed = [
        method
        for method in (GAMMA_METHOD, SMOOTH_METHOD)
        if cast(bool, gates[method]["passed_all"])
    ]
    classification = (
        "retain_current_calibration_baseline"
        if selected_method == BASELINE_METHOD
        else "sharper_calibration_candidate_validated"
    )
    return {
        "schema_version": 1,
        "report_type": REPORT_TYPE,
        "generated_at": generated_at,
        "classification": classification,
        "decision": {
            "selected_method": selected_method,
            "candidates_passing_every_gate": candidates_passed,
            "new_serving_artifact_permitted": selected_method != BASELINE_METHOD,
            "production_final": False,
            "legacy_holdout_should_be_opened_automatically": False,
        },
        "frozen_evidence": {
            "sharpness_policy_sha256": SHARPNESS_POLICY_SHA256,
            "development_diagnostics_sha256": DEVELOPMENT_DIAGNOSTICS_SHA256,
            "phase4_confirmation_sha256": PHASE4_RETAIL_CONFIRMATION_SHA256,
            "calibration_v1_policy_sha256": CALIBRATION_V1_POLICY_SHA256,
            "calibration_v1_artifact_sha256": CALIBRATION_V1_ARTIFACT_SHA256,
            "calibration_v1_report_sha256": CALIBRATION_V1_REPORT_SHA256,
            "rf05_identity_sha256": active_rf05_identity().identity_sha256,
        },
        "data_boundaries": {
            "development_rows": len(development_target),
            "calibration_rows": len(calibration_target),
            "calibration_assignment_sha256": CALIBRATION_ASSIGNMENT_SHA256,
            "development_oof_predictions_used_as_scale_targets": True,
            "calibration_targets_used_to_fit_gamma_scale": False,
            "calibration_targets_used_to_fit_or_tune_rf05": False,
            "legacy_holdout_accessed_by_modeling_code": False,
            "yoad_accessed": False,
            "river_accessed": False,
            "autotrader_accessed": False,
            "carson_shively_accessed": False,
            "raw_rows_predictions_or_residuals_persisted": False,
        },
        "reconstruction": {
            "development_point_metrics": regression_metrics(
                development_target, development_predictions
            ).to_dict(),
            "calibration_point_metrics": regression_metrics(
                calibration_target, calibration_predictions
            ).to_dict(),
            "rf05_definition_changed": False,
            "rf05_hyperparameters_tuned": False,
            "rf05_fit_received_calibration_rows": False,
        },
        "scale_models": {
            GAMMA_METHOD: {
                "version": SCALE_VERSION,
                "estimator": "GammaRegressor",
                "alpha": GAMMA_ALPHA,
                "max_iter": GAMMA_MAX_ITER,
                "tolerance": GAMMA_TOLERANCE,
                "iterations": (
                    int(gamma_regressor.n_iter_)
                    if isinstance(gamma_regressor, GammaRegressor)
                    else None
                ),
                "floor_usd": GAMMA_SCALE_FLOOR_USD,
                "cap_usd": GAMMA_SCALE_CAP_USD,
                "calibration_raw_scale_usd": _distribution(gamma_scale.raw),
                "floor_hit_count": int(np.count_nonzero(gamma_scale.raw < GAMMA_SCALE_FLOOR_USD)),
                "cap_hit_count": int(np.count_nonzero(gamma_scale.raw > GAMMA_SCALE_CAP_USD)),
            },
            SMOOTH_METHOD: {
                "formula": "1 + ln(1 + max(RF05 prediction USD, 0) / 10000)",
                "fit": "none",
            },
        },
        "methods": {method: evaluations[method].report for method in METHODS},
        "bootstrap": bootstraps,
        "acceptance_gates": gates,
        "selected_full_calibration_quantiles": full_quantiles,
        "confidence": {
            "thresholds_reused_from_v1": CONFIDENCE_THRESHOLDS.to_dict(),
            "labels_are_probabilities": False,
            "data_quality_warnings_are_separate": True,
        },
        "publication": {
            "aggregate_only": True,
            "raw_rows_predictions_residuals_or_category_vocabularies_in_report": False,
            "original_calibration_v1_modified": False,
            "legacy_holdout_remains_reserved": True,
        },
    }


def _validate_scale(scale: NDArray[np.float64], *, expected_rows: int, method: str) -> None:
    if scale.ndim != 1 or len(scale) != expected_rows:
        raise UncertaintySharpnessError(f"{method} scale must be a one-dimensional row match")
    if not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise UncertaintySharpnessError(f"{method} scale must be strictly positive finite")


def _validate_runtime_policy(policy: UncertaintySharpnessPolicy) -> None:
    """Bind every governed runtime choice to the typed immutable policy."""

    if not isinstance(policy, UncertaintySharpnessPolicy):
        raise UncertaintySharpnessError("typed uncertainty-sharpness policy is required")
    identity = active_rf05_identity()
    comparison = policy.calibration_comparison
    bootstrap = comparison.bootstrap
    gates = policy.acceptance_gates
    confidence = policy.confidence_policy
    bindings: tuple[tuple[str, object, object], ...] = (
        ("policy SHA-256", policy.policy_sha256, SHARPNESS_POLICY_SHA256),
        ("candidate methods", policy.candidate_ids, METHODS),
        (
            "Phase 4 confirmation",
            policy.frozen_inputs.phase4_retail_confirmation_sha256,
            PHASE4_RETAIL_CONFIRMATION_SHA256,
        ),
        ("RF05 identity", policy.frozen_inputs.rf05_identity_sha256, identity.identity_sha256),
        ("RF05 candidate", policy.frozen_inputs.rf05_candidate_id, identity.candidate_id),
        ("RF05 parameters", policy.frozen_inputs.rf05_parameters, identity.parameters),
        ("RF05 random state", policy.frozen_inputs.rf05_random_state, identity.random_state),
        (
            "feature contract",
            policy.frozen_inputs.feature_contract_version,
            RETAIL_TRACK.contract_version,
        ),
        (
            "calibration v1 policy",
            policy.frozen_inputs.calibration_v1_policy_sha256,
            CALIBRATION_V1_POLICY_SHA256,
        ),
        (
            "calibration v1 artifact",
            policy.frozen_inputs.calibration_v1_artifact_sha256,
            CALIBRATION_V1_ARTIFACT_SHA256,
        ),
        (
            "calibration v1 report",
            policy.frozen_inputs.calibration_v1_report_sha256,
            CALIBRATION_V1_REPORT_SHA256,
        ),
        (
            "development diagnostics",
            policy.frozen_inputs.development_residual_diagnostics_sha256,
            DEVELOPMENT_DIAGNOSTICS_SHA256,
        ),
        (
            "calibration assignment",
            policy.frozen_inputs.calibration_assignment_sha256,
            CALIBRATION_ASSIGNMENT_SHA256,
        ),
        ("development rows", policy.frozen_inputs.development_rows, DEVELOPMENT_SAMPLE_COUNT),
        ("calibration rows", policy.frozen_inputs.calibration_rows, CALIBRATION_SAMPLE_COUNT),
        (
            "baseline status support",
            policy.baseline_method.minimum_status_support,
            MINIMUM_BUCKET_SUPPORT,
        ),
        (
            "Gamma status support",
            policy.gamma_method.minimum_status_support,
            MINIMUM_BUCKET_SUPPORT,
        ),
        (
            "smooth status support",
            policy.smooth_value_method.minimum_status_support,
            MINIMUM_BUCKET_SUPPORT,
        ),
        ("Gamma alpha", policy.gamma_method.hyperparameters.alpha, GAMMA_ALPHA),
        ("Gamma max iterations", policy.gamma_method.hyperparameters.max_iter, GAMMA_MAX_ITER),
        ("Gamma tolerance", policy.gamma_method.hyperparameters.tol, GAMMA_TOLERANCE),
        ("Gamma scale floor", policy.gamma_method.scale_floor_usd, GAMMA_SCALE_FLOOR_USD),
        ("Gamma scale cap", policy.gamma_method.scale_cap_usd, GAMMA_SCALE_CAP_USD),
        ("coverage levels", comparison.coverage_levels, COVERAGE_LEVELS),
        ("calibration folds", comparison.scheme, "five_fold_predictor_group_cross_calibration"),
        ("slice support", comparison.minimum_reported_slice_support, MINIMUM_SLICE_SUPPORT),
        ("manufacturer support", comparison.manufacturer_support, MINIMUM_SLICE_SUPPORT),
        ("price cutpoints", comparison.price_bands_usd, PRICE_CUTPOINTS),
        ("mileage cutpoints", comparison.mileage_bands_miles, MILEAGE_CUTPOINTS),
        ("age reference year", comparison.vehicle_age_reference_year, RETAIL_TRACK.reference_year),
        ("age cutpoints", comparison.vehicle_age_bands_years, AGE_CUTPOINTS),
        ("bootstrap unit", bootstrap.unit, "existing retail predictor group"),
        ("bootstrap replicates", bootstrap.replicates, BOOTSTRAP_REPLICATES),
        ("bootstrap seed", bootstrap.random_state, BOOTSTRAP_SEED),
        ("bootstrap confidence", bootstrap.confidence_level, BOOTSTRAP_CONFIDENCE_LEVEL),
        (
            "invalid interval gate",
            gates.validity.invalid_or_nonfinite_intervals,
            REQUIRED_INVALID_COUNT,
        ),
        (
            "reversed interval gate",
            gates.validity.reversed_or_point_excluding_intervals,
            REQUIRED_INVALID_COUNT,
        ),
        (
            "negative lower gate",
            gates.validity.negative_displayed_lower_bounds,
            REQUIRED_INVALID_COUNT,
        ),
        (
            "coverage-match gate",
            gates.validity.clipped_and_unclipped_coverage_must_match,
            REQUIRE_CLIPPED_COVERAGE_MATCH,
        ),
        (
            "Gamma floor-hit gate",
            gates.validity.gamma_scale_floor_hit_maximum_rate,
            GAMMA_SCALE_FLOOR_MAXIMUM_RATE,
        ),
        (
            "Gamma cap-hit gate",
            gates.validity.gamma_scale_cap_hit_maximum_rate,
            GAMMA_SCALE_CAP_MAXIMUM_RATE,
        ),
        (
            "overall coverage-gap gate",
            gates.overall_coverage.minimum_gap_from_target_each_level,
            OVERALL_MINIMUM_COVERAGE_GAP,
        ),
        (
            "overall regression gate",
            gates.overall_coverage.maximum_regression_vs_baseline_each_level,
            OVERALL_MAXIMUM_COVERAGE_REGRESSION,
        ),
        (
            "bootstrap coverage gate",
            gates.overall_coverage.minimum_cluster_bootstrap_95pct_lower_delta_vs_baseline,
            BOOTSTRAP_MINIMUM_COVERAGE_DELTA_LOWER,
        ),
        (
            "80% width gate",
            gates.sharpness.minimum_unclipped_mean_width_reduction_80pct,
            MEAN_WIDTH_REDUCTION_THRESHOLDS["0.8"],
        ),
        (
            "90% width gate",
            gates.sharpness.minimum_unclipped_mean_width_reduction_90pct,
            MEAN_WIDTH_REDUCTION_THRESHOLDS["0.9"],
        ),
        (
            "95% width gate",
            gates.sharpness.minimum_unclipped_mean_width_reduction_95pct,
            MEAN_WIDTH_REDUCTION_THRESHOLDS["0.95"],
        ),
        (
            "median width gate",
            gates.sharpness.minimum_displayed_median_width_reduction_each_level,
            MINIMUM_MEDIAN_WIDTH_REDUCTION,
        ),
        (
            "bootstrap width gate",
            gates.sharpness.maximum_bootstrap_95pct_upper_mean_width_ratio_at_90pct,
            MAXIMUM_BOOTSTRAP_WIDTH_RATIO,
        ),
        (
            "p95 width gate",
            gates.sharpness.maximum_p95_width_ratio_vs_baseline_each_level,
            MAXIMUM_P95_WIDTH_RATIO,
        ),
        (
            "status gap gate",
            gates.conditional_coverage.minimum_status_gap_from_target_each_level,
            MINIMUM_STATUS_COVERAGE_GAP,
        ),
        (
            "status regression gate",
            gates.conditional_coverage.maximum_status_regression_vs_baseline_each_level,
            MAXIMUM_STATUS_COVERAGE_REGRESSION,
        ),
        (
            "broad-slice gate",
            gates.conditional_coverage.maximum_broad_slice_regression_vs_baseline_each_level,
            MAXIMUM_BROAD_SLICE_REGRESSION,
        ),
        (
            "undercoverage boundary",
            gates.conditional_coverage.new_broad_slice_undercoverage_boundary,
            BROAD_SLICE_UNDERCOVERAGE_BOUNDARY,
        ),
        (
            "manufacturer gate",
            gates.conditional_coverage.maximum_manufacturer_regression_vs_baseline_at_90pct,
            MAXIMUM_MANUFACTURER_REGRESSION,
        ),
        (
            "manufacturer count gate",
            gates.conditional_coverage.manufacturer_count_below_80pct_at_90pct_may_increase,
            False,
        ),
        (
            "focus-slice gate",
            gates.conditional_coverage.focus_90pct_maximum_regression_vs_baseline,
            MAXIMUM_FOCUS_SLICE_REGRESSION,
        ),
        ("focus slices", gates.conditional_coverage.focus_slices, FOCUS_SLICES),
        (
            "fold regression gate",
            gates.stability.maximum_fold_coverage_regression_vs_baseline_each_level,
            MAXIMUM_FOLD_COVERAGE_REGRESSION,
        ),
        (
            "fold SD gate",
            gates.stability.maximum_fold_coverage_sd_increase_vs_baseline_each_level,
            MAXIMUM_FOLD_COVERAGE_SD_INCREASE,
        ),
        (
            "fallback gate",
            gates.stability.maximum_fallback_rate_increase_vs_baseline,
            MAXIMUM_FALLBACK_RATE_INCREASE,
        ),
        (
            "p95/median gate",
            gates.stability.maximum_p95_to_median_width_ratio,
            MAXIMUM_P95_TO_MEDIAN_WIDTH_RATIO,
        ),
        (
            "maximum width gate",
            gates.stability.maximum_interval_width_usd,
            MAXIMUM_INTERVAL_WIDTH_USD,
        ),
        ("confidence coverage", confidence.coverage_level, SELECTION_COVERAGE_LEVEL),
        (
            "high-confidence width",
            confidence.high_max_relative_width,
            CONFIDENCE_THRESHOLDS.high_max_relative_width,
        ),
        (
            "moderate-confidence width",
            confidence.moderate_max_relative_width,
            CONFIDENCE_THRESHOLDS.moderate_max_relative_width,
        ),
        (
            "high-confidence support",
            confidence.high_minimum_support,
            CONFIDENCE_THRESHOLDS.high_minimum_support,
        ),
        (
            "moderate-confidence support",
            confidence.moderate_minimum_support,
            CONFIDENCE_THRESHOLDS.moderate_minimum_support,
        ),
    )
    for label, policy_value, runtime_value in bindings:
        if policy_value != runtime_value:
            raise UncertaintySharpnessError(f"runtime policy binding differs: {label}")
    semantic_bindings = (
        policy.baseline_method.scale == "1 USD",
        policy.baseline_method.score == "absolute_error_usd",
        policy.baseline_method.quantile_hierarchy == ("vehicle_status", "global"),
        policy.gamma_method.scale_target
        == "max(development_OOF_absolute_RF05_residual_usd, 1 USD)",
        policy.gamma_method.scale_inputs
        == (
            "log1p(max(RF05_prediction_usd, 0) / 10000)",
            "model_year",
            "mileage",
            "mileage_per_year",
            "mileage_missing",
            "make",
            "model",
            "vehicle_status",
        ),
        policy.gamma_method.estimator == "sklearn.linear_model.GammaRegressor",
        policy.gamma_method.hyperparameters.fit_intercept,
        policy.gamma_method.hyperparameters.solver == "lbfgs",
        not policy.gamma_method.hyperparameters.warm_start,
        policy.gamma_method.quantile_hierarchy == ("vehicle_status", "global"),
        policy.smooth_value_method.scale_formula
        == "1 + ln(1 + max(RF05_prediction_usd, 0) / 10000)",
        policy.smooth_value_method.scale_fit == "none",
        policy.smooth_value_method.quantile_hierarchy == ("vehicle_status", "global"),
        comparison.finite_sample_order == "ceil((n + 1) * coverage)",
        comparison.same_folds_and_point_predictions_for_every_method,
        comparison.fold_quantiles_use_other_four_calibration_folds_only,
        comparison.lower_bound == "max(0, point_prediction - radius)",
        comparison.upper_bound == "point_prediction + radius",
        comparison.sharpness_primary_width == "unclipped symmetric width equal to 2 times radius",
        comparison.displayed_width == "zero-clipped lower-bound width",
        bootstrap.paired_across_methods,
        policy.selection_rule.candidate_must_pass_every_gate,
        policy.selection_rule.if_neither_passes == "retain_current_calibration_baseline",
        policy.selection_rule.if_execution_or_baseline_reproduction_fails
        == "uncertainty_method_requires_further_research",
        policy.selection_rule.if_both_pass
        == (
            "prefer normalized_smooth_value_scale_v1 unless normalized_gamma_scale_v1 "
            "reduces 90pct mean width by at least 3 additional percentage points without "
            "a worse gate result"
        ),
        policy.selection_rule.coverage_is_primary,
        policy.selection_rule.do_not_force_a_winner,
        confidence.thresholds_reused_from_v1_for_direct_comparability,
        confidence.confidence_is_not_a_probability,
        confidence.data_quality_warnings_remain_separate,
        policy.publication.reports_are_aggregate_only,
        policy.publication.persist_new_serving_artifact_only_if_candidate_passes_every_gate,
        not policy.publication.legacy_holdout_access,
    )
    if not all(semantic_bindings):
        raise UncertaintySharpnessError("runtime policy semantic binding differs")


def _validate_completed_intervals(
    intervals: IntervalArrays,
    predictions: NDArray[np.float64],
) -> None:
    values = (intervals.lower, intervals.unbounded_lower, intervals.upper, intervals.radius)
    if any(not np.isfinite(value).all() for value in values):
        raise UncertaintySharpnessError("cross-calibration left invalid interval values")
    if (intervals.radius < 0.0).any() or (intervals.lower < 0.0).any():
        raise UncertaintySharpnessError("cross-calibration produced invalid interval bounds")
    if np.any(intervals.lower > predictions) or np.any(intervals.upper < predictions):
        raise UncertaintySharpnessError("cross-calibration interval excludes its point prediction")


def _validate_calibration_v1_report(report: Mapping[str, object]) -> None:
    if report.get("report_type") != "retail_rf05_split_conformal_calibration_report":
        raise UncertaintySharpnessError("calibration v1 report type differs")
    if report.get("classification") != "validated_for_calibrated_prediction_intervals":
        raise UncertaintySharpnessError("calibration v1 is not the frozen validated baseline")
    raw_decision = report.get("decision")
    if not isinstance(raw_decision, Mapping):
        raise UncertaintySharpnessError("calibration v1 decision is invalid")
    decision = cast(Mapping[str, object], raw_decision)
    if decision.get("selected_method") != "vehicle_status":
        raise UncertaintySharpnessError("calibration v1 selected method differs")


def _validate_point_metrics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    report: Mapping[str, object],
) -> None:
    point = cast(Mapping[str, object], report["point_prediction_metrics_on_calibration"])
    expected = cast(Mapping[str, object], point["overall"])
    observed = regression_metrics(target, predictions).to_dict()
    for field in ("mae", "rmse", "r2"):
        if not np.isclose(
            cast(float, observed[field]),
            cast(float, expected[field]),
            rtol=0.0,
            atol=1e-8,
        ):
            raise UncertaintySharpnessError(f"calibration point {field} differs from v1")


def _validate_baseline_reproduction(
    evaluation: MethodEvaluation,
    calibration_v1_report: Mapping[str, object],
) -> None:
    cross = _required_mapping(calibration_v1_report, "cross_calibration")
    methods = _required_mapping(cross, "methods")
    selected = _required_mapping(methods, "vehicle_status")
    expected_coverages = _required_mapping(selected, "coverages")
    observed_coverages = _required_mapping(evaluation.report, "coverages")
    for level in COVERAGE_LEVELS:
        key = str(level)
        expected = _required_mapping(expected_coverages, key)
        observed = _required_mapping(observed_coverages, key)
        _validate_reproduced_metric_block(observed, expected, context=f"overall {key}")
        _require_reproduced_number(
            observed,
            expected,
            "fold_coverage_standard_deviation",
            context=f"overall {key}",
        )
        observed_status = _required_mapping(observed, "status")
        expected_status = _required_mapping(expected, "status")
        if set(observed_status) != set(RETAIL_VEHICLE_STATUSES) or set(expected_status) != set(
            RETAIL_VEHICLE_STATUSES
        ):
            raise UncertaintySharpnessError(f"reconstructed baseline status fields differ at {key}")
        for status in RETAIL_VEHICLE_STATUSES:
            _validate_reproduced_metric_block(
                _required_mapping(observed_status, status),
                _required_mapping(expected_status, status),
                context=f"status {status} {key}",
            )
        observed_folds = _required_list(observed, "folds")
        expected_folds = _required_list(expected, "folds")
        if len(observed_folds) != CALIBRATION_FOLD_COUNT or len(expected_folds) != (
            CALIBRATION_FOLD_COUNT
        ):
            raise UncertaintySharpnessError(f"reconstructed baseline fold count differs at {key}")
        for position, (observed_fold, expected_fold) in enumerate(
            zip(observed_folds, expected_folds, strict=True), start=1
        ):
            observed_item = _as_mapping(observed_fold, label=f"observed fold {position}")
            expected_item = _as_mapping(expected_fold, label=f"expected fold {position}")
            _require_reproduced_integer(
                observed_item,
                expected_item,
                "fold_number",
                context=f"fold {position} {key}",
            )
            _validate_reproduced_metric_block(
                observed_item,
                expected_item,
                context=f"fold {position} {key}",
            )


def _validate_reproduced_metric_block(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    context: str,
) -> None:
    _require_reproduced_integer(observed, expected, "sample_count", context=context)
    for field in (
        "nominal_coverage",
        "empirical_coverage",
        "coverage_gap",
        "undercoverage_rate",
        "overcoverage_rate",
    ):
        _require_reproduced_number(observed, expected, field, context=context)
    observed_width = _required_mapping(observed, "displayed_width_usd")
    _require_cross_named_number(
        observed_width, "mean", expected, "average_width_usd", context=context
    )
    _require_cross_named_number(
        observed_width, "median", expected, "median_width_usd", context=context
    )
    expected_percentiles = _required_mapping(expected, "width_percentiles_usd")
    for percentile in ("10", "25", "75", "90", "95"):
        _require_cross_named_number(
            observed_width,
            f"p{percentile}",
            expected_percentiles,
            percentile,
            context=context,
        )


def _required_mapping(container: Mapping[str, object], field: str) -> Mapping[str, object]:
    if field not in container:
        raise UncertaintySharpnessError(f"frozen baseline is missing {field}")
    return _as_mapping(container[field], label=field)


def _as_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise UncertaintySharpnessError(f"frozen baseline {label} must be an object")
    return cast(Mapping[str, object], value)


def _required_list(container: Mapping[str, object], field: str) -> list[object]:
    value = container.get(field)
    if not isinstance(value, list):
        raise UncertaintySharpnessError(f"frozen baseline {field} must be an array")
    return cast(list[object], value)


def _require_reproduced_integer(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> None:
    observed_value = observed.get(field)
    expected_value = expected.get(field)
    if type(observed_value) is not int or type(expected_value) is not int:
        raise UncertaintySharpnessError(f"reconstructed baseline {field} is invalid at {context}")
    if observed_value != expected_value:
        raise UncertaintySharpnessError(f"reconstructed baseline {field} differs at {context}")


def _require_reproduced_number(
    observed: Mapping[str, object],
    expected: Mapping[str, object],
    field: str,
    *,
    context: str,
) -> None:
    _require_cross_named_number(observed, field, expected, field, context=context)


def _require_cross_named_number(
    observed: Mapping[str, object],
    observed_field: str,
    expected: Mapping[str, object],
    expected_field: str,
    *,
    context: str,
) -> None:
    observed_value = observed.get(observed_field)
    expected_value = expected.get(expected_field)
    if (
        isinstance(observed_value, bool)
        or isinstance(expected_value, bool)
        or not isinstance(observed_value, (int, float))
        or not isinstance(expected_value, (int, float))
        or not math.isfinite(float(observed_value))
        or not math.isfinite(float(expected_value))
    ):
        raise UncertaintySharpnessError(
            f"reconstructed baseline {expected_field} is invalid at {context}"
        )
    if not np.isclose(float(observed_value), float(expected_value), rtol=0.0, atol=1e-8):
        raise UncertaintySharpnessError(
            f"reconstructed baseline {expected_field} differs at {context}"
        )


def _v1_prediction_cutpoints(report: Mapping[str, object]) -> tuple[float, float, float]:
    values = cast(list[object], report["target_free_predicted_value_cutpoints_usd"])
    if len(values) != 3:
        raise UncertaintySharpnessError("calibration v1 predicted-value cutpoints differ")
    cutpoints = cast(
        tuple[float, float, float],
        tuple(float(cast(float, value)) for value in values),
    )
    if not cutpoints[0] < cutpoints[1] < cutpoints[2]:
        raise UncertaintySharpnessError("calibration v1 predicted-value cutpoints are invalid")
    return cutpoints


__all__ = [
    "BASELINE_METHOD",
    "GAMMA_METHOD",
    "METHODS",
    "SHARPNESS_POLICY_SHA256",
    "SMOOTH_METHOD",
    "SharpnessExperimentResult",
    "UncertaintySharpnessError",
    "build_scale_features",
    "canonical_sharpness_report_json",
    "compare_uncertainty_methods",
    "fit_gamma_residual_scale",
    "make_gamma_scale_pipeline",
    "sharpness_report_sha256",
    "smooth_value_scale",
]
