"""One-time aggregate evaluation for the frozen retail RF05 system."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_SAMPLE_COUNT,
    CALIBRATION_VERSION,
    COVERAGE_LEVELS,
    DEVELOPMENT_SAMPLE_COUNT,
    ConfidenceThresholds,
    RetailCalibrationArtifact,
    active_rf05_identity,
    calibrated_valuation,
)
from .candidates import make_random_forest_candidate
from .contracts import RETAIL_TRACK, validate_feature_frame, validate_target
from .final_evaluation_policy import (
    FINAL_EVALUATION_POLICY_SHA256,
    FinalEvaluationPolicy,
)
from .metrics import regression_metrics
from .retail_calibration_experiment import (
    AGE_CUTPOINTS,
    MILEAGE_CUTPOINTS,
    PRICE_CUTPOINTS,
    _mileage_bands,
    _normalized_statuses,
    _numeric_bands,
)

FINAL_REPORT_TYPE: Final = "retail_rf05_final_holdout_evaluation"
FINAL_GENERATED_AT: Final = "2026-09-02T23:00:00+00:00"
FINAL_HOLDOUT_ROWS: Final = 27_589
FINAL_HOLDOUT_IDENTITY_SHA256: Final = (
    "5ad3f1f9782133b5f3bba492e19a44e09dd7fd8d09326a1aa8fdf7624a982874"
)
RF05_IDENTITY_SHA256: Final = "3bbd73d6442387496b05253dd20bc749db24aa482d56fa6ba73ec2702de8b513"
CALIBRATION_ARTIFACT_SHA256: Final = (
    "b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0"
)
PRIOR_SHARPNESS_REPORT_SHA256: Final = (
    "8614bad1ccd5345c64925c11e6172a7b4ef000ed6f16856aa45b48c3e4a741dd"
)
BASELINE_METHOD: Final = "vehicle_status_absolute_residual_v1"
MINIMUM_SLICE_SUPPORT: Final = 200
FROZEN_INPUT_ALLOWLIST: Final = ("year", "make", "model", "mileage", "vehicle_status")
PREDICTED_VALUE_CUTPOINTS: Final = (
    34_749.95052955916,
    42_800.75221941281,
    72_354.82278171374,
)
CONFIDENCE_THRESHOLDS: Final = ConfidenceThresholds(
    coverage=0.9,
    high_max_relative_width=0.686682300031913,
    moderate_max_relative_width=1.094478855653873,
)
CLASSIFICATIONS: Final = (
    "final evaluation exposes significant generalization limitations",
    "final evaluation passed with material limitations",
    "final evaluation passed for portfolio/demo integration",
)


class FinalEvaluationError(ValueError):
    """A frozen final-evaluation invariant was violated."""


class FinalEstimator(Protocol):
    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> FinalEstimator: ...

    def predict(self, features: pd.DataFrame) -> object: ...


EstimatorFactory = Callable[[], FinalEstimator]


@dataclass(frozen=True, slots=True)
class IntervalArrays:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    width: NDArray[np.float64]
    support: NDArray[np.int64]
    fallback: NDArray[np.bool_]
    clipped: NDArray[np.bool_]
    confidence: NDArray[np.str_]


@dataclass(frozen=True, slots=True)
class FinalEvaluationResult:
    report: Mapping[str, object]
    classification: str


def fit_frozen_rf05_for_final(
    *,
    development_features: object,
    development_target: object,
    holdout_features: object,
    estimator_factory: EstimatorFactory | None = None,
) -> NDArray[np.float64]:
    """Fit exact RF05 on development only; no holdout target can enter this API."""

    development = validate_feature_frame(development_features, RETAIL_TRACK)
    holdout = validate_feature_frame(holdout_features, RETAIL_TRACK)
    target = validate_target(
        development_target,
        expected_rows=len(development),
        config=RETAIL_TRACK,
    )
    if len(development) != DEVELOPMENT_SAMPLE_COUNT:
        raise FinalEvaluationError("RF05 final fit requires the frozen development population")
    if len(holdout) != FINAL_HOLDOUT_ROWS:
        raise FinalEvaluationError("RF05 final scoring requires the frozen holdout population")
    estimator = (estimator_factory or _rf05_factory)()
    estimator.fit(development, target)
    predictions = _prediction_vector(
        estimator.predict(holdout),
        expected_rows=len(holdout),
        label="final RF05 predictions",
    )
    predictions.setflags(write=False)
    return predictions


def evaluate_final_holdout(
    *,
    policy: FinalEvaluationPolicy,
    holdout_features: object,
    holdout_target: object,
    holdout_predictions: object,
    calibration_artifact: RetailCalibrationArtifact,
    prior_sharpness_report: Mapping[str, object],
    generated_at: str = FINAL_GENERATED_AT,
) -> FinalEvaluationResult:
    """Evaluate once without exposing a fit, calibration, or tuning path."""

    _validate_runtime_policy(policy, calibration_artifact)
    _validate_prior_report(prior_sharpness_report)
    features = validate_feature_frame(holdout_features, RETAIL_TRACK)
    target = validate_target(
        holdout_target,
        expected_rows=len(features),
        config=RETAIL_TRACK,
    )
    predictions = _prediction_vector(
        holdout_predictions,
        expected_rows=len(features),
        label="final RF05 predictions",
    )
    if len(features) != FINAL_HOLDOUT_ROWS:
        raise FinalEvaluationError("final evaluation row count differs from frozen holdout")
    intervals = {
        str(level): _apply_calibration(
            predictions,
            _normalized_statuses(features["vehicle_status"]),
            coverage=level,
            artifact=calibration_artifact,
        )
        for level in COVERAGE_LEVELS
    }
    point = _point_metrics(target, predictions)
    interval_report = {
        str(level): _interval_metrics(
            target,
            predictions,
            intervals[str(level)],
            nominal=level,
        )
        for level in COVERAGE_LEVELS
    }
    slices = _slice_diagnostics(features, target, predictions, intervals["0.9"])
    confidence = _confidence_diagnostics(target, predictions, intervals["0.9"])
    comparison = _prior_comparison(
        point,
        interval_report,
        slices,
        prior_sharpness_report,
    )
    classification, gates = _classify(
        point=point,
        interval_report=interval_report,
        slices=slices,
        confidence=confidence,
        comparison=comparison,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "report_type": FINAL_REPORT_TYPE,
        "generated_at": generated_at,
        "classification": classification,
        "decision": {
            "frozen_system": "phase4-retail-random_forest-05 + retail-rf05-split-conformal-v1",
            "portfolio_demo_integration_allowed": classification != CLASSIFICATIONS[0],
            "production_ready_claim_allowed": False,
            "post_holdout_tuning_allowed": False,
            "automatic_followup_experiment_allowed": False,
        },
        "frozen_evidence": {
            "final_evaluation_policy_sha256": FINAL_EVALUATION_POLICY_SHA256,
            "holdout_identity_sha256": FINAL_HOLDOUT_IDENTITY_SHA256,
            "rf05_identity_sha256": RF05_IDENTITY_SHA256,
            "calibration_v1_artifact_sha256": CALIBRATION_ARTIFACT_SHA256,
            "prior_sharpness_report_sha256": PRIOR_SHARPNESS_REPORT_SHA256,
        },
        "data_boundary": {
            "source_id": "kaggle_us_sales_cars_v2",
            "domain": "historical_us_advertised_asking_price",
            "market_country": "US",
            "currency": "USD",
            "price_kind": "asking",
            "non_temporal_grouped_holdout": True,
            "holdout_rows": len(features),
            "targets_available": len(target),
            "rows_rejected_by_preexisting_validation": 0,
            "holdout_rows_fit_rf05": False,
            "holdout_rows_fit_calibration": False,
            "holdout_rows_entered_yoad_or_river": False,
            "holdout_opened_for_final_evaluation": True,
            "future_role": "permanently_evaluation_only",
        },
        "feature_and_target_coverage": _feature_coverage(features, target),
        "point_prediction": point,
        "uncertainty": {
            "calibration_version": CALIBRATION_VERSION,
            "selected_method": calibration_artifact.selected_method,
            "lower_bound_clipped_to_zero": True,
            "coverages": interval_report,
        },
        "slices": slices,
        "manufacturer_summary": _manufacturer_summary(slices),
        "confidence_labels": confidence,
        "generalization_comparison": comparison,
        "classification_gates": gates,
        "governance": {
            "rf05_retuned_or_replaced": False,
            "preprocessing_or_feature_engineering_changed": False,
            "calibration_quantiles_buckets_or_confidence_changed": False,
            "yoad_loaded_or_promoted": False,
            "river_loaded_or_trained": False,
            "autotrader_loaded": False,
            "carson_shively_loaded": False,
            "raw_rows_targets_predictions_residuals_or_identifiers_persisted": False,
            "holdout_result_used_for_post_evaluation_optimization": False,
        },
    }
    canonical_final_report_json(report)
    return FinalEvaluationResult(report=report, classification=classification)


def canonical_final_report_json(report: Mapping[str, object]) -> str:
    """Serialize aggregate final evidence deterministically."""

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
        raise FinalEvaluationError("final report is not JSON-safe") from error


def _rf05_factory() -> FinalEstimator:
    return cast(FinalEstimator, make_random_forest_candidate("retail", 5, n_jobs=4))


def _validate_runtime_policy(
    policy: FinalEvaluationPolicy,
    artifact: RetailCalibrationArtifact,
) -> None:
    if not isinstance(policy, FinalEvaluationPolicy):
        raise FinalEvaluationError("final evaluation requires the typed frozen policy")
    policy.__post_init__()
    boundary = policy.section("one_time_boundary")
    system = policy.section("frozen_system")
    if (
        boundary.get("expected_rows") != FINAL_HOLDOUT_ROWS
        or boundary.get("holdout_identity_sha256") != FINAL_HOLDOUT_IDENTITY_SHA256
        or boundary.get("market_country") != "US"
        or boundary.get("currency") != "USD"
        or boundary.get("price_kind") != "asking"
    ):
        raise FinalEvaluationError("runtime holdout boundary differs from final policy")
    identity = active_rf05_identity()
    parameters = cast(Mapping[str, object], system["parameters"])
    feature_contract = cast(Mapping[str, object], system["feature_contract"])
    training = cast(Mapping[str, object], system["training_population"])
    uncertainty = cast(Mapping[str, object], system["uncertainty"])
    confidence = cast(Mapping[str, object], system["confidence"])
    if (
        identity.identity_sha256 != RF05_IDENTITY_SHA256
        or system.get("rf05_identity_sha256") != RF05_IDENTITY_SHA256
        or system.get("candidate_id") != identity.candidate_id
        or tuple(
            parameters.get(key)
            for key in (
                "n_estimators",
                "max_leaf_nodes",
                "min_samples_leaf",
                "max_features",
                "max_samples",
            )
        )
        != identity.parameters
        or parameters.get("random_state") != identity.random_state
        or parameters.get("n_jobs_for_evaluation") != 4
        or feature_contract.get("version") != identity.feature_contract_version
        or tuple(cast(tuple[object, ...], feature_contract.get("input_allowlist")))
        != FROZEN_INPUT_ALLOWLIST
        or set(FROZEN_INPUT_ALLOWLIST) != set(RETAIL_TRACK.input_features)
        or len(FROZEN_INPUT_ALLOWLIST) != len(RETAIL_TRACK.input_features)
        or training.get("rows") != DEVELOPMENT_SAMPLE_COUNT
        or training.get("calibration_rows") != CALIBRATION_SAMPLE_COUNT
        or training.get("calibration_assignment_sha256") != CALIBRATION_ASSIGNMENT_SHA256
        or training.get("calibration_rows_fit_rf05") is not False
    ):
        raise FinalEvaluationError("runtime RF05 system differs from final policy")
    if (
        not isinstance(artifact, RetailCalibrationArtifact)
        or artifact.bound_model.identity_sha256 != RF05_IDENTITY_SHA256
        or artifact.calibration_version != CALIBRATION_VERSION
        or artifact.selected_method != "vehicle_status"
        or artifact.confidence_thresholds != CONFIDENCE_THRESHOLDS
        or artifact.predicted_value_cutpoints_usd != PREDICTED_VALUE_CUTPOINTS
        or uncertainty.get("artifact_sha256") != CALIBRATION_ARTIFACT_SHA256
        or uncertainty.get("version") != artifact.calibration_version
        or uncertainty.get("selected_method") != artifact.selected_method
        or tuple(cast(tuple[object, ...], uncertainty.get("coverage_levels"))) != COVERAGE_LEVELS
        or confidence.get("high_max_relative_width")
        != CONFIDENCE_THRESHOLDS.high_max_relative_width
        or confidence.get("moderate_max_relative_width")
        != CONFIDENCE_THRESHOLDS.moderate_max_relative_width
        or confidence.get("high_minimum_support") != CONFIDENCE_THRESHOLDS.high_minimum_support
        or confidence.get("moderate_minimum_support")
        != CONFIDENCE_THRESHOLDS.moderate_minimum_support
    ):
        raise FinalEvaluationError("runtime calibration system differs from final policy")
    evaluation = policy.section("evaluation")
    slices = cast(Mapping[str, object], evaluation["slices"])
    if (
        evaluation.get("slice_minimum_support") != MINIMUM_SLICE_SUPPORT
        or tuple(cast(tuple[object, ...], slices.get("mileage_bands_miles"))) != MILEAGE_CUTPOINTS
        or tuple(cast(tuple[object, ...], slices.get("vehicle_age_bands_years"))) != AGE_CUTPOINTS
        or slices.get("vehicle_age_reference_year") != 2023
        or tuple(cast(tuple[object, ...], slices.get("actual_price_bands_usd"))) != PRICE_CUTPOINTS
        or tuple(cast(tuple[object, ...], slices.get("predicted_value_bands_usd")))
        != PREDICTED_VALUE_CUTPOINTS
    ):
        raise FinalEvaluationError("runtime slice definitions differ from final policy")


def _validate_prior_report(report: Mapping[str, object]) -> None:
    if (
        report.get("report_type") != "retail_rf05_uncertainty_sharpness_comparison"
        or report.get("classification") != "retain_current_calibration_baseline"
    ):
        raise FinalEvaluationError("prior sharpness decision differs from frozen evidence")
    decision = cast(Mapping[str, object], report.get("decision"))
    if (
        decision.get("selected_method") != BASELINE_METHOD
        or decision.get("new_serving_artifact_permitted") is not False
    ):
        raise FinalEvaluationError("prior uncertainty selection differs")


def _prediction_vector(values: object, *, expected_rows: int, label: str) -> NDArray[np.float64]:
    inspected = np.asarray(values, dtype=object)
    if inspected.ndim != 1 or len(inspected) != expected_rows:
        raise FinalEvaluationError(f"{label} must be a one-dimensional row match")
    if any(isinstance(value, (bool, np.bool_)) for value in inspected.tolist()):
        raise FinalEvaluationError(f"{label} must be numeric, not boolean")
    try:
        result = inspected.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise FinalEvaluationError(f"{label} must be numeric") from error
    if not np.isfinite(result).all() or (result < 0.0).any():
        raise FinalEvaluationError(f"{label} must be finite and nonnegative")
    return result


def _apply_calibration(
    predictions: NDArray[np.float64],
    statuses: NDArray[np.str_],
    *,
    coverage: float,
    artifact: RetailCalibrationArtifact,
) -> IntervalArrays:
    lower = np.empty(len(predictions), dtype=np.float64)
    upper = np.empty(len(predictions), dtype=np.float64)
    support = np.empty(len(predictions), dtype=np.int64)
    fallback = np.empty(len(predictions), dtype=np.bool_)
    clipped = np.empty(len(predictions), dtype=np.bool_)
    confidence = np.empty(len(predictions), dtype="<U20")
    for index, (prediction, status) in enumerate(zip(predictions, statuses, strict=True)):
        value = calibrated_valuation(
            point_prediction=float(prediction),
            vehicle_status=str(status),
            coverage=coverage,
            artifact=artifact,
        )
        lower[index] = value.interval_lower
        upper[index] = value.interval_upper
        support[index] = value.calibration_support
        fallback[index] = value.calibration_method == "global_fallback"
        clipped[index] = prediction - (value.interval_upper - prediction) < 0.0
        confidence[index] = value.confidence_label
    width = upper - lower
    arrays = IntervalArrays(
        lower=lower,
        upper=upper,
        width=width,
        support=support,
        fallback=fallback,
        clipped=clipped,
        confidence=confidence,
    )
    _validate_intervals(arrays, predictions)
    return arrays


def _validate_intervals(
    intervals: IntervalArrays,
    predictions: NDArray[np.float64],
) -> None:
    numeric = (intervals.lower, intervals.upper, intervals.width)
    if any(not np.isfinite(values).all() for values in numeric):
        raise FinalEvaluationError("calibration produced a non-finite interval")
    if (
        (intervals.lower < 0.0).any()
        or (intervals.upper < intervals.lower).any()
        or (intervals.lower > predictions).any()
        or (intervals.upper < predictions).any()
    ):
        raise FinalEvaluationError("calibration produced an invalid interval")


def _point_metrics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
) -> dict[str, object]:
    signed = predictions - target
    absolute = np.abs(signed)
    base = regression_metrics(target, predictions).to_dict()
    return {
        **base,
        "median_absolute_error_usd": float(np.median(absolute)),
        "absolute_error_usd": _distribution(absolute),
        "mean_signed_error_usd": float(np.mean(signed)),
        "median_signed_error_usd": float(np.median(signed)),
        "underprediction_count": int(np.count_nonzero(signed < 0.0)),
        "underprediction_rate": float(np.mean(signed < 0.0)),
        "overprediction_count": int(np.count_nonzero(signed > 0.0)),
        "overprediction_rate": float(np.mean(signed > 0.0)),
        "exact_prediction_count": int(np.count_nonzero(signed == 0.0)),
        "mean_target_usd": float(np.mean(target)),
        "mean_prediction_usd": float(np.mean(predictions)),
        "mape_reported": False,
        "mape_omission_reason": "low-dollar targets make percentage errors unstable",
    }


def _interval_metrics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: IntervalArrays,
    *,
    nominal: float,
) -> dict[str, object]:
    covered = (target >= intervals.lower) & (target <= intervals.upper)
    relative = intervals.width / np.maximum(predictions, 1.0)
    invalid = ~np.isfinite(intervals.lower) | ~np.isfinite(intervals.upper)
    reversed_or_excluding = (
        (intervals.upper < intervals.lower)
        | (intervals.lower > predictions)
        | (intervals.upper < predictions)
    )
    return {
        "sample_count": len(target),
        "nominal_coverage": nominal,
        "empirical_coverage": float(np.mean(covered)),
        "coverage_gap": float(np.mean(covered)) - nominal,
        "undercoverage_rate": float(np.mean(target < intervals.lower)),
        "overcoverage_rate": float(np.mean(target > intervals.upper)),
        "displayed_width_usd": _distribution(intervals.width),
        "relative_displayed_width": _distribution(relative),
        "lower_bound_clipping_count": int(np.count_nonzero(intervals.clipped)),
        "lower_bound_clipping_rate": float(np.mean(intervals.clipped)),
        "fallback_count": int(np.count_nonzero(intervals.fallback)),
        "fallback_rate": float(np.mean(intervals.fallback)),
        "invalid_or_nonfinite_interval_count": int(np.count_nonzero(invalid)),
        "point_exclusion_or_reversed_count": int(np.count_nonzero(reversed_or_excluding)),
    }


def _distribution(values: NDArray[np.float64]) -> dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise FinalEvaluationError("distribution requires a non-empty finite vector")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10, method="linear")),
        "p25": float(np.quantile(values, 0.25, method="linear")),
        "p75": float(np.quantile(values, 0.75, method="linear")),
        "p90": float(np.quantile(values, 0.90, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "maximum": float(np.max(values)),
    }


def _feature_coverage(
    features: pd.DataFrame,
    target: NDArray[np.float64],
) -> dict[str, object]:
    mileage = pd.to_numeric(features["mileage"], errors="coerce").to_numpy(dtype=np.float64)
    year = pd.to_numeric(features["year"], errors="raise").to_numpy(dtype=np.float64)
    present_mileage = np.isfinite(mileage)
    categorical = {
        name: {
            "nonmissing_count": int(features[name].notna().sum()),
            "missing_count": int(features[name].isna().sum()),
            "unique_count": int(features[name].nunique(dropna=True)),
        }
        for name in ("make", "model", "vehicle_status")
    }
    return {
        "required_feature_columns": list(RETAIL_TRACK.input_features),
        "required_feature_columns_present": tuple(features.columns) == RETAIL_TRACK.input_features,
        "year": {
            "nonmissing_count": int(np.count_nonzero(np.isfinite(year))),
            "missing_count": int(np.count_nonzero(~np.isfinite(year))),
            "minimum": int(np.min(year)),
            "maximum": int(np.max(year)),
        },
        "mileage": {
            "nonmissing_count": int(np.count_nonzero(present_mileage)),
            "missing_count": int(np.count_nonzero(~present_mileage)),
            "missing_rate": float(np.mean(~present_mileage)),
            "minimum_miles": float(np.min(mileage[present_mileage])),
            "median_miles": float(np.median(mileage[present_mileage])),
            "maximum_miles": float(np.max(mileage[present_mileage])),
        },
        **categorical,
        "vehicle_status_counts": {
            key: int(value)
            for key, value in features["vehicle_status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .value_counts()
            .sort_index()
            .items()
        },
        "target": {
            "expected_count": len(features),
            "valid_count": len(target),
            "valid_rate": 1.0,
            "minimum_usd": float(np.min(target)),
            "median_usd": float(np.median(target)),
            "maximum_usd": float(np.max(target)),
        },
        "unsupported_requested_features": {
            "trim": "not in frozen feature contract",
            "engine": "not in frozen feature contract",
            "transmission": "not in frozen feature contract",
            "drivetrain": "not in frozen feature contract",
            "condition": "not in frozen feature contract",
            "vehicle_history": "not in frozen feature contract",
        },
    }


def _slice_diagnostics(
    features: pd.DataFrame,
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals_90: IntervalArrays,
) -> dict[str, object]:
    mileage = pd.to_numeric(features["mileage"], errors="coerce").to_numpy(dtype=np.float64)
    year = pd.to_numeric(features["year"], errors="raise").to_numpy(dtype=np.float64)
    age = np.maximum(0.0, 2023.0 - year)
    dimensions: dict[str, NDArray[np.str_]] = {
        "manufacturer": np.asarray(
            [str(value).strip().lower() for value in features["make"]],
            dtype=np.str_,
        ),
        "vehicle_status": _normalized_statuses(features["vehicle_status"]),
        "mileage_band": _mileage_bands(mileage),
        "vehicle_age_band": _numeric_bands(age, AGE_CUTPOINTS, prefix="age"),
        "actual_price_band": _numeric_bands(target, PRICE_CUTPOINTS, prefix="price"),
        "predicted_value_band": _numeric_bands(
            predictions,
            PREDICTED_VALUE_CUTPOINTS,
            prefix="predicted_value",
        ),
        "mileage_presence": np.where(
            np.isfinite(mileage),
            "mileage_present",
            "mileage_missing",
        ).astype(np.str_),
    }
    result: dict[str, object] = {}
    for dimension, labels in dimensions.items():
        entries: list[dict[str, object]] = []
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            support = int(np.count_nonzero(mask))
            if support < MINIMUM_SLICE_SUPPORT:
                continue
            point = _point_metrics(target[mask], predictions[mask])
            interval = _masked_intervals(intervals_90, mask)
            coverage = _interval_metrics(
                target[mask],
                predictions[mask],
                interval,
                nominal=0.9,
            )
            entries.append(
                {
                    "label": label,
                    "sample_count": support,
                    "mae_usd": point["mae"],
                    "rmse_usd": point["rmse"],
                    "r2": point["r2"],
                    "median_absolute_error_usd": point["median_absolute_error_usd"],
                    "mean_signed_error_usd": point["mean_signed_error_usd"],
                    "interval_90pct": {
                        "empirical_coverage": coverage["empirical_coverage"],
                        "coverage_gap": coverage["coverage_gap"],
                        "mean_displayed_width_usd": cast(
                            Mapping[str, float], coverage["displayed_width_usd"]
                        )["mean"],
                        "median_displayed_width_usd": cast(
                            Mapping[str, float], coverage["displayed_width_usd"]
                        )["median"],
                    },
                }
            )
        result[dimension] = entries
    return result


def _masked_intervals(
    intervals: IntervalArrays,
    mask: NDArray[np.bool_],
) -> IntervalArrays:
    return IntervalArrays(
        lower=intervals.lower[mask],
        upper=intervals.upper[mask],
        width=intervals.width[mask],
        support=intervals.support[mask],
        fallback=intervals.fallback[mask],
        clipped=intervals.clipped[mask],
        confidence=intervals.confidence[mask],
    )


def _confidence_diagnostics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: IntervalArrays,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for label in ("High confidence", "Moderate confidence", "Low confidence"):
        mask = intervals.confidence == label
        support = int(np.count_nonzero(mask))
        if support == 0:
            entries.append({"label": label, "sample_count": 0})
            continue
        point = _point_metrics(target[mask], predictions[mask])
        interval = _interval_metrics(
            target[mask],
            predictions[mask],
            _masked_intervals(intervals, mask),
            nominal=0.9,
        )
        entries.append(
            {
                "label": label,
                "sample_count": support,
                "mae_usd": point["mae"],
                "median_absolute_error_usd": point["median_absolute_error_usd"],
                "empirical_coverage_90pct": interval["empirical_coverage"],
                "median_displayed_width_usd_90pct": cast(
                    Mapping[str, float], interval["displayed_width_usd"]
                )["median"],
            }
        )
    usable = [entry for entry in entries if cast(int, entry["sample_count"]) >= 200]
    all_present = len(usable) == 3
    ordered = {
        key: all_present
        and all(
            float(cast(float, usable[index][key])) <= float(cast(float, usable[index + 1][key]))
            for index in range(2)
        )
        for key in (
            "mae_usd",
            "median_absolute_error_usd",
            "median_displayed_width_usd_90pct",
        )
    }
    return {
        "semantics": "precision_and_support_label_not_probability",
        "thresholds_unchanged": CONFIDENCE_THRESHOLDS.to_dict(),
        "labels": entries,
        "all_labels_have_minimum_200_support": all_present,
        "expected_order": "High <= Moderate <= Low",
        "ordering": ordered,
        "all_expected_metrics_ordered": all(ordered.values()),
    }


def _manufacturer_summary(slices: Mapping[str, object]) -> dict[str, object]:
    items = cast(list[Mapping[str, object]], slices["manufacturer"])
    ordered = sorted(items, key=lambda item: (float(cast(float, item["mae_usd"])), item["label"]))

    def select(item: Mapping[str, object]) -> dict[str, object]:
        return {
            "manufacturer": item["label"],
            "sample_count": item["sample_count"],
            "mae_usd": item["mae_usd"],
            "rmse_usd": item["rmse_usd"],
            "mean_signed_error_usd": item["mean_signed_error_usd"],
            "coverage_90pct": cast(Mapping[str, object], item["interval_90pct"])[
                "empirical_coverage"
            ],
        }

    return {
        "ranking_metric": "mae_usd",
        "minimum_support": MINIMUM_SLICE_SUPPORT,
        "supported_manufacturer_count": len(items),
        "strongest_five": [select(item) for item in ordered[:5]],
        "weakest_five": [select(item) for item in reversed(ordered[-5:])],
        "diagnostic_only_not_a_tuning_target": True,
    }


def _prior_comparison(
    point: Mapping[str, object],
    interval_report: Mapping[str, object],
    slices: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    reconstruction = cast(Mapping[str, object], prior["reconstruction"])
    development = cast(Mapping[str, object], reconstruction["development_point_metrics"])
    calibration_point = cast(Mapping[str, object], reconstruction["calibration_point_metrics"])
    methods = cast(Mapping[str, object], prior["methods"])
    calibration_method = cast(Mapping[str, object], methods[BASELINE_METHOD])
    calibration_coverages = cast(Mapping[str, object], calibration_method["coverages"])
    point_comparison = {
        "development_oof": _point_comparison(point, development),
        "calibration": _point_comparison(point, calibration_point),
    }
    interval_comparison: dict[str, object] = {}
    for level in COVERAGE_LEVELS:
        key = str(level)
        final = cast(Mapping[str, object], interval_report[key])
        baseline = cast(Mapping[str, object], calibration_coverages[key])
        final_width = cast(Mapping[str, float], final["displayed_width_usd"])
        baseline_width = cast(Mapping[str, float], baseline["displayed_width_usd"])
        interval_comparison[key] = {
            "final_empirical_coverage": final["empirical_coverage"],
            "calibration_empirical_coverage": baseline["empirical_coverage"],
            "coverage_delta_final_minus_calibration": float(
                cast(float, final["empirical_coverage"])
            )
            - float(cast(float, baseline["empirical_coverage"])),
            "final_mean_displayed_width_usd": final_width["mean"],
            "calibration_mean_displayed_width_usd": baseline_width["mean"],
            "mean_width_ratio_final_to_calibration": final_width["mean"] / baseline_width["mean"],
        }
    return {
        "point": point_comparison,
        "interval": interval_comparison,
        "slice_coverage_90pct_final_minus_calibration": _slice_comparison(
            slices,
            cast(Mapping[str, object], calibration_method["slices"]),
        ),
        "interpretation_rule": (
            "positive error/width gaps are worse; negative coverage gaps are worse"
        ),
    }


def _point_comparison(
    final: Mapping[str, object],
    prior: Mapping[str, object],
) -> dict[str, object]:
    final_mae = float(cast(float, final["mae"]))
    prior_mae = float(cast(float, prior["mae"]))
    final_rmse = float(cast(float, final["rmse"]))
    prior_rmse = float(cast(float, prior["rmse"]))
    return {
        "final_mae_usd": final_mae,
        "prior_mae_usd": prior_mae,
        "mae_difference_usd": final_mae - prior_mae,
        "mae_ratio": final_mae / prior_mae,
        "mae_relative_change": final_mae / prior_mae - 1.0,
        "final_rmse_usd": final_rmse,
        "prior_rmse_usd": prior_rmse,
        "rmse_difference_usd": final_rmse - prior_rmse,
        "rmse_ratio": final_rmse / prior_rmse,
        "final_r2": final["r2"],
        "prior_r2": prior["r2"],
        "r2_difference": float(cast(float, final["r2"])) - float(cast(float, prior["r2"])),
    }


def _slice_comparison(
    final_slices: Mapping[str, object],
    calibration_slices: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dimension in (
        "manufacturer",
        "mileage_band",
        "vehicle_age_band",
        "actual_price_band",
        "predicted_value_band",
    ):
        final_items = {
            cast(str, item["label"]): item
            for item in cast(list[Mapping[str, object]], final_slices[dimension])
        }
        prior_items = {
            cast(str, item["label"]): item
            for item in cast(list[Mapping[str, object]], calibration_slices[dimension])
        }
        comparisons: list[dict[str, object]] = []
        for label in sorted(set(final_items) & set(prior_items)):
            final = final_items[label]
            prior = prior_items[label]
            final_interval = cast(Mapping[str, object], final["interval_90pct"])
            prior_coverages = cast(Mapping[str, object], prior["coverages"])
            prior_interval = cast(Mapping[str, object], prior_coverages["0.9"])
            comparisons.append(
                {
                    "label": label,
                    "final_sample_count": final["sample_count"],
                    "calibration_sample_count": prior["sample_count"],
                    "final_mae_usd": final["mae_usd"],
                    "calibration_mae_usd": cast(Mapping[str, object], prior["point_metrics"])[
                        "mae"
                    ],
                    "mae_difference_usd": float(cast(float, final["mae_usd"]))
                    - float(
                        cast(
                            float,
                            cast(Mapping[str, object], prior["point_metrics"])["mae"],
                        )
                    ),
                    "final_coverage_90pct": final_interval["empirical_coverage"],
                    "calibration_coverage_90pct": prior_interval["empirical_coverage"],
                    "coverage_delta_final_minus_calibration": float(
                        cast(float, final_interval["empirical_coverage"])
                    )
                    - float(cast(float, prior_interval["empirical_coverage"])),
                }
            )
        result[dimension] = comparisons
    return result


def _classify(
    *,
    point: Mapping[str, object],
    interval_report: Mapping[str, object],
    slices: Mapping[str, object],
    confidence: Mapping[str, object],
    comparison: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    comparison_point = cast(Mapping[str, object], comparison["point"])
    development = cast(Mapping[str, object], comparison_point["development_oof"])
    interval_comparison = cast(Mapping[str, object], comparison["interval"])
    mean_target = float(cast(float, point["mean_target_usd"]))
    bias_ratio = abs(float(cast(float, point["mean_signed_error_usd"]))) / mean_target
    invalid = sum(
        int(
            cast(
                int,
                cast(Mapping[str, object], interval_report[str(level)])[
                    "invalid_or_nonfinite_interval_count"
                ],
            )
        )
        for level in COVERAGE_LEVELS
    )
    reversed_count = sum(
        int(
            cast(
                int,
                cast(Mapping[str, object], interval_report[str(level)])[
                    "point_exclusion_or_reversed_count"
                ],
            )
        )
        for level in COVERAGE_LEVELS
    )
    significant: list[dict[str, object]] = []
    _gate(significant, "rejected_preapproved_holdout_rows", 0, 0, "<=")
    _gate(significant, "invalid_prediction_or_interval_count", invalid, 0, "<=")
    _gate(significant, "point_exclusion_or_reversed_count", reversed_count, 0, "<=")
    _gate(significant, "mae_ratio_vs_development_oof", development["mae_ratio"], 1.5, "<=")
    _gate(significant, "rmse_ratio_vs_development_oof", development["rmse_ratio"], 1.5, "<=")
    _gate(significant, "r2", point["r2"], 0.0, ">=")
    _gate(significant, "absolute_mean_signed_error_ratio", bias_ratio, 0.25, "<=")
    for level in COVERAGE_LEVELS:
        key = str(level)
        item = cast(Mapping[str, object], interval_report[key])
        prior = cast(Mapping[str, object], interval_comparison[key])
        _gate(significant, f"coverage_gap_{key}", item["coverage_gap"], -0.10, ">=")
        _gate(
            significant,
            f"coverage_regression_vs_calibration_{key}",
            -float(cast(float, prior["coverage_delta_final_minus_calibration"])),
            0.10,
            "<=",
        )
        _gate(
            significant,
            f"mean_width_ratio_vs_calibration_{key}",
            prior["mean_width_ratio_final_to_calibration"],
            2.0,
            "<=",
        )
    portfolio: list[dict[str, object]] = []
    _gate(portfolio, "mae_ratio_vs_development_oof", development["mae_ratio"], 1.2, "<=")
    _gate(portfolio, "rmse_ratio_vs_development_oof", development["rmse_ratio"], 1.25, "<=")
    _gate(portfolio, "r2", point["r2"], 0.30, ">=")
    _gate(portfolio, "absolute_mean_signed_error_ratio", bias_ratio, 0.10, "<=")
    for level in COVERAGE_LEVELS:
        key = str(level)
        item = cast(Mapping[str, object], interval_report[key])
        prior = cast(Mapping[str, object], interval_comparison[key])
        _gate(portfolio, f"coverage_gap_{key}", item["coverage_gap"], -0.03, ">=")
        _gate(
            portfolio,
            f"coverage_regression_vs_calibration_{key}",
            -float(cast(float, prior["coverage_delta_final_minus_calibration"])),
            0.03,
            "<=",
        )
        _gate(
            portfolio,
            f"mean_width_ratio_vs_calibration_{key}",
            prior["mean_width_ratio_final_to_calibration"],
            1.35,
            "<=",
        )
    status_min = _minimum_slice_coverage(slices, ("vehicle_status",))
    broad_min = _minimum_slice_coverage(
        slices,
        (
            "mileage_band",
            "vehicle_age_band",
            "actual_price_band",
            "predicted_value_band",
            "mileage_presence",
        ),
    )
    manufacturer_min = _minimum_slice_coverage(slices, ("manufacturer",))
    _gate(portfolio, "minimum_vehicle_status_coverage_90pct", status_min, 0.80, ">=")
    _gate(portfolio, "minimum_broad_slice_coverage_90pct", broad_min, 0.70, ">=")
    _gate(portfolio, "minimum_manufacturer_coverage_90pct", manufacturer_min, 0.60, ">=")
    _gate(
        portfolio,
        "confidence_labels_have_minimum_support",
        int(bool(confidence["all_labels_have_minimum_200_support"])),
        1,
        ">=",
    )
    _gate(
        portfolio,
        "confidence_metrics_ordered",
        int(bool(confidence["all_expected_metrics_ordered"])),
        1,
        ">=",
    )
    significant_passed = all(bool(item["passed"]) for item in significant)
    portfolio_passed = all(bool(item["passed"]) for item in portfolio)
    classification = (
        CLASSIFICATIONS[0]
        if not significant_passed
        else CLASSIFICATIONS[2]
        if portfolio_passed
        else CLASSIFICATIONS[1]
    )
    return classification, {
        "significant_generalization_gates": {
            "passed_all": significant_passed,
            "failed_count": sum(not bool(item["passed"]) for item in significant),
            "outcomes": significant,
        },
        "portfolio_demo_gates": {
            "passed_all": portfolio_passed,
            "failed_count": sum(not bool(item["passed"]) for item in portfolio),
            "outcomes": portfolio,
        },
        "observed_minimum_coverage_90pct": {
            "vehicle_status": status_min,
            "broad_slice": broad_min,
            "manufacturer": manufacturer_min,
        },
    }


def _minimum_slice_coverage(
    slices: Mapping[str, object],
    dimensions: tuple[str, ...],
) -> float:
    values = [
        float(
            cast(
                float,
                cast(Mapping[str, object], item["interval_90pct"])["empirical_coverage"],
            )
        )
        for dimension in dimensions
        for item in cast(list[Mapping[str, object]], slices[dimension])
    ]
    if not values:
        raise FinalEvaluationError("classification requires supported slice coverage")
    return min(values)


def _gate(
    outcomes: list[dict[str, object]],
    name: str,
    observed: object,
    threshold: float | int,
    operator: Literal["<=", ">="],
) -> None:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise FinalEvaluationError(f"classification metric is not numeric: {name}")
    value = float(observed)
    if not math.isfinite(value):
        raise FinalEvaluationError(f"classification metric is not finite: {name}")
    passed = value <= threshold if operator == "<=" else value >= threshold
    outcomes.append(
        {
            "gate": name,
            "observed": observed,
            "operator": operator,
            "threshold": threshold,
            "passed": bool(passed),
        }
    )


__all__ = [
    "CLASSIFICATIONS",
    "FINAL_GENERATED_AT",
    "FINAL_HOLDOUT_IDENTITY_SHA256",
    "FINAL_HOLDOUT_ROWS",
    "FINAL_REPORT_TYPE",
    "FinalEvaluationError",
    "FinalEvaluationResult",
    "canonical_final_report_json",
    "evaluate_final_holdout",
    "fit_frozen_rf05_for_final",
]
