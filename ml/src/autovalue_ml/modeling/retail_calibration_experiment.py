"""Governed split-conformal calibration for the frozen Phase 4 retail RF05 model."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from .calibration import RETAIL_VEHICLE_STATUSES, retail_calibration_partition
from .calibration_artifact import (
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_POLICY_SHA256,
    CALIBRATION_SAMPLE_COUNT,
    COVERAGE_LEVELS,
    DEVELOPMENT_SAMPLE_COUNT,
    MINIMUM_BUCKET_SUPPORT,
    PHASE4_RETAIL_CONFIRMATION_SHA256,
    CalibrationMethod,
    ConditionalRadius,
    ConfidenceThresholds,
    CoverageCalibration,
    RetailCalibrationArtifact,
    active_rf05_identity,
)
from .candidates import get_candidate_spec, make_random_forest_candidate
from .contracts import RETAIL_TRACK, validate_feature_frame, validate_target
from .cv import retail_group_cv_splits
from .metrics import regression_metrics
from .phase4_confirmation import Phase4ConfirmationReport
from .phase4_protocol import Phase4Protocol
from .phase4_screening_experiment import _partition_hash

CALIBRATION_SEED: Final = 1_416_582_761
CALIBRATION_REPORT_TYPE: Final = "retail_rf05_split_conformal_calibration_report"
GENERATED_AT: Final = "2026-09-02T12:00:00+00:00"
MINIMUM_REPORTED_SLICE_SUPPORT: Final = 200
PRICE_CUTPOINTS: Final = (8_995.0, 19_995.0, 36_590.0)
MILEAGE_CUTPOINTS: Final = (38_282.0, 86_204.0, 135_803.0)
AGE_CUTPOINTS: Final = (3.0, 8.0, 13.0)
METHODS: Final = (
    "global",
    "vehicle_status",
    "vehicle_status_and_predicted_value_band_hierarchy",
)


class CalibrationExperimentError(ValueError):
    """The calibration population, model evidence, or result violated policy."""


class Regressor(Protocol):
    """Small estimator protocol used to prove fit/predict separation in tests."""

    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> Regressor: ...

    def predict(self, features: pd.DataFrame) -> object: ...


EstimatorFactory = Callable[[], Regressor]


@dataclass(frozen=True, slots=True)
class CalibrationExperimentResult:
    """Aggregate report and row-free serving artifact from one authorized run."""

    artifact: RetailCalibrationArtifact
    report: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _AppliedIntervals:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    radii: NDArray[np.float64]
    supports: NDArray[np.int64]


def run_retail_rf05_calibration(
    *,
    phase3_train_features: object,
    phase3_train_target: object,
    protocol: Phase4Protocol,
    confirmation: Phase4ConfirmationReport,
    confirmation_sha256: str,
    estimator_factory: EstimatorFactory | None = None,
    generated_at: str = GENERATED_AT,
) -> CalibrationExperimentResult:
    """Use reserved calibration rows once, without fitting or selecting an estimator on them."""

    frame = validate_feature_frame(phase3_train_features, RETAIL_TRACK)
    target = validate_target(
        phase3_train_target,
        expected_rows=len(frame),
        config=RETAIL_TRACK,
    )
    _validate_frozen_evidence(
        frame=frame,
        protocol=protocol,
        confirmation=confirmation,
        confirmation_sha256=confirmation_sha256,
    )
    partition = retail_calibration_partition(frame, seed=CALIBRATION_SEED)
    assignment_hash = _partition_hash(
        partition.calibration_indices,
        population_count=len(frame),
        selected_label="calibration",
        unselected_label="development",
    )
    if assignment_hash != CALIBRATION_ASSIGNMENT_SHA256:
        raise CalibrationExperimentError("calibration assignment differs from the frozen audit")
    if (
        len(partition.development_indices) != DEVELOPMENT_SAMPLE_COUNT
        or len(partition.calibration_indices) != CALIBRATION_SAMPLE_COUNT
    ):
        raise CalibrationExperimentError("calibration population counts differ from policy")

    development_features = frame.iloc[partition.development_indices].reset_index(drop=True)
    development_target = target[partition.development_indices]
    calibration_features = frame.iloc[partition.calibration_indices].reset_index(drop=True)
    calibration_target = target[partition.calibration_indices]
    predictions = fit_frozen_rf05_calibration_predictions(
        development_features=development_features,
        development_target=development_target,
        calibration_features=calibration_features,
        estimator_factory=estimator_factory,
    )
    residuals = np.abs(calibration_target - predictions)
    statuses = _normalized_statuses(calibration_features["vehicle_status"])
    cutpoints = cast(
        tuple[float, float, float],
        tuple(float(value) for value in np.quantile(predictions, (0.25, 0.5, 0.75))),
    )
    if not cutpoints[0] < cutpoints[1] < cutpoints[2]:
        raise CalibrationExperimentError("target-free prediction cutpoints are not distinct")
    value_bands = _numeric_bands(predictions, cutpoints, prefix="band")

    crossfit = _crossfit_diagnostics(
        features=calibration_features,
        target=calibration_target,
        predictions=predictions,
        residuals=residuals,
        statuses=statuses,
        value_bands=value_bands,
    )
    gate_results = _conditional_gate_results(crossfit)
    selected_method: CalibrationMethod = (
        "vehicle_status_and_predicted_value_band_hierarchy"
        if all(cast(bool, gate["passed"]) for gate in gate_results)
        else "vehicle_status"
    )
    full_calibrations = tuple(
        _fit_coverage_calibration(
            residuals,
            statuses,
            value_bands,
            coverage=coverage,
        )
        for coverage in COVERAGE_LEVELS
    )
    selected_crossfit = cast(
        Mapping[str, object],
        cast(Mapping[str, object], crossfit["methods"])[selected_method],
    )
    selected_90 = cast(
        Mapping[str, object],
        cast(Mapping[str, object], selected_crossfit["coverages"])["0.9"],
    )
    relative_widths = cast(NDArray[np.float64], selected_90["_relative_widths"])
    confidence_cutpoints = np.quantile(relative_widths, (0.33, 0.67), method="linear")
    confidence = ConfidenceThresholds(
        coverage=0.9,
        high_max_relative_width=float(confidence_cutpoints[0]),
        moderate_max_relative_width=float(confidence_cutpoints[1]),
    )
    artifact = RetailCalibrationArtifact(
        generated_at=generated_at,
        bound_model=active_rf05_identity(),
        selected_method=selected_method,
        predicted_value_cutpoints_usd=cutpoints,
        coverage_calibrations=full_calibrations,
        confidence_thresholds=confidence,
    )
    selected_interval_arrays = {
        key: cast(Mapping[str, object], value)
        for key, value in cast(Mapping[str, object], selected_crossfit["coverages"]).items()
    }
    diagnostics = _selected_slice_diagnostics(
        features=calibration_features,
        target=calibration_target,
        predictions=predictions,
        intervals=selected_interval_arrays,
    )
    confidence_diagnostics = _confidence_diagnostics(
        relative_widths,
        cast(NDArray[np.int64], selected_90["_supports"]),
        confidence,
    )
    report = _build_report(
        confirmation=confirmation,
        assignment_hash=assignment_hash,
        predictions=predictions,
        target=calibration_target,
        statuses=statuses,
        cutpoints=cutpoints,
        crossfit=crossfit,
        selected_method=selected_method,
        gate_results=gate_results,
        diagnostics=diagnostics,
        confidence=confidence,
        confidence_diagnostics=confidence_diagnostics,
        full_calibrations=full_calibrations,
        generated_at=generated_at,
    )
    return CalibrationExperimentResult(artifact=artifact, report=report)


def fit_frozen_rf05_calibration_predictions(
    *,
    development_features: pd.DataFrame,
    development_target: NDArray[np.float64],
    calibration_features: pd.DataFrame,
    estimator_factory: EstimatorFactory | None = None,
) -> NDArray[np.float64]:
    """Fit only on development and predict only on calibration."""

    development = validate_feature_frame(development_features, RETAIL_TRACK)
    calibration = validate_feature_frame(calibration_features, RETAIL_TRACK)
    target = validate_target(
        development_target,
        expected_rows=len(development),
        config=RETAIL_TRACK,
    )
    if len(calibration) == 0:
        raise CalibrationExperimentError("calibration features must not be empty")
    factory = estimator_factory or _rf05_factory
    estimator = factory()
    estimator.fit(development, target)
    raw_predictions = np.asarray(estimator.predict(calibration))
    if raw_predictions.ndim != 1 or len(raw_predictions) != len(calibration):
        raise CalibrationExperimentError("RF05 calibration predictions have an invalid shape")
    if np.issubdtype(raw_predictions.dtype, np.bool_):
        raise CalibrationExperimentError("RF05 calibration predictions must be numeric")
    try:
        predictions = raw_predictions.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise CalibrationExperimentError("RF05 calibration predictions must be numeric") from error
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise CalibrationExperimentError("RF05 calibration predictions must be nonnegative finite")
    predictions.setflags(write=False)
    return predictions


def canonical_calibration_report_json(report: Mapping[str, object]) -> str:
    """Serialize aggregate calibration evidence deterministically."""

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
        raise CalibrationExperimentError("calibration report is not JSON-safe") from error


def report_sha256(report: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_calibration_report_json(report).encode("utf-8")).hexdigest()


def _rf05_factory() -> Pipeline:
    return make_random_forest_candidate("retail", 5, n_jobs=4)


def _validate_frozen_evidence(
    *,
    frame: pd.DataFrame,
    protocol: Phase4Protocol,
    confirmation: Phase4ConfirmationReport,
    confirmation_sha256: str,
) -> None:
    if len(frame) != DEVELOPMENT_SAMPLE_COUNT + CALIBRATION_SAMPLE_COUNT:
        raise CalibrationExperimentError("Phase-3 retail train count differs from policy")
    track_policy = protocol.for_track("retail")
    if (
        track_policy.phase3_train_rows != len(frame)
        or track_policy.calibration_seed != CALIBRATION_SEED
    ):
        raise CalibrationExperimentError("Phase 4 retail protocol differs from calibration policy")
    if confirmation_sha256 != PHASE4_RETAIL_CONFIRMATION_SHA256:
        raise CalibrationExperimentError("Phase 4 retail confirmation checksum differs")
    if confirmation.track != "retail" or confirmation.metric_ranking[0] != (
        "phase4-retail-random_forest-05"
    ):
        raise CalibrationExperimentError("frozen retail RF05 is not the confirmed metric leader")
    spec = get_candidate_spec("retail", "random_forest", 5)
    identity = active_rf05_identity()
    if (
        spec.candidate_id != identity.candidate_id
        or spec.parameters != identity.parameters
        or spec.random_state != identity.random_state
    ):
        raise CalibrationExperimentError("current RF05 definition differs from frozen identity")


def _crossfit_diagnostics(
    *,
    features: pd.DataFrame,
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    residuals: NDArray[np.float64],
    statuses: NDArray[np.str_],
    value_bands: NDArray[np.str_],
) -> dict[str, object]:
    splits = retail_group_cv_splits(features, n_splits=5)
    method_store: dict[str, dict[str, dict[str, object]]] = {
        method: {
            str(coverage): {
                "lower": np.full(len(features), np.nan, dtype=np.float64),
                "upper": np.full(len(features), np.nan, dtype=np.float64),
                "radii": np.full(len(features), np.nan, dtype=np.float64),
                "supports": np.zeros(len(features), dtype=np.int64),
                "folds": [],
            }
            for coverage in COVERAGE_LEVELS
        }
        for method in METHODS
    }
    validation_counts = np.zeros(len(features), dtype=np.int8)
    fold_shapes: list[dict[str, int]] = []
    for fold_number, (training_indices, validation_indices) in enumerate(splits, start=1):
        validation_counts[validation_indices] += 1
        fold_shapes.append(
            {
                "fold_number": fold_number,
                "calibration_fit_rows": len(training_indices),
                "diagnostic_rows": len(validation_indices),
            }
        )
        for coverage in COVERAGE_LEVELS:
            fitted = _fit_coverage_calibration(
                residuals[training_indices],
                statuses[training_indices],
                value_bands[training_indices],
                coverage=coverage,
            )
            for method in METHODS:
                applied = _apply_intervals(
                    predictions[validation_indices],
                    statuses[validation_indices],
                    value_bands[validation_indices],
                    fitted,
                    method=method,
                    global_support=len(training_indices),
                )
                store = method_store[method][str(coverage)]
                cast(NDArray[np.float64], store["lower"])[validation_indices] = applied.lower
                cast(NDArray[np.float64], store["upper"])[validation_indices] = applied.upper
                cast(NDArray[np.float64], store["radii"])[validation_indices] = applied.radii
                cast(NDArray[np.int64], store["supports"])[validation_indices] = applied.supports
                cast(list[dict[str, object]], store["folds"]).append(
                    {
                        "fold_number": fold_number,
                        **_interval_metrics(
                            target[validation_indices],
                            applied.lower,
                            applied.upper,
                            coverage,
                        ),
                    }
                )
    if not np.all(validation_counts == 1):
        raise CalibrationExperimentError("cross-calibration did not score every row exactly once")

    methods_report: dict[str, object] = {}
    for method, coverage_store in method_store.items():
        coverage_report: dict[str, object] = {}
        for coverage in COVERAGE_LEVELS:
            store = coverage_store[str(coverage)]
            lower = cast(NDArray[np.float64], store["lower"])
            upper = cast(NDArray[np.float64], store["upper"])
            radii = cast(NDArray[np.float64], store["radii"])
            supports = cast(NDArray[np.int64], store["supports"])
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise CalibrationExperimentError("cross-calibration left unscored rows")
            widths = upper - lower
            relative_widths = widths / np.maximum(predictions, 1.0)
            status_metrics = {
                status: _interval_metrics(
                    target[statuses == status],
                    lower[statuses == status],
                    upper[statuses == status],
                    coverage,
                )
                for status in RETAIL_VEHICLE_STATUSES
            }
            coverage_report[str(coverage)] = {
                **_interval_metrics(target, lower, upper, coverage),
                "status": status_metrics,
                "folds": store["folds"],
                "fold_coverage_standard_deviation": float(
                    np.std(
                        [
                            cast(float, item["empirical_coverage"])
                            for item in cast(list[dict[str, object]], store["folds"])
                        ],
                        ddof=0,
                    )
                ),
                "_lower": lower,
                "_upper": upper,
                "_radii": radii,
                "_supports": supports,
                "_relative_widths": relative_widths,
            }
        methods_report[method] = {"coverages": coverage_report}
    return {
        "folds": fold_shapes,
        "methods": methods_report,
        "_target": target,
        "_highest_price_mask": target > PRICE_CUTPOINTS[-1],
    }


def _fit_coverage_calibration(
    residuals: NDArray[np.float64],
    statuses: NDArray[np.str_],
    value_bands: NDArray[np.str_],
    *,
    coverage: float,
) -> CoverageCalibration:
    return CoverageCalibration(
        coverage=coverage,
        global_radius_usd=_finite_sample_radius(residuals, coverage),
        status_radii=tuple(
            _conditional_radius(status, residuals[statuses == status], coverage)
            for status in RETAIL_VEHICLE_STATUSES
            if np.any(statuses == status)
        ),
        predicted_value_band_radii=tuple(
            _conditional_radius(band, residuals[value_bands == band], coverage)
            for band in ("band_1", "band_2", "band_3", "band_4")
            if np.any(value_bands == band)
        ),
        status_value_band_radii=tuple(
            _conditional_radius(
                f"{status}|{band}",
                residuals[(statuses == status) & (value_bands == band)],
                coverage,
            )
            for status in RETAIL_VEHICLE_STATUSES
            for band in ("band_1", "band_2", "band_3", "band_4")
            if np.any((statuses == status) & (value_bands == band))
        ),
    )


def _conditional_radius(
    key: str,
    residuals: NDArray[np.float64],
    coverage: float,
) -> ConditionalRadius:
    support = len(residuals)
    radius = (
        _finite_sample_radius(residuals, coverage)
        if support >= MINIMUM_BUCKET_SUPPORT and math.ceil((support + 1) * coverage) <= support
        else None
    )
    return ConditionalRadius(key=key, support=support, radius_usd=radius)


def _finite_sample_radius(residuals: NDArray[np.float64], coverage: float) -> float:
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise CalibrationExperimentError("conformal residuals must be a non-empty finite vector")
    order = math.ceil((len(residuals) + 1) * coverage)
    if order > len(residuals):
        raise CalibrationExperimentError("calibration bucket is too small for coverage")
    return float(np.partition(residuals.copy(), order - 1)[order - 1])


def _apply_intervals(
    predictions: NDArray[np.float64],
    statuses: NDArray[np.str_],
    value_bands: NDArray[np.str_],
    calibration: CoverageCalibration,
    *,
    method: str,
    global_support: int,
) -> _AppliedIntervals:
    if method not in METHODS:
        raise CalibrationExperimentError("unknown calibration method")
    status_map = {item.key: item for item in calibration.status_radii}
    band_map = {item.key: item for item in calibration.predicted_value_band_radii}
    exact_map = {item.key: item for item in calibration.status_value_band_radii}
    radii = np.empty(len(predictions), dtype=np.float64)
    supports = np.empty(len(predictions), dtype=np.int64)
    for position, (status, band) in enumerate(zip(statuses, value_bands, strict=True)):
        chosen: ConditionalRadius | None = None
        if method == "vehicle_status_and_predicted_value_band_hierarchy":
            exact = exact_map.get(f"{status}|{band}")
            if exact is not None and exact.radius_usd is not None:
                chosen = exact
        if method != "global" and chosen is None:
            status_entry = status_map.get(str(status))
            if status_entry is not None and status_entry.radius_usd is not None:
                chosen = status_entry
        if method == "vehicle_status_and_predicted_value_band_hierarchy" and chosen is None:
            band_entry = band_map.get(str(band))
            if band_entry is not None and band_entry.radius_usd is not None:
                chosen = band_entry
        if chosen is None:
            radii[position] = calibration.global_radius_usd
            supports[position] = global_support
        else:
            radii[position] = cast(float, chosen.radius_usd)
            supports[position] = chosen.support
    lower = np.maximum(0.0, predictions - radii)
    upper = predictions + radii
    return _AppliedIntervals(lower=lower, upper=upper, radii=radii, supports=supports)


def _interval_metrics(
    target: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    nominal_coverage: float,
) -> dict[str, object]:
    if len(target) == 0 or len(target) != len(lower) or len(target) != len(upper):
        raise CalibrationExperimentError("interval metrics require aligned non-empty vectors")
    covered = (target >= lower) & (target <= upper)
    widths = upper - lower
    empirical = float(np.mean(covered))
    return {
        "sample_count": len(target),
        "nominal_coverage": nominal_coverage,
        "empirical_coverage": empirical,
        "coverage_gap": empirical - nominal_coverage,
        "undercoverage_rate": float(np.mean(target < lower)),
        "overcoverage_rate": float(np.mean(target > upper)),
        "average_width_usd": float(np.mean(widths)),
        "median_width_usd": float(np.median(widths)),
        "width_percentiles_usd": {
            str(percentile): float(np.quantile(widths, percentile / 100.0, method="linear"))
            for percentile in (10, 25, 75, 90, 95)
        },
    }


def _conditional_gate_results(crossfit: Mapping[str, object]) -> list[dict[str, object]]:
    methods = cast(Mapping[str, object], crossfit["methods"])
    conditional = cast(Mapping[str, object], methods[METHODS[2]])
    status = cast(Mapping[str, object], methods[METHODS[1]])
    conditional_coverages = cast(Mapping[str, object], conditional["coverages"])
    status_coverages = cast(Mapping[str, object], status["coverages"])
    results: list[dict[str, object]] = []
    overall_gaps = [
        cast(float, cast(Mapping[str, object], conditional_coverages[str(level)])["coverage_gap"])
        for level in COVERAGE_LEVELS
    ]
    results.append(
        {
            "gate": "minimum_overall_coverage_gap",
            "threshold": -0.02,
            "observed_worst": min(overall_gaps),
            "passed": min(overall_gaps) >= -0.02,
        }
    )
    status_gaps: list[float] = []
    for level in COVERAGE_LEVELS:
        status_map = cast(
            Mapping[str, object],
            cast(Mapping[str, object], conditional_coverages[str(level)])["status"],
        )
        status_gaps.extend(
            cast(float, cast(Mapping[str, object], item)["coverage_gap"])
            for item in status_map.values()
        )
    results.append(
        {
            "gate": "minimum_status_coverage_gap",
            "threshold": -0.05,
            "observed_worst": min(status_gaps),
            "passed": min(status_gaps) >= -0.05,
        }
    )
    width_ratios = []
    for level in COVERAGE_LEVELS:
        conditional_item = cast(Mapping[str, object], conditional_coverages[str(level)])
        status_item = cast(Mapping[str, object], status_coverages[str(level)])
        width_ratios.append(
            cast(float, conditional_item["average_width_usd"])
            / cast(float, status_item["average_width_usd"])
        )
    results.append(
        {
            "gate": "maximum_average_width_ratio_vs_status",
            "threshold": 1.05,
            "observed_worst": max(width_ratios),
            "passed": max(width_ratios) <= 1.05,
        }
    )
    conditional_90 = cast(Mapping[str, object], conditional_coverages["0.9"])
    status_90 = cast(Mapping[str, object], status_coverages["0.9"])
    highest_price_mask = cast(NDArray[np.bool_], crossfit["_highest_price_mask"])
    target = cast(NDArray[np.float64], crossfit["_target"])
    conditional_high = _masked_coverage(
        cast(NDArray[np.float64], conditional_90["_lower"]),
        cast(NDArray[np.float64], conditional_90["_upper"]),
        highest_price_mask,
        target,
    )
    status_high = _masked_coverage(
        cast(NDArray[np.float64], status_90["_lower"]),
        cast(NDArray[np.float64], status_90["_upper"]),
        highest_price_mask,
        target,
    )
    regression = status_high - conditional_high
    results.append(
        {
            "gate": "maximum_90pct_high_price_coverage_regression_vs_status",
            "threshold": 0.02,
            "observed": regression,
            "passed": regression <= 0.02,
        }
    )
    return results


def _masked_coverage(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    mask: NDArray[np.bool_],
    target: NDArray[np.float64],
) -> float:
    return float(np.mean((target[mask] >= lower[mask]) & (target[mask] <= upper[mask])))


def _selected_slice_diagnostics(
    *,
    features: pd.DataFrame,
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    intervals: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    mileage = pd.to_numeric(features["mileage"], errors="coerce").to_numpy(dtype=np.float64)
    year = pd.to_numeric(features["year"], errors="raise").to_numpy(dtype=np.float64)
    age = np.maximum(0.0, 2023.0 - year)
    dimensions: dict[str, NDArray[np.str_]] = {
        "actual_price_band": _numeric_bands(target, PRICE_CUTPOINTS, prefix="price"),
        "mileage_band": _mileage_bands(mileage),
        "vehicle_age_band": _numeric_bands(age, AGE_CUTPOINTS, prefix="age"),
        "manufacturer": np.asarray(
            [str(value).strip().lower() for value in features["make"]], dtype=np.str_
        ),
    }
    report: dict[str, object] = {}
    for dimension, labels in dimensions.items():
        slices: list[dict[str, object]] = []
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            if int(np.count_nonzero(mask)) < MINIMUM_REPORTED_SLICE_SUPPORT:
                continue
            coverage_metrics: dict[str, object] = {}
            for coverage in COVERAGE_LEVELS:
                item = intervals[str(coverage)]
                coverage_metrics[str(coverage)] = _interval_metrics(
                    target[mask],
                    cast(NDArray[np.float64], item["_lower"])[mask],
                    cast(NDArray[np.float64], item["_upper"])[mask],
                    coverage,
                )
            slices.append(
                {
                    "label": label,
                    "sample_count": int(np.count_nonzero(mask)),
                    "point_metrics": regression_metrics(target[mask], predictions[mask]).to_dict(),
                    "coverages": coverage_metrics,
                }
            )
        report[dimension] = slices
    return report


def _confidence_diagnostics(
    relative_widths: NDArray[np.float64],
    supports: NDArray[np.int64],
    thresholds: ConfidenceThresholds,
) -> dict[str, object]:
    high = (supports >= thresholds.high_minimum_support) & (
        relative_widths <= thresholds.high_max_relative_width
    )
    moderate = (
        (~high)
        & (supports >= thresholds.moderate_minimum_support)
        & (relative_widths <= thresholds.moderate_max_relative_width)
    )
    low = ~(high | moderate)
    return {
        "semantics": "empirical_interval_width_and_bucket_support_label_not_probability",
        "threshold_quantile_method": "numpy_linear",
        "counts": {
            "High confidence": int(np.count_nonzero(high)),
            "Moderate confidence": int(np.count_nonzero(moderate)),
            "Low confidence": int(np.count_nonzero(low)),
        },
    }


def _build_report(
    *,
    confirmation: Phase4ConfirmationReport,
    assignment_hash: str,
    predictions: NDArray[np.float64],
    target: NDArray[np.float64],
    statuses: NDArray[np.str_],
    cutpoints: tuple[float, float, float],
    crossfit: Mapping[str, object],
    selected_method: str,
    gate_results: list[dict[str, object]],
    diagnostics: Mapping[str, object],
    confidence: ConfidenceThresholds,
    confidence_diagnostics: Mapping[str, object],
    full_calibrations: tuple[CoverageCalibration, ...],
    generated_at: str,
) -> dict[str, object]:
    public_crossfit = _public_crossfit(crossfit)
    selected_public_method = cast(
        Mapping[str, object],
        cast(Mapping[str, object], public_crossfit["methods"])[selected_method],
    )
    selected_coverages = cast(Mapping[str, object], selected_public_method["coverages"])
    selected_validated = _selected_method_validated(selected_coverages)
    classification = (
        "validated_for_calibrated_prediction_intervals"
        if selected_validated
        else "requires_methodology_revision"
    )
    point_status = {
        status: regression_metrics(
            target[statuses == status], predictions[statuses == status]
        ).to_dict()
        for status in RETAIL_VEHICLE_STATUSES
    }
    return {
        "schema_version": 1,
        "report_type": CALIBRATION_REPORT_TYPE,
        "calibration_version": "retail-rf05-split-conformal-v1",
        "generated_at": generated_at,
        "classification": classification,
        "decision": {
            "selected_method": selected_method,
            "conditional_gate_results": gate_results,
            "selection_used_calibration_for_estimator_choice_or_tuning": False,
            "model_promotion_or_replacement": False,
        },
        "frozen_model": active_rf05_identity().to_dict(),
        "frozen_evidence": {
            "policy_sha256": CALIBRATION_POLICY_SHA256,
            "phase4_confirmation_sha256": PHASE4_RETAIL_CONFIRMATION_SHA256,
            "phase4_metric_ranking": list(confirmation.metric_ranking),
        },
        "data_boundaries": {
            "source_id": "kaggle_us_sales_cars_v2",
            "target_semantics": RETAIL_TRACK.target_semantics,
            "phase3_train_rows": DEVELOPMENT_SAMPLE_COUNT + CALIBRATION_SAMPLE_COUNT,
            "development_fit_rows": DEVELOPMENT_SAMPLE_COUNT,
            "calibration_rows": CALIBRATION_SAMPLE_COUNT,
            "calibration_assignment_sha256": assignment_hash,
            "calibration_rows_used_for_estimator_fit": False,
            "calibration_rows_used_for_estimator_selection_or_tuning": False,
            "legacy_holdout_accessed": False,
            "yoad_accessed": False,
            "river_accessed": False,
            "raw_rows_predictions_or_residuals_persisted": False,
        },
        "point_prediction_metrics_on_calibration": {
            "overall": regression_metrics(target, predictions).to_dict(),
            "by_vehicle_status": point_status,
        },
        "target_free_predicted_value_cutpoints_usd": list(cutpoints),
        "cross_calibration": public_crossfit,
        "selected_method_slice_diagnostics": diagnostics,
        "full_calibration_radii": [item.to_dict() for item in full_calibrations],
        "confidence": {
            **confidence.to_dict(),
            **confidence_diagnostics,
            "data_quality_warnings_are_separate": True,
        },
        "interpretation": {
            "interval_is": "empirical split-conformal asking-price prediction interval",
            "interval_is_not": [
                "probability_that_this_vehicle_has_a_specific_value",
                "Kelley_Blue_Book_or_other_third_party_valuation",
                "guaranteed_sale_price",
            ],
            "finite_sample_formula": "ceil((n + 1) * coverage)",
            "bounds": "max(0, point_prediction - radius) to point_prediction + radius",
        },
    }


def _public_crossfit(crossfit: Mapping[str, object]) -> dict[str, object]:
    methods = cast(Mapping[str, object], crossfit["methods"])
    public_methods: dict[str, object] = {}
    for method, raw_method in methods.items():
        method_map = cast(Mapping[str, object], raw_method)
        coverages = cast(Mapping[str, object], method_map["coverages"])
        public_coverages: dict[str, object] = {}
        for coverage, raw_item in coverages.items():
            item = cast(Mapping[str, object], raw_item)
            public_coverages[coverage] = {
                key: value for key, value in item.items() if not key.startswith("_")
            }
        public_methods[method] = {"coverages": public_coverages}
    return {"folds": crossfit["folds"], "methods": public_methods}


def _selected_method_validated(coverages: Mapping[str, object]) -> bool:
    for level in COVERAGE_LEVELS:
        item = cast(Mapping[str, object], coverages[str(level)])
        if cast(float, item["coverage_gap"]) < -0.02:
            return False
        status = cast(Mapping[str, object], item["status"])
        if any(
            cast(float, cast(Mapping[str, object], metrics)["coverage_gap"]) < -0.05
            for metrics in status.values()
        ):
            return False
    return True


def _normalized_statuses(values: object) -> NDArray[np.str_]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise CalibrationExperimentError("vehicle statuses must be one-dimensional")
    result = np.asarray([str(value).strip().lower() for value in array], dtype=np.str_)
    if any(status not in RETAIL_VEHICLE_STATUSES for status in result.tolist()):
        raise CalibrationExperimentError("calibration contains an unsupported vehicle status")
    return result


def _numeric_bands(
    values: NDArray[np.float64],
    cutpoints: tuple[float, float, float],
    *,
    prefix: str,
) -> NDArray[np.str_]:
    labels = np.select(
        (values <= cutpoints[0], values <= cutpoints[1], values <= cutpoints[2]),
        (f"{prefix}_1", f"{prefix}_2", f"{prefix}_3"),
        default=f"{prefix}_4",
    )
    return np.asarray(labels, dtype=np.str_)


def _mileage_bands(values: NDArray[np.float64]) -> NDArray[np.str_]:
    result = np.full(len(values), "mileage_missing", dtype="<U24")
    present = np.isfinite(values)
    result[present] = _numeric_bands(values[present], MILEAGE_CUTPOINTS, prefix="mileage")
    return result


__all__ = [
    "CALIBRATION_REPORT_TYPE",
    "CalibrationExperimentError",
    "CalibrationExperimentResult",
    "canonical_calibration_report_json",
    "fit_frozen_rf05_calibration_predictions",
    "report_sha256",
    "run_retail_rf05_calibration",
]
