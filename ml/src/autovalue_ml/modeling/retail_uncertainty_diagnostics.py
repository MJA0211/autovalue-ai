"""Development-only RF05 residual diagnostics for uncertainty design."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Final, Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .calibration_artifact import PHASE4_RETAIL_CONFIRMATION_SHA256, active_rf05_identity
from .candidates import make_random_forest_candidate
from .contracts import RETAIL_TRACK, validate_feature_frame, validate_target
from .cv import CVSplit, retail_group_cv_splits
from .metrics import regression_metrics
from .phase4_confirmation import Phase4ConfirmationReport
from .retail_calibration_experiment import (
    AGE_CUTPOINTS,
    PRICE_CUTPOINTS,
    _mileage_bands,
    _normalized_statuses,
    _numeric_bands,
)

DEVELOPMENT_SAMPLE_COUNT: Final = 98_552
DIAGNOSTIC_REPORT_TYPE: Final = "retail_rf05_development_residual_diagnostics"
GENERATED_AT: Final = "2026-09-02T18:00:00+00:00"
MINIMUM_MANUFACTURER_SUPPORT: Final = 500
MINIMUM_MODEL_SUPPORT: Final = 500
MINIMUM_COMBINATION_SUPPORT: Final = 400
MAXIMUM_REPORTED_CATEGORIES: Final = 20


class ResidualDiagnosticsError(ValueError):
    """Development evidence or an RF05 reconstruction violated policy."""


class Regressor(Protocol):
    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> Regressor: ...

    def predict(self, features: pd.DataFrame) -> object: ...


EstimatorFactory = Callable[[], Regressor]
ProgressCallback = Callable[[int, int], None]


def reconstruct_rf05_development_oof(
    *,
    development_features: object,
    development_target: object,
    estimator_factory: EstimatorFactory | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[NDArray[np.float64], tuple[CVSplit, ...]]:
    """Reconstruct exact predictor-group OOF predictions without calibration rows."""

    features = validate_feature_frame(development_features, RETAIL_TRACK)
    target = validate_target(
        development_target,
        expected_rows=len(features),
        config=RETAIL_TRACK,
    )
    if len(features) != DEVELOPMENT_SAMPLE_COUNT:
        raise ResidualDiagnosticsError(
            "RF05 residual diagnostics require the exact development set"
        )
    if progress is not None and not callable(progress):
        raise ResidualDiagnosticsError("progress must be callable")
    splits = retail_group_cv_splits(features, n_splits=5)
    predictions = np.full(len(features), np.nan, dtype=np.float64)
    validation_counts = np.zeros(len(features), dtype=np.int8)
    factory = estimator_factory or _rf05_factory
    for fold_number, (training_indices, validation_indices) in enumerate(splits, start=1):
        estimator = factory()
        estimator.fit(features.iloc[training_indices], target[training_indices])
        fold_predictions = _prediction_vector(
            estimator.predict(features.iloc[validation_indices]),
            expected_rows=len(validation_indices),
        )
        predictions[validation_indices] = fold_predictions
        validation_counts[validation_indices] += 1
        if progress is not None:
            progress(fold_number, len(splits))
    if not np.all(validation_counts == 1) or not np.isfinite(predictions).all():
        raise ResidualDiagnosticsError("RF05 OOF reconstruction did not score every row once")
    predictions.setflags(write=False)
    return predictions, splits


def build_development_residual_diagnostics(
    *,
    development_features: object,
    development_target: object,
    confirmation: Phase4ConfirmationReport,
    confirmation_sha256: str,
    estimator_factory: EstimatorFactory | None = None,
    progress: ProgressCallback | None = None,
    generated_at: str = GENERATED_AT,
) -> dict[str, object]:
    """Create an aggregate-only residual report from the frozen development boundary."""

    features = validate_feature_frame(development_features, RETAIL_TRACK)
    target = validate_target(
        development_target,
        expected_rows=len(features),
        config=RETAIL_TRACK,
    )
    _validate_confirmation(confirmation, confirmation_sha256)
    predictions, splits = reconstruct_rf05_development_oof(
        development_features=features,
        development_target=target,
        estimator_factory=estimator_factory,
        progress=progress,
    )
    _validate_reconstruction_metrics(target, predictions, confirmation)
    residuals = np.abs(target - predictions)
    statuses = _normalized_statuses(features["vehicle_status"])
    predicted_cutpoints = _strict_quartiles(predictions, label="predicted value")
    predicted_bands = _numeric_bands(predictions, predicted_cutpoints, prefix="predicted_value")
    mileage = pd.to_numeric(features["mileage"], errors="coerce").to_numpy(dtype=np.float64)
    year = pd.to_numeric(features["year"], errors="raise").to_numpy(dtype=np.float64)
    age = np.maximum(0.0, 2023.0 - year)
    mileage_per_year = mileage / np.maximum(age, 1.0)
    mileage_per_year_cutpoints = _strict_quartiles(
        mileage_per_year[np.isfinite(mileage_per_year)],
        label="mileage per year",
    )
    mileage_per_year_bands = np.full(len(features), "mileage_per_year_missing", dtype="<U32")
    present_mileage_per_year = np.isfinite(mileage_per_year)
    mileage_per_year_bands[present_mileage_per_year] = _numeric_bands(
        mileage_per_year[present_mileage_per_year],
        mileage_per_year_cutpoints,
        prefix="mileage_per_year",
    )
    manufacturers = _normalized_text(features["make"])
    models = _normalized_text(features["model"])
    combination = np.asarray(
        [f"{status}|{band}" for status, band in zip(statuses, predicted_bands, strict=True)],
        dtype=np.str_,
    )
    dimensions: dict[str, tuple[NDArray[np.str_], int, int | None]] = {
        "predicted_value_band": (predicted_bands, 1, None),
        "actual_price_band_evaluation_only": (
            _numeric_bands(target, PRICE_CUTPOINTS, prefix="actual_price"),
            1,
            None,
        ),
        "vehicle_age_band": (_numeric_bands(age, AGE_CUTPOINTS, prefix="age"), 1, None),
        "mileage_band": (_mileage_bands(mileage), 1, None),
        "mileage_per_year_band": (mileage_per_year_bands, 1, None),
        "vehicle_status": (statuses, 1, None),
        "missing_mileage": (
            np.where(np.isfinite(mileage), "mileage_present", "mileage_missing").astype(np.str_),
            1,
            None,
        ),
        "manufacturer": (
            manufacturers,
            MINIMUM_MANUFACTURER_SUPPORT,
            MAXIMUM_REPORTED_CATEGORIES,
        ),
        "model": (models, MINIMUM_MODEL_SUPPORT, MAXIMUM_REPORTED_CATEGORIES),
        "vehicle_status_by_predicted_value_band": (
            combination,
            MINIMUM_COMBINATION_SUPPORT,
            None,
        ),
    }
    slice_report = {
        name: _dimension_report(
            labels,
            residuals=residuals,
            target=target,
            minimum_support=minimum_support,
            maximum_categories=maximum_categories,
        )
        for name, (labels, minimum_support, maximum_categories) in dimensions.items()
    }
    relationship = _predicted_value_relationship(
        predictions,
        residuals,
        predicted_bands,
    )
    return {
        "schema_version": 1,
        "report_type": DIAGNOSTIC_REPORT_TYPE,
        "generated_at": generated_at,
        "classification": "development_only_diagnostic_not_calibration_or_model_selection",
        "frozen_model": active_rf05_identity().to_dict(),
        "frozen_evidence": {
            "phase4_confirmation_sha256": PHASE4_RETAIL_CONFIRMATION_SHA256,
            "reconstruction_matches_phase4_rf05_aggregate_metrics": True,
        },
        "data_boundaries": {
            "development_rows": len(features),
            "development_oof_rows": len(predictions),
            "development_group_folds": len(splits),
            "calibration_rows_accessed": False,
            "calibration_targets_accessed": False,
            "legacy_holdout_accessed": False,
            "yoad_accessed": False,
            "river_accessed": False,
            "autotrader_accessed": False,
            "carson_shively_accessed": False,
            "raw_rows_predictions_or_residuals_persisted": False,
        },
        "point_prediction_metrics": regression_metrics(target, predictions).to_dict(),
        "overall_residual_distribution": _residual_statistics(residuals, target),
        "target_free_cutpoints": {
            "predicted_value_usd": list(predicted_cutpoints),
            "mileage_per_year": list(mileage_per_year_cutpoints),
        },
        "predicted_value_relationship": relationship,
        "slices": slice_report,
        "design_implication": {
            "supports_heteroscedastic_method": cast(
                float, relationship["highest_to_lowest_quartile_mean_residual_ratio"]
            )
            >= 1.25,
            "candidate_scope": [
                "frozen_vehicle_status_conformal_baseline",
                "development_trained_predictor_scale_normalized_conformal",
                "development_trained_log_linear_predicted_value_scale_conformal",
            ],
            "actual_price_is_evaluation_only": True,
            "calibration_outcomes_used_to_choose_diagnostics": False,
        },
    }


def canonical_diagnostics_json(report: Mapping[str, object]) -> str:
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
        raise ResidualDiagnosticsError("residual diagnostics are not JSON-safe") from error


def diagnostics_sha256(report: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_diagnostics_json(report).encode("utf-8")).hexdigest()


def _rf05_factory() -> Regressor:
    return cast(Regressor, make_random_forest_candidate("retail", 5, n_jobs=4))


def _prediction_vector(predictions: object, *, expected_rows: int) -> NDArray[np.float64]:
    inspected = np.asarray(predictions, dtype=object)
    if inspected.ndim != 1 or len(inspected) != expected_rows:
        raise ResidualDiagnosticsError("RF05 predictions must be a one-dimensional row match")
    if any(isinstance(value, (bool, np.bool_)) for value in inspected.tolist()):
        raise ResidualDiagnosticsError("RF05 predictions must be numeric, not boolean")
    try:
        values = inspected.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ResidualDiagnosticsError("RF05 predictions must be numeric") from error
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ResidualDiagnosticsError("RF05 predictions must be finite and nonnegative")
    return values


def _validate_confirmation(
    confirmation: Phase4ConfirmationReport,
    confirmation_sha256: str,
) -> None:
    if confirmation_sha256 != PHASE4_RETAIL_CONFIRMATION_SHA256:
        raise ResidualDiagnosticsError("Phase 4 retail confirmation checksum differs")
    if confirmation.track != "retail" or confirmation.metric_ranking[0] != (
        "phase4-retail-random_forest-05"
    ):
        raise ResidualDiagnosticsError("Phase 4 confirmation does not freeze retail RF05")


def _validate_reconstruction_metrics(
    target: NDArray[np.float64],
    predictions: NDArray[np.float64],
    confirmation: Phase4ConfirmationReport,
) -> None:
    expected = next(
        item
        for item in confirmation.candidates
        if item.spec.candidate_id == "phase4-retail-random_forest-05"
    ).overall
    observed = regression_metrics(target, predictions)
    for label, actual, frozen in (
        ("MAE", observed.mae, expected.mae),
        ("RMSE", observed.rmse, expected.rmse),
        ("R-squared", observed.r2, expected.r2),
    ):
        if actual is None or frozen is None or not np.isclose(actual, frozen, rtol=0.0, atol=1e-8):
            raise ResidualDiagnosticsError(
                f"reconstructed RF05 {label} differs from frozen Phase 4 evidence"
            )


def _strict_quartiles(values: NDArray[np.float64], *, label: str) -> tuple[float, float, float]:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ResidualDiagnosticsError(f"{label} values must be a non-empty finite vector")
    cutpoints = cast(
        tuple[float, float, float],
        tuple(float(value) for value in np.quantile(values, (0.25, 0.5, 0.75))),
    )
    if not cutpoints[0] < cutpoints[1] < cutpoints[2]:
        raise ResidualDiagnosticsError(f"{label} quartiles must be distinct")
    return cutpoints


def _normalized_text(values: object) -> NDArray[np.str_]:
    array = np.asarray(values, dtype=object)
    return np.asarray(
        ["__missing__" if pd.isna(value) else str(value).strip().lower() for value in array],
        dtype=np.str_,
    )


def _residual_statistics(
    residuals: NDArray[np.float64],
    target: NDArray[np.float64],
) -> dict[str, object]:
    if len(residuals) == 0 or len(residuals) != len(target):
        raise ResidualDiagnosticsError("residual statistics require aligned non-empty vectors")
    relative = residuals / np.maximum(np.abs(target), 1.0)
    return {
        "support": len(residuals),
        "median_absolute_residual_usd": float(np.median(residuals)),
        "mean_absolute_residual_usd": float(np.mean(residuals)),
        "residual_variance_usd2": float(np.var(residuals, ddof=0)),
        "absolute_residual_quantiles_usd": {
            str(level): float(np.quantile(residuals, level, method="linear"))
            for level in (0.5, 0.8, 0.9, 0.95)
        },
        "residual_to_actual_price_ratio": {
            "mean": float(np.mean(relative)),
            "median": float(np.median(relative)),
            "p90": float(np.quantile(relative, 0.9, method="linear")),
        },
    }


def _dimension_report(
    labels: NDArray[np.str_],
    *,
    residuals: NDArray[np.float64],
    target: NDArray[np.float64],
    minimum_support: int,
    maximum_categories: int | None,
) -> list[dict[str, object]]:
    counts = Counter(labels.tolist())
    eligible = [
        label
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_support
    ]
    if maximum_categories is not None:
        eligible = eligible[:maximum_categories]
    return [
        {
            "label": label,
            **_residual_statistics(residuals[labels == label], target[labels == label]),
        }
        for label in sorted(eligible)
    ]


def _predicted_value_relationship(
    predictions: NDArray[np.float64],
    residuals: NDArray[np.float64],
    bands: NDArray[np.str_],
) -> dict[str, object]:
    log_prediction = np.log1p(predictions)
    log_residual = np.log1p(residuals)
    pearson = float(np.corrcoef(log_prediction, log_residual)[0, 1])
    prediction_ranks = pd.Series(predictions).rank(method="average").to_numpy(dtype=np.float64)
    residual_ranks = pd.Series(residuals).rank(method="average").to_numpy(dtype=np.float64)
    spearman = float(np.corrcoef(prediction_ranks, residual_ranks)[0, 1])
    band_means = {
        band: float(np.mean(residuals[bands == band]))
        for band in (
            "predicted_value_1",
            "predicted_value_2",
            "predicted_value_3",
            "predicted_value_4",
        )
    }
    lowest = band_means["predicted_value_1"]
    return {
        "log_prediction_log_residual_pearson": pearson,
        "prediction_residual_spearman": spearman,
        "mean_absolute_residual_usd_by_predicted_value_quartile": band_means,
        "highest_to_lowest_quartile_mean_residual_ratio": (
            band_means["predicted_value_4"] / lowest if lowest > 0.0 else None
        ),
    }


__all__ = [
    "DEVELOPMENT_SAMPLE_COUNT",
    "DIAGNOSTIC_REPORT_TYPE",
    "ResidualDiagnosticsError",
    "build_development_residual_diagnostics",
    "canonical_diagnostics_json",
    "diagnostics_sha256",
    "reconstruct_rf05_development_oof",
]
