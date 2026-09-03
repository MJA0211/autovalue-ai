"""Leakage-safe training-weight confirmation for moderate Yoad augmentation.

The source composition, validation rows, folds, preprocessing, RF05 tuple, and
random state are inherited from the checksum-bound source-composition report.
Only fold-local training weights differ. Checkpoints contain aggregate metrics
and diagnostics, never source rows or row-level predictions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .candidates import RETAIL_RANDOM_FOREST_CONFIGS
from .metrics import regression_metrics
from .tree_preprocessing import make_tree_preprocessor
from .yoad_confirmation import deterministic_yoad_subsets
from .yoad_experiment import (
    BROAD_RETAIL_TRACK,
    MODEL_RANDOM_STATE,
    N_SPLITS,
    PreparedExperimentData,
    controlled_group_splits,
)

Treatment = Literal[
    "source_balanced_weighting",
    "source_mileage_weighting",
    "source_segment_weighting",
]
ReportArm = Literal[
    "cars_only",
    "moderate_augmentation",
    "source_balanced_weighting",
    "source_mileage_weighting",
    "source_segment_weighting",
]
ProgressCallback = Callable[[tuple[Mapping[str, object], ...]], None]

EXPERIMENT_ID: Final = "autovalue-yoad22-training-weight-confirmation-v1"
CONFIRMATION_REPORT_SHA256: Final = (
    "6ca3dd25cfb24bb0734497e4703cc516b3152e42f319286fcdd73374a6b2e5f5"
)
MODERATE_YOAD_ROWS: Final = 150_000
CARS_DEVELOPMENT_ROWS: Final = 98_552
CARS_CALIBRATION_ROWS: Final = 10_958
FULL_VALIDATION_ROWS: Final = 341_218
_RF_PARAMETERS: Final = RETAIL_RANDOM_FOREST_CONFIGS[5]
_SOURCE_NAMES: Final = ("cars_com_development", "yoad22_craigslist")
_TREATMENTS: Final[tuple[Treatment, ...]] = (
    "source_balanced_weighting",
    "source_mileage_weighting",
    "source_segment_weighting",
)
_REPORT_ARMS: Final[tuple[ReportArm, ...]] = (
    "cars_only",
    "moderate_augmentation",
    *_TREATMENTS,
)
_MILEAGE_EDGES: Final = (0.0, 38_282.0, 86_204.0, 135_803.0, 405_187.0)
_AGE_EDGES: Final = (0.0, 3.0, 8.0, 13.0, 64.0)
_PRICE_EDGES: Final = (1.0, 8_995.0, 19_995.0, 36_590.0, 8_078_160.0)
_ABSOLUTE_WEIGHT_BOUNDS: Final = (0.5, 2.0)
_MAX_CHECKPOINT_BYTES: Final = 2_000_000

_WEIGHTING_POLICY: Final[dict[str, object]] = {
    "version": "yoad22-fold-local-weighting-v1",
    "source_balance": "each source receives one half of fold training weight",
    "mileage": {
        "alignment_exponent": 0.5,
        "factor_bounds": [0.85, 1.15],
        "bands": list(_MILEAGE_EDGES),
    },
    "segments": {
        "dimensions": ["mileage_band", "vehicle_age_band", "manufacturer"],
        "alignment_exponent": 0.25,
        "manufacturer_reliability_prior_rows": BROAD_RETAIL_TRACK.one_hot_min_frequency,
        "factor_bounds": [0.9, 1.1],
        "combined_factor_bounds": [0.8, 1.25],
    },
    "absolute_weight_bounds": list(_ABSOLUTE_WEIGHT_BOUNDS),
    "target_used": False,
    "validation_weights_used": False,
}
WEIGHTING_POLICY_SHA256: Final = hashlib.sha256(
    json.dumps(_WEIGHTING_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class YoadWeightingError(RuntimeError):
    """The weighting experiment violated an evidence or leakage boundary."""


def load_confirmation_report(path: Path) -> dict[str, object]:
    """Load the exact immutable source-composition confirmation."""

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CONFIRMATION_REPORT_SHA256:
        raise YoadWeightingError("source-composition confirmation checksum differs")
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise YoadWeightingError("source-composition confirmation is invalid JSON") from error
    if not isinstance(value, dict):
        raise YoadWeightingError("source-composition confirmation must be an object")
    return cast(dict[str, object], value)


def make_training_weights(
    treatment: Treatment,
    features: pd.DataFrame,
    sources: NDArray[np.str_],
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Derive bounded weights using only one fold's training predictors and source."""

    if treatment not in _TREATMENTS:
        raise YoadWeightingError("unknown weighting treatment")
    if len(features) == 0 or len(features) != len(sources):
        raise YoadWeightingError("weight inputs are empty or unaligned")
    if set(np.unique(sources).tolist()) != set(_SOURCE_NAMES):
        raise YoadWeightingError("both approved sources are required in every training fold")

    rows = len(features)
    target_source_total = rows / 2.0
    weights = np.empty(rows, dtype=np.float64)
    for source in _SOURCE_NAMES:
        selected = sources == source
        weights[selected] = target_source_total / float(selected.sum())

    adjustment = np.ones(rows, dtype=np.float64)
    adjustment_summaries: dict[str, object] = {}
    if treatment == "source_mileage_weighting":
        mileage = _mileage_labels(features["mileage"]).to_numpy(dtype=np.str_, copy=True)
        factors, summary = _alignment_factors(
            mileage,
            sources,
            exponent=0.5,
            bounds=(0.85, 1.15),
            reliability_prior=None,
        )
        adjustment *= factors
        adjustment_summaries["mileage_band"] = summary
    elif treatment == "source_segment_weighting":
        dimensions = {
            "mileage_band": _mileage_labels(features["mileage"]).to_numpy(dtype=np.str_, copy=True),
            "vehicle_age_band": _fixed_band_labels(
                (
                    BROAD_RETAIL_TRACK.reference_year
                    - pd.to_numeric(features["year"], errors="raise").astype(float)
                ).clip(lower=0),
                _AGE_EDGES,
            ).to_numpy(dtype=np.str_, copy=True),
            "manufacturer": features["make"].astype(str).to_numpy(dtype=np.str_, copy=True),
        }
        for dimension, labels in dimensions.items():
            factors, summary = _alignment_factors(
                labels,
                sources,
                exponent=0.25,
                bounds=(0.9, 1.1),
                reliability_prior=(
                    BROAD_RETAIL_TRACK.one_hot_min_frequency
                    if dimension == "manufacturer"
                    else None
                ),
            )
            adjustment *= factors
            adjustment_summaries[dimension] = summary
        adjustment = np.clip(adjustment, 0.8, 1.25)

    weights *= adjustment
    for source in _SOURCE_NAMES:
        selected = sources == source
        weights[selected] *= target_source_total / float(weights[selected].sum())
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise YoadWeightingError("training weights are non-finite or non-positive")
    minimum, maximum = _ABSOLUTE_WEIGHT_BOUNDS
    tolerance = 1e-12
    if float(weights.min()) < minimum - tolerance or float(weights.max()) > maximum + tolerance:
        raise YoadWeightingError("training weights exceed preregistered absolute bounds")
    diagnostics = _weight_diagnostics(weights, sources, adjustment_summaries)
    return weights, diagnostics


def run_weighting_experiment(
    *,
    data: PreparedExperimentData,
    confirmation_report: Mapping[str, object],
    completed_fits: Sequence[Mapping[str, object]] = (),
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Run or resume the three weighted treatments and assemble aggregate evidence."""

    _validate_confirmation_evidence(data, confirmation_report)
    moderate_indices = deterministic_yoad_subsets(data)["moderate_augmentation"]
    moderate_mask = np.zeros(len(data.features), dtype=np.bool_)
    moderate_mask[moderate_indices] = True
    cars_mask = data.sources == "cars_com_development"
    included_mask = cars_mask | moderate_mask
    if (
        int(cars_mask.sum()) != CARS_DEVELOPMENT_ROWS
        or int(moderate_mask.sum()) != MODERATE_YOAD_ROWS
    ):
        raise YoadWeightingError("moderate composition differs from frozen confirmation")
    splits = controlled_group_splits(data.features)
    labels = _evaluation_labels(data)
    completed = [dict(item) for item in completed_fits]
    _validate_completed_fits(completed, data, splits, included_mask)

    expected_order = tuple(
        (treatment, fold_number)
        for treatment in _TREATMENTS
        for fold_number in range(1, N_SPLITS + 1)
    )
    for treatment, fold_number in expected_order[len(completed) :]:
        training, validation = splits[fold_number - 1]
        included = training[included_mask[training]]
        training_features = data.features.iloc[included]
        training_sources = data.sources[included]
        weights, diagnostics = make_training_weights(
            treatment,
            training_features,
            training_sources,
        )
        model = _make_model()
        model.fit(
            training_features,
            data.target[included],
            regressor__sample_weight=weights,
        )
        predicted = cast(
            NDArray[np.float64],
            model.predict(data.features.iloc[validation]).astype(np.float64, copy=False),
        )
        completed.append(
            _completed_fit(
                treatment=treatment,
                fold_number=fold_number,
                included=included,
                validation=validation,
                predicted=predicted,
                diagnostics=diagnostics,
                data=data,
                labels=labels,
            )
        )
        if on_progress is not None:
            on_progress(tuple(completed))

    _validate_completed_fits(completed, data, splits, included_mask)
    if len(completed) != len(expected_order):
        raise YoadWeightingError("weighting experiment did not complete every fit")
    return _assemble_report(data, confirmation_report, completed, labels)


def make_weighting_checkpoint(
    completed_fits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Create aggregate-only resumable evidence for completed RF fits."""

    return {
        "schema_version": 1,
        "report_type": "yoad22_weighting_checkpoint",
        "experiment_id": EXPERIMENT_ID,
        "confirmation_report_sha256": CONFIRMATION_REPORT_SHA256,
        "weighting_policy_sha256": WEIGHTING_POLICY_SHA256,
        "completed_fits": [dict(item) for item in completed_fits],
    }


def parse_weighting_checkpoint_json(serialized: str | bytes) -> tuple[Mapping[str, object], ...]:
    """Parse bounded checkpoint JSON and reject policy drift or reordered fits."""

    text = _bounded_text(serialized, label="weighting checkpoint")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise YoadWeightingError("weighting checkpoint is invalid JSON") from error
    required = {
        "schema_version",
        "report_type",
        "experiment_id",
        "confirmation_report_sha256",
        "weighting_policy_sha256",
        "completed_fits",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise YoadWeightingError("weighting checkpoint fields are invalid")
    if (
        value["schema_version"] != 1
        or value["report_type"] != "yoad22_weighting_checkpoint"
        or value["experiment_id"] != EXPERIMENT_ID
        or value["confirmation_report_sha256"] != CONFIRMATION_REPORT_SHA256
        or value["weighting_policy_sha256"] != WEIGHTING_POLICY_SHA256
    ):
        raise YoadWeightingError("weighting checkpoint policy metadata differs")
    completed = value["completed_fits"]
    if not isinstance(completed, list) or not all(isinstance(item, dict) for item in completed):
        raise YoadWeightingError("weighting checkpoint completed_fits is invalid")
    _validate_stable_prefix(cast(Sequence[Mapping[str, object]], completed))
    return tuple(cast(Sequence[Mapping[str, object]], completed))


def canonical_weighting_json(report: Mapping[str, object]) -> str:
    """Serialize a report or checkpoint deterministically and reject non-finite values."""

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


def _alignment_factors(
    labels: NDArray[np.str_],
    sources: NDArray[np.str_],
    *,
    exponent: float,
    bounds: tuple[float, float],
    reliability_prior: int | None,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    factors = np.ones(len(labels), dtype=np.float64)
    categories = sorted(set(labels.tolist()))
    source_counts = {source: int((sources == source).sum()) for source in _SOURCE_NAMES}
    details: dict[str, list[dict[str, object]]] = {source: [] for source in _SOURCE_NAMES}
    shares: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for source in _SOURCE_NAMES:
        source_mask = sources == source
        for category in categories:
            count = int((source_mask & (labels == category)).sum())
            counts[(source, category)] = count
            shares[(source, category)] = count / source_counts[source]
    for source in _SOURCE_NAMES:
        for category in categories:
            count = counts[(source, category)]
            if count == 0:
                continue
            target_share = sum(shares[(item, category)] for item in _SOURCE_NAMES) / 2.0
            ratio = target_share / shares[(source, category)]
            effective_exponent = exponent
            if reliability_prior is not None:
                effective_exponent *= count / (count + reliability_prior)
            factor = float(np.clip(ratio**effective_exponent, *bounds))
            selected = (sources == source) & (labels == category)
            factors[selected] = factor
            details[source].append(
                {
                    "category": category,
                    "training_rows": count,
                    "source_share": shares[(source, category)],
                    "equal_source_target_share": target_share,
                    "factor": factor,
                }
            )
    summary: dict[str, object] = {
        "formula": "clipped((mean source category share / source category share) ** exponent)",
        "exponent": exponent,
        "factor_bounds": list(bounds),
        "reliability_prior_rows": reliability_prior,
        "target_used": False,
        "by_source": {},
    }
    by_source = cast(dict[str, object], summary["by_source"])
    for source in _SOURCE_NAMES:
        rows = details[source]
        ordered = sorted(
            rows,
            key=lambda item: (cast(float, item["factor"]), str(item["category"])),
        )
        by_source[source] = {
            "categories_present": len(rows),
            "minimum_factor": min(cast(float, item["factor"]) for item in rows),
            "maximum_factor": max(cast(float, item["factor"]) for item in rows),
            "lowest_factors": ordered[:5],
            "highest_factors": list(reversed(ordered[-5:])),
        }
    return factors, summary


def _weight_diagnostics(
    weights: NDArray[np.float64],
    sources: NDArray[np.str_],
    adjustments: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {
        "minimum": float(weights.min()),
        "maximum": float(weights.max()),
        "median": float(np.median(weights)),
        "mean": float(weights.mean()),
        "weight_total": float(weights.sum()),
        "effective_sample_size": _effective_sample_size(weights),
        "effective_sample_fraction": _effective_sample_size(weights) / len(weights),
        "absolute_bounds": list(_ABSOLUTE_WEIGHT_BOUNDS),
        "by_source": {},
        "adjustment_summaries": dict(adjustments),
    }
    by_source = cast(dict[str, object], result["by_source"])
    for source in _SOURCE_NAMES:
        values = weights[sources == source]
        by_source[source] = {
            "rows": len(values),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "weight_total": float(values.sum()),
            "effective_sample_size": _effective_sample_size(values),
        }
    return result


def _effective_sample_size(weights: NDArray[np.float64]) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def _completed_fit(
    *,
    treatment: Treatment,
    fold_number: int,
    included: NDArray[np.int64],
    validation: NDArray[np.int64],
    predicted: NDArray[np.float64],
    diagnostics: Mapping[str, object],
    data: PreparedExperimentData,
    labels: Mapping[str, pd.Series],
) -> dict[str, object]:
    return {
        "treatment": treatment,
        "fold": fold_number,
        "training_rows": len(included),
        "training_source_rows": {
            source: int((data.sources[included] == source).sum()) for source in _SOURCE_NAMES
        },
        "validation_rows": len(validation),
        "metrics": _metrics_by_source(data.target[validation], predicted, data.sources[validation]),
        "slice_metrics": _fold_slice_metrics(data, validation, predicted, labels),
        "weight_diagnostics": dict(diagnostics),
    }


def _assemble_report(
    data: PreparedExperimentData,
    confirmation: Mapping[str, object],
    completed: Sequence[Mapping[str, object]],
    labels: Mapping[str, pd.Series],
) -> dict[str, object]:
    confirmation_metrics = cast(Mapping[str, object], confirmation["metrics"])
    metrics: dict[str, object] = {
        "cars_only": confirmation_metrics["cars_only"],
        "moderate_augmentation": confirmation_metrics["moderate_augmentation"],
    }
    confirmation_folds = cast(Sequence[Mapping[str, object]], confirmation["fold_metrics"])
    fold_metrics: list[dict[str, object]] = []
    for position, fold in enumerate(confirmation_folds):
        source_metrics = cast(Mapping[str, object], fold["metrics"])
        fold_metrics.append(
            {
                "fold": position + 1,
                "validation_rows": fold["validation_rows"],
                "metrics": {
                    "cars_only": source_metrics["cars_only"],
                    "moderate_augmentation": source_metrics["moderate_augmentation"],
                },
            }
        )
    confirmation_slices = cast(Mapping[str, object], confirmation["slice_metrics"])
    slices: dict[str, object] = {
        dimension: {
            "cars_only": cast(Mapping[str, object], arms)["cars_only"],
            "moderate_augmentation": cast(Mapping[str, object], arms)["moderate_augmentation"],
        }
        for dimension, arms in confirmation_slices.items()
    }

    for treatment in _TREATMENTS:
        treatment_fits = [item for item in completed if item["treatment"] == treatment]
        metrics[treatment] = _aggregate_metrics(data, treatment_fits)
        treatment_slices = _aggregate_slice_metrics(data, treatment_fits, labels)
        for dimension, value in treatment_slices.items():
            cast(dict[str, object], slices[dimension])[treatment] = value
        for fold_index, fit in enumerate(treatment_fits):
            cast(dict[str, object], fold_metrics[fold_index]["metrics"])[treatment] = fit["metrics"]

    stability = _stability_report(fold_metrics)
    comparisons = _comparison_report(metrics, slices, stability, fold_metrics)
    decision = _weighting_decision(metrics, stability, comparisons)
    return {
        "schema_version": 1,
        "report_type": "yoad22_training_weight_confirmation",
        "experiment_id": EXPERIMENT_ID,
        "reference_confirmation": {
            "path": "docs/experiments/yoad22-source-composition-confirmation-v1.json",
            "sha256": CONFIRMATION_REPORT_SHA256,
            "moderate_result_reused_without_refitting": True,
        },
        "boundaries": {
            "cars_development_rows": CARS_DEVELOPMENT_ROWS,
            "yoad_moderate_rows": MODERATE_YOAD_ROWS,
            "composition_rows": 248_552,
            "validation_rows_per_arm": len(data.features),
            "cars_calibration_rows_excluded": CARS_CALIBRATION_ROWS,
            "legacy_holdout_used": False,
            "phase4_artifacts_modified": False,
            "feature_contract_version": BROAD_RETAIL_TRACK.contract_version,
            "common_predictors": list(BROAD_RETAIL_TRACK.input_features),
            "model_field_excluded": True,
            "validation_observation_weights": "unweighted",
            "target_used_for_training_weights": False,
            "weights_recomputed_from_each_training_fold_only": True,
            "fold_method": "exact source-composition confirmation GroupKFold assignments",
        },
        "model": {
            "family": "RandomForestRegressor",
            "methodology_reference": "Phase 4 retail Random Forest 05",
            "parameters": {
                "n_estimators": _RF_PARAMETERS[0],
                "max_leaf_nodes": _RF_PARAMETERS[1],
                "min_samples_leaf": _RF_PARAMETERS[2],
                "max_features": _RF_PARAMETERS[3],
                "max_samples": _RF_PARAMETERS[4],
                "random_state": MODEL_RANDOM_STATE,
            },
        },
        "weighting_policy": _WEIGHTING_POLICY,
        "weighting_policy_sha256": WEIGHTING_POLICY_SHA256,
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "fold_stability": stability,
        "slice_metrics": slices,
        "weight_diagnostics": _aggregate_weight_diagnostics(completed),
        "comparisons": comparisons,
        "decision": decision,
        "checkpoint": {
            "aggregate_only": True,
            "completed_fit_count": len(completed),
            "raw_rows_persisted": False,
            "row_level_predictions_persisted": False,
        },
        "governance": {
            "automatic_promotion": False,
            "phase4_rf05_replaced": False,
            "moderate_reference_overwritten": False,
            "yoad_online_river_learning": "blocked",
            "carson_shively_included": False,
        },
    }


def _aggregate_metrics(
    data: PreparedExperimentData,
    fits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for source in ("overall", *_SOURCE_NAMES):
        rows = [cast(Mapping[str, object], fit["metrics"])[source] for fit in fits]
        selected = np.ones(len(data.target), dtype=np.bool_)
        if source != "overall":
            selected = data.sources == source
        result[source] = _combine_metrics(
            cast(Sequence[Mapping[str, object]], rows), data.target[selected]
        )
    return result


def _aggregate_slice_metrics(
    data: PreparedExperimentData,
    fits: Sequence[Mapping[str, object]],
    labels: Mapping[str, pd.Series],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dimension, dimension_labels in labels.items():
        source_result: dict[str, object] = {}
        for source in _SOURCE_NAMES:
            source_rows: list[dict[str, object]] = []
            all_labels = sorted(
                set(
                    dimension_labels[data.sources == source]
                    .astype(str)
                    .to_numpy(dtype=np.str_)
                    .tolist()
                )
            )
            for label in all_labels:
                fold_rows: list[Mapping[str, object]] = []
                for fit in fits:
                    fit_slices = cast(Mapping[str, object], fit["slice_metrics"])
                    dimension_value = cast(Mapping[str, object], fit_slices[dimension])
                    reported = cast(Sequence[Mapping[str, object]], dimension_value[source])
                    row = next((item for item in reported if item["slice"] == label), None)
                    if row is not None:
                        fold_rows.append(cast(Mapping[str, object], row["metrics"]))
                selected = (data.sources == source) & (
                    dimension_labels.astype(str).to_numpy(dtype=np.str_) == label
                )
                if int(selected.sum()) >= (100 if dimension == "manufacturer" else 1):
                    source_rows.append(
                        {
                            "slice": label,
                            "metrics": _combine_metrics(fold_rows, data.target[selected]),
                        }
                    )
            source_result[source] = source_rows
        result[dimension] = source_result
    return result


def _combine_metrics(
    rows: Sequence[Mapping[str, object]], target: NDArray[np.float64]
) -> dict[str, float | int]:
    count = sum(cast(int, row["sample_count"]) for row in rows)
    if count != len(target) or count == 0:
        raise YoadWeightingError("aggregate metric row accounting differs")
    absolute_error = sum(cast(float, row["mae"]) * cast(int, row["sample_count"]) for row in rows)
    squared_error = sum(
        cast(float, row["rmse"]) ** 2 * cast(int, row["sample_count"]) for row in rows
    )
    centered = target - float(target.mean())
    total_squares = float(np.dot(centered, centered))
    r2 = 1.0 - squared_error / total_squares if total_squares > 0 else float(squared_error == 0)
    return {
        "mae": absolute_error / count,
        "rmse": float(np.sqrt(squared_error / count)),
        "r2": r2,
        "sample_count": count,
    }


def _fold_slice_metrics(
    data: PreparedExperimentData,
    validation: NDArray[np.int64],
    predicted: NDArray[np.float64],
    labels: Mapping[str, pd.Series],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for dimension, dimension_labels in labels.items():
        source_result: dict[str, object] = {}
        for source in _SOURCE_NAMES:
            source_local = data.sources[validation] == source
            source_result[source] = _aggregate_slices(
                data.target[validation][source_local],
                predicted[source_local],
                dimension_labels.iloc[validation][source_local],
                minimum_count=1,
            )
        result[dimension] = source_result
    return result


def _metrics_by_source(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    sources: NDArray[np.str_],
) -> dict[str, object]:
    result: dict[str, object] = {"overall": regression_metrics(target, predicted).to_dict()}
    for source in _SOURCE_NAMES:
        selected = sources == source
        result[source] = regression_metrics(target[selected], predicted[selected]).to_dict()
    return result


def _evaluation_labels(data: PreparedExperimentData) -> dict[str, pd.Series]:
    age = (
        BROAD_RETAIL_TRACK.reference_year
        - pd.to_numeric(data.features["year"], errors="raise").astype(float)
    ).clip(lower=0)
    return {
        "price_band": _fixed_band_labels(pd.Series(data.target), _PRICE_EDGES),
        "manufacturer": data.features["make"].astype(str),
        "vehicle_age_band": _fixed_band_labels(age, _AGE_EDGES),
        "mileage_band": _mileage_labels(data.features["mileage"]),
    }


def _fixed_band_labels(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    adjusted = np.asarray(edges, dtype=np.float64)
    adjusted[0] -= max(1e-9, abs(float(adjusted[0])) * 1e-12)
    return pd.cut(values.astype(float), bins=adjusted, include_lowest=True).astype(str)


def _mileage_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    labels = pd.Series("mileage:missing", index=values.index, dtype=object)
    present = numeric.notna()
    labels.loc[present] = _fixed_band_labels(numeric.loc[present], _MILEAGE_EDGES).to_numpy()
    return labels


def _aggregate_slices(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    labels: pd.Series,
    *,
    minimum_count: int,
) -> list[dict[str, object]]:
    values = labels.astype(str).to_numpy(dtype=np.str_, copy=True)
    result: list[dict[str, object]] = []
    for label in sorted(set(values.tolist())):
        selected = values == label
        if int(selected.sum()) >= minimum_count:
            result.append(
                {
                    "slice": label,
                    "metrics": regression_metrics(target[selected], predicted[selected]).to_dict(),
                }
            )
    return result


def _stability_report(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for arm in _REPORT_ARMS:
        arm_result: dict[str, object] = {}
        for source in ("overall", *_SOURCE_NAMES):
            values = np.asarray(
                [
                    cast(
                        float,
                        cast(
                            Mapping[str, object],
                            cast(Mapping[str, object], fold["metrics"])[arm],
                        )[source]["mae"],  # type: ignore[index]
                    )
                    for fold in folds
                ],
                dtype=np.float64,
            )
            arm_result[source] = {
                "fold_mae_mean": float(values.mean()),
                "fold_mae_std": float(values.std(ddof=0)),
                "fold_mae_min": float(values.min()),
                "fold_mae_max": float(values.max()),
                "coefficient_of_variation": float(values.std(ddof=0) / values.mean()),
            }
        result[arm] = arm_result
    return result


def _comparison_report(
    metrics: Mapping[str, object],
    slices: Mapping[str, object],
    stability: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cars_only = cast(Mapping[str, object], metrics["cars_only"])
    moderate = cast(Mapping[str, object], metrics["moderate_augmentation"])
    cars_only_cars = _source_mae(cars_only, "cars_com_development")
    cars_only_yoad = _source_mae(cars_only, "yoad22_craigslist")
    moderate_yoad_gain = (
        cars_only_yoad - _source_mae(moderate, "yoad22_craigslist")
    ) / cars_only_yoad
    result: dict[str, object] = {}
    for arm in ("moderate_augmentation", *_TREATMENTS):
        arm_metrics = cast(Mapping[str, object], metrics[arm])
        focus = _focus_changes(slices, cast(ReportArm, arm))
        all_cars = _all_cars_slice_changes(slices, cast(ReportArm, arm))
        moderate_focus = _focus_changes(slices, "moderate_augmentation")
        fold_changes = _cars_fold_changes(stability, folds, cast(ReportArm, arm))
        yoad_gain = (
            cars_only_yoad - _source_mae(arm_metrics, "yoad22_craigslist")
        ) / cars_only_yoad
        result[arm] = {
            "cars_mae_relative_change_vs_cars_only": (
                _source_mae(arm_metrics, "cars_com_development") - cars_only_cars
            )
            / cars_only_cars,
            "cars_mae_relative_change_vs_moderate": (
                _source_mae(arm_metrics, "cars_com_development")
                - _source_mae(moderate, "cars_com_development")
            )
            / _source_mae(moderate, "cars_com_development"),
            "yoad_mae_relative_improvement_vs_cars_only": yoad_gain,
            "moderate_yoad_gain_retained": yoad_gain / moderate_yoad_gain,
            "focus_cars_mae_relative_changes_vs_cars_only": focus,
            "focus_slices_improved_vs_moderate": sum(
                _slice_mae(slices, cast(ReportArm, arm), dimension, label)
                < _slice_mae(slices, "moderate_augmentation", dimension, label)
                for dimension, label in _FOCUS_SLICES.values()
            ),
            "worst_focus_cars_regression": max(focus.values()),
            "worst_focus_reduction_vs_moderate": max(moderate_focus.values()) - max(focus.values()),
            "cars_manufacturer_regression_count": all_cars["manufacturer"]["regression_count"],
            "worst_cars_slice_regression": max(
                cast(float, value["worst_relative_change"]) for value in all_cars.values()
            ),
            "all_cars_slice_change_summary": all_cars,
            "cars_fold_mae_std": fold_changes["std"],
            "cars_fold_mae_relative_changes_vs_cars_only": fold_changes["relative_changes"],
            "worst_cars_fold_mae_relative_change_vs_cars_only": fold_changes[
                "worst_relative_change"
            ],
            "cars_fold_std_relative_to_moderate": cast(float, fold_changes["std"])
            / cast(
                float,
                cast(
                    Mapping[str, object],
                    cast(Mapping[str, object], stability["moderate_augmentation"])[
                        "cars_com_development"
                    ],
                )["fold_mae_std"],
            ),
        }
    return result


_FOCUS_SLICES: Final[dict[str, tuple[str, str]]] = {
    "highest_mileage": ("mileage_band", "(135803.0, 405187.0]"),
    "low_mileage": ("mileage_band", "(-0.001000001, 38282.0]"),
    "age_3_to_8": ("vehicle_age_band", "(3.0, 8.0]"),
    "cadillac": ("manufacturer", "cadillac"),
    "jaguar": ("manufacturer", "jaguar"),
    "hyundai": ("manufacturer", "hyundai"),
    "toyota": ("manufacturer", "toyota"),
    "chevrolet": ("manufacturer", "chevrolet"),
    "acura": ("manufacturer", "acura"),
}


def _focus_changes(slices: Mapping[str, object], arm: ReportArm) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, (dimension, label) in _FOCUS_SLICES.items():
        baseline = _slice_mae(slices, "cars_only", dimension, label)
        challenger = _slice_mae(slices, arm, dimension, label)
        result[name] = (challenger - baseline) / baseline
    return result


def _all_cars_slice_changes(
    slices: Mapping[str, object], arm: ReportArm
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for dimension in ("mileage_band", "vehicle_age_band", "price_band", "manufacturer"):
        baseline = _slice_map(slices, "cars_only", dimension, "cars_com_development")
        challenger = _slice_map(slices, arm, dimension, "cars_com_development")
        changes = {
            label: (challenger[label] - baseline[label]) / baseline[label]
            for label in sorted(set(baseline) & set(challenger))
        }
        result[dimension] = {
            "reported_slices": len(changes),
            "regression_count": sum(value > 0 for value in changes.values()),
            "worst_relative_change": max(changes.values()),
            "largest_regressions": [
                {"slice": label, "mae_relative_change": value}
                for label, value in sorted(changes.items(), key=lambda item: (-item[1], item[0]))
                if value > 0
            ][:10],
        }
    return result


def _weighting_decision(
    metrics: Mapping[str, object],
    stability: Mapping[str, object],
    comparisons: Mapping[str, object],
) -> dict[str, object]:
    moderate = cast(Mapping[str, object], comparisons["moderate_augmentation"])
    moderate_worst_focus = cast(float, moderate["worst_focus_cars_regression"])
    moderate_manufacturers = cast(int, moderate["cars_manufacturer_regression_count"])
    assessments: dict[str, object] = {}
    advancing: list[Treatment] = []
    for treatment in _TREATMENTS:
        comparison = cast(Mapping[str, object], comparisons[treatment])
        gates = {
            "aggregate_cars_preserved_or_improved": cast(
                float, comparison["cars_mae_relative_change_vs_moderate"]
            )
            <= 0.0,
            "at_least_90_percent_moderate_yoad_gain_retained": cast(
                float, comparison["moderate_yoad_gain_retained"]
            )
            >= 0.90,
            "at_least_five_focus_slices_improve_vs_moderate": cast(
                int, comparison["focus_slices_improved_vs_moderate"]
            )
            >= 5,
            "worst_focus_regression_reduced_at_least_10_percent": cast(
                float, comparison["worst_focus_cars_regression"]
            )
            <= moderate_worst_focus * 0.90,
            "manufacturer_regression_count_not_increased": cast(
                int, comparison["cars_manufacturer_regression_count"]
            )
            <= moderate_manufacturers,
            "no_reported_cars_slice_regresses_more_than_5_percent": cast(
                float, comparison["worst_cars_slice_regression"]
            )
            <= 0.05,
            "cars_fold_standard_deviation_within_10_percent": cast(
                float, comparison["cars_fold_std_relative_to_moderate"]
            )
            <= 1.10,
            "worst_cars_fold_degradation_no_more_than_3_percent": cast(
                float,
                comparison["worst_cars_fold_mae_relative_change_vs_cars_only"],
            )
            <= 0.03,
        }
        passes = all(gates.values())
        assessments[treatment] = {"gates": gates, "advances": passes}
        if passes:
            advancing.append(treatment)

    if not advancing:
        return {
            "classification": "weighting rejected; retain moderate baseline",
            "preferred_treatment": "moderate_augmentation",
            "automatic_promotion": False,
            "assessments": assessments,
            "rationale": (
                "No simple weighting treatment passed every preregistered Cars-slice, "
                "Yoad-retention, aggregate, and fold-stability gate."
            ),
        }
    ranked = sorted(
        advancing,
        key=lambda treatment: (
            cast(
                float,
                cast(Mapping[str, object], comparisons[treatment])["worst_focus_cars_regression"],
            ),
            cast(
                float,
                cast(Mapping[str, object], comparisons[treatment])[
                    "cars_mae_relative_change_vs_moderate"
                ],
            ),
            _TREATMENTS.index(treatment),
        ),
    )
    best = ranked[0]
    best_comparison = cast(Mapping[str, object], comparisons[best])
    near_best = [
        treatment
        for treatment in advancing
        if cast(
            float,
            cast(Mapping[str, object], comparisons[treatment])["worst_focus_cars_regression"],
        )
        <= cast(float, best_comparison["worst_focus_cars_regression"]) + 0.002
        and cast(
            float,
            cast(Mapping[str, object], comparisons[treatment])[
                "cars_mae_relative_change_vs_moderate"
            ],
        )
        <= cast(float, best_comparison["cars_mae_relative_change_vs_moderate"]) + 0.002
    ]
    preferred = min(near_best, key=_TREATMENTS.index)
    preferred_comparison = cast(Mapping[str, object], comparisons[preferred])
    final_eligible = (
        cast(float, preferred_comparison["moderate_yoad_gain_retained"]) >= 0.95
        and cast(float, preferred_comparison["worst_focus_cars_regression"]) <= 0.02
        and cast(float, preferred_comparison["worst_cars_slice_regression"]) <= 0.03
        and cast(int, preferred_comparison["cars_manufacturer_regression_count"]) <= 10
        and cast(float, preferred_comparison["cars_mae_relative_change_vs_moderate"]) <= -0.005
    )
    return {
        "classification": (
            "eligible for a separately defined final promotion evaluation"
            if final_eligible
            else "improved experimental weighting candidate"
        ),
        "preferred_treatment": preferred,
        "automatic_promotion": False,
        "assessments": assessments,
        "rationale": (
            "Selection uses Cars aggregate accuracy, focus slices, all-slice severity, "
            "Yoad-gain retention, and fold stability; pooled MAE is not a selection key. "
            "The simplest treatment wins when results are within preregistered tolerances."
        ),
    }


def _cars_fold_changes(
    stability: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    arm: ReportArm,
) -> dict[str, object]:
    arm_value = cast(Mapping[str, object], stability[arm])
    cars_value = cast(Mapping[str, object], arm_value["cars_com_development"])
    changes: list[float] = []
    for fold in folds:
        metrics = cast(Mapping[str, object], fold["metrics"])
        baseline = cast(
            float,
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], metrics["cars_only"])["cars_com_development"],
            )["mae"],
        )
        challenger = cast(
            float,
            cast(
                Mapping[str, object],
                cast(Mapping[str, object], metrics[arm])["cars_com_development"],
            )["mae"],
        )
        changes.append((challenger - baseline) / baseline)
    return {
        "std": cast(float, cars_value["fold_mae_std"]),
        "relative_changes": changes,
        "worst_relative_change": max(changes),
    }


def _aggregate_weight_diagnostics(
    completed: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for treatment in _TREATMENTS:
        diagnostics = [
            cast(Mapping[str, object], item["weight_diagnostics"])
            for item in completed
            if item["treatment"] == treatment
        ]
        result[treatment] = {
            "folds": diagnostics,
            "minimum_across_folds": min(cast(float, item["minimum"]) for item in diagnostics),
            "maximum_across_folds": max(cast(float, item["maximum"]) for item in diagnostics),
            "mean_of_fold_means": float(
                np.mean([cast(float, item["mean"]) for item in diagnostics])
            ),
            "median_of_fold_medians": float(
                np.median([cast(float, item["median"]) for item in diagnostics])
            ),
            "minimum_effective_sample_fraction": min(
                cast(float, item["effective_sample_fraction"]) for item in diagnostics
            ),
        }
    return result


def _validate_confirmation_evidence(
    data: PreparedExperimentData,
    report: Mapping[str, object],
) -> None:
    if report.get("confirmation_id") != "autovalue-yoad22-source-composition-confirmation-v1":
        raise YoadWeightingError("source-composition confirmation identity differs")
    boundaries = cast(Mapping[str, object], report.get("boundaries"))
    if (
        boundaries.get("cars_development_rows") != CARS_DEVELOPMENT_ROWS
        or boundaries.get("cars_calibration_rows_excluded") != CARS_CALIBRATION_ROWS
        or boundaries.get("legacy_holdout_used") is not False
        or boundaries.get("phase4_artifacts_modified") is not False
        or boundaries.get("feature_contract_version") != BROAD_RETAIL_TRACK.contract_version
    ):
        raise YoadWeightingError("source-composition protection boundary differs")
    model = cast(Mapping[str, object], report.get("model"))
    expected_parameters = {
        "n_estimators": _RF_PARAMETERS[0],
        "max_leaf_nodes": _RF_PARAMETERS[1],
        "min_samples_leaf": _RF_PARAMETERS[2],
        "max_features": _RF_PARAMETERS[3],
        "max_samples": _RF_PARAMETERS[4],
        "random_state": MODEL_RANDOM_STATE,
    }
    if model.get("parameters") != expected_parameters:
        raise YoadWeightingError("source-composition RF05 parameters differ")
    if (
        len(data.features) != FULL_VALIDATION_ROWS
        or dict(data.row_accounting).get("cars_development_rows") != CARS_DEVELOPMENT_ROWS
    ):
        raise YoadWeightingError("current verified experiment population differs")


def _validate_completed_fits(
    completed: Sequence[Mapping[str, object]],
    data: PreparedExperimentData,
    splits: Sequence[tuple[NDArray[np.int64], NDArray[np.int64]]],
    included_mask: NDArray[np.bool_],
) -> None:
    _validate_stable_prefix(completed)
    for item in completed:
        treatment = cast(Treatment, item["treatment"])
        fold_number = cast(int, item["fold"])
        training, validation = splits[fold_number - 1]
        included = training[included_mask[training]]
        if (
            treatment not in _TREATMENTS
            or item.get("training_rows") != len(included)
            or item.get("validation_rows") != len(validation)
        ):
            raise YoadWeightingError("checkpoint fit row accounting differs")
        source_rows = cast(Mapping[str, object], item.get("training_source_rows"))
        if any(
            source_rows.get(source) != int((data.sources[included] == source).sum())
            for source in _SOURCE_NAMES
        ):
            raise YoadWeightingError("checkpoint source row accounting differs")
        metrics = cast(Mapping[str, object], item.get("metrics"))
        for source in ("overall", *_SOURCE_NAMES):
            metric = cast(Mapping[str, object], metrics.get(source))
            expected = (
                len(validation)
                if source == "overall"
                else int((data.sources[validation] == source).sum())
            )
            if metric.get("sample_count") != expected:
                raise YoadWeightingError("checkpoint validation metric rows differ")


def _validate_stable_prefix(completed: Sequence[Mapping[str, object]]) -> None:
    expected = [(treatment, fold) for treatment in _TREATMENTS for fold in range(1, N_SPLITS + 1)]
    if len(completed) > len(expected):
        raise YoadWeightingError("checkpoint has too many completed fits")
    observed = [(item.get("treatment"), item.get("fold")) for item in completed]
    if observed != expected[: len(observed)]:
        raise YoadWeightingError("checkpoint fits must be a stable policy prefix")


def _slice_map(
    slices: Mapping[str, object],
    arm: ReportArm,
    dimension: str,
    source: str,
) -> dict[str, float]:
    dimension_value = cast(Mapping[str, object], slices[dimension])
    arm_value = cast(Mapping[str, object], dimension_value[arm])
    rows = cast(Sequence[Mapping[str, object]], arm_value[source])
    return {
        cast(str, row["slice"]): cast(float, cast(Mapping[str, object], row["metrics"])["mae"])
        for row in rows
    }


def _slice_mae(
    slices: Mapping[str, object],
    arm: ReportArm,
    dimension: str,
    label: str,
) -> float:
    try:
        return _slice_map(slices, arm, dimension, "cars_com_development")[label]
    except KeyError as error:
        raise YoadWeightingError(f"required weighting slice is absent: {label}") from error


def _source_mae(metrics: Mapping[str, object], source: str) -> float:
    return cast(float, cast(Mapping[str, object], metrics[source])["mae"])


def _make_model() -> Pipeline:
    n_estimators, max_leaf_nodes, min_samples_leaf, max_features, max_samples = _RF_PARAMETERS
    return Pipeline(
        steps=(
            ("preprocessor", make_tree_preprocessor(BROAD_RETAIL_TRACK)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    criterion="squared_error",
                    max_depth=None,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    max_leaf_nodes=max_leaf_nodes,
                    bootstrap=True,
                    max_samples=max_samples,
                    n_jobs=4,
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        )
    )


def _bounded_text(serialized: str | bytes, *, label: str) -> str:
    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_CHECKPOINT_BYTES:
            raise YoadWeightingError(f"{label} exceeds maximum size")
        try:
            return serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise YoadWeightingError(f"{label} must be UTF-8") from error
    if isinstance(serialized, str):
        if len(serialized.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
            raise YoadWeightingError(f"{label} exceeds maximum size")
        return serialized
    raise YoadWeightingError(f"{label} must be text or bytes")


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise YoadWeightingError(f"JSON has duplicate field: {key}")
        result[key] = value
    return result


__all__ = [
    "CONFIRMATION_REPORT_SHA256",
    "EXPERIMENT_ID",
    "WEIGHTING_POLICY_SHA256",
    "YoadWeightingError",
    "canonical_weighting_json",
    "load_confirmation_report",
    "make_training_weights",
    "make_weighting_checkpoint",
    "parse_weighting_checkpoint_json",
    "run_weighting_experiment",
]
