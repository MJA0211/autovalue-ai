"""Source-composition confirmation for the approved Yoad22 augmentation.

Cars-only and full-augmentation results are imported from the checksum-bound
controlled experiment. Balanced and moderate augmentation are newly fitted on
the exact same pooled predictor-group folds and scored on the full paired
validation population.
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
from .yoad_experiment import (
    BROAD_RETAIL_TRACK,
    MODEL_RANDOM_STATE,
    N_SPLITS,
    PreparedExperimentData,
    controlled_group_splits,
)

ConfirmationArm = Literal[
    "cars_only",
    "balanced_augmentation",
    "moderate_augmentation",
    "full_augmentation",
]
NewArm = Literal["balanced_augmentation", "moderate_augmentation"]
ProgressCallback = Callable[[NewArm, int, int], None]

CONFIRMATION_ID: Final = "autovalue-yoad22-source-composition-confirmation-v1"
CONTROLLED_REPORT_SHA256: Final = "30d1f6011b7f2d5e611bbae6197be4780eeabcda3daca501c0b683807cf12ec5"
BALANCED_YOAD_ROWS: Final = 98_552
MODERATE_YOAD_ROWS: Final = 150_000
FULL_YOAD_ROWS: Final = 242_666
_RF_PARAMETERS: Final = RETAIL_RANDOM_FOREST_CONFIGS[5]
_SAMPLE_DOMAIN: Final = b"autovalue-yoad22-composition-sampling-v1\x00"
_SOURCE_NAMES: Final = ("cars_com_development", "yoad22_craigslist")
_ARMS: Final[tuple[ConfirmationArm, ...]] = (
    "cars_only",
    "balanced_augmentation",
    "moderate_augmentation",
    "full_augmentation",
)
_NEW_ARMS: Final[tuple[NewArm, ...]] = (
    "balanced_augmentation",
    "moderate_augmentation",
)


class YoadConfirmationError(RuntimeError):
    """Confirmation evidence, sampling, or evaluation failed closed."""


def load_controlled_report(path: Path) -> dict[str, object]:
    """Load the exact immutable controlled report used as endpoint evidence."""

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CONTROLLED_REPORT_SHA256:
        raise YoadConfirmationError("controlled experiment checksum differs from approval")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise YoadConfirmationError("controlled experiment report is invalid JSON") from error
    if not isinstance(value, dict):
        raise YoadConfirmationError("controlled experiment report must be an object")
    return cast(dict[str, object], value)


def deterministic_yoad_subsets(
    data: PreparedExperimentData,
) -> dict[NewArm, NDArray[np.int64]]:
    """Return exact, nested, target-free Yoad samples stratified on predictors."""

    yoad_indices = np.flatnonzero(data.sources == "yoad22_craigslist").astype(np.int64, copy=False)
    if len(yoad_indices) != FULL_YOAD_ROWS:
        raise YoadConfirmationError("approved Yoad population differs from confirmation policy")
    frame = data.features.iloc[yoad_indices]
    mileage = pd.to_numeric(frame["mileage"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(mileage).all():
        raise YoadConfirmationError("Yoad confirmation mileage must be complete and finite")
    quantile_edges = np.unique(np.quantile(mileage, np.linspace(0.0, 1.0, 11)))
    mileage_bins = np.searchsorted(quantile_edges[1:-1], mileage, side="right")
    strata = np.asarray(
        [
            f"{make}\x1f{int(year)}\x1f{int(mileage_bin)}"
            for make, year, mileage_bin in zip(
                frame["make"], frame["year"], mileage_bins, strict=True
            )
        ],
        dtype=np.str_,
    )
    unique_strata, inverse, counts = np.unique(
        strata,
        return_inverse=True,
        return_counts=True,
    )
    balanced_quotas = _proportional_quotas(
        unique_strata,
        counts.astype(np.int64, copy=False),
        BALANCED_YOAD_ROWS,
    )
    moderate_quotas = _nested_quotas(
        unique_strata,
        counts.astype(np.int64, copy=False),
        MODERATE_YOAD_ROWS,
        minimum=balanced_quotas,
    )
    ranks = np.asarray(
        [
            hashlib.sha256(
                _SAMPLE_DOMAIN + f"{strata[position]}\x1f{int(yoad_indices[position])}".encode()
            ).digest()
            for position in range(len(yoad_indices))
        ],
        dtype="S32",
    )
    balanced = _select_by_stratum(yoad_indices, inverse, ranks, balanced_quotas)
    moderate = _select_by_stratum(yoad_indices, inverse, ranks, moderate_quotas)
    if len(balanced) != BALANCED_YOAD_ROWS or len(moderate) != MODERATE_YOAD_ROWS:
        raise YoadConfirmationError("deterministic subset row counts do not match policy")
    if not set(balanced.tolist()).issubset(moderate.tolist()):
        raise YoadConfirmationError("balanced Yoad subset is not nested inside moderate")
    return {
        "balanced_augmentation": balanced,
        "moderate_augmentation": moderate,
    }


def run_yoad_confirmation(
    *,
    data: PreparedExperimentData,
    controlled_report: Mapping[str, object],
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Fit new composition arms and assemble a four-arm paired confirmation."""

    _validate_controlled_evidence(data, controlled_report)
    subsets = deterministic_yoad_subsets(data)
    splits = controlled_group_splits(data.features)
    predictions = {arm: np.full(len(data.features), np.nan, dtype=np.float64) for arm in _NEW_ARMS}
    fold_reports: list[dict[str, object]] = []
    cars_mask = data.sources == "cars_com_development"
    selected_masks: dict[NewArm, NDArray[np.bool_]] = {}
    for arm, indices in subsets.items():
        mask = np.zeros(len(data.features), dtype=np.bool_)
        mask[indices] = True
        selected_masks[arm] = mask

    for fold_number, (training, validation) in enumerate(splits, start=1):
        report: dict[str, object] = {
            "fold": fold_number,
            "validation_rows": len(validation),
            "training_rows": {},
            "metrics": {},
        }
        training_rows = cast(dict[str, int], report["training_rows"])
        fold_metrics = cast(dict[str, object], report["metrics"])
        for arm in _NEW_ARMS:
            included = training[cars_mask[training] | selected_masks[arm][training]]
            model = _make_model()
            model.fit(data.features.iloc[included], data.target[included])
            predictions[arm][validation] = model.predict(data.features.iloc[validation])
            training_rows[arm] = len(included)
            fold_metrics[arm] = _metrics_by_source(
                data.target[validation],
                predictions[arm][validation],
                data.sources[validation],
            )
            if on_progress is not None:
                on_progress(arm, fold_number, N_SPLITS)
        fold_reports.append(report)

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise YoadConfirmationError("not every row received confirmation predictions")

    metrics = _endpoint_metrics(controlled_report)
    metrics.update(
        {
            arm: _metrics_by_source(data.target, values, data.sources)
            for arm, values in predictions.items()
        }
    )
    folds = _endpoint_folds(controlled_report)
    for position, report in enumerate(fold_reports):
        target_fold = folds[position]
        cast(dict[str, object], target_fold["metrics"]).update(
            cast(Mapping[str, object], report["metrics"])
        )
        cast(dict[str, object], target_fold["training_rows"]).update(
            cast(Mapping[str, object], report["training_rows"])
        )
    slices = _endpoint_slices(controlled_report)
    new_slices = _slice_report(data, predictions)
    for dimension in new_slices:
        cast(dict[str, object], slices[dimension]).update(
            cast(Mapping[str, object], new_slices[dimension])
        )
    stability = _stability_report(folds)
    subset_audit = _subset_audit(data, subsets)
    segment_comparison = _critical_segment_comparison(metrics, folds, slices)
    decision = _confirmation_decision(metrics, stability, segment_comparison)
    return {
        "schema_version": 1,
        "report_type": "yoad22_source_composition_confirmation",
        "confirmation_id": CONFIRMATION_ID,
        "controlled_experiment": {
            "path": "docs/experiments/yoad22-controlled-batch-v1.json",
            "sha256": CONTROLLED_REPORT_SHA256,
            "endpoint_results_reused_without_modification": True,
        },
        "boundaries": {
            "cars_development_rows": 98_552,
            "cars_calibration_rows_excluded": 10_958,
            "legacy_holdout_used": False,
            "phase4_artifacts_modified": False,
            "feature_contract_version": BROAD_RETAIL_TRACK.contract_version,
            "common_predictors": list(BROAD_RETAIL_TRACK.input_features),
            "model_field_excluded": True,
            "source_identity_used_as_feature": False,
            "target_used_for_sampling_or_folds": False,
            "fold_method": "exact controlled-experiment pooled GroupKFold assignments",
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
        "composition": {
            "cars_only": {"cars_rows": 98_552, "yoad_rows": 0},
            "balanced_augmentation": {
                "cars_rows": 98_552,
                "yoad_rows": BALANCED_YOAD_ROWS,
            },
            "moderate_augmentation": {
                "cars_rows": 98_552,
                "yoad_rows": MODERATE_YOAD_ROWS,
            },
            "full_augmentation": {"cars_rows": 98_552, "yoad_rows": FULL_YOAD_ROWS},
        },
        "subset_selection": subset_audit,
        "metrics": metrics,
        "fold_metrics": folds,
        "fold_stability": stability,
        "slice_metrics": slices,
        "critical_segment_comparison": segment_comparison,
        "decision": decision,
        "governance": {
            "automatic_promotion": False,
            "phase4_rf05_replaced": False,
            "yoad_online_river_learning": "blocked",
            "carson_shively_included": False,
        },
    }


def canonical_confirmation_json(report: Mapping[str, object]) -> str:
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


def _proportional_quotas(
    strata: NDArray[np.str_], counts: NDArray[np.int64], target: int
) -> NDArray[np.int64]:
    if target < 1 or target > int(counts.sum()):
        raise YoadConfirmationError("subset target is outside the population")
    exact = counts.astype(np.float64) * (target / float(counts.sum()))
    quotas = np.floor(exact).astype(np.int64)
    remaining = target - int(quotas.sum())
    order = sorted(
        range(len(strata)),
        key=lambda index: (-(exact[index] - quotas[index]), str(strata[index])),
    )
    for index in order[:remaining]:
        quotas[index] += 1
    return quotas


def _nested_quotas(
    strata: NDArray[np.str_],
    counts: NDArray[np.int64],
    target: int,
    *,
    minimum: NDArray[np.int64],
) -> NDArray[np.int64]:
    quotas = np.maximum(_proportional_quotas(strata, counts, target), minimum)
    excess = int(quotas.sum()) - target
    if excess:
        exact = counts.astype(np.float64) * (target / float(counts.sum()))
        removable = sorted(
            (index for index in range(len(strata)) if quotas[index] > minimum[index]),
            key=lambda index: (exact[index] - np.floor(exact[index]), str(strata[index])),
        )
        for index in removable[:excess]:
            quotas[index] -= 1
    if int(quotas.sum()) != target or (quotas < minimum).any() or (quotas > counts).any():
        raise YoadConfirmationError("nested proportional quota allocation failed")
    return quotas


def _select_by_stratum(
    source_indices: NDArray[np.int64],
    inverse: NDArray[np.int64],
    ranks: NDArray[np.bytes_],
    quotas: NDArray[np.int64],
) -> NDArray[np.int64]:
    selected: list[int] = []
    order = np.lexsort((ranks, inverse))
    ordered_inverse = inverse[order]
    boundaries = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(ordered_inverse) != 0).astype(np.int64) + 1,
            np.asarray([len(order)], dtype=np.int64),
        )
    )
    if len(boundaries) != len(quotas) + 1:
        raise YoadConfirmationError("stratum ordering lost an allocation group")
    for stratum_index, quota in enumerate(quotas):
        positions = order[boundaries[stratum_index] : boundaries[stratum_index] + int(quota)]
        selected.extend(int(source_indices[position]) for position in positions)
    return np.asarray(sorted(selected), dtype=np.int64)


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


def _validate_controlled_evidence(
    data: PreparedExperimentData, report: Mapping[str, object]
) -> None:
    if report.get("experiment_id") != "autovalue-yoad22-controlled-batch-v1":
        raise YoadConfirmationError("controlled experiment identity differs")
    accounting = cast(Mapping[str, object], report.get("row_accounting"))
    expected = {
        "cars_development_rows": 98_552,
        "cars_calibration_rows_excluded": 10_958,
        "yoad_approved_rows": FULL_YOAD_ROWS,
        "combined_training_rows": 341_218,
    }
    if any(accounting.get(key) != value for key, value in expected.items()):
        raise YoadConfirmationError("controlled experiment row boundary differs")
    if dict(data.row_accounting) != accounting:
        raise YoadConfirmationError("current verified row accounting differs from controlled run")
    boundaries = cast(Mapping[str, object], report.get("boundaries"))
    if (
        boundaries.get("phase4_calibration_used") is not False
        or boundaries.get("legacy_holdout_used") is not False
        or boundaries.get("feature_contract_version") != BROAD_RETAIL_TRACK.contract_version
    ):
        raise YoadConfirmationError("controlled experiment protection boundary differs")
    model = cast(Mapping[str, object], report.get("model"))
    parameters = cast(Mapping[str, object], model.get("parameters"))
    expected_parameters = {
        "n_estimators": _RF_PARAMETERS[0],
        "max_leaf_nodes": _RF_PARAMETERS[1],
        "min_samples_leaf": _RF_PARAMETERS[2],
        "max_features": _RF_PARAMETERS[3],
        "max_samples": _RF_PARAMETERS[4],
        "random_state": MODEL_RANDOM_STATE,
    }
    if parameters != expected_parameters:
        raise YoadConfirmationError("controlled experiment RF05 parameters differ")


def _metrics_by_source(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    sources: NDArray[np.str_],
) -> dict[str, object]:
    result: dict[str, object] = {"overall": regression_metrics(target, predicted).to_dict()}
    for source in _SOURCE_NAMES:
        mask = sources == source
        result[source] = regression_metrics(target[mask], predicted[mask]).to_dict()
    return result


def _endpoint_metrics(report: Mapping[str, object]) -> dict[str, object]:
    controlled = cast(Mapping[str, object], report["metrics"])
    return {
        "cars_only": controlled["cars_only"],
        "full_augmentation": controlled["cars_plus_yoad"],
    }


def _endpoint_folds(report: Mapping[str, object]) -> list[dict[str, object]]:
    controlled_folds = cast(Sequence[Mapping[str, object]], report["fold_metrics"])
    output: list[dict[str, object]] = []
    for fold in controlled_folds:
        metrics = cast(Mapping[str, object], fold["metrics"])
        training = cast(Mapping[str, object], fold["training_rows"])
        output.append(
            {
                "fold": fold["fold"],
                "validation_rows": fold["validation_rows"],
                "training_rows": {
                    "cars_only": training["cars_only"],
                    "full_augmentation": training["cars_plus_yoad"],
                },
                "metrics": {
                    "cars_only": metrics["cars_only"],
                    "full_augmentation": metrics["cars_plus_yoad"],
                },
            }
        )
    return output


def _endpoint_slices(report: Mapping[str, object]) -> dict[str, object]:
    controlled = cast(Mapping[str, object], report["slice_metrics"])
    output: dict[str, object] = {}
    for dimension, value in controlled.items():
        arms = cast(Mapping[str, object], value)
        output[dimension] = {
            "cars_only": arms["cars_only"],
            "full_augmentation": arms["cars_plus_yoad"],
        }
    return output


def _stability_report(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for arm in _ARMS:
        arm_result: dict[str, object] = {}
        for source in ("overall", *_SOURCE_NAMES):
            values = np.asarray(
                [
                    _metric_value(cast(Mapping[str, object], fold["metrics"]), arm, source, "mae")
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
        output[arm] = arm_result
    return output


def _slice_report(
    data: PreparedExperimentData,
    predictions: Mapping[NewArm, NDArray[np.float64]],
) -> dict[str, object]:
    age = (BROAD_RETAIL_TRACK.reference_year - data.features["year"].astype(float)).clip(lower=0)
    labels = {
        "price_band": _fixed_band_labels(
            pd.Series(data.target), (1.0, 8_995.0, 19_995.0, 36_590.0, 8_078_160.0)
        ),
        "manufacturer": data.features["make"].astype(str),
        "vehicle_age_band": _fixed_band_labels(age, (0.0, 3.0, 8.0, 13.0, 64.0)),
        "mileage_band": _mileage_labels(data.features["mileage"]),
    }
    result: dict[str, object] = {}
    for dimension, dimension_labels in labels.items():
        arm_result: dict[str, object] = {}
        for arm, values in predictions.items():
            arm_result[arm] = {
                source: _aggregate_slices(
                    data.target[data.sources == source],
                    values[data.sources == source],
                    dimension_labels[data.sources == source],
                    minimum_count=100 if dimension == "manufacturer" else 1,
                )
                for source in _SOURCE_NAMES
            }
        result[dimension] = arm_result
    return result


def _fixed_band_labels(values: pd.Series, edges: Sequence[float]) -> pd.Series:
    adjusted = np.asarray(edges, dtype=np.float64)
    adjusted[0] -= max(1e-9, abs(float(adjusted[0])) * 1e-12)
    return pd.cut(values.astype(float), bins=adjusted, include_lowest=True).astype(str)


def _mileage_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    labels = pd.Series("mileage:missing", index=values.index, dtype=object)
    present = numeric.notna()
    labels.loc[present] = _fixed_band_labels(
        numeric.loc[present],
        (0.0, 38_282.0, 86_204.0, 135_803.0, 405_187.0),
    ).to_numpy()
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


def _subset_audit(
    data: PreparedExperimentData,
    subsets: Mapping[NewArm, NDArray[np.int64]],
) -> dict[str, object]:
    yoad = np.flatnonzero(data.sources == "yoad22_craigslist").astype(np.int64, copy=False)
    result: dict[str, object] = {
        "method": (
            "nested proportional allocation by normalized manufacturer, exact model year, and "
            "full-Yoad mileage decile; SHA-256 predictor/position rank within stratum"
        ),
        "target_used": False,
        "full_population_rows": len(yoad),
    }
    for arm, selected in subsets.items():
        result[arm] = _distribution_preservation(
            data.features.iloc[yoad], data.features.iloc[selected]
        )
    return result


def _distribution_preservation(full: pd.DataFrame, sample: pd.DataFrame) -> dict[str, object]:
    full_mileage = pd.to_numeric(full["mileage"], errors="raise").to_numpy(dtype=np.float64)
    sample_mileage = pd.to_numeric(sample["mileage"], errors="raise").to_numpy(dtype=np.float64)
    edges = np.unique(np.quantile(full_mileage, np.linspace(0.0, 1.0, 11)))
    return {
        "rows": len(sample),
        "nested_selection": True,
        "median_year": float(sample["year"].median()),
        "median_mileage": float(np.median(sample_mileage)),
        "maximum_manufacturer_share_difference_percentage_points": _max_share_difference(
            full["make"].astype(str), sample["make"].astype(str)
        ),
        "maximum_year_share_difference_percentage_points": _max_share_difference(
            full["year"].astype(str), sample["year"].astype(str)
        ),
        "maximum_mileage_decile_share_difference_percentage_points": _max_share_difference(
            pd.Series(np.searchsorted(edges[1:-1], full_mileage, side="right")).astype(str),
            pd.Series(np.searchsorted(edges[1:-1], sample_mileage, side="right")).astype(str),
        ),
    }


def _max_share_difference(full: pd.Series, sample: pd.Series) -> float:
    full_shares = full.value_counts(normalize=True)
    sample_shares = sample.value_counts(normalize=True)
    labels = full_shares.index.union(sample_shares.index)
    difference = full_shares.reindex(labels, fill_value=0.0) - sample_shares.reindex(
        labels, fill_value=0.0
    )
    return float(difference.abs().max() * 100.0)


def _critical_segment_comparison(
    metrics: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
    slices: Mapping[str, object],
) -> dict[str, object]:
    baseline = cast(Mapping[str, object], metrics["cars_only"])
    result: dict[str, object] = {}
    focus = {
        "highest_mileage": ("mileage_band", "(135803.0, 405187.0]"),
        "low_mileage": ("mileage_band", "(-0.001000001, 38282.0]"),
        "age_3_to_8": ("vehicle_age_band", "(3.0, 8.0]"),
        "age_8_to_13": ("vehicle_age_band", "(8.0, 13.0]"),
        "highest_price": ("price_band", "(36590.0, 8078160.0]"),
        "alfa_romeo": ("manufacturer", "alfa romeo"),
        "hyundai": ("manufacturer", "hyundai"),
        "chevrolet": ("manufacturer", "chevrolet"),
        "audi": ("manufacturer", "audi"),
        "jaguar": ("manufacturer", "jaguar"),
    }
    baseline_cars_mae = _source_mae(baseline, "cars_com_development")
    baseline_yoad_mae = _source_mae(baseline, "yoad22_craigslist")
    for arm in _ARMS:
        arm_metrics = cast(Mapping[str, object], metrics[arm])
        cars_change = (
            _source_mae(arm_metrics, "cars_com_development") - baseline_cars_mae
        ) / baseline_cars_mae
        yoad_improvement = (
            baseline_yoad_mae - _source_mae(arm_metrics, "yoad22_craigslist")
        ) / baseline_yoad_mae
        segment_changes: dict[str, float] = {}
        for name, (dimension, label) in focus.items():
            baseline_value = _slice_mae(
                slices, dimension, "cars_only", "cars_com_development", label
            )
            arm_value = _slice_mae(slices, dimension, arm, "cars_com_development", label)
            segment_changes[name] = (arm_value - baseline_value) / baseline_value
        manufacturer_regressions = _manufacturer_regressions(slices, arm)
        fold_changes = []
        for fold in folds:
            fold_metrics = cast(Mapping[str, object], fold["metrics"])
            reference = _metric_value(fold_metrics, "cars_only", "cars_com_development", "mae")
            challenger = _metric_value(fold_metrics, arm, "cars_com_development", "mae")
            fold_changes.append((challenger - reference) / reference)
        result[arm] = {
            "cars_mae_relative_change": cars_change,
            "yoad_mae_relative_improvement": yoad_improvement,
            "worst_cars_fold_mae_relative_change": max(fold_changes),
            "focus_segment_cars_mae_relative_changes": segment_changes,
            "worst_focus_segment_regression": max(segment_changes.values()),
            "manufacturer_cars_regression_count": len(manufacturer_regressions),
            "largest_manufacturer_cars_regressions": manufacturer_regressions[:10],
        }
    return result


def _manufacturer_regressions(
    slices: Mapping[str, object], arm: ConfirmationArm
) -> list[dict[str, object]]:
    baseline = _slice_map(slices, "manufacturer", "cars_only", "cars_com_development")
    challenger = _slice_map(slices, "manufacturer", arm, "cars_com_development")
    rows: list[dict[str, object]] = []
    for manufacturer in sorted(set(baseline) & set(challenger)):
        change = (challenger[manufacturer] - baseline[manufacturer]) / baseline[manufacturer]
        if change > 0:
            rows.append(
                {
                    "manufacturer": manufacturer,
                    "mae_relative_change": change,
                    "baseline_mae": baseline[manufacturer],
                    "challenger_mae": challenger[manufacturer],
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            -cast(float, item["mae_relative_change"]),
            cast(str, item["manufacturer"]),
        ),
    )


def _confirmation_decision(
    metrics: Mapping[str, object],
    stability: Mapping[str, object],
    segments: Mapping[str, object],
) -> dict[str, object]:
    full_yoad_improvement = cast(
        float,
        cast(Mapping[str, object], segments["full_augmentation"])["yoad_mae_relative_improvement"],
    )
    candidates: list[tuple[tuple[float, float, float, float], ConfirmationArm]] = []
    assessments: dict[str, object] = {}
    for arm in _ARMS[1:]:
        segment = cast(Mapping[str, object], segments[arm])
        cars_change = cast(float, segment["cars_mae_relative_change"])
        yoad_improvement = cast(float, segment["yoad_mae_relative_improvement"])
        retention = yoad_improvement / full_yoad_improvement
        critical = cast(float, segment["worst_focus_segment_regression"])
        worst_fold = cast(float, segment["worst_cars_fold_mae_relative_change"])
        arm_stability = cast(Mapping[str, object], stability[arm])
        cars_stability = cast(Mapping[str, object], arm_stability["cars_com_development"])
        substantial = retention >= 0.70 and yoad_improvement >= 0.20
        stable = (
            worst_fold <= 0.03 and cast(float, cars_stability["coefficient_of_variation"]) <= 0.12
        )
        assessments[arm] = {
            "yoad_improvement_retained_vs_full": retention,
            "substantial_yoad_improvement": substantial,
            "cars_fold_stability_gate": stable,
            "cars_mae_relative_change": cars_change,
            "worst_focus_segment_regression": critical,
        }
        if substantial and stable:
            candidates.append(
                (
                    (
                        max(cars_change, 0.0),
                        max(critical, 0.0),
                        worst_fold,
                        -yoad_improvement,
                    ),
                    arm,
                )
            )
    if not candidates:
        return {
            "recommendation": "reject augmentation",
            "preferred_composition": "cars_only",
            "automatic_promotion": False,
            "assessments": assessments,
            "rationale": "No augmentation retained substantial Yoad gain with stable Cars folds.",
        }
    _, preferred = min(candidates)
    preferred_segment = cast(Mapping[str, object], segments[preferred])
    eligible = (
        cast(float, preferred_segment["cars_mae_relative_change"]) <= 0.0
        and cast(float, preferred_segment["worst_focus_segment_regression"]) <= 0.02
        and cast(int, preferred_segment["manufacturer_cars_regression_count"]) <= 10
    )
    return {
        "recommendation": (
            "eligible for final promotion evaluation"
            if eligible
            else "retain as separate experimental model"
        ),
        "preferred_composition": preferred,
        "automatic_promotion": False,
        "assessments": assessments,
        "rationale": (
            "Preference minimizes Cars MAE degradation first, then critical-segment regression "
            "and worst-fold degradation, among stable treatments retaining at least 70% of the "
            "full augmentation's Yoad MAE gain. Pooled MAE is not the selection key."
        ),
    }


def _slice_map(
    slices: Mapping[str, object],
    dimension: str,
    arm: ConfirmationArm,
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
    dimension: str,
    arm: ConfirmationArm,
    source: str,
    label: str,
) -> float:
    try:
        return _slice_map(slices, dimension, arm, source)[label]
    except KeyError as error:
        raise YoadConfirmationError(f"required confirmation slice is absent: {label}") from error


def _metric_value(
    fold_metrics: Mapping[str, object], arm: ConfirmationArm, source: str, metric: str
) -> float:
    arm_value = cast(Mapping[str, object], fold_metrics[arm])
    source_value = cast(Mapping[str, object], arm_value[source])
    return cast(float, source_value[metric])


def _source_mae(metrics: Mapping[str, object], source: str) -> float:
    return cast(float, cast(Mapping[str, object], metrics[source])["mae"])


__all__ = [
    "BALANCED_YOAD_ROWS",
    "CONFIRMATION_ID",
    "CONTROLLED_REPORT_SHA256",
    "MODERATE_YOAD_ROWS",
    "YoadConfirmationError",
    "canonical_confirmation_json",
    "deterministic_yoad_subsets",
    "load_controlled_report",
    "run_yoad_confirmation",
]
