"""Controlled Cars.com-development versus Cars.com-plus-Yoad batch experiment.

The experiment is deliberately separate from Phase 4. It uses only the frozen
retail development partition, keeps every common predictor group in one fold,
and persists aggregate metrics rather than row-level predictions.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from autovalue_ml.acquisition.huggingface_dataset import acquire_huggingface_artifact
from autovalue_ml.acquisition.sources.huggingface_candidates import YOAD22_CRAIGSLIST_SPEC
from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    prepare_kaggle_us_sales_cars_split_training_rows,
)

from .baseline_cli import (
    RetailTrainingRow,
    _collect_retail_partition,
    _expected_count,
)
from .calibration import retail_calibration_partition
from .candidates import RETAIL_RANDOM_FOREST_CONFIGS
from .contracts import TrackConfig
from .metrics import regression_metrics
from .tree_preprocessing import make_tree_preprocessor

SourceName = Literal["cars_com_development", "yoad22_craigslist"]
ArmName = Literal["cars_only", "cars_plus_yoad"]
ProgressCallback = Callable[[int, int], None]

EXPERIMENT_ID: Final = "autovalue-yoad22-controlled-batch-v1"
CALIBRATION_SEED: Final = 1_416_582_761
MODEL_RANDOM_STATE: Final = 1_254_777_149
N_SPLITS: Final = 5
_RF_PARAMETERS: Final = RETAIL_RANDOM_FOREST_CONFIGS[5]
_RETAIL_PATHS: Final = (
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json"),
    PurePosixPath("docs/data-reviews/kaggle-us-sales-cars-v2.review.json"),
)
_YOAD_COLUMNS: Final = (
    "price",
    "year",
    "manufacturer",
    "condition",
    "cylinders",
    "fuel",
    "odometer",
    "title_status",
    "transmission",
    "drive",
    "type",
    "paint_color",
    "state",
    "car_age",
)

BROAD_RETAIL_TRACK: Final = TrackConfig(
    name="retail",
    contract_version="retail-yoad22-common-coverage-v1",
    target_name="price_usd",
    target_semantics="historical_us_advertised_asking_price_usd_cross_source_experiment",
    reference_year=2023,
    required_input_features=("year", "make", "vehicle_status"),
    optional_input_features=("mileage",),
    numeric_features=("model_year", "mileage", "mileage_per_year", "mileage_missing"),
    categorical_features=("make", "vehicle_status"),
    one_hot_min_frequency=25,
    one_hot_max_categories=128,
    status_slice_feature="vehicle_status",
)


class YoadExperimentError(RuntimeError):
    """The controlled experiment violated an approval or leakage boundary."""


@dataclass(frozen=True, slots=True)
class PreparedExperimentData:
    features: pd.DataFrame
    target: NDArray[np.float64]
    sources: NDArray[np.str_]
    row_accounting: Mapping[str, int]

    def __post_init__(self) -> None:
        rows = len(self.features)
        if rows == 0 or len(self.target) != rows or len(self.sources) != rows:
            raise YoadExperimentError("prepared experiment arrays are not aligned")


def load_controlled_experiment_data(project_root: Path) -> PreparedExperimentData:
    """Load only verified Cars development rows and approved Yoad rows."""

    root = project_root.resolve()
    YOAD22_CRAIGSLIST_SPEC.approvals.require_batch_training()
    artifact = acquire_huggingface_artifact(
        YOAD22_CRAIGSLIST_SPEC,
        root / "data" / "raw",
    )

    retail_paths = tuple(root.joinpath(*path.parts) for path in _RETAIL_PATHS)
    stream = prepare_kaggle_us_sales_cars_split_training_rows(
        *retail_paths,
        partition="train",
    )
    cars_train = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], stream),
        expected_rows=_expected_count(stream, "expected_rows"),
        label="Yoad experiment Cars.com Phase-3 train",
    )
    partition = retail_calibration_partition(cars_train.features, seed=CALIBRATION_SEED)
    cars_features = cars_train.features.iloc[partition.development_indices].reset_index(drop=True)
    cars_target = cars_train.target[partition.development_indices].astype(np.float64, copy=True)
    cars_common = _common_cars_features(cars_features)

    raw_yoad = pd.read_csv(artifact.path, low_memory=False)
    if tuple(raw_yoad.columns) != _YOAD_COLUMNS:
        raise YoadExperimentError("Yoad artifact columns differ from the reviewed schema")
    raw_rows = len(raw_yoad)
    if raw_rows != YOAD22_CRAIGSLIST_SPEC.expected_row_count:
        raise YoadExperimentError("Yoad row count differs from the pinned specification")
    exact_duplicates = int(raw_yoad.duplicated(keep="first").sum())
    yoad = raw_yoad.drop_duplicates(keep="first").copy()
    normalized_make = yoad["manufacturer"].astype(str).str.strip().str.casefold()
    unknown_make_mask = normalized_make.eq("unknown") | normalized_make.eq("")
    unknown_make_rows = int(unknown_make_mask.sum())
    yoad = yoad.loc[~unknown_make_mask].copy()
    yoad["manufacturer"] = normalized_make.loc[~unknown_make_mask]
    yoad_common, yoad_target = _common_yoad_rows(yoad)

    collision_keys = _collision_keys(cars_common, cars_target)
    yoad_keys = _collision_key_values(yoad_common, yoad_target)
    collision_mask = np.fromiter(
        (key in collision_keys for key in yoad_keys),
        dtype=np.bool_,
        count=len(yoad_keys),
    )
    collision_rows = int(collision_mask.sum())
    if collision_rows:
        yoad_common = yoad_common.loc[~collision_mask].reset_index(drop=True)
        yoad_target = yoad_target[~collision_mask]

    combined = pd.concat((cars_common, yoad_common), ignore_index=True)
    target = np.concatenate((cars_target, yoad_target)).astype(np.float64, copy=False)
    sources = np.asarray(
        ["cars_com_development"] * len(cars_common) + ["yoad22_craigslist"] * len(yoad_common),
        dtype=np.str_,
    )
    accounting = {
        "cars_phase3_train_rows": len(cars_train.features),
        "cars_calibration_rows_excluded": len(partition.calibration_indices),
        "cars_development_rows": len(cars_common),
        "yoad_source_rows": raw_rows,
        "yoad_exact_duplicate_rows_removed": exact_duplicates,
        "yoad_unknown_make_rows_removed": unknown_make_rows,
        "yoad_cross_source_collision_rows_removed": collision_rows,
        "yoad_approved_rows": len(yoad_common),
        "combined_training_rows": len(combined),
    }
    return PreparedExperimentData(combined, target, sources, accounting)


def controlled_group_splits(
    features: pd.DataFrame,
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    """Create five shared target-free folds from the common predictor contract."""

    groups = _predictor_groups(features)
    if len(np.unique(groups)) < N_SPLITS:
        raise YoadExperimentError("too few common predictor groups for five-fold CV")
    rows = np.arange(len(features), dtype=np.int64)
    splitter = GroupKFold(n_splits=N_SPLITS, shuffle=False)
    splits = tuple(
        (
            train.astype(np.int64, copy=False),
            validation.astype(np.int64, copy=False),
        )
        for train, validation in splitter.split(rows, groups=groups)
    )
    for train, validation in splits:
        if set(groups[train]) & set(groups[validation]):
            raise YoadExperimentError("a common predictor group crossed a CV boundary")
    return splits


def run_controlled_experiment(
    data: PreparedExperimentData,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Fit paired Cars-only and combined models on identical grouped folds."""

    splits = controlled_group_splits(data.features)
    rows = len(data.features)
    predictions: dict[ArmName, NDArray[np.float64]] = {
        "cars_only": np.full(rows, np.nan, dtype=np.float64),
        "cars_plus_yoad": np.full(rows, np.nan, dtype=np.float64),
    }
    fold_reports: list[dict[str, object]] = []
    cars_mask = data.sources == "cars_com_development"

    for fold_number, (training, validation) in enumerate(splits, start=1):
        cars_training = training[cars_mask[training]]
        if len(cars_training) == 0:
            raise YoadExperimentError("a fold has no Cars.com training rows")
        cars_model = _make_model()
        cars_model.fit(data.features.iloc[cars_training], data.target[cars_training])
        predictions["cars_only"][validation] = cars_model.predict(data.features.iloc[validation])

        combined_model = _make_model()
        combined_model.fit(data.features.iloc[training], data.target[training])
        predictions["cars_plus_yoad"][validation] = combined_model.predict(
            data.features.iloc[validation]
        )
        fold_reports.append(
            _fold_report(
                fold_number,
                training,
                validation,
                data,
                predictions,
            )
        )
        if on_progress is not None:
            on_progress(fold_number, N_SPLITS)

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise YoadExperimentError("not every row received paired out-of-fold predictions")

    metrics = {
        arm: _metrics_by_source(data.target, values, data.sources)
        for arm, values in predictions.items()
    }
    stability = _stability_report(fold_reports)
    slices = _slice_report(data, predictions)
    shifts = _distribution_report(data)
    decision = _recommendation(cast(Mapping[str, object], metrics), fold_reports)
    return {
        "schema_version": 1,
        "report_type": "controlled_cross_source_batch_experiment",
        "experiment_id": EXPERIMENT_ID,
        "approval": {
            "source_id": YOAD22_CRAIGSLIST_SPEC.source_id,
            "batch_training": "approved_for_controlled_experiment_only",
            "online_learning": "blocked",
            "production_training": "not_approved",
            "license": YOAD22_CRAIGSLIST_SPEC.declared_license,
            "attribution": YOAD22_CRAIGSLIST_SPEC.attribution,
            "pinned_revision": YOAD22_CRAIGSLIST_SPEC.revision,
            "artifact_sha256": YOAD22_CRAIGSLIST_SPEC.expected_sha256,
        },
        "boundaries": {
            "phase4_artifacts_modified": False,
            "phase4_calibration_used": False,
            "legacy_holdout_used": False,
            "source_identity_used_as_feature": False,
            "target_used_for_fold_assignment": False,
            "feature_contract_version": BROAD_RETAIL_TRACK.contract_version,
            "common_predictors": list(BROAD_RETAIL_TRACK.input_features),
            "model_field_excluded_because_yoad_coverage_is_zero": True,
            "folds": N_SPLITS,
            "fold_method": "GroupKFold on pooled common predictor groups; paired arms share folds",
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
        "row_accounting": dict(data.row_accounting),
        "source_composition": _source_composition(data.sources),
        "metrics": metrics,
        "fold_metrics": fold_reports,
        "fold_stability": stability,
        "slice_metrics": slices,
        "distribution_shifts": shifts,
        "decision": decision,
    }


def canonical_experiment_json(report: Mapping[str, object]) -> str:
    """Serialize aggregate evidence deterministically and reject NaN/Infinity."""

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


def _common_cars_features(features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": features["year"].astype(np.int32),
            "make": features["make"].astype(str).str.strip().str.casefold(),
            "vehicle_status": features["vehicle_status"].astype(str).str.strip().str.casefold(),
            "mileage": pd.to_numeric(features["mileage"], errors="raise"),
        },
        columns=list(BROAD_RETAIL_TRACK.input_features),
    )


def _common_yoad_rows(yoad: pd.DataFrame) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    year = pd.to_numeric(yoad["year"], errors="raise")
    mileage = pd.to_numeric(yoad["odometer"], errors="raise")
    price = pd.to_numeric(yoad["price"], errors="raise").to_numpy(dtype=np.float64, copy=True)
    if (
        not np.isfinite(year.to_numpy(dtype=np.float64)).all()
        or not np.isfinite(mileage.to_numpy(dtype=np.float64)).all()
        or not np.isfinite(price).all()
    ):
        raise YoadExperimentError("Yoad common fields contain non-finite values")
    if ((year % 1) != 0).any() or ((mileage % 1) != 0).any():
        raise YoadExperimentError("Yoad year and mileage must be integral")
    if (price <= 0).any() or (mileage < 0).any():
        raise YoadExperimentError("Yoad price and mileage must be positive/nonnegative")
    states = yoad["state"].astype(str).str.strip().str.casefold()
    if states.nunique(dropna=False) != 51:
        raise YoadExperimentError("Yoad rows no longer cover the reviewed 50-state-plus-DC scope")
    frame = pd.DataFrame(
        {
            "year": year.astype(np.int32),
            "make": yoad["manufacturer"].astype(str),
            "vehicle_status": "used",
            "mileage": mileage.astype(np.float64),
        },
        columns=list(BROAD_RETAIL_TRACK.input_features),
    ).reset_index(drop=True)
    return frame, price


def _collision_keys(
    features: pd.DataFrame,
    target: NDArray[np.float64],
) -> set[tuple[int, str, int | None, float]]:
    return set(_collision_key_values(features, target))


def _collision_key_values(
    features: pd.DataFrame,
    target: NDArray[np.float64],
) -> list[tuple[int, str, int | None, float]]:
    values: list[tuple[int, str, int | None, float]] = []
    for year, make, mileage, price in zip(
        features["year"], features["make"], features["mileage"], target, strict=True
    ):
        canonical_mileage = None if pd.isna(mileage) else int(float(mileage))
        values.append((int(year), str(make), canonical_mileage, float(price)))
    return values


def _predictor_groups(features: pd.DataFrame) -> NDArray[np.str_]:
    mileage = features["mileage"].map(
        lambda value: "null" if pd.isna(value) else str(int(float(value)))
    )
    raw = (
        features["year"].astype(str)
        + "\x1f"
        + features["make"].astype(str)
        + "\x1f"
        + mileage
        + "\x1f"
        + features["vehicle_status"].astype(str)
    )
    return cast(NDArray[np.str_], raw.to_numpy(dtype=np.str_, copy=True))


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


def _metrics_by_source(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    sources: NDArray[np.str_],
) -> dict[str, object]:
    result: dict[str, object] = {"overall": regression_metrics(target, predicted).to_dict()}
    for source in ("cars_com_development", "yoad22_craigslist"):
        mask = sources == source
        result[source] = regression_metrics(target[mask], predicted[mask]).to_dict()
    return result


def _fold_report(
    fold_number: int,
    training: NDArray[np.int64],
    validation: NDArray[np.int64],
    data: PreparedExperimentData,
    predictions: Mapping[ArmName, NDArray[np.float64]],
) -> dict[str, object]:
    report: dict[str, object] = {
        "fold": fold_number,
        "training_rows": {
            "cars_only": int((data.sources[training] == "cars_com_development").sum()),
            "cars_plus_yoad": len(training),
        },
        "validation_rows": len(validation),
        "metrics": {},
    }
    fold_metrics = cast(dict[str, object], report["metrics"])
    for arm, values in predictions.items():
        source_metrics: dict[str, object] = {
            "overall": regression_metrics(data.target[validation], values[validation]).to_dict()
        }
        for source in ("cars_com_development", "yoad22_craigslist"):
            selected = validation[data.sources[validation] == source]
            source_metrics[source] = regression_metrics(
                data.target[selected], values[selected]
            ).to_dict()
        fold_metrics[arm] = source_metrics
    return report


def _stability_report(folds: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for arm in ("cars_only", "cars_plus_yoad"):
        arm_result: dict[str, object] = {}
        for source in ("overall", "cars_com_development", "yoad22_craigslist"):
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
        output[arm] = arm_result
    return output


def _slice_report(
    data: PreparedExperimentData,
    predictions: Mapping[ArmName, NDArray[np.float64]],
) -> dict[str, object]:
    age = (BROAD_RETAIL_TRACK.reference_year - data.features["year"].astype(float)).clip(lower=0)
    labels = {
        "price_band": _quantile_labels(pd.Series(data.target), label="price_usd"),
        "manufacturer": data.features["make"].astype(str),
        "vehicle_age_band": _quantile_labels(age, label="age_years"),
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
                for source in ("cars_com_development", "yoad22_craigslist")
            }
        result[dimension] = arm_result
    return result


def _quantile_labels(values: pd.Series, *, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    edges = np.unique(np.quantile(numeric, (0.0, 0.25, 0.5, 0.75, 1.0)))
    if len(edges) < 2:
        return pd.Series(f"{label}:all", index=values.index, dtype=object)
    # Avoid a subnormal ``nextafter(0, -inf)`` boundary: pandas interval label
    # formatting can turn that value into a one-sided missing endpoint.
    edges[0] -= max(1e-9, abs(float(edges[0])) * 1e-12)
    return pd.cut(numeric, bins=edges, include_lowest=True, duplicates="drop").astype(str)


def _mileage_labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    labels = pd.Series("mileage:missing", index=values.index, dtype=object)
    present = numeric.notna()
    labels.loc[present] = _quantile_labels(numeric.loc[present], label="mileage").to_numpy()
    return labels


def _aggregate_slices(
    target: NDArray[np.float64],
    predicted: NDArray[np.float64],
    labels: pd.Series,
    *,
    minimum_count: int,
) -> list[dict[str, object]]:
    label_values = labels.astype(str).to_numpy(dtype=np.str_, copy=True)
    result: list[dict[str, object]] = []
    for label in sorted(set(label_values.tolist())):
        selected = label_values == label
        if int(selected.sum()) < minimum_count:
            continue
        result.append(
            {
                "slice": label,
                "metrics": regression_metrics(target[selected], predicted[selected]).to_dict(),
            }
        )
    return result


def _distribution_report(data: PreparedExperimentData) -> dict[str, object]:
    sources: dict[str, object] = {}
    for source in ("cars_com_development", "yoad22_craigslist"):
        mask = data.sources == source
        frame = data.features.loc[mask]
        target = data.target[mask]
        mileage = pd.to_numeric(frame["mileage"], errors="raise")
        sources[source] = {
            "rows": int(mask.sum()),
            "price_usd": _distribution(target),
            "model_year": _distribution(frame["year"].to_numpy(dtype=np.float64)),
            "mileage_miles_present": _distribution(mileage.dropna().to_numpy(dtype=np.float64)),
            "mileage_present_percentage": float(mileage.notna().mean() * 100.0),
            "distinct_makes": int(frame["make"].nunique()),
            "vehicle_status_counts": {
                str(key): int(value)
                for key, value in frame["vehicle_status"].value_counts().sort_index().items()
            },
        }
    cars = cast(Mapping[str, object], sources["cars_com_development"])
    yoad = cast(Mapping[str, object], sources["yoad22_craigslist"])
    cars_price = cast(Mapping[str, float], cars["price_usd"])
    yoad_price = cast(Mapping[str, float], yoad["price_usd"])
    cars_year = cast(Mapping[str, float], cars["model_year"])
    yoad_year = cast(Mapping[str, float], yoad["model_year"])
    cars_mileage = cast(Mapping[str, float], cars["mileage_miles_present"])
    yoad_mileage = cast(Mapping[str, float], yoad["mileage_miles_present"])
    return {
        "by_source": sources,
        "important_shifts": {
            "median_price_usd_difference_yoad_minus_cars": yoad_price["median"]
            - cars_price["median"],
            "median_model_year_difference_yoad_minus_cars": yoad_year["median"]
            - cars_year["median"],
            "median_mileage_difference_yoad_minus_cars": yoad_mileage["median"]
            - cars_mileage["median"],
            "mileage_coverage_percentage_point_difference_yoad_minus_cars": cast(
                float, yoad["mileage_present_percentage"]
            )
            - cast(float, cars["mileage_present_percentage"]),
            "yoad_has_model_field": False,
            "cars_source_period": "2023 snapshot",
            "yoad_row_timestamps_available": False,
        },
    }


def _distribution(values: NDArray[np.float64]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": float(np.min(values)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _source_composition(sources: NDArray[np.str_]) -> dict[str, object]:
    total = len(sources)
    return {
        source: {
            "rows": int((sources == source).sum()),
            "percentage": float((sources == source).mean() * 100.0),
        }
        for source in ("cars_com_development", "yoad22_craigslist")
    } | {"total_rows": total}


def _recommendation(
    metrics: Mapping[str, object],
    folds: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    cars_only = cast(Mapping[str, object], metrics["cars_only"])
    combined = cast(Mapping[str, object], metrics["cars_plus_yoad"])
    overall_baseline = _mae(cars_only, "overall")
    overall_combined = _mae(combined, "overall")
    cars_baseline = _mae(cars_only, "cars_com_development")
    cars_combined = _mae(combined, "cars_com_development")
    yoad_baseline = _mae(cars_only, "yoad22_craigslist")
    yoad_combined = _mae(combined, "yoad22_craigslist")
    pooled_improvement = (overall_baseline - overall_combined) / overall_baseline
    cars_change = (cars_combined - cars_baseline) / cars_baseline
    yoad_improvement = (yoad_baseline - yoad_combined) / yoad_baseline
    cars_fold_changes: list[float] = []
    for fold in folds:
        fold_metrics = cast(Mapping[str, object], fold["metrics"])
        baseline = _mae(
            cast(Mapping[str, object], fold_metrics["cars_only"]),
            "cars_com_development",
        )
        challenger = _mae(
            cast(Mapping[str, object], fold_metrics["cars_plus_yoad"]),
            "cars_com_development",
        )
        cars_fold_changes.append((challenger - baseline) / baseline)
    gates = {
        "pooled_mae_improvement_at_least_5_percent": pooled_improvement >= 0.05,
        "cars_mae_degradation_no_more_than_5_percent": cars_change <= 0.05,
        "worst_cars_fold_degradation_no_more_than_10_percent": max(cars_fold_changes) <= 0.10,
        "yoad_mae_improves": yoad_improvement > 0.0,
    }
    passes = all(gates.values())
    return {
        "pooled_mae_relative_improvement": pooled_improvement,
        "cars_mae_relative_change": cars_change,
        "yoad_mae_relative_improvement": yoad_improvement,
        "worst_cars_fold_mae_relative_change": max(cars_fold_changes),
        "source_specific_degradation_detected": cars_change > 0.0,
        "guardrails": gates,
        "recommendation": (
            "eligible_for_separate_confirmation_not_promotion"
            if passes
            else "do_not_promote_combined_model"
        ),
        "automatic_promotion": False,
    }


def _mae(metrics: Mapping[str, object], source: str) -> float:
    return cast(float, cast(Mapping[str, object], metrics[source])["mae"])


__all__ = [
    "BROAD_RETAIL_TRACK",
    "EXPERIMENT_ID",
    "PreparedExperimentData",
    "YoadExperimentError",
    "canonical_experiment_json",
    "controlled_group_splits",
    "load_controlled_experiment_data",
    "run_controlled_experiment",
]
