from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import yoad_experiment
from autovalue_ml.modeling.yoad_experiment import (
    PreparedExperimentData,
    YoadExperimentError,
    canonical_experiment_json,
    controlled_group_splits,
    run_controlled_experiment,
)
from sklearn.dummy import DummyRegressor


def _prepared_data() -> PreparedExperimentData:
    rows = 50
    features = pd.DataFrame(
        {
            "year": np.concatenate(
                (np.asarray([2024], dtype=np.int32), np.arange(1971, 1971 + rows - 1))
            ),
            "make": [f"make-{index % 10}" for index in range(rows)],
            "vehicle_status": ["used"] * rows,
            "mileage": np.arange(1_000, 1_000 + rows * 2_000, 2_000, dtype=np.float64),
        }
    )
    target = np.linspace(2_000.0, 40_000.0, rows, dtype=np.float64)
    sources = np.asarray(
        [
            "cars_com_development" if index % 2 == 0 else "yoad22_craigslist"
            for index in range(rows)
        ],
        dtype=np.str_,
    )
    return PreparedExperimentData(
        features=features,
        target=target,
        sources=sources,
        row_accounting={"combined_training_rows": rows},
    )


def test_controlled_splits_keep_duplicate_predictors_together() -> None:
    data = _prepared_data()
    duplicate = data.features.iloc[[0]].copy()
    features = pd.concat((data.features, duplicate), ignore_index=True)

    splits = controlled_group_splits(features)

    assert len(splits) == 5
    for training, validation in splits:
        assert set(training).isdisjoint(validation)
        assert (0 in validation) == (50 in validation)


def test_controlled_experiment_reports_paired_metrics_and_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _prepared_data()
    monkeypatch.setattr(
        yoad_experiment,
        "_make_model",
        lambda: DummyRegressor(strategy="mean"),
    )
    progress: list[tuple[int, int]] = []

    report = run_controlled_experiment(
        data,
        on_progress=lambda fold, total: progress.append((fold, total)),
    )

    assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    assert report["experiment_id"] == "autovalue-yoad22-controlled-batch-v1"
    assert report["boundaries"]["phase4_calibration_used"] is False  # type: ignore[index]
    assert report["boundaries"]["legacy_holdout_used"] is False  # type: ignore[index]
    metrics = report["metrics"]
    assert metrics["cars_only"]["overall"]["sample_count"] == 50  # type: ignore[index]
    assert metrics["cars_plus_yoad"]["overall"]["sample_count"] == 50  # type: ignore[index]
    assert len(report["fold_metrics"]) == 5  # type: ignore[arg-type]
    assert report["decision"]["automatic_promotion"] is False  # type: ignore[index]
    assert "price_band" in report["slice_metrics"]  # type: ignore[operator]
    assert "important_shifts" in report["distribution_shifts"]  # type: ignore[operator]
    assert json.loads(canonical_experiment_json(report))["schema_version"] == 1


def test_prepared_data_rejects_misaligned_arrays() -> None:
    with pytest.raises(YoadExperimentError, match="not aligned"):
        PreparedExperimentData(
            features=pd.DataFrame({"year": [2020], "make": ["a"], "vehicle_status": ["used"]}),
            target=np.asarray([], dtype=np.float64),
            sources=np.asarray(["cars_com_development"], dtype=np.str_),
            row_accounting={},
        )


def test_canonical_report_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_experiment_json({"metric": float("nan")})
