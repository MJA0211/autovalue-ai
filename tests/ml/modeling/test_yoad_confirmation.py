from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import yoad_confirmation, yoad_experiment
from autovalue_ml.modeling.yoad_confirmation import (
    YoadConfirmationError,
    canonical_confirmation_json,
    load_controlled_report,
    run_yoad_confirmation,
)
from autovalue_ml.modeling.yoad_experiment import PreparedExperimentData
from sklearn.dummy import DummyRegressor


def _data() -> PreparedExperimentData:
    cars = 20
    yoad = 30
    rows = cars + yoad
    features = pd.DataFrame(
        {
            "year": np.arange(1970, 1970 + rows, dtype=np.int32),
            "make": [f"make-{index % 5}" for index in range(rows)],
            "vehicle_status": ["used"] * rows,
            "mileage": np.arange(1_000, 1_000 + rows * 2_000, 2_000, dtype=np.float64),
        }
    )
    return PreparedExperimentData(
        features=features,
        target=np.linspace(2_000.0, 40_000.0, rows, dtype=np.float64),
        sources=np.asarray(
            ["cars_com_development"] * cars + ["yoad22_craigslist"] * yoad,
            dtype=np.str_,
        ),
        row_accounting={"combined_training_rows": rows},
    )


def test_load_controlled_report_enforces_checksum(tmp_path: Path) -> None:
    path = tmp_path / "controlled.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(YoadConfirmationError, match="checksum"):
        load_controlled_report(path)


def test_quota_allocation_is_exact_and_nested() -> None:
    strata = np.asarray(["a", "b", "c"], dtype=np.str_)
    counts = np.asarray([10, 20, 30], dtype=np.int64)

    first = yoad_confirmation._proportional_quotas(strata, counts, 20)
    second = yoad_confirmation._nested_quotas(strata, counts, 40, minimum=first)

    assert int(first.sum()) == 20
    assert int(second.sum()) == 40
    assert (second >= first).all()
    assert (second <= counts).all()


def test_confirmation_runs_new_arms_without_refitting_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    monkeypatch.setattr(yoad_experiment, "_make_model", lambda: DummyRegressor())
    controlled = yoad_experiment.run_controlled_experiment(data)
    monkeypatch.setattr(yoad_confirmation, "FULL_YOAD_ROWS", 30)
    monkeypatch.setattr(yoad_confirmation, "BALANCED_YOAD_ROWS", 10)
    monkeypatch.setattr(yoad_confirmation, "MODERATE_YOAD_ROWS", 20)
    monkeypatch.setattr(yoad_confirmation, "_make_model", lambda: DummyRegressor())
    monkeypatch.setattr(
        yoad_confirmation,
        "_validate_controlled_evidence",
        lambda _data, _report: None,
    )
    monkeypatch.setattr(
        yoad_confirmation,
        "_critical_segment_comparison",
        lambda _metrics, _folds, _slices: {"checked": True},
    )
    monkeypatch.setattr(
        yoad_confirmation,
        "_confirmation_decision",
        lambda _metrics, _stability, _segments: {
            "recommendation": "retain as separate experimental model",
            "automatic_promotion": False,
        },
    )
    progress: list[tuple[str, int, int]] = []

    report = run_yoad_confirmation(
        data=data,
        controlled_report=controlled,
        on_progress=lambda arm, fold, total: progress.append((arm, fold, total)),
    )

    assert len(progress) == 10
    assert report["controlled_experiment"]["endpoint_results_reused_without_modification"]  # type: ignore[index]
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {
        "cars_only",
        "balanced_augmentation",
        "moderate_augmentation",
        "full_augmentation",
    }
    selection = report["subset_selection"]
    assert selection["balanced_augmentation"]["rows"] == 10  # type: ignore[index]
    assert selection["moderate_augmentation"]["rows"] == 20  # type: ignore[index]
    assert report["governance"]["automatic_promotion"] is False  # type: ignore[index]
    assert json.loads(canonical_confirmation_json(report))["schema_version"] == 1


def test_confirmation_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_confirmation_json({"metric": float("inf")})
