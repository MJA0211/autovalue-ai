from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import (
    yoad_confirmation,
    yoad_experiment,
    yoad_weighting,
    yoad_weighting_cli,
)
from autovalue_ml.modeling.yoad_experiment import PreparedExperimentData
from autovalue_ml.modeling.yoad_weighting import (
    YoadWeightingError,
    canonical_weighting_json,
    load_confirmation_report,
    make_training_weights,
    make_weighting_checkpoint,
    parse_weighting_checkpoint_json,
    run_weighting_experiment,
)
from sklearn.dummy import DummyRegressor


def _features(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": np.arange(1970, 1970 + rows, dtype=np.int32),
            "make": [f"make-{index % 5}" for index in range(rows)],
            "vehicle_status": ["used"] * rows,
            "mileage": np.arange(1_000, 1_000 + rows * 2_000, 2_000, dtype=np.float64),
        }
    )


def _data() -> PreparedExperimentData:
    cars = 20
    yoad = 30
    rows = cars + yoad
    return PreparedExperimentData(
        features=_features(rows),
        target=np.linspace(2_000.0, 40_000.0, rows, dtype=np.float64),
        sources=np.asarray(
            ["cars_com_development"] * cars + ["yoad22_craigslist"] * yoad,
            dtype=np.str_,
        ),
        row_accounting={"cars_development_rows": cars, "combined_training_rows": rows},
    )


class _WeightedMeanModel:
    def fit(
        self,
        _features: pd.DataFrame,
        target: np.ndarray[Any, np.dtype[np.float64]],
        **parameters: object,
    ) -> _WeightedMeanModel:
        weights = parameters["regressor__sample_weight"]
        assert isinstance(weights, np.ndarray)
        self.mean_ = float(np.average(target, weights=weights))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.full(len(features), self.mean_, dtype=np.float64)


def test_source_balancing_is_mathematical_and_target_free() -> None:
    features = _features(16)
    sources = np.asarray(
        ["cars_com_development"] * 6 + ["yoad22_craigslist"] * 10,
        dtype=np.str_,
    )

    weights, diagnostics = make_training_weights("source_balanced_weighting", features, sources)

    assert "target" not in inspect.signature(make_training_weights).parameters
    assert weights.mean() == pytest.approx(1.0)
    assert weights[sources == "cars_com_development"].sum() == pytest.approx(8.0)
    assert weights[sources == "yoad22_craigslist"].sum() == pytest.approx(8.0)
    assert np.unique(weights[sources == "cars_com_development"]).size == 1
    assert np.unique(weights[sources == "yoad22_craigslist"]).size == 1
    assert cast(float, diagnostics["effective_sample_fraction"]) <= 1.0


@pytest.mark.parametrize(
    "treatment",
    ["source_mileage_weighting", "source_segment_weighting"],
)
def test_distribution_weights_are_bounded_deterministic_and_source_balanced(
    treatment: yoad_weighting.Treatment,
) -> None:
    features = _features(40)
    features.loc[:4, "mileage"] = np.nan
    sources = np.asarray(
        ["cars_com_development"] * 15 + ["yoad22_craigslist"] * 25,
        dtype=np.str_,
    )

    first, diagnostics = make_training_weights(treatment, features, sources)
    second, _ = make_training_weights(treatment, features.copy(), sources.copy())

    np.testing.assert_array_equal(first, second)
    assert float(first.min()) >= 0.5
    assert float(first.max()) <= 2.0
    assert first[sources == "cars_com_development"].sum() == pytest.approx(20.0)
    assert first[sources == "yoad22_craigslist"].sum() == pytest.approx(20.0)
    assert diagnostics["adjustment_summaries"]


def test_weight_statistics_are_recomputed_from_the_training_fold() -> None:
    features = _features(30)
    sources = np.asarray(
        ["cars_com_development"] * 12 + ["yoad22_craigslist"] * 18,
        dtype=np.str_,
    )
    complete, _ = make_training_weights("source_mileage_weighting", features, sources)
    selected = np.asarray([*range(10), *range(12, 27)], dtype=np.int64)
    subset, _ = make_training_weights(
        "source_mileage_weighting", features.iloc[selected], sources[selected]
    )

    assert not np.allclose(complete[selected], subset)


def test_checkpoint_is_policy_bound_and_ordered() -> None:
    first = {
        "treatment": "source_balanced_weighting",
        "fold": 1,
    }
    serialized = canonical_weighting_json(make_weighting_checkpoint((first,)))

    parsed = parse_weighting_checkpoint_json(serialized)

    assert parsed[0]["fold"] == 1
    altered = json.loads(serialized)
    altered["weighting_policy_sha256"] = "0" * 64
    with pytest.raises(YoadWeightingError, match="policy metadata"):
        parse_weighting_checkpoint_json(json.dumps(altered))
    altered = json.loads(serialized)
    altered["completed_fits"][0]["fold"] = 2
    with pytest.raises(YoadWeightingError, match="stable policy prefix"):
        parse_weighting_checkpoint_json(json.dumps(altered))


def test_load_confirmation_report_rejects_unpinned_bytes(tmp_path: Path) -> None:
    report = tmp_path / "confirmation.json"
    report.write_text("{}", encoding="utf-8")

    with pytest.raises(YoadWeightingError, match="checksum"):
        load_confirmation_report(report)


def test_experiment_reuses_moderate_and_checkpoints_each_new_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data()
    monkeypatch.setattr(yoad_experiment, "_make_model", lambda: DummyRegressor())
    controlled = yoad_experiment.run_controlled_experiment(data)
    monkeypatch.setattr(yoad_confirmation, "FULL_YOAD_ROWS", 30)
    monkeypatch.setattr(yoad_confirmation, "BALANCED_YOAD_ROWS", 10)
    monkeypatch.setattr(yoad_confirmation, "MODERATE_YOAD_ROWS", 20)
    monkeypatch.setattr(yoad_confirmation, "_make_model", lambda: DummyRegressor())
    monkeypatch.setattr(yoad_confirmation, "_validate_controlled_evidence", lambda *_: None)
    monkeypatch.setattr(
        yoad_confirmation,
        "_critical_segment_comparison",
        lambda *_: {"checked": True},
    )
    monkeypatch.setattr(
        yoad_confirmation,
        "_confirmation_decision",
        lambda *_: {"recommendation": "retain as separate experimental model"},
    )
    confirmation = yoad_confirmation.run_yoad_confirmation(
        data=data,
        controlled_report=controlled,
    )
    monkeypatch.setattr(yoad_weighting, "CARS_DEVELOPMENT_ROWS", 20)
    monkeypatch.setattr(yoad_weighting, "CARS_CALIBRATION_ROWS", 0)
    monkeypatch.setattr(yoad_weighting, "MODERATE_YOAD_ROWS", 20)
    monkeypatch.setattr(yoad_weighting, "FULL_VALIDATION_ROWS", 50)
    monkeypatch.setattr(yoad_weighting, "_validate_confirmation_evidence", lambda *_: None)
    monkeypatch.setattr(
        yoad_weighting,
        "deterministic_yoad_subsets",
        lambda _data: {
            "balanced_augmentation": np.arange(20, 30, dtype=np.int64),
            "moderate_augmentation": np.arange(20, 40, dtype=np.int64),
        },
    )
    fits = 0

    def model() -> _WeightedMeanModel:
        nonlocal fits
        fits += 1
        return _WeightedMeanModel()

    monkeypatch.setattr(yoad_weighting, "_make_model", model)
    monkeypatch.setattr(
        yoad_weighting,
        "_comparison_report",
        lambda *_: {"moderate_augmentation": {"checked": True}},
    )
    monkeypatch.setattr(
        yoad_weighting,
        "_weighting_decision",
        lambda *_: {
            "classification": "weighting rejected; retain moderate baseline",
            "preferred_treatment": "moderate_augmentation",
        },
    )
    progress: list[int] = []

    report = run_weighting_experiment(
        data=data,
        confirmation_report=confirmation,
        on_progress=lambda completed: progress.append(len(completed)),
    )

    assert fits == 15
    assert progress == list(range(1, 16))
    assert report["reference_confirmation"][  # type: ignore[index]
        "moderate_result_reused_without_refitting"
    ]
    assert report["checkpoint"]["completed_fit_count"] == 15  # type: ignore[index]
    assert report["governance"]["automatic_promotion"] is False  # type: ignore[index]


def test_decision_rejects_when_any_preregistered_gate_fails() -> None:
    moderate = {
        "worst_focus_cars_regression": 0.05,
        "cars_manufacturer_regression_count": 16,
    }
    treatment = {
        "cars_mae_relative_change_vs_moderate": 0.01,
        "moderate_yoad_gain_retained": 0.95,
        "focus_slices_improved_vs_moderate": 9,
        "worst_focus_cars_regression": 0.03,
        "cars_manufacturer_regression_count": 10,
        "worst_cars_slice_regression": 0.04,
        "cars_fold_std_relative_to_moderate": 1.0,
        "worst_cars_fold_mae_relative_change_vs_cars_only": 0.02,
    }
    comparisons: dict[str, Mapping[str, object]] = {
        "moderate_augmentation": moderate,
        **{name: treatment for name in yoad_weighting._TREATMENTS},
    }

    decision = yoad_weighting._weighting_decision({}, {}, comparisons)

    assert decision["classification"] == "weighting rejected; retain moderate baseline"


def test_weighting_cli_resumes_and_writes_checkpoint_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    checkpoint = tmp_path / "progress.json"
    checkpoint.write_text("{}", encoding="utf-8")
    writes: list[tuple[Path, bool, str]] = []
    resumed: tuple[Mapping[str, object], ...] = ({"resumed": True},)
    progress: tuple[Mapping[str, object], ...] = (
        {"treatment": "source_balanced_weighting", "fold": 1},
    )
    monkeypatch.setattr(yoad_weighting_cli, "_validate_project_root", lambda path: path)
    monkeypatch.setattr(
        yoad_weighting_cli,
        "_validate_output_path",
        lambda path, **_kwargs: path,
    )
    monkeypatch.setattr(
        yoad_weighting_cli,
        "load_confirmation_report",
        lambda _path: {"confirmation": True},
    )
    monkeypatch.setattr(
        yoad_weighting_cli,
        "parse_weighting_checkpoint_json",
        lambda _payload: resumed,
    )
    monkeypatch.setattr(
        yoad_weighting_cli,
        "load_controlled_experiment_data",
        lambda _root: _data(),
    )

    def fake_run(**arguments: object) -> dict[str, object]:
        assert arguments["completed_fits"] == resumed
        callback = arguments["on_progress"]
        assert callable(callback)
        callback(progress)
        return {
            "decision": {
                "classification": "weighting rejected; retain moderate baseline",
                "preferred_treatment": "moderate_augmentation",
            }
        }

    monkeypatch.setattr(yoad_weighting_cli, "run_weighting_experiment", fake_run)
    monkeypatch.setattr(
        yoad_weighting_cli,
        "_write_atomic",
        lambda path, serialized, *, force: writes.append((path, force, serialized)),
    )

    result = yoad_weighting_cli.main(
        [
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
        ]
    )

    assert result == 0
    assert [item[0] for item in writes] == [checkpoint, output]
    assert writes[0][1] is True
    assert writes[1][1] is False
    assert "completed=1/15" in capsys.readouterr().out


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_weighting_json({"metric": float("nan")})
