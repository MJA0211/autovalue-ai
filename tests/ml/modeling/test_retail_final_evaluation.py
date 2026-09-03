from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import autovalue_ml.modeling.retail_final_evaluation as final_eval
import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import (
    RetailCalibrationArtifact,
    load_calibration_artifact,
)
from autovalue_ml.modeling.final_evaluation_policy import (
    FinalEvaluationPolicy,
    load_final_evaluation_policy_file,
)
from autovalue_ml.modeling.retail_final_evaluation_cli import (
    render_final_report,
    render_model_card,
)
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-final-evaluation-policy-v1.json"
ARTIFACT_PATH = PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-calibration-v1.artifact.json"
PRIOR_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-v1.report.json"
)


@pytest.fixture(scope="module")
def policy() -> FinalEvaluationPolicy:
    return load_final_evaluation_policy_file(POLICY_PATH)


@pytest.fixture(scope="module")
def artifact() -> RetailCalibrationArtifact:
    return load_calibration_artifact(
        ARTIFACT_PATH.read_bytes(),
        active_model_identity_sha256=final_eval.RF05_IDENTITY_SHA256,
    )


@pytest.fixture(scope="module")
def prior_report() -> Mapping[str, object]:
    import json

    value = json.loads(PRIOR_PATH.read_bytes())
    assert isinstance(value, dict)
    return cast(Mapping[str, object], value)


def _features(row_count: int) -> pd.DataFrame:
    statuses = ("certified", "new", "used")
    makes = ("ford", "toyota", "chevrolet", "honda", "bmw")
    years = (2023, 2020, 2015, 2008, 2000)
    mileage = np.arange(row_count, dtype=np.float64) * 17.0
    mileage[::19] = np.nan
    return pd.DataFrame(
        {
            "year": [years[index % len(years)] for index in range(row_count)],
            "make": [makes[index % len(makes)] for index in range(row_count)],
            "model": [f"model-{index % 20}" for index in range(row_count)],
            "vehicle_status": [statuses[index % len(statuses)] for index in range(row_count)],
            "mileage": mileage,
        }
    )


class RecordingEstimator:
    def __init__(self) -> None:
        self.fit_features: pd.DataFrame | None = None
        self.fit_target: NDArray[np.float64] | None = None
        self.predict_features: pd.DataFrame | None = None

    def fit(
        self,
        features: pd.DataFrame,
        target: NDArray[np.float64],
    ) -> RecordingEstimator:
        self.fit_features = features.copy(deep=True)
        self.fit_target = target.copy()
        return self

    def predict(self, features: pd.DataFrame) -> object:
        self.predict_features = features.copy(deep=True)
        return np.full(len(features), 25_000.0, dtype=np.float64)


def test_frozen_fit_has_no_holdout_target_path_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(final_eval, "DEVELOPMENT_SAMPLE_COUNT", 6)
    monkeypatch.setattr(final_eval, "FINAL_HOLDOUT_ROWS", 3)
    estimator = RecordingEstimator()
    development_target = np.arange(1, 7, dtype=np.float64) * 1_000.0

    predictions = final_eval.fit_frozen_rf05_for_final(
        development_features=_features(6),
        development_target=development_target,
        holdout_features=_features(3),
        estimator_factory=lambda: estimator,
    )

    assert estimator.fit_target is not None
    np.testing.assert_array_equal(estimator.fit_target, development_target)
    assert estimator.fit_features is not None and len(estimator.fit_features) == 6
    assert estimator.predict_features is not None and len(estimator.predict_features) == 3
    np.testing.assert_array_equal(predictions, np.full(3, 25_000.0))
    assert predictions.flags.writeable is False


def test_fit_rejects_wrong_boundaries_and_invalid_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(final_eval, "DEVELOPMENT_SAMPLE_COUNT", 4)
    monkeypatch.setattr(final_eval, "FINAL_HOLDOUT_ROWS", 2)

    with pytest.raises(final_eval.FinalEvaluationError, match="development population"):
        final_eval.fit_frozen_rf05_for_final(
            development_features=_features(3),
            development_target=np.ones(3),
            holdout_features=_features(2),
            estimator_factory=RecordingEstimator,
        )

    estimator = RecordingEstimator()
    estimator.predict = lambda features: np.full(len(features), np.nan)  # type: ignore[method-assign]
    with pytest.raises(final_eval.FinalEvaluationError, match="finite and nonnegative"):
        final_eval.fit_frozen_rf05_for_final(
            development_features=_features(4),
            development_target=np.ones(4),
            holdout_features=_features(2),
            estimator_factory=lambda: estimator,
        )


def test_aggregate_evaluation_is_deterministic_and_row_free(
    policy: FinalEvaluationPolicy,
    artifact: RetailCalibrationArtifact,
    prior_report: Mapping[str, object],
) -> None:
    row_count = final_eval.FINAL_HOLDOUT_ROWS
    features = _features(row_count)
    predictions = np.linspace(8_000.0, 120_000.0, row_count, dtype=np.float64)
    residual = 500.0 + (np.arange(row_count, dtype=np.float64) % 41.0) * 100.0
    direction = np.where(np.arange(row_count) % 2 == 0, 1.0, -1.0)
    target = np.maximum(1.0, predictions + direction * residual)

    first = final_eval.evaluate_final_holdout(
        policy=policy,
        holdout_features=features,
        holdout_target=target,
        holdout_predictions=predictions,
        calibration_artifact=artifact,
        prior_sharpness_report=prior_report,
    )
    second = final_eval.evaluate_final_holdout(
        policy=policy,
        holdout_features=features,
        holdout_target=target,
        holdout_predictions=predictions,
        calibration_artifact=artifact,
        prior_sharpness_report=prior_report,
    )

    assert final_eval.canonical_final_report_json(first.report) == (
        final_eval.canonical_final_report_json(second.report)
    )
    assert first.classification in final_eval.CLASSIFICATIONS
    assert first.report["data_boundary"] == second.report["data_boundary"]
    serialized = final_eval.canonical_final_report_json(first.report)
    for forbidden in ("vin", "listing_id", 'predictions":[', 'targets":[', 'residuals":['):
        assert forbidden not in serialized.lower()
    uncertainty = cast(Mapping[str, object], first.report["uncertainty"])
    coverages = cast(Mapping[str, object], uncertainty["coverages"])
    for level in ("0.8", "0.9", "0.95"):
        interval = cast(Mapping[str, object], coverages[level])
        assert interval["invalid_or_nonfinite_interval_count"] == 0
        assert interval["point_exclusion_or_reversed_count"] == 0
    markdown = render_final_report(first, report_sha256="a" * 64)
    model_card = render_model_card(first, report_sha256="a" * 64)
    assert first.classification in markdown
    assert "Point performance" in markdown
    assert "Model card" in model_card
    assert "27,589-row grouped final holdout" in model_card


def test_evaluation_rejects_policy_or_prior_decision_drift(
    policy: FinalEvaluationPolicy,
    artifact: RetailCalibrationArtifact,
    prior_report: Mapping[str, object],
) -> None:
    features = _features(final_eval.FINAL_HOLDOUT_ROWS)
    target = np.full(final_eval.FINAL_HOLDOUT_ROWS, 25_000.0)
    predictions = np.full(final_eval.FINAL_HOLDOUT_ROWS, 25_000.0)
    changed = copy.deepcopy(dict(prior_report))
    cast(dict[str, object], changed["decision"])["selected_method"] = "changed"

    with pytest.raises(final_eval.FinalEvaluationError, match="selection differs"):
        final_eval.evaluate_final_holdout(
            policy=policy,
            holdout_features=features,
            holdout_target=target,
            holdout_predictions=predictions,
            calibration_artifact=artifact,
            prior_sharpness_report=changed,
        )
