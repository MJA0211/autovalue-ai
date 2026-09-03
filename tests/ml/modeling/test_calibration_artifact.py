from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.calibration_artifact import (
    COVERAGE_LEVELS,
    CalibrationArtifactError,
    ConditionalRadius,
    ConfidenceThresholds,
    CoverageCalibration,
    PredictionDataQuality,
    RetailCalibrationArtifact,
    active_rf05_identity,
    calibrated_valuation,
    canonical_calibration_artifact_json,
    load_calibration_artifact,
)
from autovalue_ml.modeling.candidates import get_candidate_spec
from autovalue_ml.modeling.retail_calibration_experiment import (
    CalibrationExperimentError,
    fit_frozen_rf05_calibration_predictions,
)
from numpy.typing import NDArray


def _coverage(coverage: float, radius: float) -> CoverageCalibration:
    return CoverageCalibration(
        coverage=coverage,
        global_radius_usd=radius + 1_000.0,
        status_radii=(
            ConditionalRadius("certified", 300, None),
            ConditionalRadius("new", 2_000, radius + 500.0),
            ConditionalRadius("used", 3_000, radius),
        ),
        predicted_value_band_radii=(
            ConditionalRadius("band_1", 2_000, radius - 100.0),
            ConditionalRadius("band_2", 2_000, radius - 50.0),
            ConditionalRadius("band_3", 2_000, radius),
            ConditionalRadius("band_4", 2_000, radius + 100.0),
        ),
        status_value_band_radii=(ConditionalRadius("used|band_2", 500, radius - 200.0),),
    )


def _artifact() -> RetailCalibrationArtifact:
    return RetailCalibrationArtifact(
        generated_at="2026-09-02T12:00:00+00:00",
        bound_model=active_rf05_identity(),
        selected_method="vehicle_status_and_predicted_value_band_hierarchy",
        predicted_value_cutpoints_usd=(10_000.0, 20_000.0, 30_000.0),
        coverage_calibrations=tuple(
            _coverage(level, radius)
            for level, radius in zip(COVERAGE_LEVELS, (3_000.0, 5_000.0, 8_000.0), strict=True)
        ),
        confidence_thresholds=ConfidenceThresholds(
            coverage=0.9,
            high_max_relative_width=0.5,
            moderate_max_relative_width=1.0,
        ),
    )


def test_artifact_round_trip_is_canonical_and_model_bound() -> None:
    artifact = _artifact()
    serialized = canonical_calibration_artifact_json(artifact)

    loaded = load_calibration_artifact(
        serialized,
        active_model_identity_sha256=active_rf05_identity().identity_sha256,
    )

    assert loaded == artifact
    assert canonical_calibration_artifact_json(loaded) == serialized
    with pytest.raises(CalibrationArtifactError, match="active RF05"):
        load_calibration_artifact(serialized, active_model_identity_sha256="0" * 64)


def test_artifact_rejects_tampering_duplicate_fields_and_noncanonical_json() -> None:
    serialized = canonical_calibration_artifact_json(_artifact())
    payload = json.loads(serialized)
    payload["bound_model"]["candidate_id"] = "changed"
    with pytest.raises(CalibrationArtifactError, match="frozen evidence"):
        load_calibration_artifact(
            json.dumps(payload),
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
        )

    duplicate = serialized.replace('{"artifact_type"', '{"schema_version":1,"artifact_type"', 1)
    with pytest.raises(CalibrationArtifactError, match="duplicate field"):
        load_calibration_artifact(
            duplicate,
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
        )

    with pytest.raises(CalibrationArtifactError, match="canonical"):
        load_calibration_artifact(
            json.dumps(json.loads(serialized), indent=2),
            active_model_identity_sha256=active_rf05_identity().identity_sha256,
        )


def test_calibrated_valuation_uses_hierarchy_clips_lower_and_separates_warnings() -> None:
    result = calibrated_valuation(
        point_prediction=4_000.0,
        vehicle_status="certified",
        coverage=0.9,
        artifact=_artifact(),
        data_quality=PredictionDataQuality(
            mileage_missing=True,
            rare_or_unseen_category=True,
        ),
    )

    assert result.interval_lower == 0.0
    assert result.interval_upper == 8_900.0
    assert result.interval_width == 8_900.0
    assert result.calibration_method == "predicted_value_band_fallback"
    assert result.confidence_label == "Low confidence"
    assert result.warnings == ("missing_mileage", "rare_or_unseen_category")


def test_calibrated_valuation_uses_exact_supported_bucket() -> None:
    result = calibrated_valuation(
        point_prediction=15_000.0,
        vehicle_status="used",
        coverage=0.9,
        artifact=_artifact(),
    )

    assert result.interval_lower == 10_200.0
    assert result.interval_upper == 19_800.0
    assert result.calibration_support == 500
    assert result.confidence_label == "Moderate confidence"
    assert not result.warnings


def test_minimum_bucket_support_is_enforced_in_artifact() -> None:
    with pytest.raises(CalibrationArtifactError, match="undersupported"):
        ConditionalRadius("too_small", 399, 2_000.0)
    with pytest.raises(CalibrationArtifactError, match="supported"):
        ConditionalRadius("large_enough", 400, None)


def test_frozen_rf05_identity_matches_candidate_definition() -> None:
    identity = active_rf05_identity()
    spec = get_candidate_spec("retail", "random_forest", 5)

    assert identity.candidate_id == spec.candidate_id
    assert identity.parameters == spec.parameters
    assert identity.random_state == spec.random_state


class _TrackingRegressor:
    def __init__(self) -> None:
        self.fit_features: pd.DataFrame | None = None
        self.fit_target: NDArray[np.float64] | None = None
        self.predict_features: pd.DataFrame | None = None

    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> _TrackingRegressor:
        self.fit_features = features.copy(deep=True)
        self.fit_target = target.copy()
        return self

    def predict(self, features: pd.DataFrame) -> object:
        self.predict_features = features.copy(deep=True)
        return np.full(len(features), 25_000.0)


def test_calibration_rows_never_enter_estimator_fit() -> None:
    development = pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "make": ["Ford", "Honda", "Toyota"],
            "model": ["F-150", "Civic", "Camry"],
            "vehicle_status": ["used", "used", "new"],
            "mileage": [30_000.0, 20_000.0, np.nan],
        }
    )
    calibration = pd.DataFrame(
        {
            "year": [2018, 2019],
            "make": ["Subaru", "Mazda"],
            "model": ["Outback", "CX-5"],
            "vehicle_status": ["used", "certified"],
            "mileage": [80_000.0, 60_000.0],
        }
    )
    tracker = _TrackingRegressor()

    def factory() -> _TrackingRegressor:
        return tracker

    predictions = fit_frozen_rf05_calibration_predictions(
        development_features=development,
        development_target=np.asarray([30_000.0, 22_000.0, 28_000.0]),
        calibration_features=calibration,
        estimator_factory=factory,
    )

    assert tracker.fit_features is not None
    assert tracker.predict_features is not None
    assert tracker.fit_features.equals(development)
    assert tracker.predict_features.equals(calibration)
    assert predictions.tolist() == [25_000.0, 25_000.0]


def test_prediction_validation_fails_closed() -> None:
    frame = pd.DataFrame(
        {
            "year": [2020],
            "make": ["Ford"],
            "model": ["F-150"],
            "vehicle_status": ["used"],
        }
    )

    with pytest.raises(CalibrationExperimentError, match="nonnegative finite"):
        fit_frozen_rf05_calibration_predictions(
            development_features=frame,
            development_target=np.asarray([20_000.0]),
            calibration_features=frame,
            estimator_factory=lambda: _BadRegressor(),
        )


class _BadRegressor:
    def fit(self, features: pd.DataFrame, target: NDArray[np.float64]) -> _BadRegressor:
        return self

    def predict(self, features: pd.DataFrame) -> object:
        return np.asarray([-1.0] * len(features))
