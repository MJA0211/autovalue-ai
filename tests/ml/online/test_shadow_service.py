"""Lifecycle, quarantine, ordering, and idempotency tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pytest
from autovalue_ml.online.contracts import (
    ShadowOutcomeSubmission,
    ShadowPredictionRequest,
    ShadowVehicleFeatures,
)
from autovalue_ml.online.errors import DuplicatePredictionError, SourcePermissionError
from autovalue_ml.online.permissions import (
    AUTOTRADER_SOURCE_ID,
    CARSON_SHIVELY_SOURCE_ID,
    SYNTHETIC_SHADOW_SOURCE_ID,
    YOAD22_SOURCE_ID,
)
from autovalue_ml.online.service import ShadowLearningService

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class TrackingRegressor:
    model_version = "tracking-regressor-v1"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.predict_features: dict[str, str | float] | None = None
        self.learn_features: dict[str, str | float] | None = None
        self.targets: list[float] = []

    def predict_one(self, features: Mapping[str, str | float]) -> float:
        self.calls.append("predict_one")
        self.predict_features = dict(features)
        return 20_000.0

    def learn_one(self, features: Mapping[str, str | float], target: float) -> None:
        self.calls.append("learn_one")
        self.learn_features = dict(features)
        self.targets.append(target)


def _request(
    *,
    event_id: str = "event-001",
    source_id: str = SYNTHETIC_SHADOW_SOURCE_ID,
) -> ShadowPredictionRequest:
    return ShadowPredictionRequest(
        event_id=event_id,
        source_id=source_id,
        observed_at=_NOW,
        features=ShadowVehicleFeatures(
            year=2020,
            make="Toyota",
            model="Camry",
            mileage=50_000,
            condition="good",
        ),
        reference_prediction_usd=21_000.0,
    )


def _outcome(
    *,
    event_id: str = "event-001",
    outcome_id: str = "outcome-001",
    source_id: str = SYNTHETIC_SHADOW_SOURCE_ID,
    occurred_at: datetime = _NOW + timedelta(days=2),
    target: int | float | str | None = 22_000.0,
) -> ShadowOutcomeSubmission:
    return ShadowOutcomeSubmission(
        event_id=event_id,
        outcome_id=outcome_id,
        source_id=source_id,
        occurred_at=occurred_at,
        target_price_usd=target,
    )


def test_prediction_is_recorded_before_metrics_and_learning() -> None:
    model = TrackingRegressor()
    service = ShadowLearningService(model=model, rolling_window_size=5)

    prediction = service.create_shadow_prediction(_request())
    assert prediction.river_prediction_usd == 20_000.0
    assert model.calls == ["predict_one"]
    assert service.get_shadow_metrics().observations_learned == 0

    result = service.submit_actual_outcome(_outcome())

    assert result.accepted
    assert model.calls == ["predict_one", "learn_one"]
    assert model.predict_features == model.learn_features
    assert model.predict_features is not None
    assert not any("target" in key or "price" in key for key in model.predict_features)
    assert model.targets == [22_000.0]
    assert service.get_shadow_metrics().observations_learned == 1


def test_duplicate_outcome_cannot_learn_twice() -> None:
    model = TrackingRegressor()
    service = ShadowLearningService(model=model)
    service.create_shadow_prediction(_request())
    outcome = _outcome()

    first = service.submit_actual_outcome(outcome)
    second = service.submit_actual_outcome(outcome)

    assert first.accepted
    assert not second.accepted
    assert second.reason_code == "duplicate_outcome"
    assert model.calls.count("learn_one") == 1
    assert service.get_shadow_metrics().observations_learned == 1


@pytest.mark.parametrize("target", [None, "22000", 0, -1, float("nan"), float("inf")])
def test_invalid_outcomes_are_quarantined_without_learning(
    target: int | float | str | None,
) -> None:
    model = TrackingRegressor()
    service = ShadowLearningService(model=model)
    service.create_shadow_prediction(_request())

    result = service.submit_actual_outcome(_outcome(target=target))

    assert not result.accepted
    assert result.reason_code == "invalid_target"
    assert model.calls == ["predict_one"]
    assert service.get_quarantine_summary()["reason_counts"] == {"invalid_target": 1}


def test_delayed_outcome_is_accepted_but_early_outcome_is_quarantined() -> None:
    service = ShadowLearningService(model=TrackingRegressor())
    service.create_shadow_prediction(_request())

    early = service.submit_actual_outcome(
        _outcome(outcome_id="outcome-early", occurred_at=_NOW - timedelta(seconds=1))
    )
    delayed = service.submit_actual_outcome(_outcome(outcome_id="outcome-delayed"))

    assert not early.accepted
    assert early.reason_code == "timestamp_order"
    assert delayed.accepted


def test_missing_event_and_source_mismatch_are_quarantined() -> None:
    service = ShadowLearningService(model=TrackingRegressor())
    missing = service.submit_actual_outcome(_outcome(event_id="event-missing"))
    service.create_shadow_prediction(_request())
    mismatched = service.submit_actual_outcome(_outcome(source_id="different.source"))

    assert missing.reason_code == "event_not_found"
    assert mismatched.reason_code == "source_mismatch"
    assert service.get_shadow_metrics().observations_learned == 0


@pytest.mark.parametrize(
    "source_id",
    [
        YOAD22_SOURCE_ID,
        AUTOTRADER_SOURCE_ID,
        CARSON_SHIVELY_SOURCE_ID,
    ],
)
def test_blocked_real_sources_cannot_create_shadow_events(source_id: str) -> None:
    service = ShadowLearningService(model=TrackingRegressor())

    with pytest.raises(SourcePermissionError):
        service.create_shadow_prediction(_request(source_id=source_id))


def test_duplicate_prediction_event_is_rejected() -> None:
    service = ShadowLearningService(model=TrackingRegressor())
    service.create_shadow_prediction(_request())

    with pytest.raises(DuplicatePredictionError):
        service.create_shadow_prediction(_request())


def test_river_model_has_explicit_zero_cold_start() -> None:
    service = ShadowLearningService()

    prediction = service.create_shadow_prediction(_request())

    assert prediction.river_prediction_usd == 0.0
    assert prediction.mode == "shadow"
    assert prediction.status == "experimental"
