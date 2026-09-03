"""Checkpoint integrity and deterministic simulator tests."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from autovalue_ml.online.contracts import (
    ShadowOutcomeSubmission,
    ShadowPredictionRequest,
    ShadowVehicleFeatures,
)
from autovalue_ml.online.errors import CheckpointError
from autovalue_ml.online.permissions import SYNTHETIC_SHADOW_SOURCE_ID
from autovalue_ml.online.service import ShadowLearningService
from autovalue_ml.online.simulator import SimulationConfig, run_simulation_suite

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _request(event_id: str) -> ShadowPredictionRequest:
    return ShadowPredictionRequest(
        event_id=event_id,
        source_id=SYNTHETIC_SHADOW_SOURCE_ID,
        observed_at=_NOW,
        features=ShadowVehicleFeatures(
            year=2019,
            make="Honda",
            model="Civic",
            mileage=70_000,
        ),
        reference_prediction_usd=18_000.0,
    )


def _outcome(event_id: str, outcome_id: str) -> ShadowOutcomeSubmission:
    return ShadowOutcomeSubmission(
        event_id=event_id,
        outcome_id=outcome_id,
        source_id=SYNTHETIC_SHADOW_SOURCE_ID,
        occurred_at=_NOW + timedelta(days=1),
        target_price_usd=18_500.0,
    )


def test_checkpoint_restores_model_metrics_pending_and_idempotency(tmp_path: Path) -> None:
    checkpoint = tmp_path / "shadow-state.json"
    service = ShadowLearningService(rolling_window_size=5)
    service.create_shadow_prediction(_request("event-001"))
    service.submit_actual_outcome(_outcome("event-001", "outcome-001"))
    service.create_shadow_prediction(_request("event-002"))

    payload_hash = service.save_checkpoint(checkpoint)
    restored = ShadowLearningService.load_checkpoint(checkpoint)

    assert len(payload_hash) == 64
    assert restored.get_model_state()["pending_outcomes"] == 1
    assert restored.get_shadow_metrics().observations_learned == 1
    duplicate = restored.submit_actual_outcome(_outcome("event-001", "outcome-001"))
    accepted = restored.submit_actual_outcome(_outcome("event-002", "outcome-002"))
    assert duplicate.reason_code == "duplicate_outcome"
    assert accepted.accepted
    assert restored.get_shadow_metrics().observations_learned == 2


def test_corrupt_checkpoint_fails_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "shadow-state.json"
    checkpoint.write_text('{"format_version":"wrong"}', encoding="utf-8")

    with pytest.raises(CheckpointError, match="checkpoint rejected"):
        ShadowLearningService.load_checkpoint(checkpoint)


def test_simulation_is_deterministic_and_supports_delayed_outcomes() -> None:
    config = SimulationConfig(events_per_scenario=40, rolling_window_size=10)

    first = run_simulation_suite(config)
    second = run_simulation_suite(config)

    assert first == second
    assert first["classification"] == "architecture validated for shadow simulation"
    scenarios = cast(dict[str, Mapping[str, object]], first["scenarios"])
    assert set(scenarios) == {
        "stable_market",
        "gradual_price_drift",
        "abrupt_price_shift",
        "manufacturer_specific_drift",
        "mileage_related_drift",
    }
    assert all(result["outcomes_accepted"] == 40 for result in scenarios.values())
    assert all(cast(int, result["maximum_pending_outcomes"]) > 1 for result in scenarios.values())
    checkpoint = cast(Mapping[str, object], first["checkpoint_restart_verification"])
    idempotency = cast(Mapping[str, object], first["idempotency_verification"])
    assert checkpoint["passed"] is True
    assert idempotency["passed"] is True
