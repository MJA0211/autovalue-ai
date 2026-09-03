"""Deterministic delayed-outcome scenarios for shadow architecture validation."""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from autovalue_ml.online.contracts import (
    FEATURE_CONTRACT_VERSION,
    ShadowOutcomeSubmission,
    ShadowPredictionRequest,
    ShadowVehicleFeatures,
)
from autovalue_ml.online.model import MODEL_VERSION
from autovalue_ml.online.permissions import (
    SYNTHETIC_SHADOW_SOURCE_ID,
    OnlineSourcePermissionRegistry,
)
from autovalue_ml.online.service import ShadowLearningService

SIMULATION_REPORT_VERSION = "river-shadow-simulation-v1"
_SIMULATION_START = datetime(2026, 1, 1, tzinfo=UTC)


class DriftScenario(StrEnum):
    STABLE = "stable_market"
    GRADUAL = "gradual_price_drift"
    ABRUPT = "abrupt_price_shift"
    MANUFACTURER = "manufacturer_specific_drift"
    MILEAGE = "mileage_related_drift"


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Stable simulator inputs retained in aggregate reports."""

    events_per_scenario: int = 600
    seed: int = 20260902
    rolling_window_size: int = 100
    drift_delta: float = 0.002
    maximum_outcome_delay_steps: int = 5

    def validate(self) -> None:
        if type(self.events_per_scenario) is not int or self.events_per_scenario < 20:
            raise ValueError("events_per_scenario must be an integer of at least 20")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.rolling_window_size) is not int or self.rolling_window_size < 2:
            raise ValueError("rolling_window_size must be at least 2")
        if not 0 < self.drift_delta < 1:
            raise ValueError("drift_delta must be between 0 and 1")
        if self.maximum_outcome_delay_steps != 5:
            raise ValueError("v1 simulator uses a fixed five-step maximum delay")


@dataclass(frozen=True, slots=True)
class _ScheduledOutcome:
    due_at: datetime
    outcome: ShadowOutcomeSubmission


def run_simulation_suite(config: SimulationConfig | None = None) -> dict[str, object]:
    """Run all scenarios and governance checks without loading real data."""
    active_config = config or SimulationConfig()
    active_config.validate()
    scenarios = {
        scenario.value: _run_scenario(scenario, active_config) for scenario in DriftScenario
    }
    restart = _verify_checkpoint_restart(active_config)
    idempotency = _verify_idempotency(active_config)
    permissions = OnlineSourcePermissionRegistry()
    report: dict[str, object] = {
        "report_version": SIMULATION_REPORT_VERSION,
        "classification": "architecture validated for shadow simulation",
        "promotion_decision": "not promoted",
        "production_effect": False,
        "data_scope": "project-owned synthetic events only",
        "simulator_config": asdict(active_config),
        "model": {
            "model_version": MODEL_VERSION,
            "estimator": "River TargetStandardScaler + LinearRegression (SGD)",
            "incremental_preprocessing": ["StandardScaler", "OneHotEncoder"],
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "prediction_manipulation": False,
        },
        "permission_policy": {
            "version": permissions.version,
            "default": "deny",
            "decisions": permissions.public_summary(),
        },
        "lifecycle": [
            "create_prediction_event",
            "predict_one",
            "record_pre_update_prediction",
            "receive_delayed_outcome",
            "validate_eligibility",
            "update_prequential_metrics",
            "update_drift_telemetry",
            "learn_one",
            "checkpoint_state",
            "reject_duplicate_learning",
        ],
        "scenarios": scenarios,
        "checkpoint_restart_verification": restart,
        "idempotency_verification": idempotency,
        "interpretation": (
            "Synthetic results validate lifecycle plumbing only and are not evidence that the "
            "River model is superior on real vehicles."
        ),
    }
    report["report_sha256_without_self"] = _canonical_hash(report)
    return report


def _run_scenario(
    scenario: DriftScenario,
    config: SimulationConfig,
    *,
    restart_at: int | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    service = ShadowLearningService(
        rolling_window_size=config.rolling_window_size,
        drift_delta=config.drift_delta,
    )
    randomizer = random.Random(config.seed)
    pending: list[_ScheduledOutcome] = []
    maximum_pending = 0
    cold_start_prediction: float | None = None
    for index in range(config.events_per_scenario):
        observed_at = _SIMULATION_START + timedelta(minutes=index)
        features = _features_for(index, randomizer)
        baseline = _static_reference(features, observed_at=observed_at)
        target = _target_for(
            scenario,
            index=index,
            count=config.events_per_scenario,
            baseline=baseline,
            features=features,
            noise=randomizer.gauss(0.0, 1_700.0),
        )
        event_id = f"sim-{scenario.value}-{index:06d}"
        prediction = service.create_shadow_prediction(
            ShadowPredictionRequest(
                event_id=event_id,
                source_id=SYNTHETIC_SHADOW_SOURCE_ID,
                observed_at=observed_at,
                features=features,
                reference_prediction_usd=baseline,
            )
        )
        if cold_start_prediction is None:
            cold_start_prediction = prediction.river_prediction_usd
        delay = _delay_for(index)
        due_at = observed_at + timedelta(minutes=delay + 1)
        pending.append(
            _ScheduledOutcome(
                due_at=due_at,
                outcome=ShadowOutcomeSubmission(
                    outcome_id=f"out-{scenario.value}-{index:06d}",
                    event_id=event_id,
                    source_id=SYNTHETIC_SHADOW_SOURCE_ID,
                    occurred_at=due_at,
                    target_price_usd=target,
                ),
            )
        )
        maximum_pending = max(maximum_pending, len(pending))
        pending = _resolve_due(service, pending, now=observed_at)
        if restart_at == index + 1:
            if checkpoint_path is None:
                raise ValueError("restart_at requires a checkpoint_path")
            service.save_checkpoint(checkpoint_path)
            service = ShadowLearningService.load_checkpoint(checkpoint_path)

    for scheduled in sorted(pending, key=lambda item: (item.due_at, item.outcome.event_id)):
        result = service.submit_actual_outcome(scheduled.outcome)
        if not result.accepted:
            raise RuntimeError(f"simulator outcome unexpectedly quarantined: {result.reason_code}")

    metrics = service.get_shadow_metrics().to_dict()
    state = service.get_model_state()
    return {
        "scenario": scenario.value,
        "events_created": state["prediction_count"],
        "outcomes_accepted": state["observations_learned"],
        "outcomes_quarantined": state["quarantined_outcomes"],
        "maximum_pending_outcomes": maximum_pending,
        "cold_start_prediction_usd": cold_start_prediction,
        "metrics": metrics,
        "drift": service.get_drift_status(),
        "state_fingerprint": _canonical_hash(
            {
                "metrics": metrics,
                "drift": service.get_drift_status(),
                "state": state,
            }
        ),
    }


def _resolve_due(
    service: ShadowLearningService,
    pending: list[_ScheduledOutcome],
    *,
    now: datetime,
) -> list[_ScheduledOutcome]:
    remaining: list[_ScheduledOutcome] = []
    for scheduled in sorted(pending, key=lambda item: (item.due_at, item.outcome.event_id)):
        if scheduled.due_at > now:
            remaining.append(scheduled)
            continue
        result = service.submit_actual_outcome(scheduled.outcome)
        if not result.accepted:
            raise RuntimeError(f"simulator outcome unexpectedly quarantined: {result.reason_code}")
    return remaining


def _features_for(index: int, randomizer: random.Random) -> ShadowVehicleFeatures:
    makes = ("Toyota", "Ford", "Honda", "Chevrolet", "Hyundai")
    models = {
        "Toyota": ("Camry", "RAV4"),
        "Ford": ("F-150", "Escape"),
        "Honda": ("Civic", "CR-V"),
        "Chevrolet": ("Silverado", "Equinox"),
        "Hyundai": ("Elantra", "Tucson"),
    }
    make = makes[index % len(makes)]
    model = models[make][(index // len(makes)) % 2]
    year = randomizer.randint(2005, 2025)
    age = 2026 - year
    mileage = max(0, int(age * randomizer.uniform(8_000, 15_000) + randomizer.gauss(0, 9_000)))
    return ShadowVehicleFeatures(
        year=year,
        make=make,
        model=model,
        mileage=mileage,
        condition=("excellent", "good", "fair")[index % 3],
        engine=("2.0L I4", "2.5L I4", "3.5L V6")[index % 3],
        transmission=("automatic", "manual")[index % 2],
        drivetrain=("FWD", "AWD", "RWD")[index % 3],
        accident_count=0 if index % 7 else 1,
        owner_count=1 + (index % 3),
        vehicle_type=("sedan", "suv", "truck")[index % 3],
    )


def _static_reference(features: ShadowVehicleFeatures, *, observed_at: datetime) -> float:
    age = max(0, observed_at.year - features.year)
    make_adjustment = {
        "Toyota": 2_000.0,
        "Ford": 1_000.0,
        "Honda": 1_600.0,
        "Chevrolet": 800.0,
        "Hyundai": -500.0,
    }[features.make]
    type_adjustment = {"sedan": -1_000.0, "suv": 2_000.0, "truck": 5_000.0}[
        features.vehicle_type or "sedan"
    ]
    condition_adjustment = {"excellent": 2_000.0, "good": 0.0, "fair": -3_000.0}[
        features.condition or "good"
    ]
    value = (
        38_000.0
        - age * 1_150.0
        - features.mileage * 0.055
        + make_adjustment
        + type_adjustment
        + condition_adjustment
        - (features.accident_count or 0) * 2_500.0
        - max(0, (features.owner_count or 1) - 1) * 700.0
    )
    return max(1_000.0, value)


def _target_for(
    scenario: DriftScenario,
    *,
    index: int,
    count: int,
    baseline: float,
    features: ShadowVehicleFeatures,
    noise: float,
) -> float:
    halfway = count // 2
    drift_adjustment = 0.0
    if scenario is DriftScenario.GRADUAL:
        drift_adjustment = 7_000.0 * index / max(1, count - 1)
    elif scenario is DriftScenario.ABRUPT and index >= halfway:
        drift_adjustment = 8_000.0
    elif scenario is DriftScenario.MANUFACTURER and index >= halfway and features.make == "Ford":
        drift_adjustment = 10_000.0
    elif scenario is DriftScenario.MILEAGE and index >= halfway and features.mileage >= 120_000:
        drift_adjustment = -8_000.0
    return max(500.0, baseline + drift_adjustment + noise)


def _delay_for(index: int) -> int:
    return (0, 3, 1, 5, 2)[index % 5]


def _verify_checkpoint_restart(config: SimulationConfig) -> dict[str, object]:
    verification_config = SimulationConfig(
        events_per_scenario=max(40, min(config.events_per_scenario, 120)),
        seed=config.seed,
        rolling_window_size=config.rolling_window_size,
        drift_delta=config.drift_delta,
        maximum_outcome_delay_steps=config.maximum_outcome_delay_steps,
    )
    uninterrupted = _run_scenario(DriftScenario.ABRUPT, verification_config)
    with tempfile.TemporaryDirectory(prefix="autovalue-shadow-") as temporary_directory:
        checkpoint = Path(temporary_directory) / "shadow-state.json"
        restarted = _run_scenario(
            DriftScenario.ABRUPT,
            verification_config,
            restart_at=verification_config.events_per_scenario // 2,
            checkpoint_path=checkpoint,
        )
    matched = uninterrupted["state_fingerprint"] == restarted["state_fingerprint"]
    return {
        "passed": matched,
        "events": verification_config.events_per_scenario,
        "restart_after_event": verification_config.events_per_scenario // 2,
        "aggregate_state_matched": matched,
        "raw_rows_persisted_after_resolution": False,
    }


def _verify_idempotency(config: SimulationConfig) -> dict[str, object]:
    service = ShadowLearningService(
        rolling_window_size=config.rolling_window_size,
        drift_delta=config.drift_delta,
    )
    observed_at = _SIMULATION_START
    features = ShadowVehicleFeatures(year=2020, make="Toyota", model="Camry", mileage=50_000)
    reference = _static_reference(features, observed_at=observed_at)
    service.create_shadow_prediction(
        ShadowPredictionRequest(
            event_id="sim-idempotency-event",
            source_id=SYNTHETIC_SHADOW_SOURCE_ID,
            observed_at=observed_at,
            features=features,
            reference_prediction_usd=reference,
        )
    )
    outcome = ShadowOutcomeSubmission(
        outcome_id="sim-idempotency-outcome",
        event_id="sim-idempotency-event",
        source_id=SYNTHETIC_SHADOW_SOURCE_ID,
        occurred_at=observed_at + timedelta(days=1),
        target_price_usd=reference + 500.0,
    )
    first = service.submit_actual_outcome(outcome)
    learned_after_first = service.get_shadow_metrics().observations_learned
    second = service.submit_actual_outcome(outcome)
    learned_after_second = service.get_shadow_metrics().observations_learned
    passed = (
        first.accepted
        and not second.accepted
        and second.reason_code == "duplicate_outcome"
        and learned_after_first == learned_after_second == 1
    )
    return {
        "passed": passed,
        "first_delivery_accepted": first.accepted,
        "duplicate_delivery_accepted": second.accepted,
        "duplicate_reason_code": second.reason_code,
        "learn_calls": learned_after_second,
    }


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
