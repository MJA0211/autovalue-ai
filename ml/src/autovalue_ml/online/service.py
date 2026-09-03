"""Governed prediction-then-outcome-then-learning lifecycle."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from autovalue_ml.online.contracts import (
    FEATURE_CONTRACT_VERSION,
    ShadowOutcomeSubmission,
    ShadowPredictionRequest,
    validate_identifier,
    validate_utc,
)
from autovalue_ml.online.drift_monitor import ShadowDriftMonitor
from autovalue_ml.online.errors import (
    CheckpointError,
    DuplicatePredictionError,
    QuarantineReason,
    ShadowValidationError,
    SourcePermissionError,
)
from autovalue_ml.online.metrics import MetricSnapshot, PrequentialMetricTracker
from autovalue_ml.online.model import MODEL_VERSION, OnlineRegressor, RiverVehicleRegressor
from autovalue_ml.online.permissions import OnlineSourcePermissionRegistry

CHECKPOINT_FORMAT_VERSION = "autovalue-river-shadow-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class ShadowPrediction:
    """Recorded pre-update prediction, never a user-facing valuation."""

    event_id: str
    river_prediction_usd: float
    reference_prediction_usd: float
    model_version: str
    feature_contract_version: str
    mode: str = "shadow"
    status: str = "experimental"

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """Outcome rejection metadata without raw features or price."""

    outcome_id: str
    event_id: str
    reason_code: QuarantineReason
    message: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["reason_code"] = self.reason_code.value
        return payload


@dataclass(frozen=True, slots=True)
class OutcomeResult:
    """Accepted or quarantined outcome response."""

    accepted: bool
    event_id: str
    outcome_id: str
    reason_code: str | None
    observations_learned: int
    drift_detected: bool

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PendingPrediction:
    request: ShadowPredictionRequest
    prediction: ShadowPrediction
    model_features: dict[str, str | float]


class ShadowLearningService:
    """Isolated River state machine with fail-closed outcomes and persistence."""

    mode = "shadow"
    lifecycle_status = "experimental"

    def __init__(
        self,
        *,
        model: OnlineRegressor | None = None,
        permissions: OnlineSourcePermissionRegistry | None = None,
        rolling_window_size: int = 100,
        drift_delta: float = 0.002,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.model = model or RiverVehicleRegressor()
        self.permissions = permissions or OnlineSourcePermissionRegistry()
        self.metrics = PrequentialMetricTracker(rolling_window_size=rolling_window_size)
        self.drift_monitor = ShadowDriftMonitor(delta=drift_delta)
        self.checkpoint_path = checkpoint_path
        self.prediction_count = 0
        self._pending: dict[str, _PendingPrediction] = {}
        self._seen_event_ids: set[str] = set()
        self._processed_event_ids: set[str] = set()
        self._processed_outcome_ids: set[str] = set()
        self._quarantine: list[QuarantineRecord] = []

    def create_shadow_prediction(self, request: ShadowPredictionRequest) -> ShadowPrediction:
        """Validate, predict once, and retain the minimum state needed for a later outcome."""
        request.validate()
        self.permissions.require_learning_approval(request.source_id)
        if request.event_id in self._seen_event_ids:
            raise DuplicatePredictionError(f"prediction event already exists: {request.event_id}")
        features = request.features.model_features(observed_at=request.observed_at)
        raw_prediction = self.model.predict_one(features)
        if not math.isfinite(raw_prediction):
            raise ShadowValidationError("shadow model prediction must be finite")
        prediction = ShadowPrediction(
            event_id=request.event_id,
            river_prediction_usd=raw_prediction,
            reference_prediction_usd=float(request.reference_prediction_usd),
            model_version=self.model.model_version,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
        )
        self._pending[request.event_id] = _PendingPrediction(request, prediction, features)
        self._seen_event_ids.add(request.event_id)
        self.prediction_count += 1
        return prediction

    def submit_actual_outcome(self, outcome: ShadowOutcomeSubmission) -> OutcomeResult:
        """Validate, score the recorded prediction, learn once, and checkpoint."""
        duplicate = (
            outcome.outcome_id in self._processed_outcome_ids
            or outcome.event_id in self._processed_event_ids
        )
        if duplicate:
            return self._reject(outcome, QuarantineReason.DUPLICATE_OUTCOME, "outcome already used")

        pending = self._pending.get(outcome.event_id)
        if pending is None:
            return self._reject(outcome, QuarantineReason.EVENT_NOT_FOUND, "prediction not found")
        if outcome.source_id != pending.request.source_id:
            return self._reject(outcome, QuarantineReason.SOURCE_MISMATCH, "source does not match")
        try:
            validate_identifier(outcome.outcome_id, field="outcome_id")
            validate_identifier(outcome.event_id, field="event_id")
            self.permissions.require_learning_approval(outcome.source_id)
        except SourcePermissionError as error:
            return self._reject(outcome, QuarantineReason.SOURCE_NOT_APPROVED, str(error))
        except ShadowValidationError as error:
            return self._reject(outcome, QuarantineReason.INVALID_TARGET, str(error))
        try:
            validate_utc(outcome.occurred_at, field="occurred_at")
        except ShadowValidationError as error:
            return self._reject(outcome, QuarantineReason.TIMESTAMP_ORDER, str(error))
        if outcome.occurred_at < pending.request.observed_at:
            return self._reject(
                outcome,
                QuarantineReason.TIMESTAMP_ORDER,
                "outcome cannot precede its prediction event",
            )
        try:
            pending.request.features.validate(observed_at=pending.request.observed_at)
        except ShadowValidationError as error:
            return self._reject(outcome, QuarantineReason.INVALID_PREDICTORS, str(error))

        target = _positive_target(outcome.target_price_usd)
        if target is None:
            return self._reject(
                outcome,
                QuarantineReason.INVALID_TARGET,
                "target must be a finite positive numeric USD value",
            )

        prediction = pending.prediction
        self.metrics.update(
            target=target,
            river_prediction=prediction.river_prediction_usd,
            static_prediction=prediction.reference_prediction_usd,
        )
        drift_detected = self.drift_monitor.update(
            event_id=outcome.event_id,
            observation_index=self.metrics.observations_learned,
            target=target,
            prediction=prediction.river_prediction_usd,
        )
        self.model.learn_one(pending.model_features, target)
        self._processed_event_ids.add(outcome.event_id)
        self._processed_outcome_ids.add(outcome.outcome_id)
        del self._pending[outcome.event_id]
        if self.checkpoint_path is not None:
            self.save_checkpoint(self.checkpoint_path)
        return OutcomeResult(
            accepted=True,
            event_id=outcome.event_id,
            outcome_id=outcome.outcome_id,
            reason_code=None,
            observations_learned=self.metrics.observations_learned,
            drift_detected=drift_detected,
        )

    def get_shadow_metrics(self) -> MetricSnapshot:
        return self.metrics.snapshot(prediction_count=self.prediction_count)

    def get_drift_status(self) -> dict[str, object]:
        return self.drift_monitor.status()

    def get_model_state(self) -> dict[str, str | int | bool]:
        return {
            "status": self.lifecycle_status,
            "mode": self.mode,
            "user_facing": False,
            "model_version": self.model.model_version,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "permission_policy_version": self.permissions.version,
            "prediction_count": self.prediction_count,
            "observations_learned": self.metrics.observations_learned,
            "pending_outcomes": len(self._pending),
            "processed_outcomes": len(self._processed_outcome_ids),
            "quarantined_outcomes": len(self._quarantine),
        }

    def get_quarantine_summary(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        for record in self._quarantine:
            counts[record.reason_code.value] = counts.get(record.reason_code.value, 0) + 1
        return {
            "total": len(self._quarantine),
            "reason_counts": dict(sorted(counts.items())),
        }

    def save_checkpoint(self, path: Path) -> str:
        """Atomically write checksummed local state; processed rows are not retained."""
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "feature_contract_version": FEATURE_CONTRACT_VERSION,
            "model_version": self.model.model_version,
            "permission_policy_version": self.permissions.version,
            "model": self.model,
            "metrics": self.metrics,
            "drift_monitor": self.drift_monitor,
            "prediction_count": self.prediction_count,
            "pending": self._pending,
            "seen_event_ids": self._seen_event_ids,
            "processed_event_ids": self._processed_event_ids,
            "processed_outcome_ids": self._processed_outcome_ids,
            "quarantine": self._quarantine,
        }
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        envelope = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "payload_encoding": "base64-pickle-local-trusted-only",
            "payload_sha256": payload_sha256,
            "payload": base64.b64encode(payload).decode("ascii"),
        }
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        return payload_sha256

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        permissions: OnlineSourcePermissionRegistry | None = None,
        checkpoint_path: Path | None = None,
    ) -> ShadowLearningService:
        """Verify the local envelope before deserializing trusted local state."""
        active_permissions = permissions or OnlineSourcePermissionRegistry()
        try:
            raw = path.resolve().read_bytes()
            envelope = json.loads(raw)
            if not isinstance(envelope, dict):
                raise ValueError("checkpoint envelope must be an object")
            if envelope.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("checkpoint format version is not supported")
            if envelope.get("payload_encoding") != "base64-pickle-local-trusted-only":
                raise ValueError("checkpoint encoding is not supported")
            expected_hash = envelope.get("payload_sha256")
            encoded_payload = envelope.get("payload")
            if not isinstance(expected_hash, str) or not isinstance(encoded_payload, str):
                raise ValueError("checkpoint envelope is incomplete")
            payload = base64.b64decode(encoded_payload, validate=True)
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ValueError("checkpoint payload checksum does not match")
            state = pickle.loads(payload)  # noqa: S301 - verified, local-only state; never an upload
            if not isinstance(state, dict):
                raise ValueError("checkpoint state must be an object")
            _validate_checkpoint_versions(state, active_permissions)
            model = state["model"]
            metrics = state["metrics"]
            drift_monitor = state["drift_monitor"]
            if not isinstance(model, RiverVehicleRegressor):
                raise ValueError("checkpoint model type is not allowed")
            if not isinstance(metrics, PrequentialMetricTracker):
                raise ValueError("checkpoint metric type is not allowed")
            if not isinstance(drift_monitor, ShadowDriftMonitor):
                raise ValueError("checkpoint drift type is not allowed")
            service = cls(
                model=model,
                permissions=active_permissions,
                rolling_window_size=metrics.rolling_window_size,
                drift_delta=drift_monitor.delta,
                checkpoint_path=checkpoint_path,
            )
            service.metrics = metrics
            service.drift_monitor = drift_monitor
            service.prediction_count = cast(int, state["prediction_count"])
            service._pending = cast(dict[str, _PendingPrediction], state["pending"])
            service._seen_event_ids = cast(set[str], state["seen_event_ids"])
            service._processed_event_ids = cast(set[str], state["processed_event_ids"])
            service._processed_outcome_ids = cast(set[str], state["processed_outcome_ids"])
            service._quarantine = cast(list[QuarantineRecord], state["quarantine"])
            service._validate_restored_state()
            return service
        except CheckpointError:
            raise
        except Exception as error:
            raise CheckpointError(f"checkpoint rejected: {error}") from error

    def _reject(
        self,
        outcome: ShadowOutcomeSubmission,
        reason: QuarantineReason,
        message: str,
    ) -> OutcomeResult:
        record = QuarantineRecord(
            outcome_id=str(outcome.outcome_id),
            event_id=str(outcome.event_id),
            reason_code=reason,
            message=message,
        )
        self._quarantine.append(record)
        return OutcomeResult(
            accepted=False,
            event_id=str(outcome.event_id),
            outcome_id=str(outcome.outcome_id),
            reason_code=reason.value,
            observations_learned=self.metrics.observations_learned,
            drift_detected=False,
        )

    def _validate_restored_state(self) -> None:
        integer_values = (
            self.prediction_count,
            self.metrics.observations_learned,
            len(self._pending),
            len(self._processed_event_ids),
            len(self._processed_outcome_ids),
        )
        if any(type(value) is not int or value < 0 for value in integer_values):
            raise ValueError("checkpoint counters are invalid")
        if self.prediction_count != len(self._seen_event_ids):
            raise ValueError("checkpoint prediction count is inconsistent")
        if len(self._processed_event_ids) != self.metrics.observations_learned:
            raise ValueError("checkpoint learned count is inconsistent")
        if self._processed_event_ids & self._pending.keys():
            raise ValueError("checkpoint contains processed events in pending state")


def _positive_target(value: int | float | str | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    target = float(value)
    if not math.isfinite(target) or target <= 0:
        return None
    return target


def _validate_checkpoint_versions(
    state: dict[object, object],
    permissions: OnlineSourcePermissionRegistry,
) -> None:
    expected = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "model_version": MODEL_VERSION,
        "permission_policy_version": permissions.version,
    }
    for field, value in expected.items():
        if state.get(field) != value:
            raise CheckpointError(f"checkpoint {field} is not approved")
