"""Future FastAPI integration boundary for experimental shadow learning.

This module deliberately defines no ``APIRouter``. Importing it cannot make a
River prediction user-facing or register an outcome endpoint.
"""

from __future__ import annotations

from autovalue_ml.online.contracts import ShadowOutcomeSubmission, ShadowPredictionRequest
from autovalue_ml.online.service import OutcomeResult, ShadowLearningService, ShadowPrediction


class ExperimentalShadowLearningInterface:
    """Typed application facade kept separate from the public inference path."""

    mode = "shadow"
    status = "experimental"
    user_facing = False

    def __init__(self, service: ShadowLearningService) -> None:
        self._service = service

    def create_shadow_prediction(self, request: ShadowPredictionRequest) -> ShadowPrediction:
        return self._service.create_shadow_prediction(request)

    def submit_actual_outcome(self, outcome: ShadowOutcomeSubmission) -> OutcomeResult:
        return self._service.submit_actual_outcome(outcome)

    def get_shadow_metrics(self) -> dict[str, int | float | None]:
        return self._service.get_shadow_metrics().to_dict()

    def get_drift_status(self) -> dict[str, object]:
        return self._service.get_drift_status()

    def get_model_state(self) -> dict[str, str | int | bool]:
        return self._service.get_model_state()
