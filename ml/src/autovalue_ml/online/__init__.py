"""Experimental, shadow-only online-learning components."""

from autovalue_ml.online.contracts import (
    ShadowOutcomeSubmission,
    ShadowPredictionRequest,
    ShadowVehicleFeatures,
)
from autovalue_ml.online.permissions import SYNTHETIC_SHADOW_SOURCE_ID
from autovalue_ml.online.service import ShadowLearningService

__all__ = [
    "SYNTHETIC_SHADOW_SOURCE_ID",
    "ShadowLearningService",
    "ShadowOutcomeSubmission",
    "ShadowPredictionRequest",
    "ShadowVehicleFeatures",
]
