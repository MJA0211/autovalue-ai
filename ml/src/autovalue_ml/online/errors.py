"""Errors and quarantine reasons for the shadow-learning boundary."""

from enum import StrEnum


class ShadowLearningError(RuntimeError):
    """Base error for the experimental shadow-learning subsystem."""


class SourcePermissionError(ShadowLearningError):
    """Raised when an online source is not explicitly approved."""


class ShadowValidationError(ShadowLearningError):
    """Raised when a prediction event cannot safely enter shadow state."""


class DuplicatePredictionError(ShadowLearningError):
    """Raised when a prediction event ID has already been seen."""


class CheckpointError(ShadowLearningError):
    """Raised when a local checkpoint cannot be trusted or restored."""


class QuarantineReason(StrEnum):
    """Stable outcome-rejection reason codes exposed only as telemetry."""

    DUPLICATE_OUTCOME = "duplicate_outcome"
    EVENT_NOT_FOUND = "event_not_found"
    SOURCE_MISMATCH = "source_mismatch"
    SOURCE_NOT_APPROVED = "source_not_approved"
    INVALID_PREDICTORS = "invalid_predictors"
    INVALID_TARGET = "invalid_target"
    TIMESTAMP_ORDER = "timestamp_order"
