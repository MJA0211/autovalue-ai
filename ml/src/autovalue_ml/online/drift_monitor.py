"""Conservative error-stream drift telemetry with no model side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from river import drift


@dataclass(frozen=True, slots=True)
class DriftDetection:
    """Aggregate-safe record of one detector signal."""

    event_id: str
    observation_index: int
    normalized_absolute_error: float
    detector_width: float
    detector_estimation: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


class ShadowDriftMonitor:
    """Feed normalized prequential error to ADWIN and expose telemetry only."""

    detector_name = "ADWIN"

    def __init__(self, *, delta: float = 0.002) -> None:
        self.delta = delta
        self.detector = drift.ADWIN(delta=delta)  # type: ignore[no-untyped-call]
        self.detections: list[DriftDetection] = []

    def update(
        self,
        *,
        event_id: str,
        observation_index: int,
        target: float,
        prediction: float,
    ) -> bool:
        """Update telemetry; never mutate or replace a predictive model."""
        signal = abs(prediction - target) / max(target, 1.0)
        self.detector.update(signal)  # type: ignore[no-untyped-call]
        if not self.detector.drift_detected:
            return False
        self.detections.append(
            DriftDetection(
                event_id=event_id,
                observation_index=observation_index,
                normalized_absolute_error=signal,
                detector_width=float(self.detector.width),
                detector_estimation=float(self.detector.estimation),
            )
        )
        return True

    def status(self) -> dict[str, object]:
        """Return telemetry without raw predictor or target values."""
        return {
            "detector": self.detector_name,
            "delta": self.delta,
            "status": "telemetry_only",
            "automatic_action": False,
            "detections": [detection.to_dict() for detection in self.detections],
            "detection_count": len(self.detections),
            "current_width": float(self.detector.width),
            "current_estimation": float(self.detector.estimation),
        }
