"""Tests for paired rolling metrics and telemetry-only drift monitoring."""

import pytest
from autovalue_ml.online.drift_monitor import ShadowDriftMonitor
from autovalue_ml.online.metrics import PrequentialMetricTracker


def test_cumulative_and_rolling_metrics_use_paired_predictions() -> None:
    tracker = PrequentialMetricTracker(rolling_window_size=2)
    tracker.update(target=10.0, river_prediction=8.0, static_prediction=7.0)
    tracker.update(target=20.0, river_prediction=24.0, static_prediction=22.0)
    tracker.update(target=30.0, river_prediction=29.0, static_prediction=35.0)

    snapshot = tracker.snapshot(prediction_count=4)

    assert snapshot.observations_learned == 3
    assert snapshot.prediction_count == 4
    assert snapshot.rolling_observations == 2
    assert snapshot.cumulative_mae_usd == pytest.approx(7 / 3)
    assert snapshot.rolling_mae_usd == pytest.approx(2.5)
    assert snapshot.static_cumulative_mae_usd == pytest.approx(10 / 3)
    assert snapshot.mae_delta_vs_static_usd == pytest.approx(-1.0)


def test_empty_metric_snapshot_is_explicit() -> None:
    snapshot = PrequentialMetricTracker(rolling_window_size=5).snapshot(prediction_count=1)

    assert snapshot.observations_learned == 0
    assert snapshot.cumulative_mae_usd is None
    assert snapshot.rolling_mae_usd is None


def test_adwin_emits_telemetry_without_an_automatic_action() -> None:
    monitor = ShadowDriftMonitor(delta=0.01)
    detected = False
    for index in range(400):
        target = 10_000.0
        prediction = target if index < 200 else 1_000.0
        detected = (
            monitor.update(
                event_id=f"event-{index:04d}",
                observation_index=index + 1,
                target=target,
                prediction=prediction,
            )
            or detected
        )

    status = monitor.status()
    assert detected
    assert status["detection_count"]
    assert status["automatic_action"] is False
    assert status["status"] == "telemetry_only"
