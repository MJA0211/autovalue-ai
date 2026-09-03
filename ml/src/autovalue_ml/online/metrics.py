"""Prequential cumulative and rolling regression metrics."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Aggregate-only shadow and static-reference metrics."""

    observations_learned: int
    prediction_count: int
    rolling_window_size: int
    rolling_observations: int
    cumulative_mae_usd: float | None
    cumulative_rmse_usd: float | None
    rolling_mae_usd: float | None
    rolling_rmse_usd: float | None
    static_cumulative_mae_usd: float | None
    static_cumulative_rmse_usd: float | None
    static_rolling_mae_usd: float | None
    static_rolling_rmse_usd: float | None
    mae_delta_vs_static_usd: float | None
    rmse_delta_vs_static_usd: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


class PrequentialMetricTracker:
    """Track paired River and static errors on the same resolved outcomes."""

    def __init__(self, *, rolling_window_size: int = 100) -> None:
        if type(rolling_window_size) is not int or rolling_window_size < 2:
            raise ValueError("rolling_window_size must be an integer of at least 2")
        self.rolling_window_size = rolling_window_size
        self.observations_learned = 0
        self.river_absolute_error_sum = 0.0
        self.river_squared_error_sum = 0.0
        self.static_absolute_error_sum = 0.0
        self.static_squared_error_sum = 0.0
        self._rolling: deque[tuple[float, float, float, float]] = deque(maxlen=rolling_window_size)

    def update(
        self,
        *,
        target: float,
        river_prediction: float,
        static_prediction: float,
    ) -> None:
        """Update metrics from pre-update predictions only."""
        values = (target, river_prediction, static_prediction)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("metrics require finite target and predictions")
        river_error = river_prediction - target
        static_error = static_prediction - target
        row = (
            abs(river_error),
            river_error * river_error,
            abs(static_error),
            static_error * static_error,
        )
        self.observations_learned += 1
        self.river_absolute_error_sum += row[0]
        self.river_squared_error_sum += row[1]
        self.static_absolute_error_sum += row[2]
        self.static_squared_error_sum += row[3]
        self._rolling.append(row)

    def snapshot(self, *, prediction_count: int) -> MetricSnapshot:
        """Return aggregate metrics without row-level errors or outcomes."""
        count = self.observations_learned
        if count == 0:
            return MetricSnapshot(
                observations_learned=0,
                prediction_count=prediction_count,
                rolling_window_size=self.rolling_window_size,
                rolling_observations=0,
                cumulative_mae_usd=None,
                cumulative_rmse_usd=None,
                rolling_mae_usd=None,
                rolling_rmse_usd=None,
                static_cumulative_mae_usd=None,
                static_cumulative_rmse_usd=None,
                static_rolling_mae_usd=None,
                static_rolling_rmse_usd=None,
                mae_delta_vs_static_usd=None,
                rmse_delta_vs_static_usd=None,
            )

        river_mae = self.river_absolute_error_sum / count
        river_rmse = math.sqrt(self.river_squared_error_sum / count)
        static_mae = self.static_absolute_error_sum / count
        static_rmse = math.sqrt(self.static_squared_error_sum / count)
        rolling_count = len(self._rolling)
        rolling_river_mae = sum(row[0] for row in self._rolling) / rolling_count
        rolling_river_rmse = math.sqrt(sum(row[1] for row in self._rolling) / rolling_count)
        rolling_static_mae = sum(row[2] for row in self._rolling) / rolling_count
        rolling_static_rmse = math.sqrt(sum(row[3] for row in self._rolling) / rolling_count)
        return MetricSnapshot(
            observations_learned=count,
            prediction_count=prediction_count,
            rolling_window_size=self.rolling_window_size,
            rolling_observations=rolling_count,
            cumulative_mae_usd=river_mae,
            cumulative_rmse_usd=river_rmse,
            rolling_mae_usd=rolling_river_mae,
            rolling_rmse_usd=rolling_river_rmse,
            static_cumulative_mae_usd=static_mae,
            static_cumulative_rmse_usd=static_rmse,
            static_rolling_mae_usd=rolling_static_mae,
            static_rolling_rmse_usd=rolling_static_rmse,
            mae_delta_vs_static_usd=river_mae - static_mae,
            rmse_delta_vs_static_usd=river_rmse - static_rmse,
        )

    @property
    def rolling_rows(self) -> tuple[tuple[float, float, float, float], ...]:
        """Return state for trusted local checkpoint serialization."""
        return tuple(self._rolling)

    def restore_rolling(self, rows: tuple[tuple[float, float, float, float], ...]) -> None:
        """Restore bounded rolling aggregates from a verified checkpoint."""
        if len(rows) > self.rolling_window_size:
            raise ValueError("checkpoint rolling metric state exceeds its configured window")
        self._rolling.clear()
        self._rolling.extend(rows)
