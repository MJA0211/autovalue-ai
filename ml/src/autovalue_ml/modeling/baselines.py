"""Transparent baseline pipeline factories; no fitting occurs in this module."""

from __future__ import annotations

from typing import Literal

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .contracts import RETAIL_TRACK, TrackConfig, get_track_config
from .preprocessing import make_preprocessor

BaselineName = Literal["dummy_median", "linear_regression"]


def make_dummy_pipeline(config: TrackConfig = RETAIL_TRACK) -> Pipeline:
    """Return an unfitted, end-to-end median-price baseline."""

    resolved = get_track_config(config)
    return Pipeline(
        steps=(
            ("preprocessor", make_preprocessor(resolved)),
            ("regressor", DummyRegressor(strategy="median")),
        )
    )


def make_linear_pipeline(config: TrackConfig = RETAIL_TRACK) -> Pipeline:
    """Return an unfitted, sparse-compatible linear-regression baseline."""

    resolved = get_track_config(config)
    return Pipeline(
        steps=(
            ("preprocessor", make_preprocessor(resolved)),
            ("regressor", LinearRegression(copy_X=True, n_jobs=1)),
        )
    )


def make_baseline_pipeline(
    name: BaselineName,
    config: TrackConfig = RETAIL_TRACK,
) -> Pipeline:
    """Dispatch a supported baseline name without accepting arbitrary estimators."""

    if name == "dummy_median":
        return make_dummy_pipeline(config)
    if name == "linear_regression":
        return make_linear_pipeline(config)
    raise ValueError(f"unsupported baseline: {name!r}")
