"""Leakage-safe, stateless feature engineering for vehicle price models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .contracts import (
    RETAIL_TRACK,
    FeatureContractError,
    TrackConfig,
    get_track_config,
    validate_feature_frame,
)


class VehicleFeatureEngineer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Create deterministic model-year and mileage features.

    The transformer learns no values from data. Its fitted attributes only record
    the already-frozen input/output schema required by scikit-learn's estimator
    interface; medians and category vocabularies are learned later in the pipeline.

    ``model_year`` preserves the raw integral year instead of exposing clipped
    vehicle age. This keeps adjacent and future years distinguishable. Clipped age
    remains an internal denominator for the derived ``mileage_per_year`` feature.
    """

    def __init__(self, config: TrackConfig = RETAIL_TRACK) -> None:
        self.config = config

    def fit(self, X: object, y: object = None) -> VehicleFeatureEngineer:
        """Validate the feature contract without deriving any data-dependent state."""

        config = get_track_config(self.config)
        frame = validate_feature_frame(X, config)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.output_feature_names_ = np.asarray(config.engineered_features, dtype=object)
        return self

    def transform(self, X: object) -> pd.DataFrame:
        """Return a new, ordered frame and never mutate the caller's frame."""

        check_is_fitted(self, "output_feature_names_")
        config = get_track_config(self.config)
        frame = validate_feature_frame(X, config)

        year = _numeric_column(frame, "year", required_column=True, integral=True)
        mileage = _numeric_column(frame, "mileage", required_column=False, minimum=0.0)

        clipped_age = (float(config.reference_year) - year).clip(lower=0.0)
        mileage_years = clipped_age.clip(lower=1.0)
        mileage_per_year = mileage / mileage_years

        output: dict[str, pd.Series[Any]] = {
            "model_year": year,
            "mileage": mileage.astype(np.float32),
            "mileage_per_year": mileage_per_year.astype(np.float32),
            "mileage_missing": mileage.isna().astype(np.float32),
        }

        if config.name == "wholesale":
            condition = _numeric_column(frame, "condition", required_column=False)
            output["condition"] = condition.astype(np.float32)
            output["condition_missing"] = condition.isna().astype(np.float32)

        for name in config.categorical_features:
            output[name] = _categorical_column(frame, name)

        result = pd.DataFrame(output, index=frame.index)
        if tuple(result.columns) != config.engineered_features:
            raise FeatureContractError("engineered feature contract drifted")
        return result

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Expose the frozen output schema for sklearn introspection."""

        del input_features
        check_is_fitted(self, "output_feature_names_")
        return self.output_feature_names_.copy()


def _numeric_column(
    frame: pd.DataFrame,
    name: str,
    *,
    required_column: bool,
    integral: bool = False,
    minimum: float | None = None,
) -> pd.Series[float]:
    if name not in frame:
        if required_column:
            raise FeatureContractError(f"missing required numeric feature: {name}")
        return pd.Series(np.nan, index=frame.index, dtype=np.float64, name=name)

    source = frame[name]
    boolean_mask = source.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean_mask.fillna(False).any():
        raise FeatureContractError(f"feature {name!r} must not contain booleans")

    converted = pd.to_numeric(source, errors="coerce").astype(np.float64)
    invalid = source.notna() & converted.isna()
    if invalid.any():
        raise FeatureContractError(f"feature {name!r} contains non-numeric values")
    finite = converted.dropna().to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(finite).all():
        raise FeatureContractError(f"feature {name!r} contains non-finite values")
    if integral and ((finite % 1.0) != 0.0).any():
        raise FeatureContractError(f"feature {name!r} must contain integral values")
    if minimum is not None and (finite < minimum).any():
        raise FeatureContractError(f"feature {name!r} must be at least {minimum:g}")
    return converted.rename(name)


def _categorical_column(frame: pd.DataFrame, name: str) -> pd.Series[object]:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=object, name=name)

    source = frame[name]
    invalid = source.notna() & ~source.map(lambda value: isinstance(value, str))
    if invalid.any():
        raise FeatureContractError(f"feature {name!r} contains non-text values")

    def normalize(value: object) -> object:
        if pd.isna(value):
            return np.nan
        normalized = str(value).strip()
        return normalized if normalized else np.nan

    return source.map(normalize).astype(object).rename(name)
