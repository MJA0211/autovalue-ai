"""Strict, immutable feature contracts for AutoValue AI modeling tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

TrackName: TypeAlias = Literal["retail", "wholesale"]


class FeatureContractError(ValueError):
    """Raised when modeling input does not exactly match an approved contract."""


@dataclass(frozen=True, slots=True)
class TrackConfig:
    """A complete, immutable contract for one independently modeled target."""

    name: TrackName
    contract_version: str
    target_name: str
    target_semantics: str
    reference_year: int
    required_input_features: tuple[str, ...]
    optional_input_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    one_hot_min_frequency: int
    one_hot_max_categories: int = 512
    status_slice_feature: str | None = None

    def __post_init__(self) -> None:
        """Reject internally inconsistent contracts at construction time."""

        all_inputs = self.required_input_features + self.optional_input_features
        if not self.contract_version.strip():
            raise FeatureContractError("contract_version must not be empty")
        if not self.target_name.strip():
            raise FeatureContractError("target_name must not be empty")
        if self.reference_year < 1900:
            raise FeatureContractError("reference_year must be at least 1900")
        if len(all_inputs) != len(set(all_inputs)):
            raise FeatureContractError("input feature names must be unique")
        if self.target_name in all_inputs:
            raise FeatureContractError("target must not appear in the feature allowlist")
        if len(self.numeric_features) != len(set(self.numeric_features)):
            raise FeatureContractError("numeric feature names must be unique")
        if len(self.categorical_features) != len(set(self.categorical_features)):
            raise FeatureContractError("categorical feature names must be unique")
        if set(self.numeric_features) & set(self.categorical_features):
            raise FeatureContractError("numeric and categorical features must not overlap")
        if self.one_hot_min_frequency < 1:
            raise FeatureContractError("one_hot_min_frequency must be positive")
        if self.one_hot_max_categories < 2:
            raise FeatureContractError("one_hot_max_categories must be at least two")
        if (
            self.status_slice_feature is not None
            and self.status_slice_feature not in self.required_input_features
        ):
            raise FeatureContractError("status slice feature must be a required input feature")

    @property
    def input_features(self) -> tuple[str, ...]:
        """Return the exact ordered raw-feature allowlist."""

        return self.required_input_features + self.optional_input_features

    @property
    def engineered_features(self) -> tuple[str, ...]:
        """Return the exact ordered feature-engineering output contract."""

        return self.numeric_features + self.categorical_features

    @property
    def maximum_transformed_features(self) -> int:
        """Return a conservative upper bound for the sparse design matrix width."""

        categorical_bound = len(self.categorical_features) * self.one_hot_max_categories
        return len(self.numeric_features) + categorical_bound


RETAIL_TRACK: Final = TrackConfig(
    name="retail",
    contract_version="retail-historical-asking-price-v2",
    target_name="price_usd",
    target_semantics="historical_us_advertised_asking_price_usd_2023",
    reference_year=2023,
    required_input_features=("year", "make", "model", "vehicle_status"),
    optional_input_features=("mileage",),
    numeric_features=(
        "model_year",
        "mileage",
        "mileage_per_year",
        "mileage_missing",
    ),
    categorical_features=("make", "model", "vehicle_status"),
    one_hot_min_frequency=25,
    status_slice_feature="vehicle_status",
)

WHOLESALE_TRACK: Final = TrackConfig(
    name="wholesale",
    contract_version="wholesale-historical-completed-sale-v2",
    target_name="price_usd",
    target_semantics="historical_us_wholesale_auction_completed_sale_price_usd_2014_2015",
    reference_year=2015,
    required_input_features=("year", "make", "model"),
    optional_input_features=("trim", "mileage", "condition", "vehicle_type"),
    numeric_features=(
        "model_year",
        "mileage",
        "mileage_per_year",
        "mileage_missing",
        "condition",
        "condition_missing",
    ),
    categorical_features=("make", "model", "trim", "vehicle_type"),
    one_hot_min_frequency=50,
)

TRACKS: Final[dict[TrackName, TrackConfig]] = {
    RETAIL_TRACK.name: RETAIL_TRACK,
    WHOLESALE_TRACK.name: WHOLESALE_TRACK,
}


def get_track_config(track: TrackName | TrackConfig) -> TrackConfig:
    """Resolve a supported track name while preserving an explicit config."""

    if isinstance(track, TrackConfig):
        return track
    try:
        return TRACKS[track]
    except KeyError as error:
        raise FeatureContractError(f"unsupported modeling track: {track!r}") from error


def validate_feature_frame(frame: object, config: TrackConfig) -> pd.DataFrame:
    """Validate exact feature names and return a private, contract-ordered copy.

    Optional columns may be absent. Any unapproved column is rejected, even when
    it would otherwise be ignored by a downstream ``ColumnTransformer``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise FeatureContractError("features must be a pandas DataFrame")
    if not frame.columns.is_unique:
        raise FeatureContractError("feature columns must be unique")

    columns = tuple(frame.columns)
    non_text_columns = tuple(column for column in columns if not isinstance(column, str))
    if non_text_columns:
        raise FeatureContractError("feature column names must be strings")

    allowed = set(config.input_features)
    provided = set(columns)
    forbidden = sorted(provided - allowed)
    if forbidden:
        raise FeatureContractError(f"forbidden feature columns: {', '.join(forbidden)}")

    missing = sorted(set(config.required_input_features) - provided)
    if missing:
        raise FeatureContractError(f"missing required feature columns: {', '.join(missing)}")

    ordered = [name for name in config.input_features if name in provided]
    return frame.loc[:, ordered].copy(deep=True)


def validate_target(
    target: object,
    *,
    expected_rows: int,
    config: TrackConfig,
) -> NDArray[np.float64]:
    """Validate a positive, finite USD target without silently reshaping tables."""

    if isinstance(target, pd.DataFrame):
        if tuple(target.columns) != (config.target_name,):
            raise FeatureContractError(f"target frame must contain only {config.target_name!r}")
        values = target.iloc[:, 0].to_numpy(dtype=object, copy=True)
    elif isinstance(target, pd.Series):
        if target.name is not None and target.name != config.target_name:
            raise FeatureContractError(
                f"target series must be named {config.target_name!r} when named"
            )
        values = target.to_numpy(dtype=object, copy=True)
    else:
        values = np.asarray(target, dtype=object)

    if values.ndim != 1:
        raise FeatureContractError("target must be one-dimensional")
    if len(values) != expected_rows:
        raise FeatureContractError("features and target must have the same number of rows")
    if len(values) == 0:
        raise FeatureContractError("target must not be empty")
    if any(isinstance(value, (bool, np.bool_)) for value in values.flat):
        raise FeatureContractError("target must be numeric, not boolean")
    try:
        numeric: NDArray[np.float64] = np.asarray(values, dtype=np.float64).copy()
    except (TypeError, ValueError) as error:
        raise FeatureContractError("target must contain only numeric values") from error
    if not np.isfinite(numeric).all():
        raise FeatureContractError("target must contain only finite values")
    if (numeric <= 0).any():
        raise FeatureContractError("target prices must be positive")
    return numeric
