from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import (
    RETAIL_TRACK,
    WHOLESALE_TRACK,
    FeatureContractError,
    VehicleFeatureEngineer,
    get_track_config,
    validate_feature_frame,
    validate_target,
)


def _retail_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["Camry", "Leaf", "F-150"],
            "year": [2020, 2023, 2024],
            "vehicle_status": ["used", "new", "certified"],
            "make": ["Toyota", "Nissan", "Ford"],
            "mileage": [30_000, np.nan, 12_000],
        }
    )


def test_track_configs_are_frozen_distinct_contracts() -> None:
    assert RETAIL_TRACK.reference_year == 2023
    assert WHOLESALE_TRACK.reference_year == 2015
    assert RETAIL_TRACK.contract_version.endswith("-v2")
    assert WHOLESALE_TRACK.contract_version.endswith("-v2")
    assert RETAIL_TRACK.one_hot_min_frequency == 25
    assert WHOLESALE_TRACK.one_hot_min_frequency == 50
    assert "vehicle_status" in RETAIL_TRACK.input_features
    assert "condition" in WHOLESALE_TRACK.input_features
    assert RETAIL_TRACK.target_semantics != WHOLESALE_TRACK.target_semantics

    with pytest.raises(FrozenInstanceError):
        RETAIL_TRACK.reference_year = 2024  # type: ignore[misc]


def test_feature_engineering_is_stateless_and_does_not_mutate_input() -> None:
    original = _retail_frame()
    before = original.copy(deep=True)
    engineer = VehicleFeatureEngineer(RETAIL_TRACK)

    transformed = engineer.fit_transform(original)

    pd.testing.assert_frame_equal(original, before)
    assert tuple(transformed.columns) == RETAIL_TRACK.engineered_features
    assert transformed["model_year"].tolist() == [2020.0, 2023.0, 2024.0]
    assert transformed["mileage_per_year"].iloc[0] == pytest.approx(10_000)
    assert np.isnan(transformed["mileage_per_year"].iloc[1])
    assert transformed["mileage_per_year"].iloc[2] == pytest.approx(12_000)
    assert transformed["mileage_missing"].tolist() == [0.0, 1.0, 0.0]
    assert engineer.get_feature_names_out().tolist() == list(RETAIL_TRACK.engineered_features)
    assert not hasattr(engineer, "statistics_")


def test_wholesale_feature_engineering_adds_numeric_condition_and_missing_flags() -> None:
    frame = pd.DataFrame(
        {
            "year": [2010, 2015],
            "make": ["Honda", "Ford"],
            "model": ["Accord", "Escape"],
            "trim": ["EX", None],
            "mileage": [50_000, None],
            "condition": ["4.2", None],
            "vehicle_type": ["Sedan", "SUV"],
        }
    )

    transformed = VehicleFeatureEngineer(WHOLESALE_TRACK).fit_transform(frame)

    assert tuple(transformed.columns) == WHOLESALE_TRACK.engineered_features
    assert transformed["model_year"].tolist() == [2010.0, 2015.0]
    assert transformed["mileage_per_year"].iloc[0] == pytest.approx(10_000)
    assert transformed["condition"].iloc[0] == pytest.approx(4.2)
    assert transformed["condition_missing"].tolist() == [0.0, 1.0]
    assert pd.isna(transformed["trim"].iloc[1])


@pytest.mark.parametrize("config", [RETAIL_TRACK, WHOLESALE_TRACK])
def test_model_year_is_injective_for_adjacent_and_future_years(config: object) -> None:
    track = get_track_config(config)  # type: ignore[arg-type]
    years = [track.reference_year - 1, track.reference_year, track.reference_year + 1]
    frame = pd.DataFrame(
        {
            "year": years,
            "make": ["Same"] * 3,
            "model": ["Vehicle"] * 3,
        }
    )
    if track.name == "retail":
        frame["vehicle_status"] = ["new"] * 3

    transformed = VehicleFeatureEngineer(track).fit_transform(frame)

    assert transformed["model_year"].tolist() == [float(year) for year in years]
    transformed_keys = {
        tuple(transformed.loc[index, track.engineered_features]) for index in transformed.index
    }
    assert len(transformed_keys) == len(years)


@pytest.mark.parametrize(
    "forbidden", ["price_usd", "price_cents", "dealer", "vin", "seller", "mmr"]
)
def test_feature_allowlist_rejects_every_forbidden_extra(forbidden: str) -> None:
    frame = _retail_frame().assign(**{forbidden: 1})
    with pytest.raises(FeatureContractError, match="forbidden feature"):
        validate_feature_frame(frame, RETAIL_TRACK)


def test_missing_optional_feature_is_supported_but_missing_required_is_not() -> None:
    without_mileage = _retail_frame().drop(columns="mileage")
    transformed = VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(without_mileage)
    assert transformed["mileage"].isna().all()
    assert transformed["mileage_missing"].eq(1.0).all()

    with pytest.raises(FeatureContractError, match="missing required"):
        VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(without_mileage.drop(columns="make"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("year", "unknown", "non-numeric"),
        ("year", 2020.5, "integral"),
        ("year", True, "booleans"),
        ("mileage", -1, "at least"),
        ("mileage", float("inf"), "non-finite"),
        ("make", 42, "non-text"),
    ],
)
def test_feature_engineer_rejects_malformed_values(
    column: str, value: object, message: str
) -> None:
    frame = _retail_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(FeatureContractError, match=message):
        VehicleFeatureEngineer(RETAIL_TRACK).fit_transform(frame)


def test_feature_frame_and_target_validation_are_fail_closed() -> None:
    frame = _retail_frame()
    validated = validate_feature_frame(frame, RETAIL_TRACK)
    validated.iloc[0, 0] = 1999
    assert frame.iloc[0]["year"] == 2020

    target = pd.Series([10_000, 20_000, 30_000], name="price_usd")
    numeric = validate_target(target, expected_rows=len(frame), config=RETAIL_TRACK)
    assert numeric.dtype == np.float64
    target.iloc[0] = 1
    assert numeric[0] == 10_000

    with pytest.raises(FeatureContractError, match="contain only"):
        validate_target(
            pd.DataFrame({"price_usd": [1], "mmr": [1]}),
            expected_rows=1,
            config=RETAIL_TRACK,
        )
    with pytest.raises(FeatureContractError, match="positive"):
        validate_target([0, 1, 2], expected_rows=3, config=RETAIL_TRACK)
    with pytest.raises(FeatureContractError, match="same number"):
        validate_target([1], expected_rows=3, config=RETAIL_TRACK)
    with pytest.raises(FeatureContractError, match="not boolean"):
        validate_target([True, 1, 2], expected_rows=3, config=RETAIL_TRACK)


def test_config_resolution_and_internal_contract_validation() -> None:
    assert get_track_config("retail") is RETAIL_TRACK
    assert get_track_config(WHOLESALE_TRACK) is WHOLESALE_TRACK
    with pytest.raises(FeatureContractError, match="unsupported"):
        get_track_config("consumer")  # type: ignore[arg-type]
    with pytest.raises(FeatureContractError, match="target must not"):
        replace(RETAIL_TRACK, optional_input_features=("price_usd",))
