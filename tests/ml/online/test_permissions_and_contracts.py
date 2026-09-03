"""Permission and feature-contract tests for online learning."""

from datetime import UTC, datetime

import pytest
from autovalue_ml.online.contracts import ShadowVehicleFeatures
from autovalue_ml.online.errors import ShadowValidationError, SourcePermissionError
from autovalue_ml.online.permissions import (
    AUTOTRADER_SOURCE_ID,
    CARSON_SHIVELY_SOURCE_ID,
    SYNTHETIC_SHADOW_SOURCE_ID,
    YOAD22_SOURCE_ID,
    OnlineSourcePermissionRegistry,
)


def test_online_registry_allows_only_the_explicit_synthetic_source() -> None:
    registry = OnlineSourcePermissionRegistry()

    assert registry.require_learning_approval(SYNTHETIC_SHADOW_SOURCE_ID).approved
    assert sum(decision["approved"] is True for decision in registry.public_summary()) == 1


@pytest.mark.parametrize(
    "source_id",
    [
        YOAD22_SOURCE_ID,
        AUTOTRADER_SOURCE_ID,
        CARSON_SHIVELY_SOURCE_ID,
        "unregistered.real.source",
    ],
)
def test_real_and_unknown_sources_fail_closed(source_id: str) -> None:
    with pytest.raises(SourcePermissionError, match="River source blocked"):
        OnlineSourcePermissionRegistry().require_learning_approval(source_id)


def test_feature_contract_contains_predictors_and_no_target() -> None:
    features = ShadowVehicleFeatures(
        year=2020,
        make="Toyota",
        model="Camry",
        mileage=42_000,
        condition="good",
    ).model_features(observed_at=datetime(2026, 1, 1, tzinfo=UTC))

    assert {"year", "vehicle_age", "make", "model", "mileage"} <= features.keys()
    assert not any("price" in name or "target" in name for name in features)


def test_invalid_vehicle_predictors_are_rejected() -> None:
    features = ShadowVehicleFeatures(year=2020, make="Toyota", model="Camry", mileage=-1)

    with pytest.raises(ShadowValidationError, match="mileage"):
        features.validate(observed_at=datetime(2026, 1, 1, tzinfo=UTC))
