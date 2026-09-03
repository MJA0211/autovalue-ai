from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from autovalue_ml.modeling.uncertainty_sharpness_policy import (
    UNCERTAINTY_SHARPNESS_POLICY_SHA256,
    UncertaintySharpnessPolicyError,
    load_uncertainty_sharpness_policy,
    load_uncertainty_sharpness_policy_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-uncertainty-sharpness-policy-v1.json"
)


@pytest.fixture
def policy_bytes() -> bytes:
    return POLICY_PATH.read_bytes()


def test_loads_frozen_policy_as_typed_immutable_state(policy_bytes: bytes) -> None:
    policy = load_uncertainty_sharpness_policy(policy_bytes)

    assert policy.policy_sha256 == UNCERTAINTY_SHARPNESS_POLICY_SHA256
    assert policy.candidate_ids == (
        "vehicle_status_absolute_residual_v1",
        "normalized_gamma_scale_v1",
        "normalized_smooth_value_scale_v1",
    )
    assert policy.frozen_inputs.rf05_parameters == (96, 1024, 5, 1.0, 0.6)
    assert policy.frozen_inputs.development_rows == 98_552
    assert policy.gamma_method.hyperparameters.max_iter == 2_000
    assert policy.gamma_method.scale_floor_usd == 500.0
    assert policy.calibration_comparison.coverage_levels == (0.8, 0.9, 0.95)
    assert policy.calibration_comparison.bootstrap.replicates == 2_000
    assert policy.acceptance_gates.sharpness.minimum_unclipped_mean_width_reduction_90pct == (0.1)
    assert policy.confidence_policy.high_max_relative_width == pytest.approx(0.686682300031913)
    with pytest.raises(FrozenInstanceError):
        policy.schema_version = 2  # type: ignore[misc]


def test_file_loader_accepts_only_frozen_regular_file() -> None:
    policy = load_uncertainty_sharpness_policy_file(POLICY_PATH)

    assert policy.policy_id == "autovalue-retail-rf05-uncertainty-sharpness-v1"
    with pytest.raises(UncertaintySharpnessPolicyError, match="regular non-symlink"):
        load_uncertainty_sharpness_policy_file(POLICY_PATH.parent)
    with pytest.raises(UncertaintySharpnessPolicyError, match="could not be read"):
        load_uncertainty_sharpness_policy_file(POLICY_PATH.with_suffix(".missing"))


def test_text_loader_accepts_exact_utf8_text(policy_bytes: bytes) -> None:
    policy = load_uncertainty_sharpness_policy(policy_bytes.decode("utf-8"))

    assert policy.scope == "historical_us_advertised_asking_price_usd_2023"


def test_rejects_byte_drift_and_noncanonical_formatting(policy_bytes: bytes) -> None:
    drifted = policy_bytes.replace(b'"development_rows": 98552', b'"development_rows": 98553', 1)
    reformatted = policy_bytes.replace(b'"schema_version": 1', b'"schema_version" : 1', 1)

    with pytest.raises(UncertaintySharpnessPolicyError):
        load_uncertainty_sharpness_policy(drifted)
    with pytest.raises(UncertaintySharpnessPolicyError, match="immutable preregistered"):
        load_uncertainty_sharpness_policy(reformatted)


def test_rejects_duplicate_and_unknown_fields(policy_bytes: bytes) -> None:
    duplicate = policy_bytes.replace(
        b'"schema_version": 1,',
        b'"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    unknown = policy_bytes.replace(b"{\n", b'{\n  "unexpected": true,\n', 1)

    with pytest.raises(UncertaintySharpnessPolicyError, match="duplicate field"):
        load_uncertainty_sharpness_policy(duplicate)
    with pytest.raises(UncertaintySharpnessPolicyError, match="policy fields are invalid"):
        load_uncertainty_sharpness_policy(unknown)


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null"])
def test_rejects_malformed_or_wrong_root_json(payload: bytes) -> None:
    with pytest.raises(UncertaintySharpnessPolicyError):
        load_uncertainty_sharpness_policy(payload)


def test_rejects_oversized_or_non_utf8_content() -> None:
    with pytest.raises(UncertaintySharpnessPolicyError, match="maximum size"):
        load_uncertainty_sharpness_policy(b" " * 100_001)
    with pytest.raises(UncertaintySharpnessPolicyError, match="valid UTF-8"):
        load_uncertainty_sharpness_policy(b"\xff")
    with pytest.raises(UncertaintySharpnessPolicyError, match="valid UTF-8"):
        load_uncertainty_sharpness_policy("\ud800")


def test_rejects_non_text_input() -> None:
    with pytest.raises(UncertaintySharpnessPolicyError, match="text or bytes"):
        load_uncertainty_sharpness_policy(123)  # type: ignore[arg-type]
