from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest
from autovalue_ml.modeling.final_evaluation_policy import (
    FINAL_EVALUATION_POLICY_ID,
    FINAL_EVALUATION_POLICY_SHA256,
    FinalEvaluationPolicyError,
    load_final_evaluation_policy,
    load_final_evaluation_policy_file,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / "docs" / "experiments" / "retail-rf05-final-evaluation-policy-v1.json"


@pytest.fixture
def policy_bytes() -> bytes:
    return POLICY_PATH.read_bytes()


def test_loads_exact_policy_as_deeply_immutable_state(policy_bytes: bytes) -> None:
    policy = load_final_evaluation_policy(policy_bytes)

    assert policy.policy_sha256 == FINAL_EVALUATION_POLICY_SHA256
    assert policy.document["policy_id"] == FINAL_EVALUATION_POLICY_ID
    assert isinstance(policy.document, MappingProxyType)
    assert isinstance(policy.section("frozen_system"), MappingProxyType)
    boundary = policy.section("one_time_boundary")
    assert boundary["expected_rows"] == 27_589
    assert boundary["currency"] == "USD"
    with pytest.raises(TypeError):
        policy.document["scope"] = "changed"  # type: ignore[index]


def test_file_loader_accepts_only_frozen_regular_file() -> None:
    policy = load_final_evaluation_policy_file(POLICY_PATH)

    assert policy.document["objective"] == (
        "one_time_final_evaluation_of_frozen_rf05_and_calibration_v1"
    )
    with pytest.raises(FinalEvaluationPolicyError, match="regular non-symlink"):
        load_final_evaluation_policy_file(POLICY_PATH.parent)
    with pytest.raises(FinalEvaluationPolicyError, match="could not be read"):
        load_final_evaluation_policy_file(POLICY_PATH.with_suffix(".missing"))


def test_rejects_any_byte_drift(policy_bytes: bytes) -> None:
    drifted = policy_bytes.replace(b'"expected_rows": 27589', b'"expected_rows": 27590', 1)
    reformatted = policy_bytes.replace(b'"schema_version": 1', b'"schema_version" : 1', 1)

    with pytest.raises(FinalEvaluationPolicyError, match="immutable preregistration"):
        load_final_evaluation_policy(drifted)
    with pytest.raises(FinalEvaluationPolicyError, match="immutable preregistration"):
        load_final_evaluation_policy(reformatted)


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"\xff"])
def test_rejects_invalid_serialized_policy(payload: bytes) -> None:
    with pytest.raises(FinalEvaluationPolicyError):
        load_final_evaluation_policy(payload)


def test_rejects_oversized_and_non_text_input() -> None:
    with pytest.raises(FinalEvaluationPolicyError, match="maximum size"):
        load_final_evaluation_policy(b" " * 100_001)
    with pytest.raises(FinalEvaluationPolicyError, match="text or bytes"):
        load_final_evaluation_policy(123)  # type: ignore[arg-type]
