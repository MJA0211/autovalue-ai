"""Strict loader for the one-time retail RF05 final-evaluation policy."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, cast

FINAL_EVALUATION_POLICY_SHA256: Final = (
    "2be880be315f39a727bd8f1c6545b9410ea855bee63a3e72336f4da8cd7d5c33"
)
FINAL_EVALUATION_POLICY_ID: Final = "autovalue-retail-rf05-final-evaluation-v1"
MAXIMUM_POLICY_BYTES: Final = 100_000


class FinalEvaluationPolicyError(ValueError):
    """The final-evaluation policy is missing, mutable, or malformed."""


@dataclass(frozen=True, slots=True)
class FinalEvaluationPolicy:
    """Immutable typed handle to the byte-frozen policy document."""

    document: Mapping[str, object]
    policy_sha256: str = FINAL_EVALUATION_POLICY_SHA256

    def __post_init__(self) -> None:
        if self.policy_sha256 != FINAL_EVALUATION_POLICY_SHA256:
            raise FinalEvaluationPolicyError("final policy checksum identity differs")
        _validate_fixed_policy(self.document)

    def section(self, name: str) -> Mapping[str, object]:
        value = self.document.get(name)
        if not isinstance(value, Mapping):
            raise FinalEvaluationPolicyError(f"final policy section is invalid: {name}")
        return cast(Mapping[str, object], value)


def load_final_evaluation_policy(serialized: str | bytes) -> FinalEvaluationPolicy:
    """Accept only the exact preregistered bytes and return deeply frozen state."""

    payload, text = _bounded_utf8(serialized)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != FINAL_EVALUATION_POLICY_SHA256:
        raise FinalEvaluationPolicyError(
            "final policy bytes differ from the immutable preregistration"
        )
    try:
        raw = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise FinalEvaluationPolicyError("final policy is not valid JSON") from error
    if not isinstance(raw, Mapping):
        raise FinalEvaluationPolicyError("final policy root must be an object")
    frozen = _deep_freeze(raw)
    if not isinstance(frozen, Mapping):
        raise FinalEvaluationPolicyError("final policy root could not be frozen")
    return FinalEvaluationPolicy(document=cast(Mapping[str, object], frozen))


def load_final_evaluation_policy_file(path: str | Path) -> FinalEvaluationPolicy:
    """Read one stable regular policy file and verify its immutable checksum."""

    resolved = Path(path)
    try:
        before = resolved.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise FinalEvaluationPolicyError("final policy path must be a regular non-symlink file")
        payload = resolved.read_bytes()
        after = resolved.lstat()
    except OSError as error:
        raise FinalEvaluationPolicyError("final policy file could not be read") from error
    if _file_identity(before) != _file_identity(after):
        raise FinalEvaluationPolicyError("final policy file changed while it was read")
    return load_final_evaluation_policy(payload)


def _validate_fixed_policy(document: Mapping[str, object]) -> None:
    expected_root = {
        "schema_version",
        "policy_id",
        "preregistered_on",
        "objective",
        "scope",
        "one_time_boundary",
        "frozen_system",
        "implementation_bindings",
        "frozen_upstream_evidence",
        "evaluation",
        "classification",
        "publication",
        "excluded_systems",
    }
    if set(document) != expected_root:
        raise FinalEvaluationPolicyError("final policy root fields differ")
    if document.get("schema_version") != 1:
        raise FinalEvaluationPolicyError("final policy schema version differs")
    if document.get("policy_id") != FINAL_EVALUATION_POLICY_ID:
        raise FinalEvaluationPolicyError("final policy id differs")
    boundary = _mapping(document, "one_time_boundary")
    if (
        boundary.get("partition") != "test"
        or boundary.get("expected_rows") != 27_589
        or boundary.get("holdout_targets_may_be_accessed_only_after_policy_and_bindings_pass")
        is not True
        or boundary.get("holdout_rows_may_fit_or_calibrate_any_model") is not False
        or boundary.get("holdout_rows_may_enter_yoad_or_river") is not False
        or boundary.get("post_holdout_tuning_or_recalibration") is not False
    ):
        raise FinalEvaluationPolicyError("one-time holdout boundary differs")
    system = _mapping(document, "frozen_system")
    if system.get("rf05_identity_sha256") != (
        "3bbd73d6442387496b05253dd20bc749db24aa482d56fa6ba73ec2702de8b513"
    ):
        raise FinalEvaluationPolicyError("frozen RF05 identity differs")
    uncertainty = _mapping(system, "uncertainty")
    if (
        uncertainty.get("version") != "retail-rf05-split-conformal-v1"
        or uncertainty.get("selected_method") != "vehicle_status"
        or tuple(cast(tuple[object, ...], uncertainty.get("coverage_levels"))) != (0.8, 0.9, 0.95)
        or uncertainty.get("quantiles_may_be_changed_after_holdout") is not False
    ):
        raise FinalEvaluationPolicyError("frozen uncertainty system differs")
    publication = _mapping(document, "publication")
    if (
        publication.get("aggregate_only") is not True
        or publication.get("persist_raw_rows_targets_predictions_residuals_or_identifiers")
        is not False
        or publication.get("overwrite_existing_output") is not False
        or publication.get("final_holdout_is_permanently_evaluation_only_after_this_run")
        is not True
        or publication.get("automatic_followup_experiment") is not False
    ):
        raise FinalEvaluationPolicyError("final publication boundary differs")
    excluded = _mapping(document, "excluded_systems")
    if set(excluded) != {"yoad", "river", "autotrader", "carson_shively"}:
        raise FinalEvaluationPolicyError("excluded-system boundary differs")


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise FinalEvaluationPolicyError(f"final policy object is invalid: {key}")
    return cast(Mapping[str, object], value)


def _bounded_utf8(serialized: str | bytes) -> tuple[bytes, str]:
    if isinstance(serialized, str):
        try:
            payload = serialized.encode("utf-8")
        except UnicodeEncodeError as error:
            raise FinalEvaluationPolicyError("final policy must be UTF-8") from error
        text = serialized
    elif isinstance(serialized, bytes):
        payload = serialized
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise FinalEvaluationPolicyError("final policy must be UTF-8") from error
    else:
        raise FinalEvaluationPolicyError("final policy must be text or bytes")
    if len(payload) > MAXIMUM_POLICY_BYTES:
        raise FinalEvaluationPolicyError("final policy exceeds maximum size")
    return payload, text


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FinalEvaluationPolicyError(f"final policy has duplicate field: {key}")
        result[key] = value
    return result


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


__all__ = [
    "FINAL_EVALUATION_POLICY_ID",
    "FINAL_EVALUATION_POLICY_SHA256",
    "FinalEvaluationPolicy",
    "FinalEvaluationPolicyError",
    "load_final_evaluation_policy",
    "load_final_evaluation_policy_file",
]
