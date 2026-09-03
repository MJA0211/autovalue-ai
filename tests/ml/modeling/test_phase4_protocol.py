from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
from autovalue_ml.modeling.contracts import TrackName
from autovalue_ml.modeling.phase4_protocol import (
    PHASE4_PROTOCOL_SHA256,
    Phase4ProtocolError,
    derive_phase4_seed,
    load_phase4_protocol,
    parse_phase4_protocol_json,
    validate_phase4_protocol,
    verify_phase4_protocol_sha256,
)

PROJECT_ROOT = Path(__file__).parents[3]
PROTOCOL_PATH = PROJECT_ROOT / "docs" / "experiments" / "phase4-model-selection-v1.json"


@pytest.fixture
def protocol_payload() -> dict[str, Any]:
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _replace(payload: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    current: Any = payload
    for component in path[:-1]:
        current = current[component]
    current[path[-1]] = value


def test_repository_protocol_is_hash_pinned_valid_and_immutable() -> None:
    assert verify_phase4_protocol_sha256(PROTOCOL_PATH) == PHASE4_PROTOCOL_SHA256
    protocol = load_phase4_protocol(PROTOCOL_PATH)

    assert protocol.policy_id == "autovalue-phase4-model-selection-v1"
    assert protocol.final_evaluation_name == "phase3_reused_legacy_holdout"
    assert tuple(track.name for track in protocol.tracks) == ("retail", "wholesale")
    assert len(protocol.for_track("retail").random_forest_candidates) == 6
    assert len(protocol.for_track("wholesale").gradient_boosting_candidates) == 6
    assert protocol.for_track("retail").development_rows is None
    assert protocol.for_track("wholesale").development_rows == 391_641
    assert protocol.for_track("wholesale").calibration_rows == 50_489
    assert protocol.budgets.maximum_peak_rss_gb == 8
    assert protocol.budgets.maximum_private_artifact_mb == 50

    with pytest.raises(FrozenInstanceError):
        protocol.policy_id = "changed"  # type: ignore[misc]
    with pytest.raises(Phase4ProtocolError, match="unsupported Phase 4 track"):
        protocol.for_track("consumer")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("track", "purpose", "expected"),
    [
        ("retail", "calibration", 1_416_582_761),
        ("retail", "screening_sample", 1_707_037_927),
        ("retail", "random_forest", 1_254_777_149),
        ("retail", "gradient_boosting", 3_295_129_705),
        ("retail", "permutation_importance", 2_824_429_337),
        ("wholesale", "calibration", 3_061_104_204),
        ("wholesale", "screening_sample", 759_966_512),
        ("wholesale", "random_forest", 2_903_812_338),
        ("wholesale", "gradient_boosting", 177_971_163),
        ("wholesale", "permutation_importance", 3_132_861_797),
    ],
)
def test_documented_seed_derivation_is_exact(track: TrackName, purpose: str, expected: int) -> None:
    assert derive_phase4_seed("autovalue-phase4-v1", track, purpose) == expected


@pytest.mark.parametrize(
    ("master", "track", "purpose", "message"),
    [
        ("other", "retail", "calibration", "master seed"),
        ("autovalue-phase4-v1", "other", "calibration", "track"),
        ("autovalue-phase4-v1", "retail", "other", "purpose"),
    ],
)
def test_seed_derivation_rejects_unapproved_labels(
    master: str, track: str, purpose: str, message: str
) -> None:
    with pytest.raises(Phase4ProtocolError, match=message):
        derive_phase4_seed(master, track, purpose)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("policy_id",), "autovalue-phase4-model-selection-v2"),
        (("reviewed_on",), "2026-08-30"),
        (("decision",), "draft"),
        (("master_seed_label",), "other"),
        (("seed_derivation", "purpose_labels"), ["calibration"]),
        (("holdout_policy", "holdout_used_for_preprocessing"), True),
        (("holdout_policy", "holdout_used_for_hyperparameter_tuning"), 0),
        (("holdout_policy", "holdout_used_for_model_family_selection"), True),
        (("holdout_policy", "holdout_used_for_interval_calibration"), True),
        (("holdout_policy", "final_evaluation_name"), "untouched_holdout"),
        (("holdout_policy", "limitation"), "not disclosed"),
        (("preprocessing", "fit_scope"), "fit_once"),
        (("preprocessing", "matrix"), "dense_float64"),
        (("preprocessing", "dense_conversion_forbidden"), False),
        (("preprocessing", "target_clipping"), True),
        (("preprocessing", "target_log_transform"), True),
        (("candidate_families", "xgboost", "status"), "enabled"),
        (("search_budget", "parallel_candidate_fits"), 2),
        (("search_budget", "maximum_peak_rss_gb"), 16),
        (
            (
                "search_budget",
                "screen_all_explicit_candidates_on_target_free_group_or_time_safe_sample",
            ),
            False,
        ),
        (("selection", "primary_metric"), "holdout_mae"),
        (("selection", "near_tie_relative_mae"), 0.02),
        (("deployment_gates", "maximum_private_artifact_mb"), 51),
        (("deployment_gates", "serving_workers"), 2),
        (("prediction_range", "method"), "normal_approximation"),
        (("prediction_range", "alpha"), 0.2),
        (("feature_importance", "holdout_role"), "untouched"),
        (("feature_importance", "local_explanation_claim"), True),
        (("feature_importance", "raw_linear_coefficients_forbidden_as_importance"), False),
        (("artifact_policy", "storage"), "public_release"),
        (("artifact_policy", "downloadable_publication"), "approved"),
        (("artifact_policy", "trusted_local_joblib_only"), False),
        (("artifact_policy", "user_uploaded_or_remote_artifacts_forbidden"), False),
        (("artifact_policy", "verify_sha256_before_load"), False),
        (
            (
                "artifact_policy",
                "persist_source_rows_predictions_residuals_or_category_vocabulary_in_public_reports",
            ),
            True,
        ),
    ],
)
def test_policy_mutations_fail_closed(
    protocol_payload: dict[str, Any], path: tuple[str | int, ...], replacement: object
) -> None:
    _replace(protocol_payload, path, replacement)
    with pytest.raises(Phase4ProtocolError):
        validate_phase4_protocol(protocol_payload)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("tracks", "retail", "source_id"), "another_source"),
        (("tracks", "retail", "feature_contract_version"), "retail-v3"),
        (("tracks", "retail", "target_semantics"), "current_market_value"),
        (("tracks", "retail", "candidate_sha256"), "0" * 64),
        (("tracks", "retail", "split_assignment_sha256"), "a" * 64),
        (("tracks", "retail", "split_manifest_sha256"), "A" * 64),
        (("tracks", "retail", "split_artifact_set_id"), "short"),
        (("tracks", "retail", "phase3_baseline_report_sha256"), "f" * 64),
        (("tracks", "retail", "phase3_train_rows"), 109_509),
        (("tracks", "retail", "legacy_holdout_rows"), True),
        (("tracks", "retail", "screening_sample_seed"), 1),
        (("tracks", "retail", "random_forest_seed"), 1.0),
        (("tracks", "wholesale", "raw_source_sha256"), "0" * 64),
        (("tracks", "wholesale", "candidate_sha256"), "0" * 64),
        (("tracks", "wholesale", "split_assignment_sha256"), "0" * 64),
        (("tracks", "wholesale", "split_manifest_sha256"), "0" * 64),
        (("tracks", "wholesale", "split_artifact_set_id"), "0" * 64),
        (("tracks", "wholesale", "phase3_baseline_report_sha256"), "0" * 64),
        (("tracks", "wholesale", "phase3_train_rows"), 442_131),
        (("tracks", "wholesale", "legacy_holdout_rows"), 98_633),
        (
            ("tracks", "wholesale", "development_calibration_split", "development_rows"),
            391_640,
        ),
        (
            ("tracks", "wholesale", "development_calibration_split", "calibration_rows"),
            50_488,
        ),
        (
            ("tracks", "wholesale", "development_calibration_split", "calibration_bucket"),
            "2015_04",
        ),
        (("tracks", "wholesale", "development_cv", "validation_folds"), 4),
        (("tracks", "wholesale", "gradient_boosting_seed"), -1),
        (("tracks", "wholesale", "permutation_importance_seed"), True),
    ],
)
def test_track_lineage_contract_counts_and_seeds_are_fixed(
    protocol_payload: dict[str, Any], path: tuple[str | int, ...], replacement: object
) -> None:
    _replace(protocol_payload, path, replacement)
    with pytest.raises(Phase4ProtocolError):
        validate_phase4_protocol(protocol_payload)


def test_unknown_missing_and_nonobject_sections_fail_closed(
    protocol_payload: dict[str, Any],
) -> None:
    extra = copy.deepcopy(protocol_payload)
    extra["raw_rows"] = []
    with pytest.raises(Phase4ProtocolError, match="unexpected raw_rows"):
        validate_phase4_protocol(extra)

    missing = copy.deepcopy(protocol_payload)
    del missing["artifact_policy"]
    with pytest.raises(Phase4ProtocolError, match="missing artifact_policy"):
        validate_phase4_protocol(missing)

    nonobject = copy.deepcopy(protocol_payload)
    nonobject["tracks"] = []
    with pytest.raises(Phase4ProtocolError, match="tracks must be an object"):
        validate_phase4_protocol(nonobject)

    with pytest.raises(Phase4ProtocolError, match="protocol must be an object"):
        validate_phase4_protocol([])


@pytest.mark.parametrize(
    ("family", "track", "candidate_index", "value_index", "replacement"),
    [
        ("random_forest", "retail", 0, 0, True),
        ("random_forest", "retail", 0, 0, 95),
        ("random_forest", "retail", 0, 1, 4096),
        ("random_forest", "retail", 0, 2, 0),
        ("random_forest", "retail", 0, 3, 1),
        ("random_forest", "retail", 0, 3, 0.0),
        ("random_forest", "wholesale", 0, 4, 1.1),
        ("gradient_boosting", "retail", 0, 0, "absolute_error"),
        ("gradient_boosting", "retail", 0, 1, True),
        ("gradient_boosting", "retail", 0, 2, 241),
        ("gradient_boosting", "retail", 0, 3, 0.0),
        ("gradient_boosting", "retail", 0, 4, 4),
        ("gradient_boosting", "wholesale", 0, 5, 0),
        ("gradient_boosting", "wholesale", 0, 6, 1.1),
        ("gradient_boosting", "wholesale", 0, 7, float("nan")),
    ],
)
def test_candidate_values_reject_wrong_types_and_ranges(
    protocol_payload: dict[str, Any],
    family: str,
    track: str,
    candidate_index: int,
    value_index: int,
    replacement: object,
) -> None:
    protocol_payload["candidate_families"][family][track][candidate_index][value_index] = (
        replacement
    )
    with pytest.raises(Phase4ProtocolError):
        validate_phase4_protocol(protocol_payload)


@pytest.mark.parametrize("family", ["random_forest", "gradient_boosting"])
def test_candidate_count_arity_duplicates_and_exact_values_are_frozen(
    protocol_payload: dict[str, Any], family: str
) -> None:
    too_few = copy.deepcopy(protocol_payload)
    too_few["candidate_families"][family]["retail"].pop()
    with pytest.raises(Phase4ProtocolError, match="exactly six"):
        validate_phase4_protocol(too_few)

    wrong_arity = copy.deepcopy(protocol_payload)
    wrong_arity["candidate_families"][family]["retail"][0].pop()
    with pytest.raises(Phase4ProtocolError, match="exactly"):
        validate_phase4_protocol(wrong_arity)

    duplicate = copy.deepcopy(protocol_payload)
    duplicate["candidate_families"][family]["retail"][1] = copy.deepcopy(
        duplicate["candidate_families"][family]["retail"][0]
    )
    with pytest.raises(Phase4ProtocolError, match="unique"):
        validate_phase4_protocol(duplicate)

    valid_but_unapproved = copy.deepcopy(protocol_payload)
    index = 0 if family == "random_forest" else 2
    valid_but_unapproved["candidate_families"][family]["retail"][0][index] += 1
    with pytest.raises(Phase4ProtocolError, match="approved candidates"):
        validate_phase4_protocol(valid_but_unapproved)


def test_candidate_common_policy_and_tuple_field_order_are_fixed(
    protocol_payload: dict[str, Any],
) -> None:
    mutations: tuple[tuple[str, str, object], ...] = (
        (
            "random_forest",
            "tuple_fields",
            list(reversed(protocol_payload["candidate_families"]["random_forest"]["tuple_fields"])),
        ),
        (
            "random_forest",
            "common",
            {
                **protocol_payload["candidate_families"]["random_forest"]["common"],
                "bootstrap": False,
            },
        ),
        ("gradient_boosting", "tuple_fields", []),
        ("gradient_boosting", "common", {"n_iter_no_change": 10}),
    )
    for family, field, replacement in mutations:
        changed = copy.deepcopy(protocol_payload)
        changed["candidate_families"][family][field] = replacement
        with pytest.raises(Phase4ProtocolError):
            validate_phase4_protocol(changed)


@pytest.mark.parametrize(
    "serialized",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"schema_version":NaN}',
        "not-json",
        b"\xff",
        123,
    ],
)
def test_parser_rejects_ambiguous_or_malformed_json(serialized: object) -> None:
    with pytest.raises(Phase4ProtocolError):
        parse_phase4_protocol_json(serialized)  # type: ignore[arg-type]


def test_parser_rejects_oversized_text_and_bytes() -> None:
    oversized = " " * 100_001
    with pytest.raises(Phase4ProtocolError, match="maximum size"):
        parse_phase4_protocol_json(oversized)
    with pytest.raises(Phase4ProtocolError, match="maximum size"):
        parse_phase4_protocol_json(oversized.encode())


def test_file_hash_boundary_rejects_tampering_bad_digest_and_nonfiles(tmp_path: Path) -> None:
    copied = tmp_path / "protocol.json"
    copied.write_bytes(PROTOCOL_PATH.read_bytes())
    assert verify_phase4_protocol_sha256(copied) == PHASE4_PROTOCOL_SHA256

    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(Phase4ProtocolError, match="does not match"):
        load_phase4_protocol(copied)
    with pytest.raises(Phase4ProtocolError, match="lowercase SHA-256"):
        verify_phase4_protocol_sha256(copied, PHASE4_PROTOCOL_SHA256.upper())
    with pytest.raises(Phase4ProtocolError, match="regular non-symlink"):
        load_phase4_protocol(tmp_path)
    with pytest.raises(Phase4ProtocolError, match="not accessible"):
        load_phase4_protocol(tmp_path / "missing.json")

    empty = tmp_path / "empty.json"
    empty.touch()
    with pytest.raises(Phase4ProtocolError, match="file size"):
        load_phase4_protocol(empty)


def test_loader_rejects_symlink_when_supported(tmp_path: Path) -> None:
    link = tmp_path / "linked.json"
    try:
        link.symlink_to(PROTOCOL_PATH)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(Phase4ProtocolError, match="regular non-symlink"):
        load_phase4_protocol(link)
