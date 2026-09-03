"""Fail-closed loader for AutoValue AI's approved Phase 4 experiment policy.

The protocol is project-owned control data, not an extensible user configuration.
Consequently this module rejects unknown fields and any drift from the approved
lineage, split, candidate, resource, holdout, calibration, or artifact policies.
It performs no filesystem access at import time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from .contracts import RETAIL_TRACK, WHOLESALE_TRACK, TrackName

PHASE4_PROTOCOL_SHA256: Final = "6e517acb29634d676155c80fb73f4f126db492eba12a4281e9216dc568b1d384"

_MAX_PROTOCOL_BYTES: Final = 100_000
_MASTER_SEED_LABEL: Final = "autovalue-phase4-v1"
_POLICY_ID: Final = "autovalue-phase4-model-selection-v1"
_HEX_DIGITS: Final = frozenset("0123456789abcdef")
_TRACK_ORDER: Final[tuple[TrackName, TrackName]] = ("retail", "wholesale")
_PURPOSES: Final = (
    "calibration",
    "screening_sample",
    "random_forest",
    "gradient_boosting",
    "permutation_importance",
)
_SEED_FIELDS: Final = {
    "screening_sample": "screening_sample_seed",
    "random_forest": "random_forest_seed",
    "gradient_boosting": "gradient_boosting_seed",
    "permutation_importance": "permutation_importance_seed",
}
_WHOLESALE_DEVELOPMENT_BUCKETS: Final = (
    "warmup",
    "2015_01",
    "2015_02",
    "2015_03_04",
)
_RETAIL_CALIBRATION_ALGORITHM: Final = (
    "Within each vehicle_status, sort indivisible year/make/model/mileage/status predictor "
    "groups by the exact seeded rank below and select the prefix whose cumulative row count "
    "is closest to one tenth of Phase-3 training rows for that status. Prefix zero "
    "participates and ties choose the smaller prefix. Targets are excluded."
)
_RETAIL_GROUP_ID_ALGORITHM: Final = (
    "retail_predictor_groups from feature contract v2: SHA-256 hex of UTF-8 compact JSON "
    "[model_year,make,model,mileage_or_null,vehicle_status] with ensure_ascii=false and "
    "separators comma/colon"
)
_RETAIL_CALIBRATION_RANK_ALGORITHM: Final = (
    "SHA-256 bytes of b'autovalue-retail-calibration-v1\\x00' + calibration_seed as "
    "unsigned 32-bit big-endian + ASCII group_id; sort by digest bytes then group_id"
)
_RETAIL_SCREENING_ALGORITHM: Final = (
    "Within each vehicle_status, keep predictor groups indivisible, rank them by the exact "
    "seeded rank below, and select the prefix whose cumulative row count is closest to three "
    "tenths of development rows for that status. Prefix zero participates and ties choose "
    "the smaller prefix. Targets are excluded from sampling."
)
_RETAIL_SCREENING_RANK_ALGORITHM: Final = (
    "SHA-256 bytes of b'autovalue-retail-screening-v1\\x00' + screening_sample_seed as "
    "unsigned 32-bit big-endian + ASCII group_id; sort by digest bytes then group_id"
)
_WHOLESALE_SCREENING_ALGORITHM: Final = (
    "Within each ordered development bucket, rank rows by the exact seeded rank below and "
    "select the prefix whose row count is closest to one quarter of that bucket. Prefix zero "
    "participates and ties choose the smaller prefix. Targets are excluded from sampling; "
    "every sampled row retains its original bucket so CV remains forward-only."
)
_WHOLESALE_SCREENING_RANK_ALGORITHM: Final = (
    "SHA-256 bytes of b'autovalue-wholesale-screening-v1\\x00' + screening_sample_seed as "
    "unsigned 32-bit big-endian + UTF-8 bucket + b'\\x00' + zero-based Phase-3 outer-train "
    "row position as unsigned 64-bit big-endian; sort by digest bytes then row position"
)

_RETAIL_LINEAGE: Final = {
    "source_id": "kaggle_us_sales_cars_v2",
    "target_semantics": RETAIL_TRACK.target_semantics,
    "feature_contract_version": RETAIL_TRACK.contract_version,
    "candidate_sha256": "12880cfbb2cb7f600f291c077adfa247afb9774b400b21bb7eb7409d72f7fb92",
    "split_assignment_sha256": ("5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5"),
    "split_manifest_sha256": ("c60bf010fb47dff44d03b5da80b191ddb4b748661cb5cf02397422fdbaaf3466"),
    "split_artifact_set_id": ("8ab1e31f08aab9cefe3293e9a1e4bfe6ddf544da5020e5ac64b6b5fff7625edf"),
    "phase3_baseline_report_sha256": (
        "b5cae941ebb01d9766716d01a24acc75ad7d0432b05e8dde44a6200caffad28a"
    ),
}
_WHOLESALE_LINEAGE: Final = {
    "source_id": "kaggle_vehicle_sales_v1",
    "target_semantics": WHOLESALE_TRACK.target_semantics,
    "feature_contract_version": WHOLESALE_TRACK.contract_version,
    "raw_source_sha256": ("32ba3ce51664e6a12c0c927ed193b41e3c4743fdf18bc0317389892aed27f556"),
    "candidate_sha256": "ef9f77b14d0cfc2b180cca5d57e6de04b10df2b4d0e6871e00ec3c6d095c1489",
    "split_assignment_sha256": ("a96909345612f5ddc5665c4d6817d2c8f0dd6d59c3a84fc523cb82b6adeeb5f2"),
    "split_manifest_sha256": ("d0dd0c24f342a8a45c1f89419780f470d0f152d61cf5dd54b2cb786df9525bd3"),
    "split_artifact_set_id": ("75fe4bbe4b1d77c48e2e8804dfdde86a2ef037580fdf58c257e762fa53ee6d37"),
    "phase3_baseline_report_sha256": (
        "b0be8b30367f7b7adca904d80610dd161b9b33dffd9e116d5030bd34403a3030"
    ),
}

_RF_FIELDS: Final = (
    "n_estimators",
    "max_leaf_nodes",
    "min_samples_leaf",
    "max_features",
    "max_samples",
)
_GB_FIELDS: Final = (
    "loss",
    "alpha",
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "subsample",
    "max_features",
)

_EXPECTED_RF: Final = {
    "retail": (
        (96, 512, 5, 0.5, 0.6),
        (128, 1024, 15, 0.7, 0.8),
        (160, 2048, 15, 0.7, 0.8),
        (160, 1024, 30, 1.0, 0.8),
        (128, 2048, 30, 0.5, 1.0),
        (96, 1024, 5, 1.0, 0.6),
    ),
    "wholesale": (
        (96, 512, 25, 0.4, 0.5),
        (128, 1024, 50, 0.6, 0.6),
        (160, 2048, 50, 0.6, 0.7),
        (160, 1024, 100, 0.8, 0.7),
        (128, 2048, 100, 0.4, 0.6),
        (96, 1024, 25, 0.8, 0.5),
    ),
}
_EXPECTED_GB: Final = {
    "retail": (
        ("squared_error", 0.9, 120, 0.08, 2, 20, 0.8, 0.8),
        ("squared_error", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
        ("squared_error", 0.9, 240, 0.03, 3, 50, 0.65, 0.5),
        ("huber", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
        ("huber", 0.85, 240, 0.03, 2, 50, 0.8, 0.5),
        ("huber", 0.9, 120, 0.08, 3, 50, 0.65, 0.5),
    ),
    "wholesale": (
        ("squared_error", 0.9, 100, 0.08, 2, 50, 0.7, 0.7),
        ("squared_error", 0.9, 140, 0.05, 2, 100, 0.5, 0.4),
        ("squared_error", 0.9, 180, 0.03, 2, 50, 0.7, 0.7),
        ("squared_error", 0.9, 140, 0.05, 3, 100, 0.5, 0.4),
        ("squared_error", 0.9, 100, 0.08, 3, 200, 0.5, 0.7),
        ("huber", 0.9, 140, 0.05, 2, 100, 0.7, 0.4),
    ),
}


class Phase4ProtocolError(ValueError):
    """The approved Phase 4 protocol or its file boundary failed validation."""


@dataclass(frozen=True, slots=True)
class RandomForestCandidate:
    """One immutable, explicitly approved Random Forest configuration."""

    n_estimators: int
    max_leaf_nodes: int
    min_samples_leaf: int
    max_features: float
    max_samples: float


@dataclass(frozen=True, slots=True)
class GradientBoostingCandidate:
    """One immutable, explicitly approved Gradient Boosting configuration."""

    loss: Literal["squared_error", "huber"]
    alpha: float
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_leaf: int
    subsample: float
    max_features: float


@dataclass(frozen=True, slots=True)
class TrackPhase4Protocol:
    """The immutable trainer-facing subset for one independent target track."""

    name: TrackName
    source_id: str
    phase3_train_rows: int
    development_rows: int | None
    calibration_rows: int | None
    legacy_holdout_rows: int
    calibration_seed: int
    screening_sample_seed: int
    random_forest_seed: int
    gradient_boosting_seed: int
    permutation_importance_seed: int
    screening_fraction_numerator: int
    screening_fraction_denominator: int
    screening_cv_folds: int
    random_forest_candidates: tuple[RandomForestCandidate, ...]
    gradient_boosting_candidates: tuple[GradientBoostingCandidate, ...]


@dataclass(frozen=True, slots=True)
class ResourceBudgets:
    """Approved training and serving ceilings relevant to orchestration."""

    maximum_peak_rss_gb: int
    target_peak_rss_gb: int
    maximum_wall_clock_hours_per_track: int
    maximum_private_artifact_mb: int
    maximum_warm_resident_memory_mb: int
    maximum_startup_peak_mb: int
    maximum_single_row_p95_ms: int


@dataclass(frozen=True, slots=True)
class Phase4Protocol:
    """Validated immutable Phase 4 policy; source rows are never retained."""

    policy_id: str
    master_seed_label: str
    final_evaluation_name: str
    tracks: tuple[TrackPhase4Protocol, TrackPhase4Protocol]
    budgets: ResourceBudgets

    def for_track(self, name: TrackName) -> TrackPhase4Protocol:
        """Return a track policy without exposing a mutable mapping."""

        for track in self.tracks:
            if track.name == name:
                return track
        raise Phase4ProtocolError(f"unsupported Phase 4 track: {name!r}")


def derive_phase4_seed(master_seed_label: str, track: TrackName, purpose: str) -> int:
    """Derive the approved unsigned 32-bit seed from a canonical purpose label."""

    if master_seed_label != _MASTER_SEED_LABEL:
        raise Phase4ProtocolError("master seed label is not approved")
    if track not in {"retail", "wholesale"}:
        raise Phase4ProtocolError("seed track is not approved")
    if purpose not in _PURPOSES:
        raise Phase4ProtocolError("seed purpose is not approved")
    payload = f"{master_seed_label}|{track}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def verify_phase4_protocol_sha256(
    path: str | os.PathLike[str],
    expected_sha256: str = PHASE4_PROTOCOL_SHA256,
) -> str:
    """Verify one regular, non-symlink protocol file and return its SHA-256."""

    expected = _digest(expected_sha256, label="expected protocol SHA-256")
    _, actual = _read_protocol_file(Path(path))
    if actual != expected:
        raise Phase4ProtocolError("Phase 4 protocol SHA-256 does not match the approved file")
    return actual


def load_phase4_protocol(
    path: str | os.PathLike[str],
    expected_sha256: str = PHASE4_PROTOCOL_SHA256,
) -> Phase4Protocol:
    """Read, checksum, parse, and validate an explicitly supplied protocol path."""

    expected = _digest(expected_sha256, label="expected protocol SHA-256")
    serialized, actual = _read_protocol_file(Path(path))
    if actual != expected:
        raise Phase4ProtocolError("Phase 4 protocol SHA-256 does not match the approved file")
    return parse_phase4_protocol_json(serialized)


def parse_phase4_protocol_json(serialized: str | bytes) -> Phase4Protocol:
    """Parse strict JSON and validate every approved Phase 4 policy boundary."""

    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_PROTOCOL_BYTES:
            raise Phase4ProtocolError("Phase 4 protocol exceeds the maximum size")
        try:
            text = serialized.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise Phase4ProtocolError("Phase 4 protocol must be UTF-8") from error
    elif isinstance(serialized, str):
        try:
            encoded = serialized.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise Phase4ProtocolError("Phase 4 protocol must be UTF-8") from error
        if len(encoded) > _MAX_PROTOCOL_BYTES:
            raise Phase4ProtocolError("Phase 4 protocol exceeds the maximum size")
        text = serialized
    else:
        raise Phase4ProtocolError("serialized Phase 4 protocol must be text or bytes")

    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Phase4ProtocolError("Phase 4 protocol is not strict JSON") from error
    return validate_phase4_protocol(payload)


def validate_phase4_protocol(payload: object) -> Phase4Protocol:
    """Validate a decoded protocol and return only immutable typed structures."""

    root = _object(
        payload,
        keys={
            "schema_version",
            "policy_id",
            "reviewed_on",
            "decision",
            "master_seed_label",
            "seed_derivation",
            "purpose",
            "holdout_policy",
            "tracks",
            "preprocessing",
            "candidate_families",
            "search_budget",
            "selection",
            "deployment_gates",
            "prediction_range",
            "feature_importance",
            "artifact_policy",
        },
        label="protocol",
    )
    _equal(_integer(root["schema_version"], label="schema_version"), 1, "schema_version")
    _equal(_text(root["policy_id"], label="policy_id"), _POLICY_ID, "policy_id")
    _equal(_text(root["reviewed_on"], label="reviewed_on"), "2026-08-29", "reviewed_on")
    _equal(
        _text(root["decision"], label="decision"),
        "approved_for_private_local_implementation",
        "decision",
    )
    master_seed = _text(root["master_seed_label"], label="master_seed_label")
    _equal(master_seed, _MASTER_SEED_LABEL, "master_seed_label")
    _nonempty_text(root["purpose"], label="purpose")
    _validate_seed_derivation(root["seed_derivation"])
    final_evaluation_name = _validate_holdout_policy(root["holdout_policy"])

    track_cores = _validate_tracks(root["tracks"], master_seed=master_seed)
    _validate_preprocessing(root["preprocessing"])
    rf_candidates, gb_candidates = _validate_candidate_families(root["candidate_families"])
    search = _validate_search_budget(root["search_budget"])
    _validate_selection(root["selection"])
    deployment = _validate_deployment_gates(root["deployment_gates"])
    _validate_prediction_range(root["prediction_range"])
    _validate_feature_importance(root["feature_importance"], final_evaluation_name)
    _validate_artifact_policy(root["artifact_policy"])

    built_tracks: list[TrackPhase4Protocol] = []
    for name in _TRACK_ORDER:
        core = track_cores[name]
        built_tracks.append(
            TrackPhase4Protocol(
                name=cast(TrackName, core["name"]),
                source_id=cast(str, core["source_id"]),
                phase3_train_rows=cast(int, core["phase3_train_rows"]),
                development_rows=cast(int | None, core["development_rows"]),
                calibration_rows=cast(int | None, core["calibration_rows"]),
                legacy_holdout_rows=cast(int, core["legacy_holdout_rows"]),
                calibration_seed=cast(int, core["calibration_seed"]),
                screening_sample_seed=cast(int, core["screening_sample_seed"]),
                random_forest_seed=cast(int, core["random_forest_seed"]),
                gradient_boosting_seed=cast(int, core["gradient_boosting_seed"]),
                permutation_importance_seed=cast(int, core["permutation_importance_seed"]),
                screening_fraction_numerator=cast(int, core["screening_fraction_numerator"]),
                screening_fraction_denominator=cast(int, core["screening_fraction_denominator"]),
                screening_cv_folds=cast(int, core["screening_cv_folds"]),
                random_forest_candidates=rf_candidates[name],
                gradient_boosting_candidates=gb_candidates[name],
            )
        )
    tracks = tuple(built_tracks)
    typed_tracks = cast(tuple[TrackPhase4Protocol, TrackPhase4Protocol], tracks)
    budgets = ResourceBudgets(
        maximum_peak_rss_gb=search["maximum_peak_rss_gb"],
        target_peak_rss_gb=search["target_peak_rss_gb"],
        maximum_wall_clock_hours_per_track=search["maximum_wall_clock_hours_per_track"],
        **deployment,
    )
    return Phase4Protocol(
        policy_id=_POLICY_ID,
        master_seed_label=master_seed,
        final_evaluation_name=final_evaluation_name,
        tracks=typed_tracks,
        budgets=budgets,
    )


def _validate_seed_derivation(value: object) -> None:
    policy = _object(
        value,
        keys={"algorithm", "purpose_labels"},
        label="seed_derivation",
    )
    _equal(
        _text(policy["algorithm"], label="seed_derivation.algorithm"),
        "uint32 big-endian from the first four bytes of SHA-256(master_seed_label + '|' + "
        "track + '|' + purpose)",
        "seed_derivation.algorithm",
    )
    purposes = _text_tuple(policy["purpose_labels"], label="seed_derivation.purpose_labels")
    _equal(purposes, _PURPOSES, "seed_derivation.purpose_labels")


def _validate_holdout_policy(value: object) -> str:
    policy = _object(
        value,
        keys={
            "holdout_used_for_preprocessing",
            "holdout_used_for_hyperparameter_tuning",
            "holdout_used_for_model_family_selection",
            "holdout_used_for_interval_calibration",
            "final_evaluation_name",
            "limitation",
        },
        label="holdout_policy",
    )
    for field in (
        "holdout_used_for_preprocessing",
        "holdout_used_for_hyperparameter_tuning",
        "holdout_used_for_model_family_selection",
        "holdout_used_for_interval_calibration",
    ):
        if _boolean(policy[field], label=f"holdout_policy.{field}"):
            raise Phase4ProtocolError(f"holdout_policy.{field} must remain false")
    name = _text(policy["final_evaluation_name"], label="holdout_policy.final_evaluation_name")
    _equal(name, "phase3_reused_legacy_holdout", "holdout_policy.final_evaluation_name")
    limitation = _nonempty_text(policy["limitation"], label="holdout_policy.limitation")
    if "already scored" not in limitation or "must not call either untouched" not in limitation:
        raise Phase4ProtocolError("holdout_policy.limitation must disclose prior holdout use")
    return name


def _validate_tracks(
    value: object,
    *,
    master_seed: str,
) -> dict[TrackName, dict[str, object]]:
    tracks = _object(value, keys={"retail", "wholesale"}, label="tracks")
    result: dict[TrackName, dict[str, object]] = {}
    for track in _TRACK_ORDER:
        result[track] = (
            _validate_retail_track(tracks[track], master_seed=master_seed)
            if track == "retail"
            else _validate_wholesale_track(tracks[track], master_seed=master_seed)
        )
    return result


def _validate_retail_track(value: object, *, master_seed: str) -> dict[str, object]:
    label = "tracks.retail"
    track = _object(
        value,
        keys={
            *_RETAIL_LINEAGE,
            "phase3_train_rows",
            "legacy_holdout_rows",
            "development_calibration_split",
            "development_cv",
            "screening",
            *_SEED_FIELDS.values(),
        },
        label=label,
    )
    _validate_lineage(track, _RETAIL_LINEAGE, label=label)
    phase3_rows = _exact_integer(
        track["phase3_train_rows"], 109_510, label=f"{label}.phase3_train_rows"
    )
    holdout_rows = _exact_integer(
        track["legacy_holdout_rows"], 27_589, label=f"{label}.legacy_holdout_rows"
    )
    split = _object(
        track["development_calibration_split"],
        keys={
            "algorithm",
            "group_id_algorithm",
            "seeded_rank_algorithm",
            "calibration_fraction_numerator",
            "calibration_fraction_denominator",
            "calibration_seed",
            "expected_calibration_rows_approximate",
        },
        label=f"{label}.development_calibration_split",
    )
    _exact_text(
        split["algorithm"], _RETAIL_CALIBRATION_ALGORITHM, label="retail calibration algorithm"
    )
    _exact_text(
        split["group_id_algorithm"],
        _RETAIL_GROUP_ID_ALGORITHM,
        label="retail calibration group ID algorithm",
    )
    _exact_text(
        split["seeded_rank_algorithm"],
        _RETAIL_CALIBRATION_RANK_ALGORITHM,
        label="retail calibration rank algorithm",
    )
    _exact_integer(split["calibration_fraction_numerator"], 1, label="retail numerator")
    _exact_integer(split["calibration_fraction_denominator"], 10, label="retail denominator")
    calibration_seed = _derived_seed(
        split["calibration_seed"], master_seed, "retail", "calibration"
    )
    _exact_integer(
        split["expected_calibration_rows_approximate"],
        10_951,
        label="retail expected calibration rows",
    )
    cv = _object(
        track["development_cv"],
        keys={"scheme", "folds", "shuffle"},
        label=f"{label}.development_cv",
    )
    _exact_text(cv["scheme"], "predictor_group_kfold", label="retail CV scheme")
    _exact_integer(cv["folds"], 5, label="retail CV folds")
    if _boolean(cv["shuffle"], label="retail CV shuffle"):
        raise Phase4ProtocolError("retail CV shuffle must remain false")
    screening_numerator, screening_denominator, screening_folds = _validate_retail_screening(
        track["screening"]
    )
    seeds = _track_seeds(track, master_seed=master_seed, track="retail")
    return {
        "name": "retail",
        "source_id": _RETAIL_LINEAGE["source_id"],
        "phase3_train_rows": phase3_rows,
        "development_rows": None,
        "calibration_rows": None,
        "legacy_holdout_rows": holdout_rows,
        "calibration_seed": calibration_seed,
        "screening_fraction_numerator": screening_numerator,
        "screening_fraction_denominator": screening_denominator,
        "screening_cv_folds": screening_folds,
        **seeds,
    }


def _validate_wholesale_track(value: object, *, master_seed: str) -> dict[str, object]:
    label = "tracks.wholesale"
    track = _object(
        value,
        keys={
            *_WHOLESALE_LINEAGE,
            "phase3_train_rows",
            "legacy_holdout_rows",
            "development_calibration_split",
            "development_cv",
            "screening",
            *_SEED_FIELDS.values(),
        },
        label=label,
    )
    _validate_lineage(track, _WHOLESALE_LINEAGE, label=label)
    phase3_rows = _exact_integer(track["phase3_train_rows"], 442_130, label="wholesale train rows")
    holdout_rows = _exact_integer(
        track["legacy_holdout_rows"], 98_634, label="wholesale holdout rows"
    )
    split = _object(
        track["development_calibration_split"],
        keys={
            "development_buckets",
            "development_rows",
            "calibration_bucket",
            "calibration_rows",
            "calibration_seed",
        },
        label=f"{label}.development_calibration_split",
    )
    buckets = _text_tuple(split["development_buckets"], label="wholesale development buckets")
    _equal(buckets, _WHOLESALE_DEVELOPMENT_BUCKETS, "wholesale development buckets")
    development_rows = _exact_integer(
        split["development_rows"], 391_641, label="wholesale development rows"
    )
    _exact_text(split["calibration_bucket"], "2015_05", label="wholesale calibration bucket")
    calibration_rows = _exact_integer(
        split["calibration_rows"], 50_489, label="wholesale calibration rows"
    )
    if development_rows + calibration_rows != phase3_rows:
        raise Phase4ProtocolError(
            "wholesale development and calibration row counts do not reconcile"
        )
    calibration_seed = _derived_seed(
        split["calibration_seed"], master_seed, "wholesale", "calibration"
    )
    cv = _object(
        track["development_cv"],
        keys={"scheme", "bucket_order", "validation_folds"},
        label=f"{label}.development_cv",
    )
    _exact_text(cv["scheme"], "forward_chaining_cv_bucket", label="wholesale CV scheme")
    _equal(
        _text_tuple(cv["bucket_order"], label="wholesale CV bucket_order"),
        buckets,
        "wholesale CV bucket_order",
    )
    _exact_integer(cv["validation_folds"], 3, label="wholesale validation folds")
    screening_numerator, screening_denominator, screening_folds = _validate_wholesale_screening(
        track["screening"]
    )
    seeds = _track_seeds(track, master_seed=master_seed, track="wholesale")
    return {
        "name": "wholesale",
        "source_id": _WHOLESALE_LINEAGE["source_id"],
        "phase3_train_rows": phase3_rows,
        "development_rows": development_rows,
        "calibration_rows": calibration_rows,
        "legacy_holdout_rows": holdout_rows,
        "calibration_seed": calibration_seed,
        "screening_fraction_numerator": screening_numerator,
        "screening_fraction_denominator": screening_denominator,
        "screening_cv_folds": screening_folds,
        **seeds,
    }


def _validate_retail_screening(value: object) -> tuple[int, int, int]:
    screening = _object(
        value,
        keys={
            "input",
            "algorithm",
            "group_id_algorithm",
            "seeded_rank_algorithm",
            "sample_fraction_numerator",
            "sample_fraction_denominator",
            "cv_scheme",
            "cv_folds",
            "shuffle",
        },
        label="tracks.retail.screening",
    )
    _exact_text(screening["input"], "development_partition_only", label="retail screening input")
    _exact_text(
        screening["algorithm"], _RETAIL_SCREENING_ALGORITHM, label="retail screening algorithm"
    )
    _exact_text(
        screening["group_id_algorithm"],
        "same retail_predictor_groups v2 group_id used by development_calibration_split",
        label="retail screening group ID algorithm",
    )
    _exact_text(
        screening["seeded_rank_algorithm"],
        _RETAIL_SCREENING_RANK_ALGORITHM,
        label="retail screening rank algorithm",
    )
    numerator = _exact_integer(
        screening["sample_fraction_numerator"], 3, label="retail screening numerator"
    )
    denominator = _exact_integer(
        screening["sample_fraction_denominator"], 10, label="retail screening denominator"
    )
    _exact_text(screening["cv_scheme"], "predictor_group_kfold", label="retail screening CV scheme")
    folds = _exact_integer(screening["cv_folds"], 5, label="retail screening CV folds")
    if _boolean(screening["shuffle"], label="retail screening shuffle"):
        raise Phase4ProtocolError("retail screening CV shuffle must remain false")
    return numerator, denominator, folds


def _validate_wholesale_screening(value: object) -> tuple[int, int, int]:
    screening = _object(
        value,
        keys={
            "input",
            "algorithm",
            "seeded_rank_algorithm",
            "sample_fraction_numerator",
            "sample_fraction_denominator",
            "bucket_order",
            "cv_scheme",
            "validation_folds",
        },
        label="tracks.wholesale.screening",
    )
    _exact_text(screening["input"], "development_partition_only", label="wholesale screening input")
    _exact_text(
        screening["algorithm"],
        _WHOLESALE_SCREENING_ALGORITHM,
        label="wholesale screening algorithm",
    )
    _exact_text(
        screening["seeded_rank_algorithm"],
        _WHOLESALE_SCREENING_RANK_ALGORITHM,
        label="wholesale screening rank algorithm",
    )
    numerator = _exact_integer(
        screening["sample_fraction_numerator"], 1, label="wholesale screening numerator"
    )
    denominator = _exact_integer(
        screening["sample_fraction_denominator"], 4, label="wholesale screening denominator"
    )
    _equal(
        _text_tuple(screening["bucket_order"], label="wholesale screening bucket order"),
        _WHOLESALE_DEVELOPMENT_BUCKETS,
        "wholesale screening bucket order",
    )
    _exact_text(
        screening["cv_scheme"],
        "forward_chaining_cv_bucket",
        label="wholesale screening CV scheme",
    )
    folds = _exact_integer(
        screening["validation_folds"], 3, label="wholesale screening validation folds"
    )
    return numerator, denominator, folds


def _validate_lineage(
    actual: Mapping[str, object], expected: Mapping[str, str], *, label: str
) -> None:
    for field, expected_value in expected.items():
        observed = _text(actual[field], label=f"{label}.{field}")
        if field.endswith("sha256") or field == "split_artifact_set_id":
            _digest(observed, label=f"{label}.{field}")
        _equal(observed, expected_value, f"{label}.{field}")


def _track_seeds(
    value: Mapping[str, object], *, master_seed: str, track: TrackName
) -> dict[str, int]:
    return {
        field: _derived_seed(value[field], master_seed, track, purpose)
        for purpose, field in _SEED_FIELDS.items()
    }


def _derived_seed(value: object, master_seed: str, track: TrackName, purpose: str) -> int:
    observed = _integer(value, label=f"{track}.{purpose}_seed", minimum=0)
    expected = derive_phase4_seed(master_seed, track, purpose)
    if observed != expected:
        raise Phase4ProtocolError(f"{track}.{purpose}_seed does not match SHA-256 derivation")
    return observed


def _validate_preprocessing(value: object) -> None:
    policy = _object(
        value,
        keys={
            "fit_scope",
            "feature_engineering",
            "numeric",
            "categorical",
            "matrix",
            "dense_conversion_forbidden",
            "target_clipping",
            "target_log_transform",
        },
        label="preprocessing",
    )
    expected_text = {
        "fit_scope": "inside_each_development_fold",
        "feature_engineering": "existing_track_v2",
        "numeric": "median_imputation_without_scaling",
        "categorical": "constant_missing_imputation_then_capped_infrequent_one_hot_encoding",
        "matrix": "csr_float32",
    }
    for field, expected in expected_text.items():
        _exact_text(policy[field], expected, label=f"preprocessing.{field}")
    if not _boolean(
        policy["dense_conversion_forbidden"], label="preprocessing.dense_conversion_forbidden"
    ):
        raise Phase4ProtocolError("dense matrix conversion must remain forbidden")
    for field in ("target_clipping", "target_log_transform"):
        if _boolean(policy[field], label=f"preprocessing.{field}"):
            raise Phase4ProtocolError(f"preprocessing.{field} must remain false")


def _validate_candidate_families(
    value: object,
) -> tuple[
    dict[TrackName, tuple[RandomForestCandidate, ...]],
    dict[TrackName, tuple[GradientBoostingCandidate, ...]],
]:
    families = _object(
        value,
        keys={"linear_regression_incumbent", "random_forest", "gradient_boosting", "xgboost"},
        label="candidate_families",
    )
    linear = _object(
        families["linear_regression_incumbent"],
        keys={"role", "configuration"},
        label="linear_regression_incumbent",
    )
    _exact_text(linear["role"], "same-fold_reference", label="linear role")
    _exact_text(linear["configuration"], "phase3_v2_linear_pipeline", label="linear configuration")
    rf = _validate_random_forest(families["random_forest"])
    gb = _validate_gradient_boosting(families["gradient_boosting"])
    xgboost = _object(
        families["xgboost"], keys={"status", "reason"}, label="candidate_families.xgboost"
    )
    _exact_text(xgboost["status"], "deferred_optional", label="xgboost status")
    reason = _nonempty_text(xgboost["reason"], label="xgboost reason")
    if (
        "not currently a project dependency" not in reason
        or "Review and pin it separately" not in reason
    ):
        raise Phase4ProtocolError("XGBoost must remain deferred pending a separate review")
    return rf, gb


def _validate_random_forest(
    value: object,
) -> dict[TrackName, tuple[RandomForestCandidate, ...]]:
    family = _object(
        value,
        keys={"common", "retail", "wholesale", "tuple_fields"},
        label="random_forest",
    )
    _equal(
        _text_tuple(family["tuple_fields"], label="RF tuple_fields"), _RF_FIELDS, "RF tuple_fields"
    )
    common = _object(
        family["common"],
        keys={"criterion", "bootstrap", "max_depth", "n_jobs_training", "n_jobs_serving"},
        label="random_forest.common",
    )
    _exact_text(common["criterion"], "squared_error", label="RF criterion")
    if not _boolean(common["bootstrap"], label="RF bootstrap"):
        raise Phase4ProtocolError("Random Forest bootstrap must remain enabled")
    if common["max_depth"] is not None:
        raise Phase4ProtocolError("Random Forest max_depth must be null")
    _exact_integer(common["n_jobs_training"], 4, label="RF training jobs")
    _exact_integer(common["n_jobs_serving"], 1, label="RF serving jobs")

    result: dict[TrackName, tuple[RandomForestCandidate, ...]] = {}
    for name in _TRACK_ORDER:
        rows = _list(family[name], label=f"random_forest.{name}")
        if len(rows) != 6:
            raise Phase4ProtocolError(f"random_forest.{name} must contain exactly six candidates")
        candidates = tuple(
            _random_forest_candidate(row, label=f"random_forest.{name}[{index}]")
            for index, row in enumerate(rows)
        )
        _unique_candidates(candidates, label=f"random_forest.{name}")
        expected = tuple(RandomForestCandidate(*item) for item in _EXPECTED_RF[name])
        _equal(candidates, expected, f"random_forest.{name} approved candidates")
        result[name] = candidates
    return result


def _random_forest_candidate(value: object, *, label: str) -> RandomForestCandidate:
    values = _list(value, label=label)
    if len(values) != len(_RF_FIELDS):
        raise Phase4ProtocolError(f"{label} must contain exactly {len(_RF_FIELDS)} values")
    n_estimators = _integer(values[0], label=f"{label}.n_estimators", minimum=32, maximum=160)
    max_leaf_nodes = _integer(values[1], label=f"{label}.max_leaf_nodes", minimum=128, maximum=2048)
    min_samples_leaf = _integer(
        values[2], label=f"{label}.min_samples_leaf", minimum=1, maximum=200
    )
    max_features = _float(values[3], label=f"{label}.max_features", minimum=0.1, maximum=1.0)
    max_samples = _float(values[4], label=f"{label}.max_samples", minimum=0.1, maximum=1.0)
    return RandomForestCandidate(
        n_estimators, max_leaf_nodes, min_samples_leaf, max_features, max_samples
    )


def _validate_gradient_boosting(
    value: object,
) -> dict[TrackName, tuple[GradientBoostingCandidate, ...]]:
    family = _object(
        value,
        keys={"common", "retail", "wholesale", "tuple_fields"},
        label="gradient_boosting",
    )
    _equal(
        _text_tuple(family["tuple_fields"], label="GB tuple_fields"), _GB_FIELDS, "GB tuple_fields"
    )
    common = _object(family["common"], keys={"n_iter_no_change"}, label="gradient_boosting.common")
    if common["n_iter_no_change"] is not None:
        raise Phase4ProtocolError("Gradient Boosting built-in early stopping must remain disabled")

    result: dict[TrackName, tuple[GradientBoostingCandidate, ...]] = {}
    for name in _TRACK_ORDER:
        rows = _list(family[name], label=f"gradient_boosting.{name}")
        if len(rows) != 6:
            raise Phase4ProtocolError(
                f"gradient_boosting.{name} must contain exactly six candidates"
            )
        candidates = tuple(
            _gradient_boosting_candidate(row, label=f"gradient_boosting.{name}[{index}]")
            for index, row in enumerate(rows)
        )
        _unique_candidates(candidates, label=f"gradient_boosting.{name}")
        expected = tuple(
            _gradient_boosting_candidate(list(item), label="approved Gradient Boosting candidate")
            for item in _EXPECTED_GB[name]
        )
        _equal(candidates, expected, f"gradient_boosting.{name} approved candidates")
        result[name] = candidates
    return result


def _gradient_boosting_candidate(value: object, *, label: str) -> GradientBoostingCandidate:
    values = _list(value, label=label)
    if len(values) != len(_GB_FIELDS):
        raise Phase4ProtocolError(f"{label} must contain exactly {len(_GB_FIELDS)} values")
    loss = _text(values[0], label=f"{label}.loss")
    if loss not in {"squared_error", "huber"}:
        raise Phase4ProtocolError(f"{label}.loss is not approved")
    alpha = _float(values[1], label=f"{label}.alpha", minimum=0.5, maximum=0.99)
    n_estimators = _integer(values[2], label=f"{label}.n_estimators", minimum=32, maximum=240)
    learning_rate = _float(values[3], label=f"{label}.learning_rate", minimum=0.001, maximum=0.2)
    max_depth = _integer(values[4], label=f"{label}.max_depth", minimum=1, maximum=3)
    min_samples_leaf = _integer(
        values[5], label=f"{label}.min_samples_leaf", minimum=1, maximum=500
    )
    subsample = _float(values[6], label=f"{label}.subsample", minimum=0.1, maximum=1.0)
    max_features = _float(values[7], label=f"{label}.max_features", minimum=0.1, maximum=1.0)
    return GradientBoostingCandidate(
        cast(Literal["squared_error", "huber"], loss),
        alpha,
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
        subsample,
        max_features,
    )


def _validate_search_budget(value: object) -> dict[str, int]:
    budget = _object(
        value,
        keys={
            "parallel_candidate_fits",
            "pre_dispatch",
            "native_threads",
            "random_forest_internal_jobs",
            "maximum_peak_rss_gb",
            "target_peak_rss_gb",
            "maximum_wall_clock_hours_per_track",
            "screen_all_explicit_candidates_on_target_free_group_or_time_safe_sample",
            "confirm_top_candidates_per_family_on_full_development_cv",
            "screening_shortlist",
        },
        label="search_budget",
    )
    exact = {
        "parallel_candidate_fits": 1,
        "pre_dispatch": 1,
        "native_threads": 1,
        "random_forest_internal_jobs": 4,
        "maximum_peak_rss_gb": 8,
        "target_peak_rss_gb": 6,
        "maximum_wall_clock_hours_per_track": 4,
        "confirm_top_candidates_per_family_on_full_development_cv": 2,
    }
    values = {
        field: _exact_integer(budget[field], expected, label=f"search_budget.{field}")
        for field, expected in exact.items()
    }
    if values["target_peak_rss_gb"] > values["maximum_peak_rss_gb"]:
        raise Phase4ProtocolError("target peak RSS cannot exceed the hard maximum")
    if not _boolean(
        budget["screen_all_explicit_candidates_on_target_free_group_or_time_safe_sample"],
        label="search_budget.safe_screening",
    ):
        raise Phase4ProtocolError("candidate screening must remain target-free and split-safe")
    _validate_screening_shortlist(budget["screening_shortlist"])
    return values


def _validate_screening_shortlist(value: object) -> None:
    shortlist = _object(
        value,
        keys={
            "metric",
            "ranking",
            "separate_by_family",
            "challengers_per_family",
            "linear_incumbent",
            "promotion_metrics",
        },
        label="search_budget.screening_shortlist",
    )
    expected_text = {
        "metric": "micro_out_of_fold_mae_usd",
        "ranking": "ascending exact finite float, then ascending stable candidate_id",
        "linear_incumbent": (
            "evaluate on the same screening and full-development folds as a reference; it is "
            "not part of either family shortlist"
        ),
        "promotion_metrics": "only full-development CV metrics may promote a model",
    }
    for field, expected in expected_text.items():
        _exact_text(shortlist[field], expected, label=f"screening_shortlist.{field}")
    if not _boolean(shortlist["separate_by_family"], label="shortlist separate_by_family"):
        raise Phase4ProtocolError("screening shortlists must remain separate by family")
    _exact_integer(shortlist["challengers_per_family"], 2, label="shortlist candidate count")


def _validate_selection(value: object) -> None:
    selection = _object(
        value,
        keys={
            "primary_metric",
            "retail_material_gain",
            "wholesale_material_gain",
            "near_tie_relative_mae",
            "near_tie_breakers",
            "fallback",
        },
        label="selection",
    )
    _exact_text(selection["primary_metric"], "micro_out_of_fold_mae_usd", label="selection metric")
    _validate_gain(
        selection["retail_material_gain"],
        label="retail_material_gain",
        expected={
            "minimum_relative_mae_improvement": 0.03,
            "minimum_absolute_mae_improvement_usd": 300,
            "maximum_status_mae_regression": 0.05,
            "maximum_rmse_regression": 0.05,
        },
    )
    _validate_gain(
        selection["wholesale_material_gain"],
        label="wholesale_material_gain",
        expected={
            "minimum_relative_mae_improvement": 0.02,
            "minimum_absolute_mae_improvement_usd": 50,
            "maximum_latest_fold_mae_regression": 0.05,
            "maximum_rmse_regression": 0.05,
        },
    )
    _exact_float(selection["near_tie_relative_mae"], 0.01, label="near tie MAE")
    _equal(
        _text_tuple(selection["near_tie_breakers"], label="near_tie_breakers"),
        ("smaller_artifact", "lower_single_row_p95_latency", "stable_model_id"),
        "near_tie_breakers",
    )
    fallback = _nonempty_text(selection["fallback"], label="selection.fallback")
    if not fallback.startswith("Keep Linear Regression"):
        raise Phase4ProtocolError("selection fallback must keep the incumbent")


def _validate_gain(value: object, *, label: str, expected: Mapping[str, int | float]) -> None:
    gain = _object(value, keys=set(expected), label=label)
    for field, expected_value in expected.items():
        if type(expected_value) is int:
            _exact_integer(gain[field], expected_value, label=f"{label}.{field}")
        else:
            _exact_float(gain[field], cast(float, expected_value), label=f"{label}.{field}")


def _validate_deployment_gates(value: object) -> dict[str, int]:
    gates = _object(
        value,
        keys={
            "maximum_private_artifact_mb",
            "maximum_warm_resident_memory_mb",
            "maximum_startup_peak_mb",
            "maximum_single_row_p95_ms",
            "serving_workers",
            "serving_native_threads",
        },
        label="deployment_gates",
    )
    expected = {
        "maximum_private_artifact_mb": 50,
        "maximum_warm_resident_memory_mb": 350,
        "maximum_startup_peak_mb": 450,
        "maximum_single_row_p95_ms": 500,
        "serving_workers": 1,
        "serving_native_threads": 1,
    }
    parsed = {
        field: _exact_integer(gates[field], expected_value, label=f"deployment_gates.{field}")
        for field, expected_value in expected.items()
    }
    return {
        field: parsed[field] for field in ResourceBudgets.__dataclass_fields__ if field in parsed
    }


def _validate_prediction_range(value: object) -> None:
    policy = _object(
        value,
        keys={
            "method",
            "alpha",
            "finite_sample_order",
            "lower_bound",
            "upper_bound",
            "retail",
            "wholesale",
            "minimum_overall_observed_coverage",
            "minimum_retail_status_coverage",
            "label",
        },
        label="prediction_range",
    )
    expected_text = {
        "method": "split_conformal_absolute_residual",
        "finite_sample_order": "ceil((n_calibration + 1) * (1 - alpha))",
        "lower_bound": "max(0, prediction - quantile)",
        "upper_bound": "prediction + quantile",
        "retail": "status-specific quantiles for certified/new/used with a global fallback",
        "wholesale": "one quantile from the 2015_05 calibration bucket",
        "label": "estimated_prediction_range_not_statistical_confidence",
    }
    for field, expected in expected_text.items():
        _exact_text(policy[field], expected, label=f"prediction_range.{field}")
    _exact_float(policy["alpha"], 0.1, label="prediction_range.alpha")
    _exact_float(
        policy["minimum_overall_observed_coverage"],
        0.88,
        label="prediction_range.minimum_overall_observed_coverage",
    )
    _exact_float(
        policy["minimum_retail_status_coverage"],
        0.85,
        label="prediction_range.minimum_retail_status_coverage",
    )


def _validate_feature_importance(value: object, holdout_name: str) -> None:
    policy = _object(
        value,
        keys={
            "method",
            "holdout_role",
            "retail_sample_rows",
            "wholesale_sample_rows",
            "repeats",
            "n_jobs",
            "output",
            "local_explanation_claim",
            "raw_linear_coefficients_forbidden_as_importance",
        },
        label="feature_importance",
    )
    _exact_text(
        policy["method"],
        "permutation_drop_in_negative_mae_on_raw_feature_families",
        label="feature importance method",
    )
    _exact_text(policy["holdout_role"], holdout_name, label="feature importance holdout")
    _exact_integer(policy["retail_sample_rows"], 10_000, label="retail importance rows")
    _exact_integer(policy["wholesale_sample_rows"], 20_000, label="wholesale importance rows")
    _exact_integer(policy["repeats"], 5, label="feature importance repeats")
    _exact_integer(policy["n_jobs"], 1, label="feature importance jobs")
    _exact_text(
        policy["output"], "aggregate_mean_and_standard_deviation_only", label="importance output"
    )
    if _boolean(policy["local_explanation_claim"], label="local explanation claim"):
        raise Phase4ProtocolError("permutation importance cannot be a local explanation claim")
    if not _boolean(
        policy["raw_linear_coefficients_forbidden_as_importance"],
        label="raw coefficient importance policy",
    ):
        raise Phase4ProtocolError("raw linear coefficients must remain forbidden as importance")


def _validate_artifact_policy(value: object) -> None:
    policy = _object(
        value,
        keys={
            "storage",
            "downloadable_publication",
            "hosted_inference",
            "trusted_local_joblib_only",
            "user_uploaded_or_remote_artifacts_forbidden",
            "verify_sha256_before_load",
            "persist_source_rows_predictions_residuals_or_category_vocabulary_in_public_reports",
        },
        label="artifact_policy",
    )
    expected_text = {
        "storage": "private_git_ignored_models_directory",
        "downloadable_publication": "pending_new_permission_review",
        "hosted_inference": "approved_with_existing_scoped_reviews",
    }
    for field, expected in expected_text.items():
        _exact_text(policy[field], expected, label=f"artifact_policy.{field}")
    for field in (
        "trusted_local_joblib_only",
        "user_uploaded_or_remote_artifacts_forbidden",
        "verify_sha256_before_load",
    ):
        if not _boolean(policy[field], label=f"artifact_policy.{field}"):
            raise Phase4ProtocolError(f"artifact_policy.{field} must remain true")
    public_rows = (
        "persist_source_rows_predictions_residuals_or_category_vocabulary_in_public_reports"
    )
    if _boolean(policy[public_rows], label=f"artifact_policy.{public_rows}"):
        raise Phase4ProtocolError("public reports must remain aggregate-only")


def _read_protocol_file(path: Path) -> tuple[bytes, str]:
    try:
        before = path.lstat()
    except OSError as error:
        raise Phase4ProtocolError("Phase 4 protocol file is not accessible") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Phase4ProtocolError("Phase 4 protocol must be a regular non-symlink file")
    if before.st_size <= 0 or before.st_size > _MAX_PROTOCOL_BYTES:
        raise Phase4ProtocolError("Phase 4 protocol file size is invalid")
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise Phase4ProtocolError("Phase 4 protocol changed before it was read")
            serialized = stream.read(_MAX_PROTOCOL_BYTES + 1)
        after = path.lstat()
    except Phase4ProtocolError:
        raise
    except OSError as error:
        raise Phase4ProtocolError("Phase 4 protocol could not be read") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_opened or identity_opened != identity_after:
        raise Phase4ProtocolError("Phase 4 protocol changed while it was read")
    if len(serialized) != before.st_size or len(serialized) > _MAX_PROTOCOL_BYTES:
        raise Phase4ProtocolError("Phase 4 protocol changed size while it was read")
    return serialized, hashlib.sha256(serialized).hexdigest()


def _object(value: object, *, keys: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise Phase4ProtocolError(f"{label} must be an object")
    result = cast(dict[object, object], value)
    if any(type(key) is not str for key in result):
        raise Phase4ProtocolError(f"{label} keys must be strings")
    typed = cast(dict[str, object], result)
    actual = set(typed)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected {', '.join(extra)}")
        raise Phase4ProtocolError(f"{label} has invalid fields: {'; '.join(detail)}")
    return typed


def _list(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        raise Phase4ProtocolError(f"{label} must be an array")
    return cast(list[object], value)


def _text_tuple(value: object, *, label: str) -> tuple[str, ...]:
    values = _list(value, label=label)
    return tuple(_text(item, label=f"{label}[{index}]") for index, item in enumerate(values))


def _text(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise Phase4ProtocolError(f"{label} must be text")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    result = _text(value, label=label)
    if not result or result != result.strip() or "\x00" in result:
        raise Phase4ProtocolError(f"{label} must be canonical non-empty text")
    return result


def _exact_text(value: object, expected: str, *, label: str) -> str:
    observed = _text(value, label=label)
    _equal(observed, expected, label)
    return observed


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise Phase4ProtocolError(f"{label} must be a boolean")
    return value


def _integer(
    value: object,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise Phase4ProtocolError(f"{label} must be an integer")
    result = value
    if minimum is not None and result < minimum:
        raise Phase4ProtocolError(f"{label} is below its approved minimum")
    if maximum is not None and result > maximum:
        raise Phase4ProtocolError(f"{label} exceeds its approved maximum")
    return result


def _exact_integer(value: object, expected: int, *, label: str) -> int:
    observed = _integer(value, label=label)
    _equal(observed, expected, label)
    return observed


def _float(value: object, *, label: str, minimum: float, maximum: float) -> float:
    if type(value) is not float:
        raise Phase4ProtocolError(f"{label} must be a JSON floating-point number")
    result = value
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise Phase4ProtocolError(f"{label} is outside its approved range")
    return result


def _exact_float(value: object, expected: float, *, label: str) -> float:
    observed = _float(value, label=label, minimum=0.0, maximum=1.0)
    _equal(observed, expected, label)
    return observed


def _digest(value: object, *, label: str) -> str:
    digest = _text(value, label=label)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise Phase4ProtocolError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _unique_candidates(value: Sequence[object], *, label: str) -> None:
    if len(value) != len(set(value)):
        raise Phase4ProtocolError(f"{label} candidates must be unique")


def _equal(observed: object, expected: object, label: str) -> None:
    if observed != expected:
        raise Phase4ProtocolError(f"{label} differs from the approved Phase 4 policy")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise Phase4ProtocolError(f"non-finite JSON constant is forbidden: {value}")


__all__ = [
    "GradientBoostingCandidate",
    "PHASE4_PROTOCOL_SHA256",
    "Phase4Protocol",
    "Phase4ProtocolError",
    "RandomForestCandidate",
    "ResourceBudgets",
    "TrackPhase4Protocol",
    "derive_phase4_seed",
    "load_phase4_protocol",
    "parse_phase4_protocol_json",
    "validate_phase4_protocol",
    "verify_phase4_protocol_sha256",
]
