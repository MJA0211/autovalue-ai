"""Frozen, development-only orchestration for Phase 4 candidate screening."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .calibration import retail_calibration_partition, wholesale_calibration_partition
from .candidates import candidate_specs
from .contracts import (
    RETAIL_TRACK,
    TRACKS,
    TrackConfig,
    TrackName,
    validate_feature_frame,
    validate_target,
)
from .cv import retail_group_cv_splits, wholesale_forward_cv_splits
from .phase4_evaluation import (
    Phase4CandidateCVResult,
    Phase4Shortlist,
    evaluate_phase4_candidate_cv,
    parse_phase4_candidate_cv_result,
    shortlist_phase4_candidates,
)
from .phase4_protocol import PHASE4_PROTOCOL_SHA256, Phase4Protocol, TrackPhase4Protocol
from .screening import retail_screening_sample, wholesale_screening_sample

ScreeningCVScheme: TypeAlias = Literal[
    "retail_predictor_group_kfold_v1",
    "wholesale_forward_chaining_cv_bucket_v1",
]
CandidateProgressCallback: TypeAlias = Callable[[tuple[Phase4CandidateCVResult, ...]], None]

_POLICY_ID: Final = "autovalue-phase4-model-selection-v1"
_RETAIL_STATUSES: Final = ("certified", "new", "used")
_WHOLESALE_BUCKET_ORDER: Final = ("warmup", "2015_01", "2015_02", "2015_03_04")
_RETAIL_COUNTS: Final = {
    "phase3": 109_510,
    "development": 98_552,
    "calibration": 10_958,
    "screening": 29_619,
}
_WHOLESALE_COUNTS: Final = {
    "phase3": 442_130,
    "development": 391_641,
    "calibration": 50_489,
    "screening": 97_909,
}
_RETAIL_CALIBRATION_ASSIGNMENT_SHA256: Final = (
    "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
)
_RETAIL_SCREENING_ASSIGNMENT_SHA256: Final = (
    "fe8954d81f681c0d3ce7253d8a23f7995e9789693d7a261c851bc4078e173988"
)
_WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256: Final = (
    "f359c455accdfd8dc2de37ceab0ad218d81b5ee0e612d1e15fcd84fedd30f0d4"
)
_WHOLESALE_SCREENING_ASSIGNMENT_SHA256: Final = (
    "cecdf8d34fc7d549024dfdb22ae83a371855808a2b405f35bcb858e665d01bc1"
)
_CHECKPOINT_KEYS: Final = {
    "schema_version",
    "report_type",
    "policy_sha256",
    "track",
    "calibration_assignment_sha256",
    "screening_assignment_sha256",
    "completed_candidates",
}
_MAX_CHECKPOINT_BYTES: Final = 250_000
_MAX_SCREENING_REPORT_BYTES: Final = 500_000


class Phase4ScreeningError(ValueError):
    """The screening boundary, evidence, or report violated the frozen policy."""


@dataclass(frozen=True, slots=True)
class ScreeningSliceCount:
    """One canonical status or time-bucket count in the screening sample."""

    label: str
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label or self.label != self.label.strip():
            raise Phase4ScreeningError("screening slice label must be canonical text")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise Phase4ScreeningError("screening slice count must be a positive integer")

    def to_dict(self) -> dict[str, str | int]:
        return {"label": self.label, "sample_count": self.sample_count}


@dataclass(frozen=True, slots=True)
class Phase4ScreeningCheckpoint:
    """Aggregate-only, hash-bound progress that can safely resume candidate fits."""

    track: TrackName
    calibration_assignment_sha256: str
    screening_assignment_sha256: str
    completed_candidates: tuple[Phase4CandidateCVResult, ...]

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise Phase4ScreeningError("screening checkpoint track is invalid")
        expected_calibration = (
            _RETAIL_CALIBRATION_ASSIGNMENT_SHA256
            if self.track == "retail"
            else _WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256
        )
        expected_screening = (
            _RETAIL_SCREENING_ASSIGNMENT_SHA256
            if self.track == "retail"
            else _WHOLESALE_SCREENING_ASSIGNMENT_SHA256
        )
        if self.calibration_assignment_sha256 != expected_calibration:
            raise Phase4ScreeningError("checkpoint calibration hash differs from frozen audit")
        if self.screening_assignment_sha256 != expected_screening:
            raise Phase4ScreeningError("checkpoint screening hash differs from frozen audit")
        if not isinstance(self.completed_candidates, tuple) or not self.completed_candidates:
            raise Phase4ScreeningError("checkpoint must contain at least one completed candidate")
        expected = candidate_specs(self.track)
        if len(self.completed_candidates) > len(expected):
            raise Phase4ScreeningError("checkpoint contains too many candidates")
        expected_ids = tuple(
            spec.candidate_id for spec in expected[: len(self.completed_candidates)]
        )
        observed_ids = tuple(result.spec.candidate_id for result in self.completed_candidates)
        if observed_ids != expected_ids:
            raise Phase4ScreeningError("checkpoint candidates must be a stable policy prefix")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "report_type": "phase4_screening_checkpoint",
            "policy_sha256": PHASE4_PROTOCOL_SHA256,
            "track": self.track,
            "calibration_assignment_sha256": self.calibration_assignment_sha256,
            "screening_assignment_sha256": self.screening_assignment_sha256,
            "completed_candidates": [item.to_dict() for item in self.completed_candidates],
        }


@dataclass(frozen=True, slots=True)
class Phase4ScreeningReport:
    """Strict aggregate-only evidence from all 13 approved screening candidates."""

    track: TrackName
    phase3_train_sample_count: int
    development_sample_count: int
    calibration_sample_count: int
    screening_sample_count: int
    calibration_assignment_sha256: str
    screening_assignment_sha256: str
    screening_slices: tuple[ScreeningSliceCount, ...]
    cv_scheme: ScreeningCVScheme
    bucket_order: tuple[str, ...]
    candidates: tuple[Phase4CandidateCVResult, ...]
    shortlist: Phase4Shortlist

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise Phase4ScreeningError("screening report track is invalid")
        expected_counts = _RETAIL_COUNTS if self.track == "retail" else _WHOLESALE_COUNTS
        observed_counts = {
            "phase3": self.phase3_train_sample_count,
            "development": self.development_sample_count,
            "calibration": self.calibration_sample_count,
            "screening": self.screening_sample_count,
        }
        if observed_counts != expected_counts:
            raise Phase4ScreeningError("screening report row counts differ from the audit")
        if (
            self.development_sample_count + self.calibration_sample_count
            != self.phase3_train_sample_count
        ):
            raise Phase4ScreeningError("development and calibration must cover Phase-3 train")
        expected_calibration_hash = (
            _RETAIL_CALIBRATION_ASSIGNMENT_SHA256
            if self.track == "retail"
            else _WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256
        )
        expected_screening_hash = (
            _RETAIL_SCREENING_ASSIGNMENT_SHA256
            if self.track == "retail"
            else _WHOLESALE_SCREENING_ASSIGNMENT_SHA256
        )
        if self.calibration_assignment_sha256 != expected_calibration_hash:
            raise Phase4ScreeningError("calibration assignment hash differs from the audit")
        if self.screening_assignment_sha256 != expected_screening_hash:
            raise Phase4ScreeningError("screening assignment hash differs from the audit")
        _validate_digest(self.calibration_assignment_sha256, label="calibration assignment")
        _validate_digest(self.screening_assignment_sha256, label="screening assignment")
        _validate_screening_slices(self.track, self.screening_slices, self.screening_sample_count)
        _validate_cv_contract(self.track, self.cv_scheme, self.bucket_order)
        expected_ids = tuple(spec.candidate_id for spec in candidate_specs(self.track))
        if (
            not isinstance(self.candidates, tuple)
            or tuple(result.spec.candidate_id for result in self.candidates) != expected_ids
        ):
            raise Phase4ScreeningError(
                "screening report must contain every approved candidate in stable order"
            )
        if any(result.spec.track != self.track for result in self.candidates):
            raise Phase4ScreeningError("screening candidate track differs from report track")
        if not isinstance(self.shortlist, Phase4Shortlist) or self.shortlist.track != self.track:
            raise Phase4ScreeningError("screening shortlist does not match report track")
        if self.shortlist != shortlist_phase4_candidates(self.track, self.candidates):
            raise Phase4ScreeningError("screening shortlist does not match candidate metrics")

    @property
    def source_id(self) -> str:
        return "kaggle_us_sales_cars_v2" if self.track == "retail" else "kaggle_vehicle_sales_v1"

    @property
    def config(self) -> TrackConfig:
        return TRACKS[self.track]

    def to_dict(self) -> dict[str, object]:
        """Return the exact public row-free screening schema."""

        return {
            "schema_version": 1,
            "report_type": "phase4_screening",
            "policy_id": _POLICY_ID,
            "policy_sha256": PHASE4_PROTOCOL_SHA256,
            "track": self.track,
            "source_id": self.source_id,
            "feature_contract_version": self.config.contract_version,
            "target_semantics": self.config.target_semantics,
            "data_boundaries": {
                "phase3_train_sample_count": self.phase3_train_sample_count,
                "development_sample_count": self.development_sample_count,
                "calibration_sample_count": self.calibration_sample_count,
                "screening_sample_count": self.screening_sample_count,
                "calibration_assignment_sha256": self.calibration_assignment_sha256,
                "screening_assignment_sha256": self.screening_assignment_sha256,
                "target_used_for_partition_or_sampling": False,
                "calibration_used_for_fitting_or_selection": False,
                "legacy_holdout_used": False,
            },
            "cross_validation": {
                "scheme": self.cv_scheme,
                "bucket_order": list(self.bucket_order),
                "screening_slices": [item.to_dict() for item in self.screening_slices],
                "candidates": [candidate.to_dict() for candidate in self.candidates],
            },
            "shortlist": self.shortlist.to_dict(),
        }


def run_retail_phase4_screening(
    *,
    phase3_train_features: object,
    phase3_train_target: object,
    protocol: Phase4Protocol,
    completed_candidates: tuple[Phase4CandidateCVResult, ...] = (),
    on_progress: CandidateProgressCallback | None = None,
) -> Phase4ScreeningReport:
    """Evaluate every approved candidate on only the target-free retail sample."""

    track_policy = _validated_protocol(protocol, "retail")
    frame = validate_feature_frame(phase3_train_features, RETAIL_TRACK)
    if len(frame) != track_policy.phase3_train_rows:
        raise Phase4ScreeningError("retail Phase-3 train count differs from protocol")
    calibration = retail_calibration_partition(frame, seed=track_policy.calibration_seed)
    calibration_hash = _partition_hash(
        calibration.calibration_indices,
        population_count=len(frame),
        selected_label="calibration",
        unselected_label="development",
    )
    if calibration_hash != _RETAIL_CALIBRATION_ASSIGNMENT_SHA256:
        raise Phase4ScreeningError("retail calibration assignment failed the frozen audit")

    development = frame.iloc[calibration.development_indices].reset_index(drop=True)
    development_target = _development_target(
        phase3_train_target,
        calibration.development_indices,
        full_frame=frame,
        config=RETAIL_TRACK,
    )
    screening = retail_screening_sample(
        development,
        seed=track_policy.screening_sample_seed,
    )
    screening_hash = _partition_hash(
        screening.sample_indices,
        population_count=len(development),
        selected_label="screening",
        unselected_label="not_screening",
    )
    if screening_hash != _RETAIL_SCREENING_ASSIGNMENT_SHA256:
        raise Phase4ScreeningError("retail screening assignment failed the frozen audit")

    sample = development.iloc[screening.sample_indices].reset_index(drop=True)
    sample_target = development_target[screening.sample_indices]
    splits = retail_group_cv_splits(sample, n_splits=track_policy.screening_cv_folds)
    results = _evaluate_candidate_sequence(
        track="retail",
        features=sample,
        target=sample_target,
        splits=splits,
        expected_oof_mask=np.ones(len(sample), dtype=np.bool_),
        validation_buckets=(None,) * len(splits),
        completed_candidates=completed_candidates,
        on_progress=on_progress,
    )
    counts = Counter(cast(str, value) for value in sample["vehicle_status"].tolist())
    slices = tuple(ScreeningSliceCount(status, counts[status]) for status in _RETAIL_STATUSES)
    return Phase4ScreeningReport(
        track="retail",
        phase3_train_sample_count=len(frame),
        development_sample_count=len(development),
        calibration_sample_count=len(calibration.calibration_indices),
        screening_sample_count=len(sample),
        calibration_assignment_sha256=calibration_hash,
        screening_assignment_sha256=screening_hash,
        screening_slices=slices,
        cv_scheme="retail_predictor_group_kfold_v1",
        bucket_order=(),
        candidates=results,
        shortlist=shortlist_phase4_candidates("retail", results),
    )


def run_wholesale_phase4_screening(
    *,
    phase3_train_features: object,
    phase3_train_target: object,
    phase3_train_cv_buckets: object,
    protocol: Phase4Protocol,
    completed_candidates: tuple[Phase4CandidateCVResult, ...] = (),
    on_progress: CandidateProgressCallback | None = None,
) -> Phase4ScreeningReport:
    """Evaluate every approved candidate on the forward-only wholesale sample."""

    track_policy = _validated_protocol(protocol, "wholesale")
    config = TRACKS["wholesale"]
    frame = validate_feature_frame(phase3_train_features, config)
    if len(frame) != track_policy.phase3_train_rows:
        raise Phase4ScreeningError("wholesale Phase-3 train count differs from protocol")
    buckets = _aligned_bucket_series(phase3_train_cv_buckets, frame)
    calibration = wholesale_calibration_partition(buckets)
    calibration_hash = _partition_hash(
        calibration.calibration_indices,
        population_count=len(frame),
        selected_label="calibration",
        unselected_label="development",
    )
    if calibration_hash != _WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256:
        raise Phase4ScreeningError("wholesale calibration assignment failed the frozen audit")

    development = frame.iloc[calibration.development_indices].reset_index(drop=True)
    development_target = _development_target(
        phase3_train_target,
        calibration.development_indices,
        full_frame=frame,
        config=config,
    )
    development_buckets = buckets.iloc[calibration.development_indices].reset_index(drop=True)
    screening = wholesale_screening_sample(
        development_buckets,
        calibration.development_indices,
        seed=track_policy.screening_sample_seed,
    )
    screening_hash = _partition_hash(
        screening.sample_indices,
        population_count=len(development),
        selected_label="screening",
        unselected_label="not_screening",
        positions=calibration.development_indices,
    )
    if screening_hash != _WHOLESALE_SCREENING_ASSIGNMENT_SHA256:
        raise Phase4ScreeningError("wholesale screening assignment failed the frozen audit")

    sample = development.iloc[screening.sample_indices].reset_index(drop=True)
    sample_target = development_target[screening.sample_indices]
    sample_buckets = development_buckets.iloc[screening.sample_indices].reset_index(drop=True)
    splits = wholesale_forward_cv_splits(sample_buckets, bucket_order=_WHOLESALE_BUCKET_ORDER)
    expected_mask = np.asarray(sample_buckets != _WHOLESALE_BUCKET_ORDER[0], dtype=np.bool_)
    results = _evaluate_candidate_sequence(
        track="wholesale",
        features=sample,
        target=sample_target,
        splits=splits,
        expected_oof_mask=expected_mask,
        validation_buckets=_WHOLESALE_BUCKET_ORDER[1:],
        completed_candidates=completed_candidates,
        on_progress=on_progress,
    )
    counts = Counter(cast(str, value) for value in sample_buckets.tolist())
    slices = tuple(
        ScreeningSliceCount(bucket, counts[bucket]) for bucket in _WHOLESALE_BUCKET_ORDER
    )
    return Phase4ScreeningReport(
        track="wholesale",
        phase3_train_sample_count=len(frame),
        development_sample_count=len(development),
        calibration_sample_count=len(calibration.calibration_indices),
        screening_sample_count=len(sample),
        calibration_assignment_sha256=calibration_hash,
        screening_assignment_sha256=screening_hash,
        screening_slices=slices,
        cv_scheme="wholesale_forward_chaining_cv_bucket_v1",
        bucket_order=_WHOLESALE_BUCKET_ORDER,
        candidates=results,
        shortlist=shortlist_phase4_candidates("wholesale", results),
    )


def canonical_phase4_screening_json(report: Phase4ScreeningReport) -> str:
    """Serialize a validated screening report as deterministic aggregate JSON."""

    if not isinstance(report, Phase4ScreeningReport):
        raise Phase4ScreeningError("report must be a Phase4ScreeningReport")
    report.__post_init__()
    return (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def make_phase4_screening_checkpoint(
    track: TrackName,
    completed_candidates: tuple[Phase4CandidateCVResult, ...],
) -> Phase4ScreeningCheckpoint:
    """Bind an ordered candidate prefix to the exact audited row assignments."""

    return Phase4ScreeningCheckpoint(
        track=track,
        calibration_assignment_sha256=(
            _RETAIL_CALIBRATION_ASSIGNMENT_SHA256
            if track == "retail"
            else _WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256
        ),
        screening_assignment_sha256=(
            _RETAIL_SCREENING_ASSIGNMENT_SHA256
            if track == "retail"
            else _WHOLESALE_SCREENING_ASSIGNMENT_SHA256
        ),
        completed_candidates=completed_candidates,
    )


def canonical_phase4_checkpoint_json(checkpoint: Phase4ScreeningCheckpoint) -> str:
    """Serialize resumable aggregate-only progress deterministically."""

    if not isinstance(checkpoint, Phase4ScreeningCheckpoint):
        raise Phase4ScreeningError("checkpoint must be Phase4ScreeningCheckpoint")
    checkpoint.__post_init__()
    return (
        json.dumps(
            checkpoint.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def parse_phase4_checkpoint_json(serialized: str | bytes) -> Phase4ScreeningCheckpoint:
    """Parse a bounded checkpoint while rejecting duplicate keys and policy drift."""

    if isinstance(serialized, bytes):
        if len(serialized) > _MAX_CHECKPOINT_BYTES:
            raise Phase4ScreeningError("checkpoint exceeds maximum size")
        try:
            text = serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Phase4ScreeningError("checkpoint must be UTF-8") from error
    elif isinstance(serialized, str):
        text = serialized
        if len(text.encode("utf-8")) > _MAX_CHECKPOINT_BYTES:
            raise Phase4ScreeningError("checkpoint exceeds maximum size")
    else:
        raise Phase4ScreeningError("checkpoint must be text or bytes")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise Phase4ScreeningError("checkpoint is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_KEYS:
        raise Phase4ScreeningError("checkpoint root fields are invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["report_type"] != "phase4_screening_checkpoint"
        or value["policy_sha256"] != PHASE4_PROTOCOL_SHA256
    ):
        raise Phase4ScreeningError("checkpoint policy metadata is invalid")
    track = value["track"]
    if track not in TRACKS:
        raise Phase4ScreeningError("checkpoint track is invalid")
    calibration_hash = value["calibration_assignment_sha256"]
    screening_hash = value["screening_assignment_sha256"]
    completed = value["completed_candidates"]
    if not isinstance(calibration_hash, str) or not isinstance(screening_hash, str):
        raise Phase4ScreeningError("checkpoint assignment hashes must be text")
    if not isinstance(completed, list):
        raise Phase4ScreeningError("checkpoint completed_candidates must be an array")
    return Phase4ScreeningCheckpoint(
        track=cast(TrackName, track),
        calibration_assignment_sha256=calibration_hash,
        screening_assignment_sha256=screening_hash,
        completed_candidates=tuple(parse_phase4_candidate_cv_result(item) for item in completed),
    )


def parse_phase4_screening_json(serialized: str | bytes) -> Phase4ScreeningReport:
    """Parse a canonical aggregate screening report and reject any field drift."""

    text = _bounded_json_text(
        serialized,
        maximum_bytes=_MAX_SCREENING_REPORT_BYTES,
        label="screening report",
    )
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise Phase4ScreeningError("screening report is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise Phase4ScreeningError("screening report root must be an object")
    track = value.get("track")
    if track not in TRACKS:
        raise Phase4ScreeningError("screening report track is invalid")
    cross_validation = value.get("cross_validation")
    if not isinstance(cross_validation, Mapping):
        raise Phase4ScreeningError("screening report cross_validation must be an object")
    candidates_value = cross_validation.get("candidates")
    if not isinstance(candidates_value, list):
        raise Phase4ScreeningError("screening report candidates must be an array")
    candidates = tuple(parse_phase4_candidate_cv_result(item) for item in candidates_value)
    resolved_track = cast(TrackName, track)
    if resolved_track == "retail":
        report = Phase4ScreeningReport(
            track="retail",
            phase3_train_sample_count=_RETAIL_COUNTS["phase3"],
            development_sample_count=_RETAIL_COUNTS["development"],
            calibration_sample_count=_RETAIL_COUNTS["calibration"],
            screening_sample_count=_RETAIL_COUNTS["screening"],
            calibration_assignment_sha256=_RETAIL_CALIBRATION_ASSIGNMENT_SHA256,
            screening_assignment_sha256=_RETAIL_SCREENING_ASSIGNMENT_SHA256,
            screening_slices=(
                ScreeningSliceCount("certified", 1_640),
                ScreeningSliceCount("new", 17_562),
                ScreeningSliceCount("used", 10_417),
            ),
            cv_scheme="retail_predictor_group_kfold_v1",
            bucket_order=(),
            candidates=candidates,
            shortlist=shortlist_phase4_candidates("retail", candidates),
        )
    else:
        report = Phase4ScreeningReport(
            track="wholesale",
            phase3_train_sample_count=_WHOLESALE_COUNTS["phase3"],
            development_sample_count=_WHOLESALE_COUNTS["development"],
            calibration_sample_count=_WHOLESALE_COUNTS["calibration"],
            screening_sample_count=_WHOLESALE_COUNTS["screening"],
            calibration_assignment_sha256=_WHOLESALE_CALIBRATION_ASSIGNMENT_SHA256,
            screening_assignment_sha256=_WHOLESALE_SCREENING_ASSIGNMENT_SHA256,
            screening_slices=(
                ScreeningSliceCount("warmup", 12_896),
                ScreeningSliceCount("2015_01", 33_612),
                ScreeningSliceCount("2015_02", 39_608),
                ScreeningSliceCount("2015_03_04", 11_793),
            ),
            cv_scheme="wholesale_forward_chaining_cv_bucket_v1",
            bucket_order=_WHOLESALE_BUCKET_ORDER,
            candidates=candidates,
            shortlist=shortlist_phase4_candidates("wholesale", candidates),
        )
    try:
        observed_canonical = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except ValueError as error:
        raise Phase4ScreeningError("screening report contains non-finite JSON numbers") from error
    if observed_canonical != canonical_phase4_screening_json(report):
        raise Phase4ScreeningError("screening report fields differ from canonical evidence")
    return report


def _evaluate_candidate_sequence(
    *,
    track: TrackName,
    features: pd.DataFrame,
    target: NDArray[np.float64],
    splits: tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...],
    expected_oof_mask: NDArray[np.bool_],
    validation_buckets: tuple[str | None, ...],
    completed_candidates: tuple[Phase4CandidateCVResult, ...],
    on_progress: CandidateProgressCallback | None,
) -> tuple[Phase4CandidateCVResult, ...]:
    specs = candidate_specs(track)
    if not isinstance(completed_candidates, tuple) or any(
        not isinstance(result, Phase4CandidateCVResult) for result in completed_candidates
    ):
        raise Phase4ScreeningError("completed candidates must be an immutable result tuple")
    if len(completed_candidates) > len(specs):
        raise Phase4ScreeningError("completed candidate count exceeds the frozen policy")
    expected_prefix = tuple(spec.candidate_id for spec in specs[: len(completed_candidates)])
    if tuple(result.spec.candidate_id for result in completed_candidates) != expected_prefix:
        raise Phase4ScreeningError("completed candidates must be a stable policy prefix")
    _validate_completed_evidence(
        track=track,
        features=features,
        splits=splits,
        expected_oof_mask=expected_oof_mask,
        validation_buckets=validation_buckets,
        results=completed_candidates,
    )
    if on_progress is not None and not callable(on_progress):
        raise Phase4ScreeningError("on_progress must be callable")

    results = list(completed_candidates)
    for spec in specs[len(results) :]:
        results.append(
            evaluate_phase4_candidate_cv(
                features=features,
                target=target,
                spec=spec,
                splits=splits,
                expected_oof_mask=expected_oof_mask,
                validation_buckets=validation_buckets,
            )
        )
        if on_progress is not None:
            on_progress(tuple(results))
    return tuple(results)


def _validate_completed_evidence(
    *,
    track: TrackName,
    features: pd.DataFrame,
    splits: tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...],
    expected_oof_mask: NDArray[np.bool_],
    validation_buckets: tuple[str | None, ...],
    results: tuple[Phase4CandidateCVResult, ...],
) -> None:
    expected_folds = tuple(
        (len(training), len(validation), bucket)
        for (training, validation), bucket in zip(splits, validation_buckets, strict=True)
    )
    expected_oof_count = int(np.count_nonzero(expected_oof_mask))
    status_counts = (
        Counter(cast(str, value) for value in features["vehicle_status"].tolist())
        if track == "retail"
        else Counter()
    )
    expected_slices = (
        tuple((status, status_counts[status]) for status in _RETAIL_STATUSES)
        if track == "retail"
        else ()
    )
    for result in results:
        observed_folds = tuple(
            (
                fold.training_sample_count,
                fold.validation_sample_count,
                fold.validation_bucket,
            )
            for fold in result.folds
        )
        observed_slices = tuple(
            (item.status, item.metrics.sample_count) for item in result.status_slices
        )
        if result.spec.track != track or result.overall.sample_count != expected_oof_count:
            raise Phase4ScreeningError("completed candidate boundary differs from current sample")
        if observed_folds != expected_folds or observed_slices != expected_slices:
            raise Phase4ScreeningError("completed candidate folds differ from current sample")


def _validated_protocol(protocol: object, track: TrackName) -> TrackPhase4Protocol:
    if not isinstance(protocol, Phase4Protocol):
        raise Phase4ScreeningError("protocol must be a validated Phase4Protocol")
    if protocol.policy_id != _POLICY_ID:
        raise Phase4ScreeningError("protocol policy_id is invalid")
    track_policy = protocol.for_track(track)
    expected = _RETAIL_COUNTS if track == "retail" else _WHOLESALE_COUNTS
    if track_policy.phase3_train_rows != expected["phase3"]:
        raise Phase4ScreeningError("protocol Phase-3 row count is invalid")
    if track_policy.screening_cv_folds != (5 if track == "retail" else 3):
        raise Phase4ScreeningError("protocol screening fold count is invalid")
    return track_policy


def _development_target(
    target: object,
    development_indices: NDArray[np.int64],
    *,
    full_frame: pd.DataFrame,
    config: TrackConfig,
) -> NDArray[np.float64]:
    if isinstance(target, pd.DataFrame):
        if not target.index.equals(full_frame.index):
            raise Phase4ScreeningError("Phase-3 feature and target indexes must align")
        selected: object = target.iloc[development_indices]
    elif isinstance(target, pd.Series):
        if not target.index.equals(full_frame.index):
            raise Phase4ScreeningError("Phase-3 feature and target indexes must align")
        selected = target.iloc[development_indices]
    else:
        values = np.asarray(target, dtype=object)
        if values.ndim != 1 or len(values) != len(full_frame):
            raise Phase4ScreeningError("Phase-3 target must be a one-dimensional row match")
        selected = values[development_indices]
    return validate_target(selected, expected_rows=len(development_indices), config=config)


def _aligned_bucket_series(value: object, frame: pd.DataFrame) -> pd.Series:
    if isinstance(value, pd.Series):
        if not value.index.equals(frame.index):
            raise Phase4ScreeningError("wholesale features and CV buckets must align")
        return value.copy(deep=True)
    values = np.asarray(value, dtype=object)
    if values.ndim != 1 or len(values) != len(frame):
        raise Phase4ScreeningError("wholesale CV buckets must be a one-dimensional row match")
    return pd.Series(values, index=frame.index, name="cv_bucket", dtype=object)


def _partition_hash(
    selected_indices: NDArray[np.int64],
    *,
    population_count: int,
    selected_label: str,
    unselected_label: str,
    positions: NDArray[np.int64] | None = None,
) -> str:
    if population_count < 1:
        raise Phase4ScreeningError("partition hash population must be positive")
    if (
        not isinstance(selected_indices, np.ndarray)
        or selected_indices.ndim != 1
        or selected_indices.dtype.kind not in "iu"
        or len(selected_indices) == 0
    ):
        raise Phase4ScreeningError("selected partition indices must be a non-empty integer array")
    if (selected_indices < 0).any() or (selected_indices >= population_count).any():
        raise Phase4ScreeningError("selected partition index is outside the population")
    if len(np.unique(selected_indices)) != len(selected_indices):
        raise Phase4ScreeningError("selected partition indices must be unique")
    mask = np.zeros(population_count, dtype=np.bool_)
    mask[selected_indices] = True
    reported_positions = (
        np.arange(population_count, dtype=np.int64) if positions is None else positions
    )
    if reported_positions.shape != (population_count,):
        raise Phase4ScreeningError("partition hash positions must match population")
    if reported_positions.dtype.kind not in "iu":
        raise Phase4ScreeningError("partition hash positions must be integers")
    if len(np.unique(reported_positions)) != population_count:
        raise Phase4ScreeningError("partition hash positions must be unique")
    digest = hashlib.sha256()
    for position, selected in zip(reported_positions, mask, strict=True):
        label = selected_label if selected else unselected_label
        digest.update(f"{int(position)},{label}\n".encode("ascii"))
    return digest.hexdigest()


def _validate_screening_slices(
    track: TrackName,
    slices: object,
    screening_count: int,
) -> None:
    if not isinstance(slices, tuple) or any(
        not isinstance(item, ScreeningSliceCount) for item in slices
    ):
        raise Phase4ScreeningError("screening slices must be an immutable count tuple")
    expected_labels = _RETAIL_STATUSES if track == "retail" else _WHOLESALE_BUCKET_ORDER
    if tuple(item.label for item in slices) != expected_labels:
        raise Phase4ScreeningError("screening slice labels differ from the track contract")
    if sum(item.sample_count for item in slices) != screening_count:
        raise Phase4ScreeningError("screening slice counts must sum to screening sample count")


def _validate_cv_contract(
    track: TrackName,
    scheme: object,
    bucket_order: object,
) -> None:
    expected_scheme = (
        "retail_predictor_group_kfold_v1"
        if track == "retail"
        else "wholesale_forward_chaining_cv_bucket_v1"
    )
    if scheme != expected_scheme:
        raise Phase4ScreeningError("screening CV scheme differs from track contract")
    expected_order = () if track == "retail" else _WHOLESALE_BUCKET_ORDER
    if bucket_order != expected_order:
        raise Phase4ScreeningError("screening bucket order differs from track contract")


def _validate_digest(value: object, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4ScreeningError(f"{label} must be a lowercase SHA-256 digest")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4ScreeningError(f"checkpoint contains duplicate field: {key}")
        result[key] = value
    return result


def _bounded_json_text(
    serialized: str | bytes,
    *,
    maximum_bytes: int,
    label: str,
) -> str:
    if isinstance(serialized, bytes):
        if len(serialized) > maximum_bytes:
            raise Phase4ScreeningError(f"{label} exceeds maximum size")
        try:
            return serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Phase4ScreeningError(f"{label} must be UTF-8") from error
    if isinstance(serialized, str):
        if len(serialized.encode("utf-8")) > maximum_bytes:
            raise Phase4ScreeningError(f"{label} exceeds maximum size")
        return serialized
    raise Phase4ScreeningError(f"{label} must be text or bytes")


__all__ = [
    "CandidateProgressCallback",
    "Phase4ScreeningError",
    "Phase4ScreeningCheckpoint",
    "Phase4ScreeningReport",
    "ScreeningCVScheme",
    "ScreeningSliceCount",
    "canonical_phase4_screening_json",
    "canonical_phase4_checkpoint_json",
    "make_phase4_screening_checkpoint",
    "parse_phase4_checkpoint_json",
    "parse_phase4_screening_json",
    "run_retail_phase4_screening",
    "run_wholesale_phase4_screening",
]
