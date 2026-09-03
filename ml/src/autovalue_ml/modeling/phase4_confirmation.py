"""Full-development CV confirmation for the frozen Phase 4 shortlists."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .calibration import retail_calibration_partition, wholesale_calibration_partition
from .candidates import CandidateSpec, candidate_specs
from .contracts import RETAIL_TRACK, TRACKS, TrackName, validate_feature_frame
from .cv import CVSplit, retail_group_cv_splits, wholesale_forward_cv_splits
from .phase4_evaluation import (
    Phase4CandidateCVResult,
    evaluate_phase4_candidate_cv,
    parse_phase4_candidate_cv_result,
)
from .phase4_protocol import PHASE4_PROTOCOL_SHA256, Phase4Protocol
from .phase4_screening_experiment import (
    Phase4ScreeningReport,
    _aligned_bucket_series,
    _development_target,
    _partition_hash,
    _validated_protocol,
)

ConfirmationProgressCallback = Callable[[tuple[Phase4CandidateCVResult, ...]], None]

_POLICY_ID: Final = "autovalue-phase4-model-selection-v1"
_WHOLESALE_BUCKET_ORDER: Final = ("warmup", "2015_01", "2015_02", "2015_03_04")
_RETAIL_CALIBRATION_HASH: Final = "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
_WHOLESALE_CALIBRATION_HASH: Final = (
    "f359c455accdfd8dc2de37ceab0ad218d81b5ee0e612d1e15fcd84fedd30f0d4"
)
_SCREENING_REPORT_HASHES: Final[dict[TrackName, str]] = {
    "retail": "62dcd2c1c41d30d49a4c98eab98e82529170aedd6f9313b46d00ffa50fdc4c9c",
    "wholesale": "0b0bb79ce82138215e6e8920f7b4ba57086e0f75e60cfc2095f8ab93e6e240c7",
}
_FULL_DEVELOPMENT_IDS: Final[dict[TrackName, tuple[str, ...]]] = {
    "retail": (
        "phase4-retail-linear_regression_incumbent-00",
        "phase4-retail-random_forest-05",
        "phase4-retail-random_forest-00",
        "phase4-retail-gradient_boosting-05",
        "phase4-retail-gradient_boosting-02",
    ),
    "wholesale": (
        "phase4-wholesale-linear_regression_incumbent-00",
        "phase4-wholesale-random_forest-05",
        "phase4-wholesale-random_forest-00",
        "phase4-wholesale-gradient_boosting-03",
        "phase4-wholesale-gradient_boosting-04",
    ),
}
_COUNTS: Final[dict[TrackName, dict[str, int]]] = {
    "retail": {
        "phase3": 109_510,
        "development": 98_552,
        "calibration": 10_958,
        "oof": 98_552,
    },
    "wholesale": {
        "phase3": 442_130,
        "development": 391_641,
        "calibration": 50_489,
        "oof": 340_055,
    },
}
_RETAIL_DEVELOPMENT_STATUS_COUNTS: Final = {
    "certified": 5_467,
    "new": 58_360,
    "used": 34_725,
}
_CHECKPOINT_KEYS: Final = {
    "schema_version",
    "report_type",
    "policy_sha256",
    "track",
    "screening_report_sha256",
    "calibration_assignment_sha256",
    "completed_candidates",
}
_MAX_CHECKPOINT_BYTES: Final = 150_000
_MAX_REPORT_BYTES: Final = 250_000


class Phase4ConfirmationError(ValueError):
    """Full-development evidence violated the frozen protocol or shortlist."""


@dataclass(frozen=True, slots=True)
class Phase4ConfirmationCheckpoint:
    """Aggregate-only resumable progress for the five confirmation candidates."""

    track: TrackName
    screening_report_sha256: str
    calibration_assignment_sha256: str
    completed_candidates: tuple[Phase4CandidateCVResult, ...]

    def __post_init__(self) -> None:
        _validate_track_hashes(
            self.track,
            self.screening_report_sha256,
            self.calibration_assignment_sha256,
        )
        if not isinstance(self.completed_candidates, tuple) or not self.completed_candidates:
            raise Phase4ConfirmationError("checkpoint requires at least one completed candidate")
        expected = _FULL_DEVELOPMENT_IDS[self.track]
        observed = tuple(item.spec.candidate_id for item in self.completed_candidates)
        if len(observed) > len(expected) or observed != expected[: len(observed)]:
            raise Phase4ConfirmationError("checkpoint candidates must be a stable shortlist prefix")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "report_type": "phase4_full_development_checkpoint",
            "policy_sha256": PHASE4_PROTOCOL_SHA256,
            "track": self.track,
            "screening_report_sha256": self.screening_report_sha256,
            "calibration_assignment_sha256": self.calibration_assignment_sha256,
            "completed_candidates": [item.to_dict() for item in self.completed_candidates],
        }


@dataclass(frozen=True, slots=True)
class Phase4ConfirmationReport:
    """Aggregate full-development CV evidence for exactly five frozen candidates."""

    track: TrackName
    screening_report_sha256: str
    calibration_assignment_sha256: str
    candidates: tuple[Phase4CandidateCVResult, ...]

    def __post_init__(self) -> None:
        _validate_track_hashes(
            self.track,
            self.screening_report_sha256,
            self.calibration_assignment_sha256,
        )
        expected_ids = _FULL_DEVELOPMENT_IDS[self.track]
        if (
            not isinstance(self.candidates, tuple)
            or tuple(item.spec.candidate_id for item in self.candidates) != expected_ids
        ):
            raise Phase4ConfirmationError(
                "confirmation report must contain the exact five candidates in shortlist order"
            )
        _validate_same_fold_results(self.track, self.candidates)

    @property
    def metric_ranking(self) -> tuple[str, ...]:
        return tuple(
            item.spec.candidate_id
            for item in sorted(
                self.candidates,
                key=lambda candidate: (candidate.overall.mae, candidate.spec.candidate_id),
            )
        )

    def to_dict(self) -> dict[str, object]:
        counts = _COUNTS[self.track]
        return {
            "schema_version": 1,
            "report_type": "phase4_full_development_confirmation",
            "policy_id": _POLICY_ID,
            "policy_sha256": PHASE4_PROTOCOL_SHA256,
            "track": self.track,
            "source_id": (
                "kaggle_us_sales_cars_v2" if self.track == "retail" else "kaggle_vehicle_sales_v1"
            ),
            "feature_contract_version": TRACKS[self.track].contract_version,
            "target_semantics": TRACKS[self.track].target_semantics,
            "screening_report_sha256": self.screening_report_sha256,
            "data_boundaries": {
                "phase3_train_sample_count": counts["phase3"],
                "development_sample_count": counts["development"],
                "calibration_sample_count": counts["calibration"],
                "oof_scored_sample_count": counts["oof"],
                "calibration_assignment_sha256": self.calibration_assignment_sha256,
                "calibration_used_for_fitting_or_selection": False,
                "legacy_holdout_used": False,
                "screening_rows_used_for_full_development_fit": True,
            },
            "selection_scope": {
                "candidate_ids": list(_FULL_DEVELOPMENT_IDS[self.track]),
                "metric": "micro_out_of_fold_mae_usd",
                "metric_ranking": list(self.metric_ranking),
                "promotion_status": "pending_deployment_measurements_and_gates",
            },
            "candidates": [item.to_dict() for item in self.candidates],
        }


def run_retail_phase4_confirmation(
    *,
    phase3_train_features: object,
    phase3_train_target: object,
    protocol: Phase4Protocol,
    screening_report: Phase4ScreeningReport,
    screening_report_sha256: str,
    completed_candidates: tuple[Phase4CandidateCVResult, ...] = (),
    on_progress: ConfirmationProgressCallback | None = None,
) -> Phase4ConfirmationReport:
    """Confirm the retail shortlist on all development groups."""

    track_policy = _validated_protocol(protocol, "retail")
    _validate_screening_input("retail", screening_report, screening_report_sha256)
    frame = validate_feature_frame(phase3_train_features, RETAIL_TRACK)
    if len(frame) != track_policy.phase3_train_rows:
        raise Phase4ConfirmationError("retail Phase-3 train count differs from protocol")
    calibration = retail_calibration_partition(frame, seed=track_policy.calibration_seed)
    calibration_hash = _partition_hash(
        calibration.calibration_indices,
        population_count=len(frame),
        selected_label="calibration",
        unselected_label="development",
    )
    if calibration_hash != _RETAIL_CALIBRATION_HASH:
        raise Phase4ConfirmationError("retail calibration assignment differs from frozen audit")
    development = frame.iloc[calibration.development_indices].reset_index(drop=True)
    target = _development_target(
        phase3_train_target,
        calibration.development_indices,
        full_frame=frame,
        config=RETAIL_TRACK,
    )
    splits = retail_group_cv_splits(development, n_splits=track_policy.screening_cv_folds)
    mask = np.ones(len(development), dtype=np.bool_)
    results = _evaluate_confirmation_sequence(
        track="retail",
        features=development,
        target=target,
        splits=splits,
        expected_oof_mask=mask,
        validation_buckets=(None,) * len(splits),
        completed_candidates=completed_candidates,
        on_progress=on_progress,
    )
    return Phase4ConfirmationReport(
        track="retail",
        screening_report_sha256=screening_report_sha256,
        calibration_assignment_sha256=calibration_hash,
        candidates=results,
    )


def run_wholesale_phase4_confirmation(
    *,
    phase3_train_features: object,
    phase3_train_target: object,
    phase3_train_cv_buckets: object,
    protocol: Phase4Protocol,
    screening_report: Phase4ScreeningReport,
    screening_report_sha256: str,
    completed_candidates: tuple[Phase4CandidateCVResult, ...] = (),
    on_progress: ConfirmationProgressCallback | None = None,
) -> Phase4ConfirmationReport:
    """Confirm the wholesale shortlist with full forward-only development CV."""

    track_policy = _validated_protocol(protocol, "wholesale")
    _validate_screening_input("wholesale", screening_report, screening_report_sha256)
    config = TRACKS["wholesale"]
    frame = validate_feature_frame(phase3_train_features, config)
    if len(frame) != track_policy.phase3_train_rows:
        raise Phase4ConfirmationError("wholesale Phase-3 train count differs from protocol")
    buckets = _aligned_bucket_series(phase3_train_cv_buckets, frame)
    calibration = wholesale_calibration_partition(buckets)
    calibration_hash = _partition_hash(
        calibration.calibration_indices,
        population_count=len(frame),
        selected_label="calibration",
        unselected_label="development",
    )
    if calibration_hash != _WHOLESALE_CALIBRATION_HASH:
        raise Phase4ConfirmationError("wholesale calibration assignment differs from audit")
    development = frame.iloc[calibration.development_indices].reset_index(drop=True)
    target = _development_target(
        phase3_train_target,
        calibration.development_indices,
        full_frame=frame,
        config=config,
    )
    development_buckets = buckets.iloc[calibration.development_indices].reset_index(drop=True)
    splits = wholesale_forward_cv_splits(
        development_buckets,
        bucket_order=_WHOLESALE_BUCKET_ORDER,
    )
    mask = np.asarray(development_buckets != "warmup", dtype=np.bool_)
    results = _evaluate_confirmation_sequence(
        track="wholesale",
        features=development,
        target=target,
        splits=splits,
        expected_oof_mask=mask,
        validation_buckets=_WHOLESALE_BUCKET_ORDER[1:],
        completed_candidates=completed_candidates,
        on_progress=on_progress,
    )
    return Phase4ConfirmationReport(
        track="wholesale",
        screening_report_sha256=screening_report_sha256,
        calibration_assignment_sha256=calibration_hash,
        candidates=results,
    )


def make_phase4_confirmation_checkpoint(
    track: TrackName,
    completed_candidates: tuple[Phase4CandidateCVResult, ...],
) -> Phase4ConfirmationCheckpoint:
    if track not in TRACKS:
        raise Phase4ConfirmationError("confirmation checkpoint track is invalid")
    return Phase4ConfirmationCheckpoint(
        track=track,
        screening_report_sha256=_SCREENING_REPORT_HASHES[track],
        calibration_assignment_sha256=(
            _RETAIL_CALIBRATION_HASH if track == "retail" else _WHOLESALE_CALIBRATION_HASH
        ),
        completed_candidates=completed_candidates,
    )


def canonical_phase4_confirmation_json(report: Phase4ConfirmationReport) -> str:
    if not isinstance(report, Phase4ConfirmationReport):
        raise Phase4ConfirmationError("report must be Phase4ConfirmationReport")
    report.__post_init__()
    return _canonical_json(report.to_dict())


def canonical_phase4_confirmation_checkpoint_json(
    checkpoint: Phase4ConfirmationCheckpoint,
) -> str:
    if not isinstance(checkpoint, Phase4ConfirmationCheckpoint):
        raise Phase4ConfirmationError("checkpoint must be Phase4ConfirmationCheckpoint")
    checkpoint.__post_init__()
    return _canonical_json(checkpoint.to_dict())


def parse_phase4_confirmation_checkpoint_json(
    serialized: str | bytes,
) -> Phase4ConfirmationCheckpoint:
    text = _bounded_text(serialized)
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise Phase4ConfirmationError("confirmation checkpoint is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_KEYS:
        raise Phase4ConfirmationError("confirmation checkpoint root fields are invalid")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["report_type"] != "phase4_full_development_checkpoint"
        or value["policy_sha256"] != PHASE4_PROTOCOL_SHA256
    ):
        raise Phase4ConfirmationError("confirmation checkpoint policy metadata is invalid")
    track = value["track"]
    if track not in TRACKS:
        raise Phase4ConfirmationError("confirmation checkpoint track is invalid")
    completed = value["completed_candidates"]
    if not isinstance(completed, list):
        raise Phase4ConfirmationError("completed_candidates must be an array")
    screening_hash = value["screening_report_sha256"]
    calibration_hash = value["calibration_assignment_sha256"]
    if not isinstance(screening_hash, str) or not isinstance(calibration_hash, str):
        raise Phase4ConfirmationError("confirmation checkpoint hashes must be text")
    return Phase4ConfirmationCheckpoint(
        track=cast(TrackName, track),
        screening_report_sha256=screening_hash,
        calibration_assignment_sha256=calibration_hash,
        completed_candidates=tuple(parse_phase4_candidate_cv_result(item) for item in completed),
    )


def parse_phase4_confirmation_json(serialized: str | bytes) -> Phase4ConfirmationReport:
    """Parse exact canonical full-development evidence for later promotion stages."""

    text = _bounded_text(serialized, maximum_bytes=_MAX_REPORT_BYTES, label="confirmation report")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise Phase4ConfirmationError("confirmation report is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise Phase4ConfirmationError("confirmation report root must be an object")
    track = value.get("track")
    if track not in TRACKS:
        raise Phase4ConfirmationError("confirmation report track is invalid")
    candidates_value = value.get("candidates")
    if not isinstance(candidates_value, list):
        raise Phase4ConfirmationError("confirmation report candidates must be an array")
    resolved_track = cast(TrackName, track)
    report = Phase4ConfirmationReport(
        track=resolved_track,
        screening_report_sha256=_SCREENING_REPORT_HASHES[resolved_track],
        calibration_assignment_sha256=(
            _RETAIL_CALIBRATION_HASH if resolved_track == "retail" else _WHOLESALE_CALIBRATION_HASH
        ),
        candidates=tuple(parse_phase4_candidate_cv_result(item) for item in candidates_value),
    )
    try:
        observed = _canonical_json(value)
    except ValueError as error:
        raise Phase4ConfirmationError("confirmation report has non-finite numbers") from error
    if observed != canonical_phase4_confirmation_json(report):
        raise Phase4ConfirmationError("confirmation report fields differ from canonical evidence")
    return report


def _evaluate_confirmation_sequence(
    *,
    track: TrackName,
    features: pd.DataFrame,
    target: NDArray[np.float64],
    splits: tuple[CVSplit, ...],
    expected_oof_mask: NDArray[np.bool_],
    validation_buckets: tuple[str | None, ...],
    completed_candidates: tuple[Phase4CandidateCVResult, ...],
    on_progress: ConfirmationProgressCallback | None,
) -> tuple[Phase4CandidateCVResult, ...]:
    specs = _confirmation_specs(track)
    if not isinstance(completed_candidates, tuple) or any(
        not isinstance(item, Phase4CandidateCVResult) for item in completed_candidates
    ):
        raise Phase4ConfirmationError("completed candidates must be an immutable result tuple")
    expected_prefix = tuple(spec.candidate_id for spec in specs[: len(completed_candidates)])
    if (
        len(completed_candidates) > len(specs)
        or tuple(item.spec.candidate_id for item in completed_candidates) != expected_prefix
    ):
        raise Phase4ConfirmationError("completed candidates must be a stable shortlist prefix")
    _validate_current_evidence(
        track,
        features,
        splits,
        expected_oof_mask,
        validation_buckets,
        completed_candidates,
    )
    if on_progress is not None and not callable(on_progress):
        raise Phase4ConfirmationError("on_progress must be callable")
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


def _confirmation_specs(track: TrackName) -> tuple[CandidateSpec, ...]:
    by_id = {spec.candidate_id: spec for spec in candidate_specs(track)}
    return tuple(by_id[candidate_id] for candidate_id in _FULL_DEVELOPMENT_IDS[track])


def _validate_current_evidence(
    track: TrackName,
    features: pd.DataFrame,
    splits: tuple[CVSplit, ...],
    expected_oof_mask: NDArray[np.bool_],
    validation_buckets: tuple[str | None, ...],
    results: tuple[Phase4CandidateCVResult, ...],
) -> None:
    expected_folds = tuple(
        (len(training), len(validation), bucket)
        for (training, validation), bucket in zip(splits, validation_buckets, strict=True)
    )
    status_counts = (
        Counter(cast(str, value) for value in features["vehicle_status"].tolist())
        if track == "retail"
        else Counter()
    )
    expected_slices = (
        tuple((status, status_counts[status]) for status in ("certified", "new", "used"))
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
        if result.overall.sample_count != int(np.count_nonzero(expected_oof_mask)):
            raise Phase4ConfirmationError("completed candidate OOF boundary differs")
        if observed_folds != expected_folds or observed_slices != expected_slices:
            raise Phase4ConfirmationError("completed candidate folds differ from development CV")


def _validate_same_fold_results(
    track: TrackName,
    results: tuple[Phase4CandidateCVResult, ...],
) -> None:
    reference = results[0]
    if reference.overall.sample_count != _COUNTS[track]["oof"]:
        raise Phase4ConfirmationError("confirmation OOF count differs from audit")
    fold_shape = tuple(
        (fold.training_sample_count, fold.validation_sample_count, fold.validation_bucket)
        for fold in reference.folds
    )
    slice_shape = tuple(
        (item.status, item.metrics.sample_count) for item in reference.status_slices
    )
    if track == "retail" and dict(slice_shape) != _RETAIL_DEVELOPMENT_STATUS_COUNTS:
        raise Phase4ConfirmationError("retail development status counts differ from audit")
    if track == "wholesale" and (
        reference.latest_fold.validation_bucket != "2015_03_04"
        or reference.latest_fold.validation_sample_count != 47_174
    ):
        raise Phase4ConfirmationError("wholesale latest development fold differs from audit")
    for result in results[1:]:
        current_folds = tuple(
            (fold.training_sample_count, fold.validation_sample_count, fold.validation_bucket)
            for fold in result.folds
        )
        current_slices = tuple(
            (item.status, item.metrics.sample_count) for item in result.status_slices
        )
        if current_folds != fold_shape or current_slices != slice_shape:
            raise Phase4ConfirmationError("confirmation candidates must use identical folds")


def _validate_screening_input(
    track: TrackName,
    report: object,
    report_hash: object,
) -> None:
    if not isinstance(report, Phase4ScreeningReport) or report.track != track:
        raise Phase4ConfirmationError("screening report does not match confirmation track")
    if report_hash != _SCREENING_REPORT_HASHES[track]:
        raise Phase4ConfirmationError("screening report hash differs from frozen evidence")
    if report.shortlist.full_development_candidate_ids != _FULL_DEVELOPMENT_IDS[track]:
        raise Phase4ConfirmationError("screening shortlist differs from frozen evidence")


def _validate_track_hashes(track: object, screening_hash: object, calibration_hash: object) -> None:
    if track not in TRACKS:
        raise Phase4ConfirmationError("confirmation track is invalid")
    resolved = track
    expected_calibration = (
        _RETAIL_CALIBRATION_HASH if resolved == "retail" else _WHOLESALE_CALIBRATION_HASH
    )
    if screening_hash != _SCREENING_REPORT_HASHES[resolved]:
        raise Phase4ConfirmationError("confirmation screening hash differs from frozen evidence")
    if calibration_hash != expected_calibration:
        raise Phase4ConfirmationError("confirmation calibration hash differs from audit")


def _canonical_json(value: Mapping[str, object]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _bounded_text(
    serialized: str | bytes,
    *,
    maximum_bytes: int = _MAX_CHECKPOINT_BYTES,
    label: str = "confirmation checkpoint",
) -> str:
    if isinstance(serialized, bytes):
        if len(serialized) > maximum_bytes:
            raise Phase4ConfirmationError(f"{label} exceeds maximum size")
        try:
            return serialized.decode("utf-8")
        except UnicodeDecodeError as error:
            raise Phase4ConfirmationError(f"{label} must be UTF-8") from error
    if isinstance(serialized, str):
        if len(serialized.encode("utf-8")) > maximum_bytes:
            raise Phase4ConfirmationError(f"{label} exceeds maximum size")
        return serialized
    raise Phase4ConfirmationError(f"{label} must be text or bytes")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise Phase4ConfirmationError(f"confirmation checkpoint has duplicate field: {key}")
        result[key] = value
    return result


__all__ = [
    "ConfirmationProgressCallback",
    "Phase4ConfirmationCheckpoint",
    "Phase4ConfirmationError",
    "Phase4ConfirmationReport",
    "canonical_phase4_confirmation_checkpoint_json",
    "canonical_phase4_confirmation_json",
    "make_phase4_confirmation_checkpoint",
    "parse_phase4_confirmation_checkpoint_json",
    "parse_phase4_confirmation_json",
    "run_retail_phase4_confirmation",
    "run_wholesale_phase4_confirmation",
]
