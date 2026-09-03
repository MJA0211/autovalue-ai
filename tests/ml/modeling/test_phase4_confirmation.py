from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from autovalue_ml.modeling.candidates import CandidateSpec, candidate_specs
from autovalue_ml.modeling.experiment import FoldAggregate
from autovalue_ml.modeling.metrics import RegressionMetrics, StatusSliceMetrics
from autovalue_ml.modeling.phase4_confirmation import (
    Phase4ConfirmationError,
    Phase4ConfirmationReport,
    canonical_phase4_confirmation_checkpoint_json,
    canonical_phase4_confirmation_json,
    make_phase4_confirmation_checkpoint,
    parse_phase4_confirmation_checkpoint_json,
    parse_phase4_confirmation_json,
)
from autovalue_ml.modeling.phase4_evaluation import Phase4CandidateCVResult

_SCREENING_HASHES = {
    "retail": "62dcd2c1c41d30d49a4c98eab98e82529170aedd6f9313b46d00ffa50fdc4c9c",
    "wholesale": "0b0bb79ce82138215e6e8920f7b4ba57086e0f75e60cfc2095f8ab93e6e240c7",
}
_CALIBRATION_HASHES = {
    "retail": "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992",
    "wholesale": "f359c455accdfd8dc2de37ceab0ad218d81b5ee0e612d1e15fcd84fedd30f0d4",
}
_IDS = {
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
PROJECT_ROOT = Path(__file__).parents[3]


def _specs(track: str) -> tuple[CandidateSpec, ...]:
    by_id = {
        spec.candidate_id: spec
        for spec in candidate_specs(track)  # type: ignore[arg-type]
    }
    return tuple(by_id[candidate_id] for candidate_id in _IDS[track])


def _result(spec: CandidateSpec, mae: float) -> Phase4CandidateCVResult:
    rmse = mae + 100.0
    validation_counts: tuple[int, ...]
    training_counts: tuple[int, ...]
    buckets: tuple[str | None, ...]
    slices: tuple[StatusSliceMetrics, ...]
    if spec.track == "retail":
        validation_counts = (19_711, 19_711, 19_710, 19_710, 19_710)
        training_counts = tuple(98_552 - count for count in validation_counts)
        buckets = (None,) * 5
        slices = tuple(
            StatusSliceMetrics(
                status=status,
                metrics=RegressionMetrics(count, mae, rmse, 0.0),
            )
            for status, count in (("certified", 5_467), ("new", 58_360), ("used", 34_725))
        )
    else:
        validation_counts = (134_449, 158_432, 47_174)
        training_counts = (51_586, 186_035, 344_467)
        buckets = ("2015_01", "2015_02", "2015_03_04")
        slices = ()
    folds = tuple(
        FoldAggregate(
            fold_number=number,
            training_sample_count=training_count,
            validation_sample_count=validation_count,
            validation_bucket=bucket,
            metrics=RegressionMetrics(validation_count, mae, rmse, 0.0),
        )
        for number, (training_count, validation_count, bucket) in enumerate(
            zip(training_counts, validation_counts, buckets, strict=True),
            start=1,
        )
    )
    return Phase4CandidateCVResult(
        spec=spec,
        overall=RegressionMetrics(sum(validation_counts), mae, rmse, 0.0),
        status_slices=slices,
        folds=folds,
    )


def _results(track: str) -> tuple[Phase4CandidateCVResult, ...]:
    return tuple(
        _result(spec, 1_000.0 + position * 10) for position, spec in enumerate(_specs(track))
    )


@pytest.mark.parametrize("track", ["retail", "wholesale"])
def test_confirmation_report_is_aggregate_only_and_ranks_exact_mae(track: str) -> None:
    results = _results(track)
    report = Phase4ConfirmationReport(
        track=track,  # type: ignore[arg-type]
        screening_report_sha256=_SCREENING_HASHES[track],
        calibration_assignment_sha256=_CALIBRATION_HASHES[track],
        candidates=results,
    )
    serialized = canonical_phase4_confirmation_json(report)
    payload = json.loads(serialized)

    assert report.metric_ranking == _IDS[track]
    assert payload["target_semantics"].endswith(("_2023", "_2014_2015"))
    assert payload["data_boundaries"]["calibration_used_for_fitting_or_selection"] is False
    assert payload["data_boundaries"]["legacy_holdout_used"] is False
    assert payload["selection_scope"]["promotion_status"] == (
        "pending_deployment_measurements_and_gates"
    )
    assert all(name not in serialized for name in ("raw_rows", "predictions", "residuals"))


def test_confirmation_checkpoint_round_trips_stable_prefix() -> None:
    results = _results("retail")
    checkpoint = make_phase4_confirmation_checkpoint("retail", results[:3])
    serialized = canonical_phase4_confirmation_checkpoint_json(checkpoint)

    assert parse_phase4_confirmation_checkpoint_json(serialized) == checkpoint
    changed = json.loads(serialized)
    changed["screening_report_sha256"] = "0" * 64
    with pytest.raises(Phase4ConfirmationError, match="screening hash"):
        parse_phase4_confirmation_checkpoint_json(json.dumps(changed))
    duplicate = serialized.replace('"track":"retail"', '"track":"retail","track":"retail"')
    with pytest.raises(Phase4ConfirmationError, match="duplicate"):
        parse_phase4_confirmation_checkpoint_json(duplicate)


def test_confirmation_rejects_reordered_candidates_and_wrong_fold_boundary() -> None:
    results = _results("wholesale")
    with pytest.raises(Phase4ConfirmationError, match="exact five"):
        Phase4ConfirmationReport(
            track="wholesale",
            screening_report_sha256=_SCREENING_HASHES["wholesale"],
            calibration_assignment_sha256=_CALIBRATION_HASHES["wholesale"],
            candidates=(results[1], results[0], *results[2:]),
        )
    changed_fold = replace(
        results[1].folds[-1],
        training_sample_count=results[1].folds[-1].training_sample_count + 1,
    )
    changed_result = replace(results[1], folds=(*results[1].folds[:-1], changed_fold))
    with pytest.raises(Phase4ConfirmationError, match="identical folds"):
        Phase4ConfirmationReport(
            track="wholesale",
            screening_report_sha256=_SCREENING_HASHES["wholesale"],
            calibration_assignment_sha256=_CALIBRATION_HASHES["wholesale"],
            candidates=(results[0], changed_result, *results[2:]),
        )


def test_confirmation_checkpoint_requires_nonempty_shortlist_prefix() -> None:
    results = _results("retail")
    with pytest.raises(Phase4ConfirmationError, match="at least one"):
        make_phase4_confirmation_checkpoint("retail", ())
    with pytest.raises(Phase4ConfirmationError, match="stable shortlist prefix"):
        make_phase4_confirmation_checkpoint("retail", (results[1],))


@pytest.mark.parametrize("track", ["retail", "wholesale"])
def test_repository_confirmation_report_parses_as_canonical_evidence(track: str) -> None:
    path = PROJECT_ROOT / "docs" / "experiments" / f"phase4-{track}-full-development-v1.json"

    report = parse_phase4_confirmation_json(path.read_bytes())

    assert report.track == track
    assert len(report.candidates) == 5
    assert canonical_phase4_confirmation_json(report) == path.read_text(encoding="utf-8")


def test_confirmation_report_parser_rejects_ranking_drift() -> None:
    path = PROJECT_ROOT / "docs" / "experiments" / "phase4-retail-full-development-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["selection_scope"]["metric_ranking"].reverse()

    with pytest.raises(Phase4ConfirmationError, match="canonical evidence"):
        parse_phase4_confirmation_json(json.dumps(value))
