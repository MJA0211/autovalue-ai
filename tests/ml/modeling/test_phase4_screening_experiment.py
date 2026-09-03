from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from autovalue_ml.modeling.candidates import CandidateSpec, candidate_specs
from autovalue_ml.modeling.experiment import FoldAggregate
from autovalue_ml.modeling.metrics import RegressionMetrics, StatusSliceMetrics
from autovalue_ml.modeling.phase4_evaluation import (
    Phase4CandidateCVResult,
    shortlist_phase4_candidates,
)
from autovalue_ml.modeling.phase4_screening_experiment import (
    Phase4ScreeningError,
    Phase4ScreeningReport,
    ScreeningSliceCount,
    _partition_hash,
    canonical_phase4_checkpoint_json,
    canonical_phase4_screening_json,
    make_phase4_screening_checkpoint,
    parse_phase4_checkpoint_json,
    parse_phase4_screening_json,
)

PROJECT_ROOT = Path(__file__).parents[3]

_RETAIL_CALIBRATION_HASH = "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
_RETAIL_SCREENING_HASH = "fe8954d81f681c0d3ce7253d8a23f7995e9789693d7a261c851bc4078e173988"
_WHOLESALE_CALIBRATION_HASH = "f359c455accdfd8dc2de37ceab0ad218d81b5ee0e612d1e15fcd84fedd30f0d4"
_WHOLESALE_SCREENING_HASH = "cecdf8d34fc7d549024dfdb22ae83a371855808a2b405f35bcb858e665d01bc1"


def _candidate_result(
    spec: CandidateSpec,
    *,
    mae: float,
) -> Phase4CandidateCVResult:
    rmse = mae + 1.0
    validation_counts: tuple[int, ...]
    training_counts: tuple[int, ...]
    bucket_labels: tuple[str | None, ...]
    slices: tuple[StatusSliceMetrics, ...]
    if spec.track == "retail":
        validation_counts = (5_924, 5_924, 5_924, 5_924, 5_923)
        training_counts = tuple(29_619 - count for count in validation_counts)
        bucket_labels = (None,) * 5
        status_counts = (("certified", 1_640), ("new", 17_562), ("used", 10_417))
        slices = tuple(
            StatusSliceMetrics(
                status=status,
                metrics=RegressionMetrics(count, mae, rmse, 0.0),
            )
            for status, count in status_counts
        )
    else:
        validation_counts = (33_612, 39_608, 11_793)
        training_counts = (12_896, 46_508, 86_116)
        bucket_labels = ("2015_01", "2015_02", "2015_03_04")
        slices = ()
    overall_count = sum(validation_counts)
    folds = tuple(
        FoldAggregate(
            fold_number=fold_number,
            training_sample_count=training_count,
            validation_sample_count=validation_count,
            validation_bucket=bucket,
            metrics=RegressionMetrics(validation_count, mae, rmse, 0.0),
        )
        for fold_number, (training_count, validation_count, bucket) in enumerate(
            zip(training_counts, validation_counts, bucket_labels, strict=True),
            start=1,
        )
    )
    return Phase4CandidateCVResult(
        spec=spec,
        overall=RegressionMetrics(overall_count, mae, rmse, 0.0),
        status_slices=slices,
        folds=folds,
    )


def _results(track: str) -> tuple[Phase4CandidateCVResult, ...]:
    return tuple(
        _candidate_result(spec, mae=1_000.0 + spec.index + family_offset)
        for spec in candidate_specs(track)  # type: ignore[arg-type]
        for family_offset in (
            200.0
            if spec.family == "linear_regression_incumbent"
            else 100.0
            if spec.family == "random_forest"
            else 0.0,
        )
    )


def _retail_report() -> Phase4ScreeningReport:
    results = _results("retail")
    return Phase4ScreeningReport(
        track="retail",
        phase3_train_sample_count=109_510,
        development_sample_count=98_552,
        calibration_sample_count=10_958,
        screening_sample_count=29_619,
        calibration_assignment_sha256=_RETAIL_CALIBRATION_HASH,
        screening_assignment_sha256=_RETAIL_SCREENING_HASH,
        screening_slices=(
            ScreeningSliceCount("certified", 1_640),
            ScreeningSliceCount("new", 17_562),
            ScreeningSliceCount("used", 10_417),
        ),
        cv_scheme="retail_predictor_group_kfold_v1",
        bucket_order=(),
        candidates=results,
        shortlist=shortlist_phase4_candidates("retail", results),
    )


def test_screening_report_is_deterministic_aggregate_only_evidence() -> None:
    report = _retail_report()

    first = canonical_phase4_screening_json(report)
    second = canonical_phase4_screening_json(report)
    payload = json.loads(first)

    assert first == second
    assert first.endswith("\n")
    assert payload["data_boundaries"]["calibration_used_for_fitting_or_selection"] is False
    assert payload["data_boundaries"]["legacy_holdout_used"] is False
    assert payload["shortlist"]["gradient_boosting_candidate_ids"] == [
        "phase4-retail-gradient_boosting-00",
        "phase4-retail-gradient_boosting-01",
    ]
    assert all(
        forbidden not in first
        for forbidden in ("raw_row", "prediction_values", "residual_values", "target_values")
    )


def test_wholesale_report_preserves_warmup_without_scoring_it() -> None:
    results = _results("wholesale")
    report = Phase4ScreeningReport(
        track="wholesale",
        phase3_train_sample_count=442_130,
        development_sample_count=391_641,
        calibration_sample_count=50_489,
        screening_sample_count=97_909,
        calibration_assignment_sha256=_WHOLESALE_CALIBRATION_HASH,
        screening_assignment_sha256=_WHOLESALE_SCREENING_HASH,
        screening_slices=(
            ScreeningSliceCount("warmup", 12_896),
            ScreeningSliceCount("2015_01", 33_612),
            ScreeningSliceCount("2015_02", 39_608),
            ScreeningSliceCount("2015_03_04", 11_793),
        ),
        cv_scheme="wholesale_forward_chaining_cv_bucket_v1",
        bucket_order=("warmup", "2015_01", "2015_02", "2015_03_04"),
        candidates=results,
        shortlist=shortlist_phase4_candidates("wholesale", results),
    )

    assert report.candidates[0].overall.sample_count == 85_013
    assert report.screening_sample_count == 97_909


def test_screening_report_rejects_changed_boundary_or_shortlist() -> None:
    report = _retail_report()
    with pytest.raises(Phase4ScreeningError, match="row counts"):
        replace(report, screening_sample_count=29_618)
    with pytest.raises(Phase4ScreeningError, match="assignment hash"):
        replace(report, screening_assignment_sha256="0" * 64)
    with pytest.raises(Phase4ScreeningError, match="shortlist"):
        replace(
            report,
            shortlist=replace(
                report.shortlist,
                random_forest_candidate_ids=(
                    "phase4-retail-random_forest-04",
                    "phase4-retail-random_forest-05",
                ),
            ),
        )


def test_partition_hash_has_exact_lineage_format_and_rejects_bad_indices() -> None:
    selected = np.asarray([1, 3], dtype=np.int64)
    expected = hashlib.sha256(
        b"0,development\n1,calibration\n2,development\n3,calibration\n"
    ).hexdigest()

    assert (
        _partition_hash(
            selected,
            population_count=4,
            selected_label="calibration",
            unselected_label="development",
        )
        == expected
    )
    with pytest.raises(Phase4ScreeningError, match="unique"):
        _partition_hash(
            np.asarray([1, 1], dtype=np.int64),
            population_count=4,
            selected_label="yes",
            unselected_label="no",
        )
    with pytest.raises(Phase4ScreeningError, match="outside"):
        _partition_hash(
            np.asarray([4], dtype=np.int64),
            population_count=4,
            selected_label="yes",
            unselected_label="no",
        )
    with pytest.raises(Phase4ScreeningError, match="integers"):
        _partition_hash(
            selected,
            population_count=4,
            selected_label="yes",
            unselected_label="no",
            positions=np.asarray([10.0, 11.0, 12.0, 13.0]),
        )


def test_checkpoint_round_trips_candidate_prefix_and_rejects_tampering() -> None:
    candidates = _results("retail")[:4]
    checkpoint = make_phase4_screening_checkpoint("retail", candidates)
    serialized = canonical_phase4_checkpoint_json(checkpoint)

    assert parse_phase4_checkpoint_json(serialized) == checkpoint
    assert serialized == canonical_phase4_checkpoint_json(checkpoint)
    assert "raw_row" not in serialized

    changed = json.loads(serialized)
    changed["policy_sha256"] = "0" * 64
    with pytest.raises(Phase4ScreeningError, match="policy metadata"):
        parse_phase4_checkpoint_json(json.dumps(changed))
    changed_schema = json.loads(serialized)
    changed_schema["schema_version"] = True
    with pytest.raises(Phase4ScreeningError, match="policy metadata"):
        parse_phase4_checkpoint_json(json.dumps(changed_schema))
    duplicate = serialized.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    with pytest.raises(Phase4ScreeningError, match="duplicate"):
        parse_phase4_checkpoint_json(duplicate)


def test_checkpoint_requires_a_stable_nonempty_candidate_prefix() -> None:
    results = _results("retail")
    with pytest.raises(Phase4ScreeningError, match="at least one"):
        make_phase4_screening_checkpoint("retail", ())
    with pytest.raises(Phase4ScreeningError, match="stable policy prefix"):
        make_phase4_screening_checkpoint("retail", (results[1],))


@pytest.mark.parametrize("track", ["retail", "wholesale"])
def test_repository_screening_reports_parse_as_exact_canonical_evidence(track: str) -> None:
    path = PROJECT_ROOT / "docs" / "experiments" / f"phase4-{track}-screening-v1.json"

    report = parse_phase4_screening_json(path.read_bytes())

    assert report.track == track
    assert len(report.candidates) == 13
    assert canonical_phase4_screening_json(report) == path.read_text(encoding="utf-8")


def test_screening_report_parser_rejects_shortlist_or_extra_field_drift() -> None:
    path = PROJECT_ROOT / "docs" / "experiments" / "phase4-retail-screening-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["shortlist"]["random_forest_candidate_ids"].reverse()
    with pytest.raises(Phase4ScreeningError, match="canonical evidence"):
        parse_phase4_screening_json(json.dumps(value))

    value = json.loads(path.read_text(encoding="utf-8"))
    value["raw_rows"] = []
    with pytest.raises(Phase4ScreeningError, match="canonical evidence"):
        parse_phase4_screening_json(json.dumps(value))
