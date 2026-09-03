from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from autovalue_ml.modeling.phase4_selection import (
    DeploymentMeasurements,
    LatestFoldMAEAggregate,
    MicroOOFMetrics,
    Phase4CandidateAggregate,
    Phase4SelectionValidationError,
    RetailStatus,
    StatusMAEAggregate,
    select_phase4_candidate,
)


def _deployment(
    *,
    artifact_mb: float = 25.0,
    warm_rss_mb: float = 200.0,
    startup_peak_mb: float = 300.0,
    p95_ms: float = 100.0,
) -> DeploymentMeasurements:
    return DeploymentMeasurements(
        artifact_mb=artifact_mb,
        warm_rss_mb=warm_rss_mb,
        startup_peak_mb=startup_peak_mb,
        p95_ms=p95_ms,
    )


def _retail(
    candidate_id: str,
    *,
    mae: float = 9_600.0,
    rmse: float | None = None,
    status_maes: tuple[float, float, float] | None = None,
    counts: tuple[int, int, int] = (20, 30, 50),
    deployment: DeploymentMeasurements | None = None,
) -> Phase4CandidateAggregate:
    resolved_status_maes = (mae, mae, mae) if status_maes is None else status_maes
    resolved_rmse = max(12_000.0, mae * 1.2) if rmse is None else rmse
    statuses: tuple[RetailStatus, ...] = ("certified", "new", "used")
    return Phase4CandidateAggregate(
        candidate_id=candidate_id,
        track="retail",
        oof=MicroOOFMetrics(sample_count=sum(counts), mae_usd=mae, rmse_usd=resolved_rmse),
        deployment=_deployment() if deployment is None else deployment,
        status_mae=tuple(
            StatusMAEAggregate(status=status, sample_count=count, mae_usd=status_mae)
            for status, count, status_mae in zip(
                statuses, counts, resolved_status_maes, strict=True
            )
        ),
    )


def _retail_incumbent(
    *, mae: float = 10_000.0, rmse: float | None = None
) -> Phase4CandidateAggregate:
    return _retail(
        "phase4-retail-linear_regression_incumbent-00",
        mae=mae,
        rmse=rmse,
    )


def _wholesale(
    candidate_id: str,
    *,
    mae: float = 2_400.0,
    rmse: float = 3_000.0,
    sample_count: int = 100,
    latest_mae: float = 2_000.0,
    latest_count: int = 30,
    deployment: DeploymentMeasurements | None = None,
) -> Phase4CandidateAggregate:
    return Phase4CandidateAggregate(
        candidate_id=candidate_id,
        track="wholesale",
        oof=MicroOOFMetrics(sample_count=sample_count, mae_usd=mae, rmse_usd=rmse),
        deployment=_deployment() if deployment is None else deployment,
        latest_fold_mae=LatestFoldMAEAggregate(
            sample_count=latest_count,
            mae_usd=latest_mae,
        ),
    )


def _wholesale_incumbent(
    *, mae: float = 2_500.0, rmse: float = 3_000.0, latest_mae: float = 2_000.0
) -> Phase4CandidateAggregate:
    return _wholesale(
        "phase4-wholesale-linear_regression_incumbent-00",
        mae=mae,
        rmse=rmse,
        latest_mae=latest_mae,
    )


def test_retail_all_threshold_equalities_pass_including_deployment() -> None:
    incumbent = _retail_incumbent()
    challenger = _retail(
        "phase4-retail-random_forest-00",
        mae=9_700.0,
        rmse=12_600.0,
        status_maes=(9_000.0, 10_500.0, 9_500.0),
        deployment=_deployment(
            artifact_mb=50.0,
            warm_rss_mb=350.0,
            startup_peak_mb=450.0,
            p95_ms=500.0,
        ),
    )

    decision = select_phase4_candidate("retail", incumbent, (challenger,))
    result = decision.challenger_results[0]

    assert result.eligible is True
    assert result.mae_gain_usd == 300.0
    assert result.relative_mae_gain == pytest.approx(0.03)
    assert all(value is not False for value in result.gates.to_dict().values())
    assert decision.selected_candidate_id == challenger.candidate_id
    assert decision.incumbent_retained is False


def test_retail_requires_both_relative_and_absolute_mae_gains() -> None:
    relative_failure = select_phase4_candidate(
        "retail",
        _retail_incumbent(mae=20_000.0),
        (_retail("phase4-retail-random_forest-00", mae=19_500.0),),
    ).challenger_results[0]
    absolute_failure = select_phase4_candidate(
        "retail",
        _retail_incumbent(mae=5_000.0),
        (_retail("phase4-retail-random_forest-01", mae=4_800.0),),
    ).challenger_results[0]

    assert relative_failure.gates.absolute_mae_gain is True
    assert relative_failure.gates.relative_mae_gain is False
    assert relative_failure.eligible is False
    assert absolute_failure.gates.relative_mae_gain is True
    assert absolute_failure.gates.absolute_mae_gain is False
    assert absolute_failure.eligible is False


def test_retail_rejects_overall_or_any_status_mae_regression_above_five_percent() -> None:
    incumbent = _retail_incumbent()
    rmse_failure = _retail(
        "phase4-retail-random_forest-00",
        mae=9_600.0,
        rmse=12_600.01,
    )
    status_failure = _retail(
        "phase4-retail-random_forest-01",
        mae=9_600.0,
        status_maes=(9_000.0, 10_500.01, 9_299.994),
    )

    decision = select_phase4_candidate("retail", incumbent, (rmse_failure, status_failure))
    by_id = {result.candidate_id: result for result in decision.challenger_results}

    assert by_id[rmse_failure.candidate_id].gates.overall_rmse is False
    assert by_id[status_failure.candidate_id].gates.retail_status_mae is False
    assert "new_mae_regression_above_5_percent" in by_id[status_failure.candidate_id].reasons
    assert decision.incumbent_retained is True


def test_wholesale_all_threshold_equalities_pass_and_latest_fold_is_guarded() -> None:
    incumbent = _wholesale_incumbent()
    equality = _wholesale(
        "phase4-wholesale-gradient_boosting-00",
        mae=2_450.0,
        rmse=3_150.0,
        latest_mae=2_100.0,
    )
    latest_failure = _wholesale(
        "phase4-wholesale-gradient_boosting-01",
        mae=2_400.0,
        latest_mae=2_100.01,
    )

    decision = select_phase4_candidate("wholesale", incumbent, (equality, latest_failure))
    by_id = {result.candidate_id: result for result in decision.challenger_results}

    assert by_id[equality.candidate_id].eligible is True
    assert by_id[equality.candidate_id].relative_mae_gain == pytest.approx(0.02)
    assert by_id[equality.candidate_id].mae_gain_usd == 50.0
    assert by_id[equality.candidate_id].gates.overall_rmse is True
    assert by_id[equality.candidate_id].gates.wholesale_latest_fold_mae is True
    assert by_id[latest_failure.candidate_id].gates.wholesale_latest_fold_mae is False
    assert decision.selected_candidate_id == equality.candidate_id


@pytest.mark.parametrize(
    ("field", "value", "gate"),
    [
        ("artifact_mb", 50.01, "artifact_mb"),
        ("warm_rss_mb", 350.01, "warm_rss_mb"),
        ("startup_peak_mb", 450.01, "startup_peak_mb"),
        ("p95_ms", 500.01, "p95_ms"),
    ],
)
def test_each_deployment_limit_is_a_required_gate(field: str, value: float, gate: str) -> None:
    measurements = replace(_deployment(), **{field: value})
    challenger = _retail(
        "phase4-retail-random_forest-00",
        deployment=measurements,
    )

    result = select_phase4_candidate(
        "retail", _retail_incumbent(), (challenger,)
    ).challenger_results[0]

    assert result.gates.to_dict()[gate] is False
    assert result.eligible is False


def test_near_ties_use_artifact_then_latency_then_stable_id() -> None:
    incumbent = _retail_incumbent()
    best_mae = _retail(
        "phase4-retail-random_forest-00",
        mae=9_000.0,
        deployment=_deployment(artifact_mb=40.0, p95_ms=80.0),
    )
    smaller_near_tie = _retail(
        "phase4-retail-gradient_boosting-01",
        mae=9_080.0,
        deployment=_deployment(artifact_mb=30.0, p95_ms=200.0),
    )
    outside_near_tie = _retail(
        "phase4-retail-random_forest-02",
        mae=9_100.0,
        deployment=_deployment(artifact_mb=1.0, p95_ms=1.0),
    )

    decision = select_phase4_candidate(
        "retail",
        incumbent,
        (outside_near_tie, best_mae, smaller_near_tie),
    )
    assert decision.selected_candidate_id == smaller_near_tie.candidate_id

    lower_latency = _retail(
        "phase4-retail-random_forest-03",
        mae=9_050.0,
        deployment=_deployment(artifact_mb=30.0, p95_ms=50.0),
    )
    decision = select_phase4_candidate("retail", incumbent, (smaller_near_tie, lower_latency))
    assert decision.selected_candidate_id == lower_latency.candidate_id

    lexical_first = replace(
        lower_latency,
        candidate_id="phase4-retail-gradient_boosting-00",
    )
    decision = select_phase4_candidate("retail", incumbent, (lower_latency, lexical_first))
    assert decision.selected_candidate_id == lexical_first.candidate_id


def test_no_eligible_challenger_falls_back_to_incumbent() -> None:
    incumbent = _retail_incumbent()
    empty = select_phase4_candidate("retail", incumbent, ())
    worse = _retail("phase4-retail-random_forest-00", mae=10_500.0)
    failed = select_phase4_candidate("retail", incumbent, (worse,))

    assert empty.selected_candidate_id == incumbent.candidate_id
    assert empty.incumbent_retained is True
    assert failed.selected_candidate_id == incumbent.candidate_id
    assert failed.incumbent_retained is True
    assert failed.challenger_results[0].mae_gain_usd == 0.0
    assert "no_challenger_passed_all_gates" in failed.reasons


@pytest.mark.parametrize(
    "bad_metrics",
    [
        {"sample_count": True, "mae_usd": 1.0, "rmse_usd": 1.0},
        {"sample_count": 0, "mae_usd": 1.0, "rmse_usd": 1.0},
        {"sample_count": 1, "mae_usd": -1.0, "rmse_usd": 1.0},
        {"sample_count": 1, "mae_usd": float("nan"), "rmse_usd": 1.0},
        {"sample_count": 1, "mae_usd": 1.0, "rmse_usd": float("inf")},
        {"sample_count": 1, "mae_usd": True, "rmse_usd": 1.0},
        {"sample_count": 1, "mae_usd": 2.0, "rmse_usd": 1.0},
    ],
)
def test_metric_contract_rejects_booleans_nonfinite_negative_and_bad_counts(
    bad_metrics: dict[str, object],
) -> None:
    with pytest.raises(Phase4SelectionValidationError):
        MicroOOFMetrics(**bad_metrics)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_mb", -1.0),
        ("warm_rss_mb", float("nan")),
        ("startup_peak_mb", float("inf")),
        ("p95_ms", True),
    ],
)
def test_deployment_contract_rejects_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "artifact_mb": 1.0,
        "warm_rss_mb": 1.0,
        "startup_peak_mb": 1.0,
        "p95_ms": 1.0,
    }
    values[field] = value
    with pytest.raises(Phase4SelectionValidationError):
        DeploymentMeasurements(**values)  # type: ignore[arg-type]


def test_candidate_contract_rejects_missing_statuses_wrong_shapes_and_invalid_ids() -> None:
    valid = _retail("phase4-retail-random_forest-00")
    with pytest.raises(Phase4SelectionValidationError, match="certified, new, and used"):
        replace(valid, status_mae=valid.status_mae[:-1], oof=replace(valid.oof, sample_count=50))
    with pytest.raises(Phase4SelectionValidationError, match="sum"):
        replace(
            valid,
            status_mae=(replace(valid.status_mae[0], sample_count=19), *valid.status_mae[1:]),
        )
    with pytest.raises(Phase4SelectionValidationError, match="count-weighted"):
        replace(
            valid,
            status_mae=(replace(valid.status_mae[0], mae_usd=9_601.0), *valid.status_mae[1:]),
        )
    with pytest.raises(Phase4SelectionValidationError, match="must not carry"):
        replace(valid, latest_fold_mae=LatestFoldMAEAggregate(1, 1.0))
    with pytest.raises(Phase4SelectionValidationError, match="candidate ID"):
        replace(valid, candidate_id="Random Forest")

    wholesale = _wholesale("phase4-wholesale-random_forest-00")
    with pytest.raises(Phase4SelectionValidationError, match="must not carry retail"):
        replace(wholesale, status_mae=valid.status_mae)
    with pytest.raises(Phase4SelectionValidationError, match="require a latest-fold"):
        replace(wholesale, latest_fold_mae=None)
    with pytest.raises(Phase4SelectionValidationError, match="must not exceed"):
        replace(
            wholesale,
            latest_fold_mae=LatestFoldMAEAggregate(
                sample_count=wholesale.oof.sample_count + 1,
                mae_usd=1.0,
            ),
        )


def test_selection_rejects_track_count_and_candidate_id_mismatches() -> None:
    incumbent = _retail_incumbent()
    challenger = _retail("phase4-retail-random_forest-00")
    different_counts = _retail(
        "phase4-retail-random_forest-01",
        counts=(21, 30, 50),
    )
    different_status_counts = _retail(
        "phase4-retail-random_forest-02",
        counts=(21, 29, 50),
    )

    with pytest.raises(Phase4SelectionValidationError, match="OOF sample counts"):
        select_phase4_candidate("retail", incumbent, (different_counts,))
    with pytest.raises(Phase4SelectionValidationError, match="status sample counts"):
        select_phase4_candidate("retail", incumbent, (different_status_counts,))
    with pytest.raises(Phase4SelectionValidationError, match="unique"):
        select_phase4_candidate("retail", incumbent, (challenger, challenger))
    with pytest.raises(Phase4SelectionValidationError, match="track"):
        select_phase4_candidate(
            "retail",
            incumbent,
            (_wholesale("phase4-wholesale-random_forest-00"),),
        )
    with pytest.raises(Phase4SelectionValidationError, match="immutable tuple"):
        select_phase4_candidate("retail", incumbent, [challenger])  # type: ignore[arg-type]

    wholesale_incumbent = _wholesale_incumbent()
    latest_count_mismatch = _wholesale(
        "phase4-wholesale-random_forest-00",
        latest_count=29,
    )
    with pytest.raises(Phase4SelectionValidationError, match="latest-fold sample counts"):
        select_phase4_candidate("wholesale", wholesale_incumbent, (latest_count_mismatch,))


def test_selection_rejects_an_undeployable_incumbent() -> None:
    incumbent = replace(
        _retail_incumbent(),
        deployment=_deployment(artifact_mb=50.01),
    )
    with pytest.raises(Phase4SelectionValidationError, match="incumbent is not deployable"):
        select_phase4_candidate("retail", incumbent, ())


def test_decision_is_immutable_deterministic_and_aggregate_only() -> None:
    incumbent = _retail_incumbent()
    challenger = _retail("phase4-retail-random_forest-00")

    first = select_phase4_candidate("retail", incumbent, (challenger,))
    second = select_phase4_candidate("retail", incumbent, (challenger,))
    serialized = json.dumps(first.to_dict(), sort_keys=True)

    assert first == second
    assert all(word not in serialized for word in ("rows", "predictions", "residuals"))
    with pytest.raises(FrozenInstanceError):
        first.selected_candidate_id = incumbent.candidate_id  # type: ignore[misc]
