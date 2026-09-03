"""Pure, aggregate-only Phase 4 promotion and model-selection policy."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from .contracts import TRACKS, TrackName

RetailStatus: TypeAlias = Literal["certified", "new", "used"]

_RETAIL_STATUSES: Final[tuple[RetailStatus, ...]] = ("certified", "new", "used")
_CANDIDATE_ID_PATTERN = re.compile(
    r"^phase4-(retail|wholesale)-"
    r"(?:linear_regression_incumbent-00|random_forest-0[0-5]|gradient_boosting-0[0-5])$"
)
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_:.]{0,127}$")

_RETAIL_RELATIVE_MAE_GAIN: Final = 0.03
_RETAIL_ABSOLUTE_MAE_GAIN_USD: Final = 300.0
_WHOLESALE_RELATIVE_MAE_GAIN: Final = 0.02
_WHOLESALE_ABSOLUTE_MAE_GAIN_USD: Final = 50.0
_MAXIMUM_ERROR_REGRESSION: Final = 0.05
_NEAR_TIE_RELATIVE_MAE: Final = 0.01

_MAXIMUM_ARTIFACT_MB: Final = 50.0
_MAXIMUM_WARM_RSS_MB: Final = 350.0
_MAXIMUM_STARTUP_PEAK_MB: Final = 450.0
_MAXIMUM_P95_MS: Final = 500.0


class Phase4SelectionValidationError(ValueError):
    """Phase 4 selection input or derived output violated the frozen policy."""


@dataclass(frozen=True, slots=True)
class MicroOOFMetrics:
    """Micro-aggregated out-of-fold accuracy with no row-level values."""

    sample_count: int
    mae_usd: float
    rmse_usd: float

    def __post_init__(self) -> None:
        _positive_integer(self.sample_count, label="OOF sample_count")
        _nonnegative_number(self.mae_usd, label="OOF MAE")
        _nonnegative_number(self.rmse_usd, label="OOF RMSE")
        if self.rmse_usd < self.mae_usd:
            raise Phase4SelectionValidationError("OOF RMSE must be at least OOF MAE")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "mae_usd": self.mae_usd,
            "rmse_usd": self.rmse_usd,
        }


@dataclass(frozen=True, slots=True)
class StatusMAEAggregate:
    """One retail status's aggregate out-of-fold MAE and count."""

    status: RetailStatus
    sample_count: int
    mae_usd: float

    def __post_init__(self) -> None:
        if self.status not in _RETAIL_STATUSES:
            raise Phase4SelectionValidationError("retail status is invalid")
        _positive_integer(self.sample_count, label="status sample_count")
        _nonnegative_number(self.mae_usd, label="status MAE")

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "status": self.status,
            "sample_count": self.sample_count,
            "mae_usd": self.mae_usd,
        }


@dataclass(frozen=True, slots=True)
class LatestFoldMAEAggregate:
    """Wholesale latest-fold aggregate used by the regression guardrail."""

    sample_count: int
    mae_usd: float

    def __post_init__(self) -> None:
        _positive_integer(self.sample_count, label="latest-fold sample_count")
        _nonnegative_number(self.mae_usd, label="latest-fold MAE")

    def to_dict(self) -> dict[str, int | float]:
        return {"sample_count": self.sample_count, "mae_usd": self.mae_usd}


@dataclass(frozen=True, slots=True)
class DeploymentMeasurements:
    """Aggregate deployment measurements required by every candidate gate."""

    artifact_mb: float
    warm_rss_mb: float
    startup_peak_mb: float
    p95_ms: float

    def __post_init__(self) -> None:
        _nonnegative_number(self.artifact_mb, label="artifact_mb")
        _nonnegative_number(self.warm_rss_mb, label="warm_rss_mb")
        _nonnegative_number(self.startup_peak_mb, label="startup_peak_mb")
        _nonnegative_number(self.p95_ms, label="p95_ms")

    def to_dict(self) -> dict[str, float]:
        return {
            "artifact_mb": self.artifact_mb,
            "warm_rss_mb": self.warm_rss_mb,
            "startup_peak_mb": self.startup_peak_mb,
            "p95_ms": self.p95_ms,
        }


@dataclass(frozen=True, slots=True)
class Phase4CandidateAggregate:
    """All row-free evidence needed to evaluate one frozen candidate."""

    candidate_id: str
    track: TrackName
    oof: MicroOOFMetrics
    deployment: DeploymentMeasurements
    status_mae: tuple[StatusMAEAggregate, ...] = ()
    latest_fold_mae: LatestFoldMAEAggregate | None = None

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id, self.track)
        if not isinstance(self.oof, MicroOOFMetrics):
            raise Phase4SelectionValidationError("oof must be MicroOOFMetrics")
        if not isinstance(self.deployment, DeploymentMeasurements):
            raise Phase4SelectionValidationError("deployment must be DeploymentMeasurements")
        if not isinstance(self.status_mae, tuple):
            raise Phase4SelectionValidationError("status_mae must be an immutable tuple")
        if self.track == "retail":
            if any(not isinstance(item, StatusMAEAggregate) for item in self.status_mae):
                raise Phase4SelectionValidationError(
                    "retail status values must be StatusMAEAggregate"
                )
            statuses = tuple(item.status for item in self.status_mae)
            if statuses != _RETAIL_STATUSES:
                raise Phase4SelectionValidationError(
                    "retail candidates require certified, new, and used status aggregates"
                )
            if sum(item.sample_count for item in self.status_mae) != self.oof.sample_count:
                raise Phase4SelectionValidationError(
                    "retail status counts must sum to the OOF sample_count"
                )
            weighted_status_mae = (
                math.fsum(item.sample_count * item.mae_usd for item in self.status_mae)
                / self.oof.sample_count
            )
            if not math.isclose(
                weighted_status_mae,
                self.oof.mae_usd,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise Phase4SelectionValidationError(
                    "retail OOF MAE must equal the count-weighted status MAEs"
                )
            if self.latest_fold_mae is not None:
                raise Phase4SelectionValidationError(
                    "retail candidates must not carry a wholesale latest fold"
                )
        else:
            if self.status_mae:
                raise Phase4SelectionValidationError(
                    "wholesale candidates must not carry retail status aggregates"
                )
            if not isinstance(self.latest_fold_mae, LatestFoldMAEAggregate):
                raise Phase4SelectionValidationError(
                    "wholesale candidates require a latest-fold aggregate"
                )
            if self.latest_fold_mae.sample_count > self.oof.sample_count:
                raise Phase4SelectionValidationError(
                    "latest-fold count must not exceed the OOF sample_count"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "track": self.track,
            "oof": self.oof.to_dict(),
            "deployment": self.deployment.to_dict(),
            "status_mae": [item.to_dict() for item in self.status_mae],
            "latest_fold_mae": (
                None if self.latest_fold_mae is None else self.latest_fold_mae.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class PromotionGateResults:
    """Explicit booleans for every accuracy and deployment promotion gate."""

    relative_mae_gain: bool
    absolute_mae_gain: bool
    overall_rmse: bool
    retail_status_mae: bool | None
    certified_mae: bool | None
    new_mae: bool | None
    used_mae: bool | None
    wholesale_latest_fold_mae: bool | None
    artifact_mb: bool
    warm_rss_mb: bool
    startup_peak_mb: bool
    p95_ms: bool

    def __post_init__(self) -> None:
        for name in (
            "relative_mae_gain",
            "absolute_mae_gain",
            "overall_rmse",
            "artifact_mb",
            "warm_rss_mb",
            "startup_peak_mb",
            "p95_ms",
        ):
            if type(getattr(self, name)) is not bool:
                raise Phase4SelectionValidationError(f"{name} gate must be boolean")
        for name in (
            "retail_status_mae",
            "certified_mae",
            "new_mae",
            "used_mae",
            "wholesale_latest_fold_mae",
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise Phase4SelectionValidationError(f"{name} gate must be boolean or null")
        status_values = (self.certified_mae, self.new_mae, self.used_mae)
        if self.retail_status_mae is None:
            if any(value is not None for value in status_values):
                raise Phase4SelectionValidationError(
                    "retail status detail gates require the combined retail gate"
                )
        elif any(value is None for value in status_values):
            raise Phase4SelectionValidationError(
                "the combined retail gate requires all three status gates"
            )
        elif self.retail_status_mae != all(status_values):
            raise Phase4SelectionValidationError(
                "the combined retail gate must equal all three status gates"
            )

    @property
    def all_passed(self) -> bool:
        values = (
            self.relative_mae_gain,
            self.absolute_mae_gain,
            self.overall_rmse,
            self.retail_status_mae,
            self.certified_mae,
            self.new_mae,
            self.used_mae,
            self.wholesale_latest_fold_mae,
            self.artifact_mb,
            self.warm_rss_mb,
            self.startup_peak_mb,
            self.p95_ms,
        )
        return all(value for value in values if value is not None)

    def to_dict(self) -> dict[str, bool | None]:
        return {
            "relative_mae_gain": self.relative_mae_gain,
            "absolute_mae_gain": self.absolute_mae_gain,
            "overall_rmse": self.overall_rmse,
            "retail_status_mae": self.retail_status_mae,
            "certified_mae": self.certified_mae,
            "new_mae": self.new_mae,
            "used_mae": self.used_mae,
            "wholesale_latest_fold_mae": self.wholesale_latest_fold_mae,
            "artifact_mb": self.artifact_mb,
            "warm_rss_mb": self.warm_rss_mb,
            "startup_peak_mb": self.startup_peak_mb,
            "p95_ms": self.p95_ms,
        }


@dataclass(frozen=True, slots=True)
class CandidatePromotionResult:
    """One challenger's aggregate gate outcome and report-safe reasons."""

    track: TrackName
    candidate_id: str
    mae_gain_usd: float
    relative_mae_gain: float
    gates: PromotionGateResults
    eligible: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_candidate_id(self.candidate_id, self.track)
        _nonnegative_number(self.mae_gain_usd, label="mae_gain_usd")
        _nonnegative_number(self.relative_mae_gain, label="relative_mae_gain")
        if not isinstance(self.gates, PromotionGateResults):
            raise Phase4SelectionValidationError("gates must be PromotionGateResults")
        if type(self.eligible) is not bool or self.eligible != self.gates.all_passed:
            raise Phase4SelectionValidationError("eligible must exactly match all gate results")
        _validate_reasons(self.reasons)
        if self.reasons != _gate_reasons(self.gates):
            raise Phase4SelectionValidationError("reasons must exactly match failed gates")
        if self.track == "retail":
            if self.gates.retail_status_mae is None:
                raise Phase4SelectionValidationError("retail status gate is required")
            if self.gates.wholesale_latest_fold_mae is not None:
                raise Phase4SelectionValidationError("retail result has a wholesale gate")
        elif self.gates.retail_status_mae is not None:
            raise Phase4SelectionValidationError("wholesale result has a retail gate")
        elif self.gates.wholesale_latest_fold_mae is None:
            raise Phase4SelectionValidationError("wholesale latest-fold gate is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "candidate_id": self.candidate_id,
            "mae_gain_usd": self.mae_gain_usd,
            "relative_mae_gain": self.relative_mae_gain,
            "gates": self.gates.to_dict(),
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class Phase4SelectionDecision:
    """Deterministic aggregate-only promotion decision for one track."""

    track: TrackName
    incumbent_candidate_id: str
    selected_candidate_id: str
    incumbent_retained: bool
    challenger_results: tuple[CandidatePromotionResult, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_candidate_id(self.incumbent_candidate_id, self.track)
        _validate_candidate_id(self.selected_candidate_id, self.track)
        if self.incumbent_candidate_id != _incumbent_id(self.track):
            raise Phase4SelectionValidationError("incumbent candidate ID is invalid")
        if type(self.incumbent_retained) is not bool:
            raise Phase4SelectionValidationError("incumbent_retained must be boolean")
        if not isinstance(self.challenger_results, tuple):
            raise Phase4SelectionValidationError("challenger_results must be an immutable tuple")
        result_ids = tuple(result.candidate_id for result in self.challenger_results)
        if result_ids != tuple(sorted(result_ids)) or len(result_ids) != len(set(result_ids)):
            raise Phase4SelectionValidationError(
                "challenger results must have unique candidate IDs in stable order"
            )
        if any(result.track != self.track for result in self.challenger_results):
            raise Phase4SelectionValidationError("challenger result track does not match")
        eligible_ids = {
            result.candidate_id for result in self.challenger_results if result.eligible
        }
        expected_retained = not eligible_ids
        if self.incumbent_retained != expected_retained:
            raise Phase4SelectionValidationError(
                "incumbent retention must exactly reflect eligible challengers"
            )
        if self.incumbent_retained:
            if self.selected_candidate_id != self.incumbent_candidate_id:
                raise Phase4SelectionValidationError("fallback must select the incumbent")
        elif self.selected_candidate_id not in eligible_ids:
            raise Phase4SelectionValidationError("selected challenger must pass every gate")
        _validate_reasons(self.reasons)

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "selected_candidate_id": self.selected_candidate_id,
            "incumbent_retained": self.incumbent_retained,
            "challenger_results": [result.to_dict() for result in self.challenger_results],
            "reasons": list(self.reasons),
        }


def select_phase4_candidate(
    track: TrackName,
    incumbent: Phase4CandidateAggregate,
    challengers: tuple[Phase4CandidateAggregate, ...],
) -> Phase4SelectionDecision:
    """Apply every frozen promotion gate and choose one deterministic winner."""

    resolved_track = _validate_track(track)
    if not isinstance(incumbent, Phase4CandidateAggregate):
        raise Phase4SelectionValidationError("incumbent must be a candidate aggregate")
    if incumbent.track != resolved_track:
        raise Phase4SelectionValidationError("incumbent track does not match selection track")
    if incumbent.candidate_id != _incumbent_id(resolved_track):
        raise Phase4SelectionValidationError(
            "selection incumbent must be the same-fold Linear Regression candidate"
        )
    if not all(_deployment_gate_values(incumbent.deployment)):
        raise Phase4SelectionValidationError(
            "Linear Regression incumbent is not deployable under the frozen budgets"
        )
    if not isinstance(challengers, tuple):
        raise Phase4SelectionValidationError("challengers must be an immutable tuple")

    seen_ids = {incumbent.candidate_id}
    evaluated: list[tuple[Phase4CandidateAggregate, CandidatePromotionResult]] = []
    for challenger in challengers:
        if not isinstance(challenger, Phase4CandidateAggregate):
            raise Phase4SelectionValidationError("every challenger must be a candidate aggregate")
        if challenger.track != resolved_track:
            raise Phase4SelectionValidationError("challenger track does not match")
        if challenger.candidate_id in seen_ids:
            raise Phase4SelectionValidationError("candidate IDs must be unique")
        if "-linear_regression_incumbent-" in challenger.candidate_id:
            raise Phase4SelectionValidationError("the incumbent cannot also be a challenger")
        seen_ids.add(challenger.candidate_id)
        _validate_same_fold_shape(incumbent, challenger)
        evaluated.append((challenger, _evaluate_challenger(incumbent, challenger)))

    eligible = [(candidate, result) for candidate, result in evaluated if result.eligible]
    decision_reasons: tuple[str, ...]
    if not eligible:
        selected_id = incumbent.candidate_id
        retained = True
        decision_reasons = ("no_challenger_passed_all_gates",)
    else:
        best_mae = min(candidate.oof.mae_usd for candidate, _ in eligible)
        near_ties = [
            (candidate, result)
            for candidate, result in eligible
            if candidate.oof.mae_usd <= best_mae * (1.0 + _NEAR_TIE_RELATIVE_MAE)
        ]
        selected, _ = min(
            near_ties,
            key=lambda item: (
                item[0].deployment.artifact_mb,
                item[0].deployment.p95_ms,
                item[0].candidate_id,
            ),
        )
        selected_id = selected.candidate_id
        retained = False
        decision_reasons = (
            "challenger_passed_all_promotion_gates",
            "near_tie_order:artifact_mb:p95_ms:candidate_id",
        )

    results = tuple(sorted((result for _, result in evaluated), key=lambda item: item.candidate_id))
    return Phase4SelectionDecision(
        track=resolved_track,
        incumbent_candidate_id=incumbent.candidate_id,
        selected_candidate_id=selected_id,
        incumbent_retained=retained,
        challenger_results=results,
        reasons=decision_reasons,
    )


def _evaluate_challenger(
    incumbent: Phase4CandidateAggregate,
    challenger: Phase4CandidateAggregate,
) -> CandidatePromotionResult:
    track = incumbent.track
    raw_mae_gain = incumbent.oof.mae_usd - challenger.oof.mae_usd
    mae_gain = max(0.0, raw_mae_gain)
    relative_gain = 0.0 if incumbent.oof.mae_usd == 0.0 else mae_gain / incumbent.oof.mae_usd
    if track == "retail":
        relative_threshold = _RETAIL_RELATIVE_MAE_GAIN
        absolute_threshold = _RETAIL_ABSOLUTE_MAE_GAIN_USD
    else:
        relative_threshold = _WHOLESALE_RELATIVE_MAE_GAIN
        absolute_threshold = _WHOLESALE_ABSOLUTE_MAE_GAIN_USD

    relative_gate = raw_mae_gain >= incumbent.oof.mae_usd * relative_threshold
    absolute_gate = raw_mae_gain >= absolute_threshold
    rmse_gate = challenger.oof.rmse_usd <= incumbent.oof.rmse_usd * (
        1.0 + _MAXIMUM_ERROR_REGRESSION
    )

    retail_status_gate: bool | None = None
    wholesale_latest_gate: bool | None = None
    status_gate_values: dict[RetailStatus, bool] = {}
    if track == "retail":
        incumbent_status = {item.status: item for item in incumbent.status_mae}
        for status in challenger.status_mae:
            passed = status.mae_usd <= incumbent_status[status.status].mae_usd * (
                1.0 + _MAXIMUM_ERROR_REGRESSION
            )
            status_gate_values[status.status] = passed
        retail_status_gate = all(status_gate_values.values())
    else:
        if incumbent.latest_fold_mae is None or challenger.latest_fold_mae is None:
            raise Phase4SelectionValidationError("wholesale latest-fold aggregate is missing")
        wholesale_latest_gate = (
            challenger.latest_fold_mae.mae_usd
            <= incumbent.latest_fold_mae.mae_usd * (1.0 + _MAXIMUM_ERROR_REGRESSION)
        )

    deployment = challenger.deployment
    artifact_gate, warm_gate, startup_gate, latency_gate = _deployment_gate_values(deployment)
    gates = PromotionGateResults(
        relative_mae_gain=relative_gate,
        absolute_mae_gain=absolute_gate,
        overall_rmse=rmse_gate,
        retail_status_mae=retail_status_gate,
        certified_mae=status_gate_values.get("certified"),
        new_mae=status_gate_values.get("new"),
        used_mae=status_gate_values.get("used"),
        wholesale_latest_fold_mae=wholesale_latest_gate,
        artifact_mb=artifact_gate,
        warm_rss_mb=warm_gate,
        startup_peak_mb=startup_gate,
        p95_ms=latency_gate,
    )

    return CandidatePromotionResult(
        track=track,
        candidate_id=challenger.candidate_id,
        mae_gain_usd=mae_gain,
        relative_mae_gain=relative_gain,
        gates=gates,
        eligible=gates.all_passed,
        reasons=_gate_reasons(gates),
    )


def _validate_same_fold_shape(
    incumbent: Phase4CandidateAggregate,
    challenger: Phase4CandidateAggregate,
) -> None:
    if challenger.oof.sample_count != incumbent.oof.sample_count:
        raise Phase4SelectionValidationError("candidate OOF sample counts must match")
    if incumbent.track == "retail":
        incumbent_counts = tuple(item.sample_count for item in incumbent.status_mae)
        challenger_counts = tuple(item.sample_count for item in challenger.status_mae)
        if challenger_counts != incumbent_counts:
            raise Phase4SelectionValidationError("candidate retail status sample counts must match")
    else:
        if incumbent.latest_fold_mae is None or challenger.latest_fold_mae is None:
            raise Phase4SelectionValidationError("wholesale latest-fold aggregate is missing")
        if challenger.latest_fold_mae.sample_count != incumbent.latest_fold_mae.sample_count:
            raise Phase4SelectionValidationError("candidate latest-fold sample counts must match")


def _validate_track(track: object) -> TrackName:
    if not isinstance(track, str) or track not in TRACKS:
        raise Phase4SelectionValidationError("selection track is invalid")
    return track


def _validate_candidate_id(candidate_id: object, track: object) -> None:
    resolved_track = _validate_track(track)
    if not isinstance(candidate_id, str):
        raise Phase4SelectionValidationError("candidate ID must be stable lowercase text")
    match = _CANDIDATE_ID_PATTERN.fullmatch(candidate_id)
    if match is None:
        raise Phase4SelectionValidationError("candidate ID is invalid")
    if match.group(1) != resolved_track:
        raise Phase4SelectionValidationError("candidate ID track does not match")


def _incumbent_id(track: TrackName) -> str:
    return f"phase4-{track}-linear_regression_incumbent-00"


def _deployment_gate_values(
    deployment: DeploymentMeasurements,
) -> tuple[bool, bool, bool, bool]:
    return (
        deployment.artifact_mb <= _MAXIMUM_ARTIFACT_MB,
        deployment.warm_rss_mb <= _MAXIMUM_WARM_RSS_MB,
        deployment.startup_peak_mb <= _MAXIMUM_STARTUP_PEAK_MB,
        deployment.p95_ms <= _MAXIMUM_P95_MS,
    )


def _gate_reasons(gates: PromotionGateResults) -> tuple[str, ...]:
    reasons: list[str] = []
    if not gates.relative_mae_gain:
        reasons.append("relative_mae_gain_below_threshold")
    if not gates.absolute_mae_gain:
        reasons.append("absolute_mae_gain_below_threshold")
    if not gates.overall_rmse:
        reasons.append("overall_rmse_regression_above_5_percent")
    for status, passed in (
        ("certified", gates.certified_mae),
        ("new", gates.new_mae),
        ("used", gates.used_mae),
    ):
        if passed is False:
            reasons.append(f"{status}_mae_regression_above_5_percent")
    if gates.wholesale_latest_fold_mae is False:
        reasons.append("latest_fold_mae_regression_above_5_percent")
    if not gates.artifact_mb:
        reasons.append("artifact_mb_above_50")
    if not gates.warm_rss_mb:
        reasons.append("warm_rss_mb_above_350")
    if not gates.startup_peak_mb:
        reasons.append("startup_peak_mb_above_450")
    if not gates.p95_ms:
        reasons.append("p95_ms_above_500")
    if gates.all_passed:
        reasons.append("all_promotion_gates_passed")
    return tuple(reasons)


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Phase4SelectionValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4SelectionValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise Phase4SelectionValidationError(f"{label} must be finite and nonnegative")
    return number


def _validate_reasons(reasons: object) -> None:
    if not isinstance(reasons, tuple) or not reasons:
        raise Phase4SelectionValidationError("reasons must be a non-empty immutable tuple")
    if any(
        not isinstance(reason, str) or not _REASON_PATTERN.fullmatch(reason) for reason in reasons
    ):
        raise Phase4SelectionValidationError("reasons must be stable lowercase identifiers")
    if len(reasons) != len(set(reasons)):
        raise Phase4SelectionValidationError("reasons must be unique")


__all__ = [
    "CandidatePromotionResult",
    "DeploymentMeasurements",
    "LatestFoldMAEAggregate",
    "MicroOOFMetrics",
    "Phase4CandidateAggregate",
    "Phase4SelectionDecision",
    "Phase4SelectionValidationError",
    "PromotionGateResults",
    "RetailStatus",
    "StatusMAEAggregate",
    "select_phase4_candidate",
]
