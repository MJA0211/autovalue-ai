"""Development-only candidate evaluation and deterministic Phase 4 shortlists."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .candidates import (
    RANDOM_FOREST_TRAINING_N_JOBS,
    CandidateSpec,
    candidate_specs,
    get_candidate_spec,
    make_candidate_pipeline,
)
from .contracts import (
    TRACKS,
    FeatureContractError,
    TrackName,
    validate_feature_frame,
    validate_target,
)
from .cv import CVSplit
from .experiment import FoldAggregate
from .metrics import (
    RegressionMetrics,
    StatusSliceMetrics,
    regression_metrics,
    retail_status_metrics,
)

_RETAIL_STATUSES: Final = ("certified", "new", "used")
_SHORTLIST_SIZE: Final = 2
_SHORTLIST_METRIC: Final = "micro_out_of_fold_mae_usd"
_CANDIDATE_RESULT_KEYS: Final = {
    "candidate_id",
    "family",
    "index",
    "parameters",
    "random_state",
    "overall",
    "status_slices",
    "folds",
}
_METRIC_KEYS: Final = {"sample_count", "mae", "rmse", "r2"}
_SLICE_KEYS: Final = {"status", "metrics"}
_FOLD_KEYS: Final = {
    "fold_number",
    "training_sample_count",
    "validation_sample_count",
    "validation_bucket",
    "metrics",
}


class Phase4EvaluationError(ValueError):
    """Candidate evaluation inputs or aggregate outputs violated the protocol."""


@dataclass(frozen=True, slots=True)
class Phase4CandidateCVResult:
    """Aggregate-only out-of-fold result for one exact candidate."""

    spec: CandidateSpec
    overall: RegressionMetrics
    status_slices: tuple[StatusSliceMetrics, ...]
    folds: tuple[FoldAggregate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, CandidateSpec):
            raise Phase4EvaluationError("spec must be a CandidateSpec")
        try:
            canonical_spec = get_candidate_spec(self.spec.track, self.spec.family, self.spec.index)
        except ValueError as error:
            raise Phase4EvaluationError("candidate spec is outside the frozen policy") from error
        if self.spec != canonical_spec:
            raise Phase4EvaluationError("candidate spec differs from the frozen policy")
        _validate_metrics(self.overall, label="candidate overall")
        if not isinstance(self.folds, tuple) or not self.folds:
            raise Phase4EvaluationError("candidate folds must be a non-empty immutable tuple")
        if any(not isinstance(fold, FoldAggregate) for fold in self.folds):
            raise Phase4EvaluationError("candidate folds must contain FoldAggregate values")
        if tuple(fold.fold_number for fold in self.folds) != tuple(range(1, len(self.folds) + 1)):
            raise Phase4EvaluationError("candidate fold numbers must be consecutive from one")
        if sum(fold.validation_sample_count for fold in self.folds) != self.overall.sample_count:
            raise Phase4EvaluationError("fold validation counts must sum to the OOF sample count")
        expected_fold_count = 5 if self.spec.track == "retail" else 3
        if len(self.folds) != expected_fold_count:
            raise Phase4EvaluationError(
                f"{self.spec.track} candidate results require {expected_fold_count} folds"
            )
        if self.spec.track == "retail":
            if any(fold.validation_bucket is not None for fold in self.folds):
                raise Phase4EvaluationError("retail candidate fold bucket labels must be null")
        elif any(fold.validation_bucket is None for fold in self.folds):
            raise Phase4EvaluationError("wholesale candidate folds require bucket labels")
        weighted_fold_mae = (
            math.fsum(fold.validation_sample_count * fold.metrics.mae for fold in self.folds)
            / self.overall.sample_count
        )
        weighted_fold_mse = (
            math.fsum(fold.validation_sample_count * fold.metrics.rmse**2 for fold in self.folds)
            / self.overall.sample_count
        )
        if not math.isclose(weighted_fold_mae, self.overall.mae, rel_tol=1e-12, abs_tol=1e-9):
            raise Phase4EvaluationError("candidate overall MAE must equal weighted fold MAEs")
        if not math.isclose(
            math.sqrt(weighted_fold_mse),
            self.overall.rmse,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise Phase4EvaluationError("candidate overall RMSE must equal pooled fold RMSE")
        _validate_status_slices(self.spec.track, self.status_slices, self.overall)

    @property
    def latest_fold(self) -> FoldAggregate:
        """Return the final ordered fold used by the wholesale guardrail."""

        return self.folds[-1]

    def to_dict(self) -> dict[str, object]:
        """Return the row-free candidate schema in stable field order."""

        return {
            "candidate_id": self.spec.candidate_id,
            "family": self.spec.family,
            "index": self.spec.index,
            "parameters": list(self.spec.parameters),
            "random_state": self.spec.random_state,
            "overall": self.overall.to_dict(),
            "status_slices": [item.to_dict() for item in self.status_slices],
            "folds": [fold.to_dict() for fold in self.folds],
        }


@dataclass(frozen=True, slots=True)
class Phase4Shortlist:
    """Two deterministic full-development candidates from each challenger family."""

    track: TrackName
    linear_reference_id: str
    random_forest_candidate_ids: tuple[str, str]
    gradient_boosting_candidate_ids: tuple[str, str]
    metric: str = _SHORTLIST_METRIC

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise Phase4EvaluationError("shortlist track is invalid")
        if self.metric != _SHORTLIST_METRIC:
            raise Phase4EvaluationError("shortlist metric is invalid")
        expected_linear = get_candidate_spec(self.track, "linear_regression_incumbent", 0)
        if self.linear_reference_id != expected_linear.candidate_id:
            raise Phase4EvaluationError("shortlist Linear reference is invalid")
        _validate_shortlist_family(
            self.track,
            "random_forest",
            self.random_forest_candidate_ids,
        )
        _validate_shortlist_family(
            self.track,
            "gradient_boosting",
            self.gradient_boosting_candidate_ids,
        )

    @property
    def full_development_candidate_ids(self) -> tuple[str, ...]:
        """Return the Linear reference followed by both family shortlists."""

        return (
            self.linear_reference_id,
            *self.random_forest_candidate_ids,
            *self.gradient_boosting_candidate_ids,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "ranking": "ascending_exact_finite_float_then_candidate_id",
            "linear_reference_id": self.linear_reference_id,
            "random_forest_candidate_ids": list(self.random_forest_candidate_ids),
            "gradient_boosting_candidate_ids": list(self.gradient_boosting_candidate_ids),
        }


def evaluate_phase4_candidate_cv(
    *,
    features: object,
    target: object,
    spec: CandidateSpec,
    splits: tuple[CVSplit, ...],
    expected_oof_mask: object,
    validation_buckets: tuple[str | None, ...],
    random_forest_n_jobs: int = RANDOM_FOREST_TRAINING_N_JOBS,
) -> Phase4CandidateCVResult:
    """Fit one fresh fold-local pipeline per approved development CV fold."""

    if not isinstance(spec, CandidateSpec):
        raise Phase4EvaluationError("spec must be a CandidateSpec")
    try:
        canonical_spec = get_candidate_spec(spec.track, spec.family, spec.index)
    except ValueError as error:
        raise Phase4EvaluationError("candidate spec is outside the frozen policy") from error
    if spec != canonical_spec:
        raise Phase4EvaluationError("candidate spec differs from the frozen policy")

    config = TRACKS[spec.track]
    frame = validate_feature_frame(features, config)
    _validate_index_alignment(frame, target)
    y = validate_target(target, expected_rows=len(frame), config=config)
    if spec.track == "retail":
        _validate_retail_statuses(frame)

    mask = _boolean_mask(expected_oof_mask, expected_rows=len(frame))
    _validate_splits(splits, expected_oof_mask=mask, row_count=len(frame))
    if not isinstance(validation_buckets, tuple) or len(validation_buckets) != len(splits):
        raise Phase4EvaluationError("every CV fold must have one validation bucket label")
    _validate_bucket_labels(spec.track, validation_buckets)

    oof_predictions = np.full(len(frame), np.nan, dtype=np.float64)
    fold_results: list[FoldAggregate] = []
    for fold_number, ((training_indices, validation_indices), validation_bucket) in enumerate(
        zip(splits, validation_buckets, strict=True),
        start=1,
    ):
        kwargs = (
            {"random_forest_n_jobs": random_forest_n_jobs} if spec.family == "random_forest" else {}
        )
        estimator = make_candidate_pipeline(spec.track, spec.family, spec.index, **kwargs)
        estimator.fit(frame.iloc[training_indices], y[training_indices])
        predictions = _prediction_vector(
            estimator.predict(frame.iloc[validation_indices]),
            expected_rows=len(validation_indices),
        )
        oof_predictions[validation_indices] = predictions
        fold_results.append(
            FoldAggregate(
                fold_number=fold_number,
                training_sample_count=len(training_indices),
                validation_sample_count=len(validation_indices),
                validation_bucket=validation_bucket,
                metrics=regression_metrics(y[validation_indices], predictions),
            )
        )

    actual_mask = np.isfinite(oof_predictions)
    if not np.array_equal(actual_mask, mask):
        raise Phase4EvaluationError("candidate predictions do not match the approved OOF rows")
    if spec.track == "retail":
        evaluation = retail_status_metrics(
            y[actual_mask],
            oof_predictions[actual_mask],
            frame.loc[actual_mask, "vehicle_status"],
        )
        overall = evaluation.overall
        slices = evaluation.status_slices
    else:
        overall = regression_metrics(y[actual_mask], oof_predictions[actual_mask])
        slices = ()
    return Phase4CandidateCVResult(
        spec=spec,
        overall=overall,
        status_slices=slices,
        folds=tuple(fold_results),
    )


def shortlist_phase4_candidates(
    track: TrackName,
    results: tuple[Phase4CandidateCVResult, ...],
) -> Phase4Shortlist:
    """Rank the complete screening field by exact micro OOF MAE then stable ID."""

    if track not in TRACKS:
        raise Phase4EvaluationError("shortlist track is invalid")
    if not isinstance(results, tuple):
        raise Phase4EvaluationError("screening results must be an immutable tuple")
    expected_specs = candidate_specs(track)
    expected_ids = tuple(spec.candidate_id for spec in expected_specs)
    actual_ids = tuple(result.spec.candidate_id for result in results)
    if actual_ids != expected_ids:
        raise Phase4EvaluationError(
            "screening results must contain every approved candidate in stable order"
        )
    _validate_same_fold_evidence(results)

    by_family: dict[str, tuple[str, str]] = {}
    for family in ("random_forest", "gradient_boosting"):
        ranked = sorted(
            (result for result in results if result.spec.family == family),
            key=lambda result: (result.overall.mae, result.spec.candidate_id),
        )
        selected = tuple(result.spec.candidate_id for result in ranked[:_SHORTLIST_SIZE])
        by_family[family] = (selected[0], selected[1])
    return Phase4Shortlist(
        track=track,
        linear_reference_id=get_candidate_spec(
            track, "linear_regression_incumbent", 0
        ).candidate_id,
        random_forest_candidate_ids=by_family["random_forest"],
        gradient_boosting_candidate_ids=by_family["gradient_boosting"],
    )


def parse_phase4_candidate_cv_result(value: object) -> Phase4CandidateCVResult:
    """Parse one strict aggregate candidate object from a trusted JSON decoder."""

    payload = _exact_mapping(value, _CANDIDATE_RESULT_KEYS, label="candidate result")
    candidate_id = payload["candidate_id"]
    if not isinstance(candidate_id, str):
        raise Phase4EvaluationError("candidate result ID must be text")
    matching = tuple(
        spec
        for track in TRACKS
        for spec in candidate_specs(track)
        if spec.candidate_id == candidate_id
    )
    if len(matching) != 1:
        raise Phase4EvaluationError("candidate result ID is outside the frozen policy")
    spec = matching[0]
    if (
        payload["family"] != spec.family
        or not _same_typed_json(payload["index"], spec.index)
        or not _same_typed_json(payload["parameters"], list(spec.parameters))
        or not _same_typed_json(payload["random_state"], spec.random_state)
    ):
        raise Phase4EvaluationError("candidate result metadata differs from frozen policy")
    slices_value = payload["status_slices"]
    folds_value = payload["folds"]
    if not isinstance(slices_value, list):
        raise Phase4EvaluationError("candidate status_slices must be an array")
    if not isinstance(folds_value, list):
        raise Phase4EvaluationError("candidate folds must be an array")
    slices = tuple(_parse_status_slice(item) for item in slices_value)
    folds = tuple(_parse_fold(item) for item in folds_value)
    return Phase4CandidateCVResult(
        spec=spec,
        overall=_parse_metrics(payload["overall"], label="candidate overall"),
        status_slices=slices,
        folds=folds,
    )


def _validate_same_fold_evidence(results: tuple[Phase4CandidateCVResult, ...]) -> None:
    reference = results[0]
    reference_folds = tuple(
        (
            fold.training_sample_count,
            fold.validation_sample_count,
            fold.validation_bucket,
        )
        for fold in reference.folds
    )
    reference_slices = tuple(
        (item.status, item.metrics.sample_count) for item in reference.status_slices
    )
    for result in results[1:]:
        if result.spec.track != reference.spec.track:
            raise Phase4EvaluationError("screening candidate tracks must match")
        if result.overall.sample_count != reference.overall.sample_count:
            raise Phase4EvaluationError("screening candidate OOF sample counts must match")
        fold_shape = tuple(
            (
                fold.training_sample_count,
                fold.validation_sample_count,
                fold.validation_bucket,
            )
            for fold in result.folds
        )
        if fold_shape != reference_folds:
            raise Phase4EvaluationError("screening candidates must use identical CV folds")
        slice_shape = tuple(
            (item.status, item.metrics.sample_count) for item in result.status_slices
        )
        if slice_shape != reference_slices:
            raise Phase4EvaluationError("screening candidates must use identical status slices")


def _validate_shortlist_family(
    track: TrackName,
    family: str,
    candidate_ids: object,
) -> None:
    if not isinstance(candidate_ids, tuple) or len(candidate_ids) != _SHORTLIST_SIZE:
        raise Phase4EvaluationError("each challenger shortlist must contain exactly two IDs")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise Phase4EvaluationError("challenger shortlist IDs must be unique")
    allowed = {spec.candidate_id for spec in candidate_specs(track) if spec.family == family}
    if any(
        not isinstance(candidate_id, str) or candidate_id not in allowed
        for candidate_id in candidate_ids
    ):
        raise Phase4EvaluationError(f"{family} shortlist contains an invalid candidate ID")


def _validate_metrics(metrics: object, *, label: str) -> None:
    if not isinstance(metrics, RegressionMetrics):
        raise Phase4EvaluationError(f"{label} must be RegressionMetrics")
    if (
        isinstance(metrics.sample_count, bool)
        or not isinstance(metrics.sample_count, int)
        or metrics.sample_count < 1
    ):
        raise Phase4EvaluationError(f"{label} sample_count must be a positive integer")
    for name, value in (("MAE", metrics.mae), ("RMSE", metrics.rmse)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise Phase4EvaluationError(f"{label} {name} must be finite and nonnegative")
    if metrics.rmse < metrics.mae:
        raise Phase4EvaluationError(f"{label} RMSE must be at least MAE")
    if metrics.r2 is not None and (
        isinstance(metrics.r2, bool)
        or not isinstance(metrics.r2, (int, float))
        or not math.isfinite(metrics.r2)
        or metrics.r2 > 1.0
    ):
        raise Phase4EvaluationError(f"{label} R-squared must be null or finite and at most one")


def _validate_status_slices(
    track: TrackName,
    slices: object,
    overall: RegressionMetrics,
) -> None:
    if not isinstance(slices, tuple):
        raise Phase4EvaluationError("status_slices must be an immutable tuple")
    if track == "wholesale":
        if slices:
            raise Phase4EvaluationError("wholesale results must not contain retail status slices")
        return
    if any(not isinstance(item, StatusSliceMetrics) for item in slices):
        raise Phase4EvaluationError("retail slices must contain StatusSliceMetrics")
    statuses = tuple(item.status for item in slices)
    if statuses != _RETAIL_STATUSES:
        raise Phase4EvaluationError("retail results require certified, new, and used slices")
    for item in slices:
        _validate_metrics(item.metrics, label=f"{item.status} slice")
    if sum(item.metrics.sample_count for item in slices) != overall.sample_count:
        raise Phase4EvaluationError("retail slice counts must sum to the OOF sample count")
    weighted_mae = (
        math.fsum(item.metrics.sample_count * item.metrics.mae for item in slices)
        / overall.sample_count
    )
    if not math.isclose(weighted_mae, overall.mae, rel_tol=1e-12, abs_tol=1e-9):
        raise Phase4EvaluationError("retail overall MAE must equal weighted status MAEs")


def _validate_index_alignment(frame: pd.DataFrame, target: object) -> None:
    if isinstance(target, (pd.Series, pd.DataFrame)) and not target.index.equals(frame.index):
        raise FeatureContractError("candidate feature and target indexes must align")


def _validate_retail_statuses(frame: pd.DataFrame) -> None:
    values = frame["vehicle_status"].tolist()
    if any(
        not isinstance(value, str)
        or value != value.strip().lower()
        or value not in _RETAIL_STATUSES
        for value in values
    ):
        raise Phase4EvaluationError(
            "retail vehicle_status must contain exact certified, new, or used values"
        )
    missing = tuple(status for status in _RETAIL_STATUSES if status not in set(values))
    if missing:
        raise Phase4EvaluationError(
            "retail evaluation is missing status slices: " + ", ".join(missing)
        )


def _boolean_mask(value: object, *, expected_rows: int) -> NDArray[np.bool_]:
    array = np.asarray(value)
    if array.shape != (expected_rows,) or array.dtype != np.bool_:
        raise Phase4EvaluationError("expected_oof_mask must be a one-dimensional boolean row match")
    return array.astype(np.bool_, copy=True)


def _validate_splits(
    splits: object,
    *,
    expected_oof_mask: NDArray[np.bool_],
    row_count: int,
) -> None:
    if not isinstance(splits, tuple) or not splits:
        raise Phase4EvaluationError("cross-validation splits must be a non-empty tuple")
    validation_counts = np.zeros(row_count, dtype=np.int64)
    for split in splits:
        if not isinstance(split, tuple) or len(split) != 2:
            raise Phase4EvaluationError(
                "each CV split must contain training and validation indices"
            )
        training_indices, validation_indices = split
        for indices, label in ((training_indices, "training"), (validation_indices, "validation")):
            if (
                not isinstance(indices, np.ndarray)
                or indices.dtype.kind not in "iu"
                or indices.ndim != 1
                or len(indices) == 0
            ):
                raise Phase4EvaluationError(f"each CV fold needs non-empty integer {label} indices")
            if (indices < 0).any() or (indices >= row_count).any():
                raise Phase4EvaluationError("CV index is outside the evaluation population")
            if len(np.unique(indices)) != len(indices):
                raise Phase4EvaluationError(f"CV {label} indices must be unique")
        if np.intersect1d(training_indices, validation_indices).size:
            raise Phase4EvaluationError("CV training and validation rows must be disjoint")
        validation_counts[validation_indices] += 1
    if (validation_counts > 1).any():
        raise Phase4EvaluationError("a row must not be predicted in multiple CV folds")
    if not np.array_equal(validation_counts == 1, expected_oof_mask):
        raise Phase4EvaluationError("CV validation rows do not match expected_oof_mask")


def _validate_bucket_labels(track: TrackName, labels: tuple[str | None, ...]) -> None:
    if track == "retail":
        if any(label is not None for label in labels):
            raise Phase4EvaluationError("retail validation bucket labels must be null")
        return
    if any(not isinstance(label, str) or not label or label != label.strip() for label in labels):
        raise Phase4EvaluationError("wholesale validation bucket labels must be canonical text")
    if len(set(labels)) != len(labels):
        raise Phase4EvaluationError("wholesale validation bucket labels must be unique")


def _prediction_vector(predictions: object, *, expected_rows: int) -> NDArray[np.float64]:
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) != expected_rows:
        raise Phase4EvaluationError("model predictions must be a one-dimensional row match")
    inspected = np.asarray(predictions, dtype=object)
    if any(isinstance(value, (bool, np.bool_)) for value in inspected.flat):
        raise Phase4EvaluationError("model predictions must be numeric, not boolean")
    try:
        numeric = values.astype(np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise Phase4EvaluationError("model predictions must be numeric") from error
    if not np.isfinite(numeric).all():
        raise Phase4EvaluationError("model predictions must be finite")
    return numeric


def _parse_fold(value: object) -> FoldAggregate:
    payload = _exact_mapping(value, _FOLD_KEYS, label="candidate fold")
    bucket = payload["validation_bucket"]
    if bucket is not None and not isinstance(bucket, str):
        raise Phase4EvaluationError("candidate fold validation_bucket must be text or null")
    return FoldAggregate(
        fold_number=_strict_integer(payload["fold_number"], label="fold_number"),
        training_sample_count=_strict_integer(
            payload["training_sample_count"], label="training_sample_count"
        ),
        validation_sample_count=_strict_integer(
            payload["validation_sample_count"], label="validation_sample_count"
        ),
        validation_bucket=bucket,
        metrics=_parse_metrics(payload["metrics"], label="candidate fold metrics"),
    )


def _parse_status_slice(value: object) -> StatusSliceMetrics:
    payload = _exact_mapping(value, _SLICE_KEYS, label="candidate status slice")
    status = payload["status"]
    if not isinstance(status, str):
        raise Phase4EvaluationError("candidate status slice label must be text")
    return StatusSliceMetrics(
        status=status,
        metrics=_parse_metrics(payload["metrics"], label="candidate status slice metrics"),
    )


def _parse_metrics(value: object, *, label: str) -> RegressionMetrics:
    payload = _exact_mapping(value, _METRIC_KEYS, label=label)
    r2_value = payload["r2"]
    if r2_value is not None:
        r2_value = _strict_number(r2_value, label=f"{label} r2")
    return RegressionMetrics(
        sample_count=_strict_integer(payload["sample_count"], label=f"{label} sample_count"),
        mae=_strict_number(payload["mae"], label=f"{label} mae"),
        rmse=_strict_number(payload["rmse"], label=f"{label} rmse"),
        r2=r2_value,
    )


def _exact_mapping(value: object, keys: set[str], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Phase4EvaluationError(f"{label} must be an object")
    if set(value) != keys:
        raise Phase4EvaluationError(f"{label} fields are invalid")
    return value


def _strict_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4EvaluationError(f"{label} must be an integer")
    return value


def _strict_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4EvaluationError(f"{label} must be numeric")
    return float(value)


def _same_typed_json(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, list):
        if not isinstance(observed, list):
            return False
        return len(observed) == len(expected) and all(
            _same_typed_json(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected, strict=True)
        )
    return observed == expected


__all__ = [
    "Phase4CandidateCVResult",
    "Phase4EvaluationError",
    "Phase4Shortlist",
    "evaluate_phase4_candidate_cv",
    "parse_phase4_candidate_cv_result",
    "shortlist_phase4_candidates",
]
