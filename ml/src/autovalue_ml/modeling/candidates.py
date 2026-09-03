"""Frozen Phase 4 model-candidate definitions and unfitted pipeline factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

from .baselines import make_linear_pipeline
from .contracts import RETAIL_TRACK, TRACKS, TrackConfig, TrackName
from .tree_preprocessing import make_tree_preprocessor

CandidateFamily: TypeAlias = Literal[
    "linear_regression_incumbent",
    "random_forest",
    "gradient_boosting",
]
GradientBoostingLoss: TypeAlias = Literal["squared_error", "huber"]
RandomForestParameterTuple: TypeAlias = tuple[int, int, int, float, float]
GradientBoostingParameterTuple: TypeAlias = tuple[
    GradientBoostingLoss,
    float,
    int,
    float,
    int,
    int,
    float,
    float,
]
CandidateParameterTuple: TypeAlias = (
    tuple[()] | RandomForestParameterTuple | GradientBoostingParameterTuple
)

CANDIDATE_FAMILIES: Final[tuple[CandidateFamily, ...]] = (
    "linear_regression_incumbent",
    "random_forest",
    "gradient_boosting",
)
RANDOM_FOREST_TRAINING_N_JOBS: Final = 4
RANDOM_FOREST_SERVING_N_JOBS: Final = 1

RETAIL_RANDOM_FOREST_CONFIGS: Final[tuple[RandomForestParameterTuple, ...]] = (
    (96, 512, 5, 0.5, 0.6),
    (128, 1024, 15, 0.7, 0.8),
    (160, 2048, 15, 0.7, 0.8),
    (160, 1024, 30, 1.0, 0.8),
    (128, 2048, 30, 0.5, 1.0),
    (96, 1024, 5, 1.0, 0.6),
)
WHOLESALE_RANDOM_FOREST_CONFIGS: Final[tuple[RandomForestParameterTuple, ...]] = (
    (96, 512, 25, 0.4, 0.5),
    (128, 1024, 50, 0.6, 0.6),
    (160, 2048, 50, 0.6, 0.7),
    (160, 1024, 100, 0.8, 0.7),
    (128, 2048, 100, 0.4, 0.6),
    (96, 1024, 25, 0.8, 0.5),
)
RETAIL_GRADIENT_BOOSTING_CONFIGS: Final[tuple[GradientBoostingParameterTuple, ...]] = (
    ("squared_error", 0.9, 120, 0.08, 2, 20, 0.8, 0.8),
    ("squared_error", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
    ("squared_error", 0.9, 240, 0.03, 3, 50, 0.65, 0.5),
    ("huber", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
    ("huber", 0.85, 240, 0.03, 2, 50, 0.8, 0.5),
    ("huber", 0.9, 120, 0.08, 3, 50, 0.65, 0.5),
)
WHOLESALE_GRADIENT_BOOSTING_CONFIGS: Final[tuple[GradientBoostingParameterTuple, ...]] = (
    ("squared_error", 0.9, 100, 0.08, 2, 50, 0.7, 0.7),
    ("squared_error", 0.9, 140, 0.05, 2, 100, 0.5, 0.4),
    ("squared_error", 0.9, 180, 0.03, 2, 50, 0.7, 0.7),
    ("squared_error", 0.9, 140, 0.05, 3, 100, 0.5, 0.4),
    ("squared_error", 0.9, 100, 0.08, 3, 200, 0.5, 0.7),
    ("huber", 0.9, 140, 0.05, 2, 100, 0.7, 0.4),
)

_RANDOM_FOREST_SEEDS: Final[dict[TrackName, int]] = {
    "retail": 1_254_777_149,
    "wholesale": 2_903_812_338,
}
_GRADIENT_BOOSTING_SEEDS: Final[dict[TrackName, int]] = {
    "retail": 3_295_129_705,
    "wholesale": 177_971_163,
}


class CandidateConfigurationError(ValueError):
    """A requested candidate falls outside the frozen Phase 4 policy."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """Immutable identity and exact policy tuple for one Phase 4 candidate."""

    candidate_id: str
    track: TrackName
    family: CandidateFamily
    index: int
    parameters: CandidateParameterTuple
    random_state: int | None


def candidate_specs(
    track: TrackName | TrackConfig,
    family: CandidateFamily | None = None,
) -> tuple[CandidateSpec, ...]:
    """Return frozen candidate metadata in stable policy order."""

    resolved = _resolve_track(track)
    families = CANDIDATE_FAMILIES if family is None else (_resolve_family(family),)
    return tuple(
        _candidate_spec(resolved.name, selected_family, index)
        for selected_family in families
        for index in range(_candidate_count(resolved.name, selected_family))
    )


def get_candidate_spec(
    track: TrackName | TrackConfig,
    family: CandidateFamily,
    index: int = 0,
) -> CandidateSpec:
    """Validate and return one exact candidate definition."""

    resolved = _resolve_track(track)
    resolved_family = _resolve_family(family)
    resolved_index = _validate_index(index, _candidate_count(resolved.name, resolved_family))
    return _candidate_spec(resolved.name, resolved_family, resolved_index)


def make_linear_incumbent(track: TrackName | TrackConfig) -> Pipeline:
    """Build the unfitted Phase 3 v2 Linear Regression reference pipeline."""

    return make_linear_pipeline(_resolve_track(track))


def make_random_forest_candidate(
    track: TrackName | TrackConfig,
    index: int,
    *,
    n_jobs: int = RANDOM_FOREST_TRAINING_N_JOBS,
) -> Pipeline:
    """Build one exact Random Forest candidate with bounded local parallelism."""

    spec = get_candidate_spec(track, "random_forest", index)
    safe_n_jobs = _validate_random_forest_n_jobs(n_jobs)
    n_estimators, max_leaf_nodes, min_samples_leaf, max_features, max_samples = cast(
        RandomForestParameterTuple, spec.parameters
    )
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion="squared_error",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        bootstrap=True,
        max_samples=max_samples,
        n_jobs=safe_n_jobs,
        random_state=spec.random_state,
    )
    return Pipeline(
        steps=(
            ("preprocessor", make_tree_preprocessor(TRACKS[spec.track])),
            ("regressor", regressor),
        )
    )


def make_gradient_boosting_candidate(
    track: TrackName | TrackConfig,
    index: int,
) -> Pipeline:
    """Build one exact Gradient Boosting candidate from the frozen grid."""

    spec = get_candidate_spec(track, "gradient_boosting", index)
    (
        loss,
        alpha,
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_leaf,
        subsample,
        max_features,
    ) = cast(GradientBoostingParameterTuple, spec.parameters)
    regressor = GradientBoostingRegressor(
        loss=loss,
        alpha=alpha,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        n_iter_no_change=None,
        random_state=spec.random_state,
    )
    return Pipeline(
        steps=(
            ("preprocessor", make_tree_preprocessor(TRACKS[spec.track])),
            ("regressor", regressor),
        )
    )


def make_candidate_pipeline(
    track: TrackName | TrackConfig,
    family: CandidateFamily,
    index: int = 0,
    *,
    random_forest_n_jobs: int | None = None,
) -> Pipeline:
    """Dispatch one validated candidate without permitting arbitrary estimators."""

    resolved = _resolve_track(track)
    resolved_family = _resolve_family(family)
    get_candidate_spec(resolved, resolved_family, index)
    if resolved_family == "linear_regression_incumbent":
        if random_forest_n_jobs is not None:
            raise CandidateConfigurationError(
                "random_forest_n_jobs is valid only for Random Forest candidates"
            )
        return make_linear_incumbent(resolved)
    if resolved_family == "random_forest":
        n_jobs = (
            RANDOM_FOREST_TRAINING_N_JOBS if random_forest_n_jobs is None else random_forest_n_jobs
        )
        return make_random_forest_candidate(resolved, index, n_jobs=n_jobs)
    if random_forest_n_jobs is not None:
        raise CandidateConfigurationError(
            "random_forest_n_jobs is valid only for Random Forest candidates"
        )
    return make_gradient_boosting_candidate(resolved, index)


def _candidate_spec(
    track: TrackName,
    family: CandidateFamily,
    index: int,
) -> CandidateSpec:
    configurations = _candidate_configurations(track, family)
    parameters = configurations[index]
    seed: int | None
    if family == "random_forest":
        seed = _RANDOM_FOREST_SEEDS[track]
    elif family == "gradient_boosting":
        seed = _GRADIENT_BOOSTING_SEEDS[track]
    else:
        seed = None
    return CandidateSpec(
        candidate_id=f"phase4-{track}-{family}-{index:02d}",
        track=track,
        family=family,
        index=index,
        parameters=parameters,
        random_state=seed,
    )


def _candidate_configurations(
    track: TrackName,
    family: CandidateFamily,
) -> tuple[CandidateParameterTuple, ...]:
    if family == "linear_regression_incumbent":
        return ((),)
    if family == "random_forest":
        return cast(
            tuple[CandidateParameterTuple, ...],
            RETAIL_RANDOM_FOREST_CONFIGS if track == "retail" else WHOLESALE_RANDOM_FOREST_CONFIGS,
        )
    return cast(
        tuple[CandidateParameterTuple, ...],
        RETAIL_GRADIENT_BOOSTING_CONFIGS
        if track == "retail"
        else WHOLESALE_GRADIENT_BOOSTING_CONFIGS,
    )


def _candidate_count(track: TrackName, family: CandidateFamily) -> int:
    return len(_candidate_configurations(track, family))


def _resolve_track(track: TrackName | TrackConfig) -> TrackConfig:
    if isinstance(track, TrackConfig):
        canonical = TRACKS.get(track.name)
        if canonical is None or track != canonical:
            raise CandidateConfigurationError("Phase 4 requires a canonical v2 track contract")
        return canonical
    if track == "retail":
        return RETAIL_TRACK
    if track == "wholesale":
        return TRACKS["wholesale"]
    raise CandidateConfigurationError(f"unsupported Phase 4 track: {track!r}")


def _resolve_family(family: CandidateFamily) -> CandidateFamily:
    if not isinstance(family, str) or family not in CANDIDATE_FAMILIES:
        raise CandidateConfigurationError(f"unsupported Phase 4 candidate family: {family!r}")
    return family


def _validate_index(index: int, candidate_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise CandidateConfigurationError("candidate index must be an integer")
    if not 0 <= index < candidate_count:
        raise CandidateConfigurationError(
            f"candidate index must be between 0 and {candidate_count - 1}"
        )
    return index


def _validate_random_forest_n_jobs(n_jobs: int) -> int:
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int):
        raise CandidateConfigurationError("Random Forest n_jobs must be an integer")
    if n_jobs not in {RANDOM_FOREST_SERVING_N_JOBS, RANDOM_FOREST_TRAINING_N_JOBS}:
        raise CandidateConfigurationError(
            "Random Forest n_jobs must be 1 (serving) or 4 (training)"
        )
    return n_jobs


__all__ = [
    "CANDIDATE_FAMILIES",
    "RANDOM_FOREST_SERVING_N_JOBS",
    "RANDOM_FOREST_TRAINING_N_JOBS",
    "RETAIL_GRADIENT_BOOSTING_CONFIGS",
    "RETAIL_RANDOM_FOREST_CONFIGS",
    "WHOLESALE_GRADIENT_BOOSTING_CONFIGS",
    "WHOLESALE_RANDOM_FOREST_CONFIGS",
    "CandidateConfigurationError",
    "CandidateFamily",
    "CandidateSpec",
    "GradientBoostingParameterTuple",
    "RandomForestParameterTuple",
    "candidate_specs",
    "get_candidate_spec",
    "make_candidate_pipeline",
    "make_gradient_boosting_candidate",
    "make_linear_incumbent",
    "make_random_forest_candidate",
]
