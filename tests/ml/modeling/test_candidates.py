from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from autovalue_ml.modeling.candidates import (
    RANDOM_FOREST_SERVING_N_JOBS,
    RANDOM_FOREST_TRAINING_N_JOBS,
    RETAIL_GRADIENT_BOOSTING_CONFIGS,
    RETAIL_RANDOM_FOREST_CONFIGS,
    WHOLESALE_GRADIENT_BOOSTING_CONFIGS,
    WHOLESALE_RANDOM_FOREST_CONFIGS,
    CandidateConfigurationError,
    candidate_specs,
    get_candidate_spec,
    make_candidate_pipeline,
    make_gradient_boosting_candidate,
    make_linear_incumbent,
    make_random_forest_candidate,
)
from autovalue_ml.modeling.contracts import RETAIL_TRACK
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def test_explicit_candidate_tuples_match_the_frozen_policy() -> None:
    assert RETAIL_RANDOM_FOREST_CONFIGS == (
        (96, 512, 5, 0.5, 0.6),
        (128, 1024, 15, 0.7, 0.8),
        (160, 2048, 15, 0.7, 0.8),
        (160, 1024, 30, 1.0, 0.8),
        (128, 2048, 30, 0.5, 1.0),
        (96, 1024, 5, 1.0, 0.6),
    )
    assert WHOLESALE_RANDOM_FOREST_CONFIGS == (
        (96, 512, 25, 0.4, 0.5),
        (128, 1024, 50, 0.6, 0.6),
        (160, 2048, 50, 0.6, 0.7),
        (160, 1024, 100, 0.8, 0.7),
        (128, 2048, 100, 0.4, 0.6),
        (96, 1024, 25, 0.8, 0.5),
    )
    assert RETAIL_GRADIENT_BOOSTING_CONFIGS == (
        ("squared_error", 0.9, 120, 0.08, 2, 20, 0.8, 0.8),
        ("squared_error", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
        ("squared_error", 0.9, 240, 0.03, 3, 50, 0.65, 0.5),
        ("huber", 0.9, 180, 0.05, 2, 20, 0.65, 0.8),
        ("huber", 0.85, 240, 0.03, 2, 50, 0.8, 0.5),
        ("huber", 0.9, 120, 0.08, 3, 50, 0.65, 0.5),
    )
    assert WHOLESALE_GRADIENT_BOOSTING_CONFIGS == (
        ("squared_error", 0.9, 100, 0.08, 2, 50, 0.7, 0.7),
        ("squared_error", 0.9, 140, 0.05, 2, 100, 0.5, 0.4),
        ("squared_error", 0.9, 180, 0.03, 2, 50, 0.7, 0.7),
        ("squared_error", 0.9, 140, 0.05, 3, 100, 0.5, 0.4),
        ("squared_error", 0.9, 100, 0.08, 3, 200, 0.5, 0.7),
        ("huber", 0.9, 140, 0.05, 2, 100, 0.7, 0.4),
    )


@pytest.mark.parametrize("track", ["retail", "wholesale"])
def test_candidate_counts_ids_and_specs_are_deterministic(track: str) -> None:
    first = candidate_specs(track)  # type: ignore[arg-type]
    second = candidate_specs(track)  # type: ignore[arg-type]

    assert first == second
    assert len(first) == 13
    assert [spec.family for spec in first].count("linear_regression_incumbent") == 1
    assert [spec.family for spec in first].count("random_forest") == 6
    assert [spec.family for spec in first].count("gradient_boosting") == 6
    assert len({spec.candidate_id for spec in first}) == 13
    assert all(spec.candidate_id.startswith(f"phase4-{track}-") for spec in first)
    with pytest.raises(FrozenInstanceError):
        first[0].index = 1  # type: ignore[misc]


def test_random_forest_builder_uses_exact_common_tuple_seed_and_job_modes() -> None:
    training = make_random_forest_candidate("retail", 2)
    serving = make_random_forest_candidate(
        "retail",
        2,
        n_jobs=RANDOM_FOREST_SERVING_N_JOBS,
    )
    regressor = training.named_steps["regressor"]
    serving_regressor = serving.named_steps["regressor"]

    assert isinstance(regressor, RandomForestRegressor)
    assert regressor.n_estimators == 160
    assert regressor.max_leaf_nodes == 2048
    assert regressor.min_samples_leaf == 15
    assert regressor.max_features == 0.7
    assert regressor.max_samples == 0.8
    assert regressor.criterion == "squared_error"
    assert regressor.bootstrap is True
    assert regressor.max_depth is None
    assert regressor.random_state == 1_254_777_149
    assert regressor.n_jobs == RANDOM_FOREST_TRAINING_N_JOBS == 4
    assert serving_regressor.n_jobs == RANDOM_FOREST_SERVING_N_JOBS == 1


def test_gradient_boosting_builder_uses_exact_common_tuple_and_seed() -> None:
    pipeline = make_gradient_boosting_candidate("wholesale", 5)
    regressor = pipeline.named_steps["regressor"]

    assert isinstance(regressor, GradientBoostingRegressor)
    assert regressor.loss == "huber"
    assert regressor.alpha == 0.9
    assert regressor.n_estimators == 140
    assert regressor.learning_rate == 0.05
    assert regressor.max_depth == 2
    assert regressor.min_samples_leaf == 100
    assert regressor.subsample == 0.7
    assert regressor.max_features == 0.4
    assert regressor.n_iter_no_change is None
    assert regressor.random_state == 177_971_163


def test_linear_incumbent_reuses_phase3_v2_pipeline() -> None:
    pipeline = make_linear_incumbent("retail")

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["regressor"], LinearRegression)
    numeric = pipeline.named_steps["preprocessor"].named_steps["columns"].transformers[0][1]
    assert tuple(numeric.named_steps) == ("imputer", "scaler")


def test_dispatch_returns_fresh_deterministic_candidates() -> None:
    first = make_candidate_pipeline("wholesale", "random_forest", 4)
    second = make_candidate_pipeline("wholesale", "random_forest", 4)

    assert first is not second
    assert first.named_steps["regressor"] is not second.named_steps["regressor"]
    assert (
        first.named_steps["regressor"].get_params() == second.named_steps["regressor"].get_params()
    )
    assert get_candidate_spec("wholesale", "random_forest", 4) == get_candidate_spec(
        "wholesale", "random_forest", 4
    )


@pytest.mark.parametrize("bad_track", ["consumer", 42, None])
def test_candidate_calls_reject_unsupported_tracks(bad_track: object) -> None:
    with pytest.raises(CandidateConfigurationError, match="unsupported"):
        candidate_specs(bad_track)  # type: ignore[arg-type]

    modified = replace(RETAIL_TRACK, one_hot_min_frequency=26)
    with pytest.raises(CandidateConfigurationError, match="canonical"):
        candidate_specs(modified)


def test_candidate_calls_reject_bad_families_and_indices() -> None:
    with pytest.raises(CandidateConfigurationError, match="family"):
        make_candidate_pipeline("retail", "xgboost", 0)  # type: ignore[arg-type]
    for bad_index in (-1, 6, True, 1.5):
        with pytest.raises(CandidateConfigurationError, match="index"):
            get_candidate_spec("retail", "random_forest", bad_index)  # type: ignore[arg-type]
    with pytest.raises(CandidateConfigurationError, match="between 0 and 0"):
        get_candidate_spec("retail", "linear_regression_incumbent", 1)


@pytest.mark.parametrize("bad_n_jobs", [-1, 0, 2, 8, True])
def test_random_forest_jobs_are_limited_to_frozen_safe_modes(bad_n_jobs: object) -> None:
    with pytest.raises(CandidateConfigurationError, match="n_jobs"):
        make_random_forest_candidate("retail", 0, n_jobs=bad_n_jobs)  # type: ignore[arg-type]

    with pytest.raises(CandidateConfigurationError, match="only for Random Forest"):
        make_candidate_pipeline(
            "retail",
            "gradient_boosting",
            0,
            random_forest_n_jobs=1,
        )
