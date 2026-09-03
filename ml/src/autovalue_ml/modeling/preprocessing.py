"""Sparse, fold-local preprocessing pipelines for approved vehicle features."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .contracts import RETAIL_TRACK, TrackConfig, get_track_config
from .feature_engineering import VehicleFeatureEngineer


def make_column_transformer(config: TrackConfig = RETAIL_TRACK) -> ColumnTransformer:
    """Build an unfitted transformer whose learned state is confined to ``fit`` data."""

    resolved = get_track_config(config)
    numeric_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True, copy=True),
            ),
            ("scaler", StandardScaler(with_mean=False, copy=True)),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                    copy=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=resolved.one_hot_min_frequency,
                    max_categories=resolved.one_hot_max_categories,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        )
    )
    return ColumnTransformer(
        transformers=(
            ("numeric", numeric_pipeline, list(resolved.numeric_features)),
            ("categorical", categorical_pipeline, list(resolved.categorical_features)),
        ),
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_preprocessor(config: TrackConfig = RETAIL_TRACK) -> Pipeline:
    """Build the complete feature-engineering and always-sparse preprocessing path."""

    resolved = get_track_config(config)
    return Pipeline(
        steps=(
            ("feature_engineering", VehicleFeatureEngineer(resolved)),
            ("columns", make_column_transformer(resolved)),
            (
                "to_csr",
                FunctionTransformer(
                    _to_csr,
                    accept_sparse=True,
                    feature_names_out="one-to-one",
                ),
            ),
        )
    )


def _to_csr(matrix: object) -> sparse.csr_matrix:
    """Keep the model boundary sparse even for unusually dense tiny folds."""

    return sparse.csr_matrix(matrix)
