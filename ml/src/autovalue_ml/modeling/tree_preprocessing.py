"""Sparse, fold-local preprocessing for AutoValue AI tree candidates."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from .contracts import RETAIL_TRACK, TrackConfig, get_track_config
from .feature_engineering import VehicleFeatureEngineer


def make_tree_column_transformer(config: TrackConfig = RETAIL_TRACK) -> ColumnTransformer:
    """Build the unfitted Phase 4 tree transformer without numeric scaling."""

    resolved = get_track_config(config)
    numeric_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True, copy=True),
            ),
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


def make_tree_preprocessor(config: TrackConfig = RETAIL_TRACK) -> Pipeline:
    """Build the Phase 4 tree preprocessing path with a CSR float32 boundary."""

    resolved = get_track_config(config)
    return Pipeline(
        steps=(
            ("feature_engineering", VehicleFeatureEngineer(resolved)),
            ("columns", make_tree_column_transformer(resolved)),
            (
                "to_csr_float32",
                FunctionTransformer(
                    _to_csr_float32,
                    accept_sparse=True,
                    feature_names_out="one-to-one",
                ),
            ),
        )
    )


def _to_csr_float32(matrix: object) -> sparse.csr_matrix:
    """Normalize either sparse format to CSR float32 without a dense conversion."""

    return sparse.csr_matrix(matrix, dtype=np.float32, copy=False)


__all__ = ["make_tree_column_transformer", "make_tree_preprocessor"]
