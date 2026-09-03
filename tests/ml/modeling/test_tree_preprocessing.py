from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling.candidates import (
    make_gradient_boosting_candidate,
    make_random_forest_candidate,
)
from autovalue_ml.modeling.tree_preprocessing import (
    make_tree_column_transformer,
    make_tree_preprocessor,
)
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _retail_rows(row_count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2016 + index % 9 for index in range(row_count)],
            "make": [f"Make {index % 4}" for index in range(row_count)],
            "model": [f"Model {index % 7}" for index in range(row_count)],
            "mileage": [None if index % 6 == 0 else index * 1_250 for index in range(row_count)],
            "vehicle_status": [
                ("new", "used", "certified")[index % 3] for index in range(row_count)
            ],
        }
    )


def test_tree_preprocessor_returns_csr_float32_without_numeric_scaling() -> None:
    frame = _retail_rows()
    before = frame.copy(deep=True)
    preprocessor = make_tree_preprocessor()

    transformed = preprocessor.fit_transform(frame)

    pd.testing.assert_frame_equal(frame, before)
    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.float32
    columns = preprocessor.named_steps["columns"]
    numeric = columns.named_transformers_["numeric"]
    categorical = columns.named_transformers_["categorical"]
    assert isinstance(numeric, Pipeline)
    assert tuple(numeric.named_steps) == ("imputer",)
    assert not any(isinstance(step, StandardScaler) for step in numeric.named_steps.values())
    assert isinstance(categorical, Pipeline)
    encoder = categorical.named_steps["encoder"]
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "infrequent_if_exist"
    assert encoder.min_frequency == 25
    assert encoder.max_categories == 512
    assert encoder.sparse_output is True
    assert encoder.dtype == np.float32


def test_tree_preprocessor_keeps_dense_tiny_inputs_at_a_sparse_boundary() -> None:
    one_row = _retail_rows(1)

    transformed = make_tree_preprocessor().fit_transform(one_row)

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.float32
    assert transformed.shape[0] == 1


def test_tree_column_transformer_matches_track_specific_category_caps() -> None:
    retail = make_tree_column_transformer("retail")  # type: ignore[arg-type]
    wholesale = make_tree_column_transformer("wholesale")  # type: ignore[arg-type]

    retail_encoder = retail.transformers[1][1].named_steps["encoder"]
    wholesale_encoder = wholesale.transformers[1][1].named_steps["encoder"]
    assert retail_encoder.min_frequency == 25
    assert wholesale_encoder.min_frequency == 50
    assert retail_encoder.max_categories == wholesale_encoder.max_categories == 512


@pytest.mark.parametrize("family", ["random_forest", "gradient_boosting"])
def test_sparse_tree_candidate_fit_predict_smoke(family: str) -> None:
    features = _retail_rows()
    target = np.asarray(
        [18_000 + row * 325 + (row % 3) * 2_000 for row in range(len(features))],
        dtype=np.float64,
    )
    if family == "random_forest":
        candidate = make_random_forest_candidate("retail", 0, n_jobs=1)
    else:
        candidate = make_gradient_boosting_candidate("retail", 0)

    predictions = candidate.fit(features, target).predict(features.iloc[:3])

    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()
    transformed = candidate.named_steps["preprocessor"].transform(features.iloc[:3])
    assert sparse.isspmatrix_csr(transformed)
    assert transformed.dtype == np.float32
