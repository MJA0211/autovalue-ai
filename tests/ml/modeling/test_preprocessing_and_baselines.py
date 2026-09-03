from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autovalue_ml.modeling import (
    RETAIL_TRACK,
    make_baseline_pipeline,
    make_dummy_pipeline,
    make_linear_pipeline,
    make_preprocessor,
)
from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2020, 2021, 2019, 2022, 2018, 2023],
            "make": ["Toyota", "Toyota", "Honda", "Ford", "Honda", "Nissan"],
            "model": ["Camry", "Corolla", "Civic", "F-150", "Accord", "Leaf"],
            "mileage": [30_000, 20_000, np.nan, 10_000, 50_000, 0],
            "vehicle_status": ["used", "used", "used", "certified", "used", "new"],
        }
    )


def test_preprocessor_learns_imputation_and_vocabulary_from_fit_rows_only() -> None:
    train = _training_frame().iloc[:4].copy()
    validation = pd.DataFrame(
        {
            "year": [2020],
            "make": ["ValidationOnlyMake"],
            "model": ["ValidationOnlyModel"],
            "mileage": [np.nan],
            "vehicle_status": ["unknown-at-fit"],
        }
    )
    before = train.copy(deep=True)
    preprocessor = make_preprocessor(RETAIL_TRACK)

    preprocessor.fit(train)
    transformed = preprocessor.transform(validation)

    pd.testing.assert_frame_equal(train, before)
    columns = preprocessor.named_steps["columns"]
    numeric_imputer = columns.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_[1] == pytest.approx(20_000)
    encoder = columns.named_transformers_["categorical"].named_steps["encoder"]
    learned_categories = {str(value) for categories in encoder.categories_ for value in categories}
    assert "ValidationOnlyMake" not in learned_categories
    assert "ValidationOnlyModel" not in learned_categories
    assert sparse.isspmatrix_csr(transformed)
    assert np.isfinite(transformed.data).all()


def test_sparse_output_is_finite_and_bounded_for_unknown_categories() -> None:
    train = _training_frame()
    unknown = train.iloc[[0]].copy()
    unknown.loc[:, "make"] = "Never Seen Motors"
    unknown.loc[:, "model"] = "Never Seen Model"
    unknown.loc[:, "vehicle_status"] = "never-seen-status"
    preprocessor = make_preprocessor(RETAIL_TRACK).fit(train)

    transformed = preprocessor.transform(unknown)

    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] <= RETAIL_TRACK.maximum_transformed_features
    assert np.isfinite(transformed.data).all()


def test_dummy_and_linear_factories_return_unfitted_full_pipelines() -> None:
    features = _training_frame()
    target = np.asarray([20_000, 22_000, 18_000, 35_000, 15_000, 30_000], dtype=float)

    dummy = make_dummy_pipeline(RETAIL_TRACK)
    linear = make_linear_pipeline(RETAIL_TRACK)
    assert isinstance(dummy.named_steps["regressor"], DummyRegressor)
    assert isinstance(linear.named_steps["regressor"], LinearRegression)
    assert linear.named_steps["regressor"].n_jobs == 1

    dummy.fit(features, target)
    linear.fit(features, target)
    assert np.all(dummy.predict(features) == np.median(target))
    assert np.isfinite(linear.predict(features)).all()


def test_named_baseline_dispatch_is_strict() -> None:
    assert isinstance(
        make_baseline_pipeline("dummy_median").named_steps["regressor"], DummyRegressor
    )
    assert isinstance(
        make_baseline_pipeline("linear_regression").named_steps["regressor"], LinearRegression
    )
    with pytest.raises(ValueError, match="unsupported baseline"):
        make_baseline_pipeline("forest")  # type: ignore[arg-type]
