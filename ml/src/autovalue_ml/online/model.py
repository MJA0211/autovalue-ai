"""River-native incremental preprocessing and simple regression."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from river import linear_model, optim, preprocessing

MODEL_VERSION = "river-target-scaled-linear-regression-v1"


class OnlineRegressor(Protocol):
    """Minimal model interface used to test lifecycle ordering."""

    model_version: str

    def predict_one(self, features: Mapping[str, str | float]) -> float: ...

    def learn_one(self, features: Mapping[str, str | float], target: float) -> None: ...


class RiverVehicleRegressor:
    """Incremental standardization, one-hot encoding, and linear regression.

    Preprocessors are updated only during ``learn_one``. This avoids River's
    optional learn-during-predict pipeline behavior and preserves an auditable
    test-then-train boundary.
    """

    model_version = MODEL_VERSION

    def __init__(self) -> None:
        self.numeric_scaler = preprocessing.StandardScaler()
        self.category_encoder = preprocessing.OneHotEncoder(drop_zeros=True)
        base_regressor = linear_model.LinearRegression(
            optimizer=optim.SGD(0.02),
            l2=0.0001,
            intercept_lr=0.01,
        )
        self.regressor = preprocessing.TargetStandardScaler(base_regressor)
        self.observations_learned = 0

    def predict_one(self, features: Mapping[str, str | float]) -> float:
        """Predict without mutating the scaler, encoder, or regressor."""
        transformed = self._transform(features)
        prediction = float(self.regressor.predict_one(transformed))  # type: ignore[no-untyped-call]
        if not math.isfinite(prediction):
            raise ValueError("River emitted a non-finite prediction")
        return prediction

    def learn_one(self, features: Mapping[str, str | float], target: float) -> None:
        """Increment preprocessing and estimator state after evaluation."""
        numeric, categorical = self._split(features)
        self.numeric_scaler.learn_one(numeric)  # type: ignore[no-untyped-call]
        self.category_encoder.learn_one(categorical)  # type: ignore[no-untyped-call]
        transformed = self._merge_transforms(numeric, categorical)
        self.regressor.learn_one(transformed, target)  # type: ignore[no-untyped-call]
        self.observations_learned += 1

    def _transform(self, features: Mapping[str, str | float]) -> dict[str, float]:
        numeric, categorical = self._split(features)
        return self._merge_transforms(numeric, categorical)

    def _merge_transforms(
        self,
        numeric: dict[str, float],
        categorical: dict[str, str],
    ) -> dict[str, float]:
        transformed = {
            f"num::{name}": float(value)
            for name, value in self.numeric_scaler.transform_one(numeric).items()  # type: ignore[no-untyped-call]
        }
        transformed.update(
            {
                f"cat::{name}": float(value)
                for name, value in self.category_encoder.transform_one(categorical).items()  # type: ignore[no-untyped-call]
            }
        )
        return transformed

    @staticmethod
    def _split(
        features: Mapping[str, str | float],
    ) -> tuple[dict[str, float], dict[str, str]]:
        numeric: dict[str, float] = {}
        categorical: dict[str, str] = {}
        for name, value in features.items():
            if isinstance(value, str):
                categorical[name] = value
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"feature {name} is not finite")
                numeric[name] = number
        return numeric, categorical
