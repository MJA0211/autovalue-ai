"""Authenticated, fail-closed serving for the frozen RF05 reference system."""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Protocol, cast

import joblib
import pandas as pd
from autovalue_ml.modeling.calibration_artifact import (
    FEATURE_CONTRACT_VERSION,
    RF05_CANDIDATE_ID,
    RF05_PARAMETERS,
    RF05_RANDOM_STATE,
    PredictionDataQuality,
    RetailCalibrationArtifact,
    active_rf05_identity,
    calibrated_valuation,
    load_calibration_artifact,
)
from autovalue_ml.modeling.rf05_serving_bundle import (
    BUNDLE_ARTIFACT_TYPE,
    BUNDLE_SCHEMA_VERSION,
    CALIBRATION_ASSIGNMENT_SHA256,
    CALIBRATION_SHA256,
    DEVELOPMENT_ROWS,
    JOBLIB_COMPRESSION,
    MODEL_VERSION,
    PICKLE_PROTOCOL,
    POLICY_SHA256,
    RF05_INDEX,
    current_runtime_versions,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from autovalue_api.schemas import PredictionResponse, VehicleValuationRequest
from autovalue_api.schemas.prediction import ModelInformation

TRUSTED_MANIFEST_SHA256: Final = "dd31703302dce38d1a85907d3f818439e70c00f179155609be9bb93f41aaf3a2"
TRUSTED_MODEL_SHA256: Final = "00ceb2680639a555a4705717e21ffe993a04e5731a3143e147d92d43b082e4fd"
DEVELOPMENT_IDENTITY_SHA256: Final = (
    "c131c5b9f2561401e7545f65b491b2f0fd98f5788f9f92ea4faac19abc28b58b"
)
CANDIDATE_SHA256: Final = "12880cfbb2cb7f600f291c077adfa247afb9774b400b21bb7eb7409d72f7fb92"
SPLIT_ASSIGNMENTS_SHA256: Final = "5b3e39d0ef418c07b0c4d08ecc18700fc9f387518a21dbd604f515463cb5ebe5"
MAX_MANIFEST_BYTES: Final = 64_000
MAX_MODEL_BYTES: Final = 300_000_000
_BUNDLE_FILES: Final = frozenset(("manifest.json", "model.joblib"))
_DIGEST_CHARACTERS: Final = frozenset("0123456789abcdef")
_MANIFEST_KEYS: Final = {
    "schema_version",
    "artifact_type",
    "model_file",
    "model_version",
    "candidate_id",
    "feature_contract_version",
    "rf05_identity_sha256",
    "model_sha256",
    "reconstruction_policy_sha256",
    "specification",
    "specification_sha256",
    "training_data",
    "calibration_binding",
    "serialization",
    "runtime",
    "contents",
    "publication",
}


class ValuationUnavailableError(RuntimeError):
    """Raised when the trusted frozen serving bundle is unavailable."""


class ValuationEngine(Protocol):
    """Narrow route-facing valuation boundary."""

    @property
    def ready(self) -> bool: ...

    def predict(self, request: VehicleValuationRequest) -> PredictionResponse: ...


class FrozenRF05Service:
    """Load RF05 only after authenticating its bundle, runtime, and calibration."""

    def __init__(
        self,
        *,
        bundle_dir: Path,
        calibration_path: Path,
        trusted_models_root: Path | None = None,
        trusted_calibration_root: Path | None = None,
        expected_manifest_sha256: str = TRUSTED_MANIFEST_SHA256,
        expected_model_sha256: str = TRUSTED_MODEL_SHA256,
    ) -> None:
        self._pipeline: Pipeline | None = None
        self._calibration: RetailCalibrationArtifact | None = None
        self._load_error = "trusted RF05 artifact unavailable"
        try:
            calibration_bytes = _read_trusted_artifact(
                calibration_path,
                trusted_root=trusted_calibration_root or calibration_path.parent,
                expected_name="retail-rf05-calibration-v1.artifact.json",
                maximum_bytes=MAX_MANIFEST_BYTES,
            )
            if hashlib.sha256(calibration_bytes).hexdigest() != CALIBRATION_SHA256:
                raise ValuationUnavailableError("calibration checksum mismatch")
            calibration = load_calibration_artifact(
                calibration_bytes,
                active_model_identity_sha256=active_rf05_identity().identity_sha256,
            )
            pipeline = _load_pipeline(
                bundle_dir,
                trusted_models_root=trusted_models_root or bundle_dir.parent,
                expected_manifest_sha256=expected_manifest_sha256,
                expected_model_sha256=expected_model_sha256,
            )
        except Exception:
            return
        self._pipeline = pipeline
        self._calibration = calibration
        self._load_error = ""

    @property
    def ready(self) -> bool:
        return self._pipeline is not None and self._calibration is not None

    @property
    def public_unavailable_reason(self) -> str | None:
        return None if self.ready else self._load_error

    def predict(self, request: VehicleValuationRequest) -> PredictionResponse:
        if self._pipeline is None or self._calibration is None:
            raise ValuationUnavailableError(self._load_error)
        features = pd.DataFrame(
            [
                {
                    "year": request.year,
                    "make": request.make,
                    "model": request.model,
                    "vehicle_status": request.vehicle_status,
                    "mileage": request.mileage,
                }
            ]
        )
        prediction = self._pipeline.predict(features)
        if len(prediction) != 1:
            raise ValuationUnavailableError("trusted RF05 inference failed")
        point = float(prediction[0])
        if not math.isfinite(point) or point < 0.0:
            raise ValuationUnavailableError("trusted RF05 inference failed")
        quality = PredictionDataQuality(
            mileage_missing=request.mileage is None,
            rare_or_unseen_category=_has_unseen_category(self._pipeline, request),
            unsupported_feature_combination=(
                request.mileage is not None and request.mileage > 300_000
            ),
        )
        valuation = calibrated_valuation(
            point_prediction=point,
            vehicle_status=request.vehicle_status,
            coverage=request.interval_coverage,
            artifact=self._calibration,
            data_quality=quality,
        )
        lower = round(valuation.interval_lower, 2)
        upper = round(valuation.interval_upper, 2)
        return PredictionResponse(
            predicted_value=round(valuation.predicted_value, 2),
            interval_lower=lower,
            interval_upper=upper,
            interval_coverage=valuation.interval_coverage,
            interval_width=round(upper - lower, 2),
            confidence_label=valuation.confidence_label,
            calibration_version=valuation.calibration_version,
            warnings=list(valuation.warnings),
            model_information=ModelInformation(
                candidate_id=RF05_CANDIDATE_ID,
                feature_contract_version=FEATURE_CONTRACT_VERSION,
            ),
        )


def _load_pipeline(
    bundle_dir: Path,
    *,
    trusted_models_root: Path,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
) -> Pipeline:
    trusted_bundle = _trusted_bundle_directory(bundle_dir, trusted_models_root)
    manifest_bytes = _read_regular_file(
        trusted_bundle / "manifest.json",
        maximum_bytes=MAX_MANIFEST_BYTES,
    )
    if not _is_sha256(expected_manifest_sha256) or (
        hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
    ):
        raise ValuationUnavailableError("model manifest authentication failed")
    manifest = _parse_manifest(
        manifest_bytes,
        expected_model_sha256=expected_model_sha256,
    )
    model_bytes = _read_regular_file(
        trusted_bundle / "model.joblib",
        maximum_bytes=MAX_MODEL_BYTES,
    )
    if hashlib.sha256(model_bytes).hexdigest() != manifest["model_sha256"]:
        raise ValuationUnavailableError("model checksum mismatch")
    loaded = joblib.load(io.BytesIO(model_bytes))
    if not isinstance(loaded, Pipeline):
        raise ValuationUnavailableError("model artifact is not a pipeline")
    _verify_rf05_pipeline(loaded)
    return loaded


def _parse_manifest(
    serialized: bytes,
    *,
    expected_model_sha256: str = TRUSTED_MODEL_SHA256,
) -> Mapping[str, object]:
    try:
        value = json.loads(serialized, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValuationUnavailableError("model manifest is invalid") from error
    if not isinstance(value, dict) or set(value) != _MANIFEST_KEYS:
        raise ValuationUnavailableError("model manifest fields are invalid")
    expected_root = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": BUNDLE_ARTIFACT_TYPE,
        "model_file": "model.joblib",
        "model_version": MODEL_VERSION,
        "candidate_id": RF05_CANDIDATE_ID,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "rf05_identity_sha256": active_rf05_identity().identity_sha256,
        "model_sha256": expected_model_sha256,
        "reconstruction_policy_sha256": POLICY_SHA256,
        "publication": "deployment_private_binary",
    }
    if not _is_sha256(expected_model_sha256) or any(
        value.get(key) != expected for key, expected in expected_root.items()
    ):
        raise ValuationUnavailableError("model manifest identity mismatch")
    specification = _expected_specification()
    if value["specification"] != specification:
        raise ValuationUnavailableError("model specification mismatch")
    specification_sha256 = hashlib.sha256(_canonical_json(specification)).hexdigest()
    if value["specification_sha256"] != specification_sha256:
        raise ValuationUnavailableError("model specification checksum mismatch")
    if value["training_data"] != _expected_training_data():
        raise ValuationUnavailableError("model training-data binding mismatch")
    if value["calibration_binding"] != _expected_calibration_binding():
        raise ValuationUnavailableError("model calibration binding mismatch")
    if value["serialization"] != {
        "format": "joblib",
        "pickle_protocol": PICKLE_PROTOCOL,
        "compression": JOBLIB_COMPRESSION,
        "trusted_local_only": True,
    }:
        raise ValuationUnavailableError("model serialization policy mismatch")
    if value["runtime"] != current_runtime_versions():
        raise ValuationUnavailableError("model runtime compatibility mismatch")
    if value["contents"] != ["manifest.json", "model.joblib"]:
        raise ValuationUnavailableError("model bundle contents mismatch")
    return cast(Mapping[str, object], value)


def _expected_specification() -> dict[str, object]:
    return {
        "candidate_id": RF05_CANDIDATE_ID,
        "candidate_index": RF05_INDEX,
        "parameters": list(RF05_PARAMETERS),
        "random_state": RF05_RANDOM_STATE,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "rf05_identity_sha256": active_rf05_identity().identity_sha256,
        "preprocessing_factory": "make_tree_preprocessor(retail)",
        "training_n_jobs": 4,
        "serving_n_jobs": 1,
    }


def _expected_training_data() -> dict[str, object]:
    return {
        "source_id": "kaggle_us_sales_cars_v2",
        "target_semantics": "historical_us_advertised_asking_price_usd_2023",
        "development_rows": DEVELOPMENT_ROWS,
        "development_identity_algorithm": (
            "sha256 length-prefixed canonical ordered predictors+target v1"
        ),
        "development_identity_sha256": DEVELOPMENT_IDENTITY_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "split_assignments_sha256": SPLIT_ASSIGNMENTS_SHA256,
        "calibration_assignment_sha256": CALIBRATION_ASSIGNMENT_SHA256,
        "calibration_rows_used_for_fit": False,
        "final_holdout_rows_used_for_fit_or_verification": False,
    }


def _expected_calibration_binding() -> dict[str, object]:
    return {
        "version": "retail-rf05-split-conformal-v1",
        "artifact_sha256": CALIBRATION_SHA256,
        "rf05_identity_sha256": active_rf05_identity().identity_sha256,
    }


def _verify_rf05_pipeline(pipeline: Pipeline) -> None:
    if tuple(pipeline.named_steps) != ("preprocessor", "regressor"):
        raise ValuationUnavailableError("RF05 pipeline structure mismatch")
    regressor = pipeline.named_steps["regressor"]
    if not isinstance(regressor, RandomForestRegressor):
        raise ValuationUnavailableError("RF05 estimator type mismatch")
    expected = {
        "n_estimators": RF05_PARAMETERS[0],
        "max_leaf_nodes": RF05_PARAMETERS[1],
        "min_samples_leaf": RF05_PARAMETERS[2],
        "max_features": RF05_PARAMETERS[3],
        "max_samples": RF05_PARAMETERS[4],
        "random_state": RF05_RANDOM_STATE,
        "criterion": "squared_error",
        "n_jobs": 1,
    }
    parameters = regressor.get_params(deep=False)
    if any(parameters[key] != expected_value for key, expected_value in expected.items()):
        raise ValuationUnavailableError("RF05 estimator parameters mismatch")
    try:
        check_is_fitted(pipeline)
    except (TypeError, ValueError) as error:
        raise ValuationUnavailableError("RF05 pipeline is not fitted") from error


def _trusted_bundle_directory(bundle_dir: Path, trusted_models_root: Path) -> Path:
    root = _resolved_real_directory(trusted_models_root, label="trusted models root")
    bundle = _resolved_real_directory(bundle_dir, label="model bundle")
    if bundle.parent != root or bundle.name != MODEL_VERSION:
        raise ValuationUnavailableError("model bundle path is outside the trusted root")
    try:
        children = {child.name for child in bundle.iterdir()}
    except OSError as error:
        raise ValuationUnavailableError("model bundle is unavailable") from error
    if children != _BUNDLE_FILES:
        raise ValuationUnavailableError("model bundle contains unexpected files")
    return bundle


def _read_trusted_artifact(
    path: Path,
    *,
    trusted_root: Path,
    expected_name: str,
    maximum_bytes: int,
) -> bytes:
    root = _resolved_real_directory(trusted_root, label="trusted artifact root")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValuationUnavailableError("trusted artifact is unavailable") from error
    if resolved.parent != root or resolved.name != expected_name:
        raise ValuationUnavailableError("trusted artifact path is invalid")
    return _read_regular_file(path, maximum_bytes=maximum_bytes)


def _resolved_real_directory(path: Path, *, label: str) -> Path:
    try:
        information = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValuationUnavailableError(f"{label} is unavailable") from error
    if not stat.S_ISDIR(information.st_mode) or path.is_symlink():
        raise ValuationUnavailableError(f"{label} must be a real directory")
    return resolved


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        information = path.lstat()
    except OSError as error:
        raise ValuationUnavailableError("trusted artifact is unavailable") from error
    if not stat.S_ISREG(information.st_mode) or path.is_symlink():
        raise ValuationUnavailableError("trusted artifact must be a real regular file")
    if information.st_size <= 0 or information.st_size > maximum_bytes:
        raise ValuationUnavailableError("trusted artifact size is invalid")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValuationUnavailableError("trusted artifact is unavailable") from error
    if len(payload) != information.st_size:
        raise ValuationUnavailableError("trusted artifact changed while reading")
    return payload


def _has_unseen_category(pipeline: Pipeline, request: VehicleValuationRequest) -> bool:
    try:
        columns = pipeline.named_steps["preprocessor"].named_steps["columns"]
        encoder = columns.named_transformers_["categorical"].named_steps["encoder"]
        known = [set(str(item).casefold() for item in values) for values in encoder.categories_]
    except (AttributeError, KeyError, TypeError):
        return True
    requested = (
        request.make.casefold(),
        request.model.casefold(),
        request.vehicle_status.casefold(),
    )
    return len(known) != 3 or any(
        value not in categories for value, categories in zip(requested, known, strict=True)
    )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValuationUnavailableError("model manifest has duplicate fields")
        result[key] = value
    return result


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_DIGEST_CHARACTERS)


__all__ = [
    "TRUSTED_MANIFEST_SHA256",
    "TRUSTED_MODEL_SHA256",
    "FrozenRF05Service",
    "ValuationEngine",
    "ValuationUnavailableError",
]
