"""Governed deterministic reconstruction of the frozen retail RF05 serving bundle."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Final, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from autovalue_ml.acquisition.sources.kaggle_us_sales_cars_split import (
    prepare_kaggle_us_sales_cars_split_training_rows,
)

from .baseline_cli import RetailTrainingRow, _collect_retail_partition, _expected_count
from .calibration import retail_calibration_partition
from .calibration_artifact import (
    CALIBRATION_VERSION,
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
from .candidates import get_candidate_spec, make_random_forest_candidate
from .contracts import RETAIL_TRACK, validate_feature_frame, validate_target
from .cv import retail_group_cv_splits
from .phase4_confirmation import parse_phase4_confirmation_json
from .phase4_evaluation import Phase4CandidateCVResult, evaluate_phase4_candidate_cv
from .phase4_screening_experiment import _partition_hash

MODEL_VERSION: Final = "retail-rf05-v1"
BUNDLE_SCHEMA_VERSION: Final = 2
BUNDLE_ARTIFACT_TYPE: Final = "trusted_local_rf05_pipeline"
POLICY_SHA256: Final = "becb895893f81cf04744786b722fc2a3c40be9e52257590989e1f9f2b44c831b"
CALIBRATION_SHA256: Final = "b7eb5970b164ec68fb76cf8314f36080d085cda02968d3570d11f724490a6da0"
CALIBRATION_ASSIGNMENT_SHA256: Final = (
    "caa743681158c4eaccb2ec75ce17a1c5e20327a311f66c5e8e0d0c630c48e992"
)
DEVELOPMENT_ROWS: Final = 98_552
PHASE3_TRAIN_ROWS: Final = 109_510
CALIBRATION_ROWS: Final = 10_958
RF05_INDEX: Final = 5
PICKLE_PROTOCOL: Final = 5
JOBLIB_COMPRESSION: Final = 0
_METRIC_REL_TOLERANCE: Final = 1e-12
_METRIC_ABS_TOLERANCE: Final = 1e-9
_POLICY_PATH: Final = PurePosixPath(
    "docs/experiments/retail-rf05-serving-reconstruction-policy-v1.json"
)
_CONFIRMATION_PATH: Final = PurePosixPath("docs/experiments/phase4-retail-full-development-v1.json")
_CALIBRATION_PATH: Final = PurePosixPath(
    "docs/experiments/retail-rf05-calibration-v1.artifact.json"
)
_SOURCE_PATHS: Final = (
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.manifest.json"),
    PurePosixPath("data/processed/kaggle_us_sales_cars_v2/split/split_assignments.manifest.json"),
    PurePosixPath("docs/data-reviews/kaggle-us-sales-cars-v2.review.json"),
)
_EXPECTED_STATUS_COUNTS: Final = {"certified": 5_467, "new": 58_360, "used": 34_725}
_EXPECTED_POLICY_KEYS: Final = {
    "schema_version",
    "policy_id",
    "authorized_on",
    "classification",
    "purpose",
    "model",
    "training_boundary",
    "reproduction_gate",
    "determinism_gate",
    "serialization",
    "calibration_binding",
    "upstream_sha256",
    "publication",
    "prohibited_actions",
}
_DIGEST_CHARACTERS: Final = frozenset("0123456789abcdef")


class RF05ServingBundleError(RuntimeError):
    """A reconstruction input, proof, or output violated the frozen policy."""


@dataclass(frozen=True, slots=True)
class AuthorizedDevelopmentData:
    """The exact development-only population authorized for the packaging fit."""

    features: pd.DataFrame
    target: NDArray[np.float64]
    identity_sha256: str
    calibration_assignment_sha256: str
    status_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ReconstructedBundle:
    """Paths and aggregate evidence emitted by one successful reconstruction."""

    bundle_dir: Path
    model_sha256: str
    manifest_sha256: str
    bundle_sha256: str
    golden_fixture_sha256: str
    report: Mapping[str, object]


def reconstruct_rf05_serving_bundle(
    *,
    project_root: Path,
    bundle_dir: Path,
    report_path: Path,
    golden_fixture_path: Path,
    force: bool = False,
) -> ReconstructedBundle:
    """Prove, refit, authenticate, and package only the already-frozen RF05 system."""

    root = _validated_root(project_root)
    output = _validated_bundle_output(root, bundle_dir, force=force)
    report_output = _validated_text_output(root, report_path, force=force)
    golden_output = _validated_text_output(root, golden_fixture_path, force=force)
    policy, upstream = _verify_policy_and_upstream(root)
    development = _load_authorized_development(root)
    reference = _load_rf05_reference(root)
    reproduced = _reproduce_oof(development)
    reproduction = _compare_reproduction(reference, reproduced)

    primary = _fit_full_development(development)
    independent = _fit_full_development(development)
    primary.named_steps["regressor"].set_params(n_jobs=1)
    independent.named_steps["regressor"].set_params(n_jobs=1)
    deterministic = _verify_determinism(primary, independent, development.features)
    primary_bytes = _serialize_pipeline(primary)
    independent_bytes = _serialize_pipeline(independent)
    deterministic["serialized_byte_identical"] = primary_bytes == independent_bytes
    deterministic["independent_model_sha256"] = hashlib.sha256(independent_bytes).hexdigest()

    model_sha256 = hashlib.sha256(primary_bytes).hexdigest()
    manifest = _bundle_manifest(
        development=development,
        model_sha256=model_sha256,
        upstream=upstream,
    )
    manifest_bytes = _canonical_json(manifest).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    bundle_sha256 = _bundle_sha256(manifest_bytes, primary_bytes)

    calibration_bytes = _trusted_file(root, _CALIBRATION_PATH).read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != CALIBRATION_SHA256:
        raise RF05ServingBundleError("frozen calibration checksum changed")
    calibration = load_calibration_artifact(
        calibration_bytes,
        active_model_identity_sha256=active_rf05_identity().identity_sha256,
    )
    golden = _golden_fixture_payload(
        primary,
        calibration=calibration,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
    )
    golden_text = _canonical_json(golden)
    golden_sha256 = hashlib.sha256(golden_text.encode("utf-8")).hexdigest()
    report = _reconstruction_report(
        policy=policy,
        upstream=upstream,
        development=development,
        reproduction=reproduction,
        deterministic=deterministic,
        manifest=manifest,
        model_bytes=primary_bytes,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        bundle_sha256=bundle_sha256,
        golden_sha256=golden_sha256,
        golden_count=len(cast(list[object], golden["fixtures"])),
    )

    _write_bundle(output, manifest_bytes=manifest_bytes, model_bytes=primary_bytes, force=force)
    _write_atomic_text(golden_output, golden_text, force=force)
    _write_atomic_text(report_output, _canonical_json(report), force=force)
    return ReconstructedBundle(
        bundle_dir=output,
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        bundle_sha256=bundle_sha256,
        golden_fixture_sha256=golden_sha256,
        report=report,
    )


def current_runtime_versions() -> dict[str, str]:
    """Return exact serialization-relevant runtime versions in stable key order."""

    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scipy": metadata.version("scipy"),
        "scikit_learn": metadata.version("scikit-learn"),
        "joblib": metadata.version("joblib"),
    }


def development_identity_sha256(
    features: object,
    target: object,
) -> str:
    """Hash ordered development predictors and target without emitting source rows."""

    frame = validate_feature_frame(features, RETAIL_TRACK)
    y = validate_target(target, expected_rows=len(frame), config=RETAIL_TRACK)
    digest = hashlib.sha256(b"autovalue-retail-rf05-development-identity-v1\x00")
    for position, row in enumerate(frame.itertuples(index=False, name=None)):
        year, make, model, status, mileage = row
        mileage_value = None if pd.isna(mileage) else int(cast(float, mileage))
        payload = [
            position,
            int(cast(int, year)),
            cast(str, make),
            cast(str, model),
            cast(str, status),
            mileage_value,
            float(y[position]).hex(),
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def bundle_sha256(manifest_bytes: bytes, model_bytes: bytes) -> str:
    """Return the domain-separated fingerprint for both serving files."""

    return _bundle_sha256(manifest_bytes, model_bytes)


def _verify_policy_and_upstream(
    root: Path,
) -> tuple[Mapping[str, object], dict[str, str]]:
    policy_path = _trusted_file(root, _POLICY_PATH)
    policy_bytes = policy_path.read_bytes()
    if hashlib.sha256(policy_bytes).hexdigest() != POLICY_SHA256:
        raise RF05ServingBundleError("serving reconstruction policy checksum changed")
    try:
        policy = json.loads(policy_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RF05ServingBundleError("serving reconstruction policy is invalid") from error
    if not isinstance(policy, dict) or set(policy) != _EXPECTED_POLICY_KEYS:
        raise RF05ServingBundleError("serving reconstruction policy fields changed")
    _verify_policy_model(policy)
    upstream_value = policy["upstream_sha256"]
    if not isinstance(upstream_value, dict) or not upstream_value:
        raise RF05ServingBundleError("upstream checksum policy is invalid")
    verified: dict[str, str] = {}
    for relative, expected in upstream_value.items():
        if not isinstance(relative, str) or not _is_sha256(expected):
            raise RF05ServingBundleError("upstream checksum policy is invalid")
        path = _trusted_file(root, PurePosixPath(relative))
        actual = _hash_file(path)
        if actual != expected:
            raise RF05ServingBundleError(f"frozen upstream checksum changed: {relative}")
        verified[relative] = actual
    return cast(Mapping[str, object], policy), verified


def _verify_policy_model(policy: Mapping[str, object]) -> None:
    expected_model = {
        "model_version": MODEL_VERSION,
        "candidate_id": RF05_CANDIDATE_ID,
        "candidate_index": RF05_INDEX,
        "parameters": list(RF05_PARAMETERS),
        "random_state": RF05_RANDOM_STATE,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
    }
    if policy.get("schema_version") != 1 or policy.get("model") != expected_model:
        raise RF05ServingBundleError("serving reconstruction model policy changed")
    boundary = policy.get("training_boundary")
    if not isinstance(boundary, dict):
        raise RF05ServingBundleError("serving reconstruction boundary is invalid")
    if (
        boundary.get("phase3_train_rows") != PHASE3_TRAIN_ROWS
        or boundary.get("calibration_rows_excluded") != CALIBRATION_ROWS
        or boundary.get("development_rows_authorized") != DEVELOPMENT_ROWS
        or boundary.get("calibration_assignment_sha256") != CALIBRATION_ASSIGNMENT_SHA256
        or boundary.get("development_only_fit") is not True
    ):
        raise RF05ServingBundleError("serving reconstruction boundary changed")


def _load_authorized_development(root: Path) -> AuthorizedDevelopmentData:
    source_paths = tuple(_trusted_file(root, relative) for relative in _SOURCE_PATHS)
    stream = prepare_kaggle_us_sales_cars_split_training_rows(
        *source_paths,
        partition="train",
    )
    train = _collect_retail_partition(
        cast(Iterable[RetailTrainingRow], stream),
        expected_rows=_expected_count(stream, "expected_rows"),
        label="authorized retail Phase-3 train",
    )
    if len(train.features) != PHASE3_TRAIN_ROWS:
        raise RF05ServingBundleError("Phase-3 train row count differs from frozen evidence")
    partition = retail_calibration_partition(train.features, seed=1_416_582_761)
    assignment_sha256 = _partition_hash(
        partition.calibration_indices,
        population_count=len(train.features),
        selected_label="calibration",
        unselected_label="development",
    )
    if assignment_sha256 != CALIBRATION_ASSIGNMENT_SHA256:
        raise RF05ServingBundleError("calibration assignment differs from frozen evidence")
    if len(partition.calibration_indices) != CALIBRATION_ROWS:
        raise RF05ServingBundleError("calibration exclusion count differs from frozen evidence")
    features = train.features.iloc[partition.development_indices].reset_index(drop=True)
    target = train.target[partition.development_indices].astype(np.float64, copy=True)
    if len(features) != DEVELOPMENT_ROWS:
        raise RF05ServingBundleError("development row count differs from frozen evidence")
    status_counts = {
        status: int((features["vehicle_status"] == status).sum())
        for status in ("certified", "new", "used")
    }
    if status_counts != _EXPECTED_STATUS_COUNTS:
        raise RF05ServingBundleError("development status counts differ from frozen evidence")
    identity = development_identity_sha256(features, target)
    return AuthorizedDevelopmentData(
        features=features,
        target=target,
        identity_sha256=identity,
        calibration_assignment_sha256=assignment_sha256,
        status_counts=status_counts,
    )


def _load_rf05_reference(root: Path) -> Phase4CandidateCVResult:
    report = parse_phase4_confirmation_json(_trusted_file(root, _CONFIRMATION_PATH).read_bytes())
    matches = tuple(
        result for result in report.candidates if result.spec.candidate_id == RF05_CANDIDATE_ID
    )
    if len(matches) != 1:
        raise RF05ServingBundleError("frozen RF05 reference result is unavailable")
    reference = matches[0]
    expected_spec = get_candidate_spec("retail", "random_forest", RF05_INDEX)
    if reference.spec != expected_spec:
        raise RF05ServingBundleError("frozen RF05 reference definition changed")
    return reference


def _reproduce_oof(development: AuthorizedDevelopmentData) -> Phase4CandidateCVResult:
    splits = retail_group_cv_splits(development.features, n_splits=5)
    expected_mask = np.ones(DEVELOPMENT_ROWS, dtype=np.bool_)
    return evaluate_phase4_candidate_cv(
        features=development.features,
        target=development.target,
        spec=get_candidate_spec("retail", "random_forest", RF05_INDEX),
        splits=splits,
        expected_oof_mask=expected_mask,
        validation_buckets=(None,) * 5,
    )


def _compare_reproduction(
    expected: Phase4CandidateCVResult,
    actual: Phase4CandidateCVResult,
) -> dict[str, object]:
    expected_metrics = _all_metric_values(expected)
    actual_metrics = _all_metric_values(actual)
    if expected_metrics.keys() != actual_metrics.keys():
        raise RF05ServingBundleError("reproduced RF05 metric shape differs")
    deltas: dict[str, float] = {}
    for name, expected_value in expected_metrics.items():
        actual_value = actual_metrics[name]
        delta = abs(actual_value - expected_value)
        deltas[name] = delta
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=_METRIC_REL_TOLERANCE,
            abs_tol=_METRIC_ABS_TOLERANCE,
        ):
            raise RF05ServingBundleError(f"reproduced RF05 evidence differs: {name}")
    return {
        "passed": True,
        "method": "existing five-fold predictor-group out-of-fold evaluation",
        "sample_count": actual.overall.sample_count,
        "fold_count": len(actual.folds),
        "expected": _metric_payload(expected),
        "reproduced": _metric_payload(actual),
        "maximum_absolute_metric_delta": max(deltas.values(), default=0.0),
        "relative_tolerance": _METRIC_REL_TOLERANCE,
        "absolute_tolerance": _METRIC_ABS_TOLERANCE,
        "fold_and_status_metrics_compared": True,
    }


def _all_metric_values(result: Phase4CandidateCVResult) -> dict[str, float]:
    values = {
        "overall.mae": result.overall.mae,
        "overall.rmse": result.overall.rmse,
        "overall.r2": cast(float, result.overall.r2),
    }
    for fold in result.folds:
        prefix = f"fold_{fold.fold_number}"
        values[f"{prefix}.mae"] = fold.metrics.mae
        values[f"{prefix}.rmse"] = fold.metrics.rmse
        values[f"{prefix}.r2"] = cast(float, fold.metrics.r2)
    for status in result.status_slices:
        prefix = f"status_{status.status}"
        values[f"{prefix}.mae"] = status.metrics.mae
        values[f"{prefix}.rmse"] = status.metrics.rmse
        values[f"{prefix}.r2"] = cast(float, status.metrics.r2)
    return values


def _metric_payload(result: Phase4CandidateCVResult) -> dict[str, float | int]:
    return {
        "sample_count": result.overall.sample_count,
        "mae_usd": result.overall.mae,
        "rmse_usd": result.overall.rmse,
        "r_squared": cast(float, result.overall.r2),
    }


def _fit_full_development(development: AuthorizedDevelopmentData) -> Pipeline:
    pipeline = make_random_forest_candidate("retail", RF05_INDEX, n_jobs=4)
    pipeline.fit(development.features, development.target)
    return pipeline


def _verify_determinism(
    primary: Pipeline,
    independent: Pipeline,
    development_features: pd.DataFrame,
) -> dict[str, object]:
    fixture = _golden_input_frame()
    primary_development = np.asarray(primary.predict(development_features), dtype=np.float64)
    second_development = np.asarray(independent.predict(development_features), dtype=np.float64)
    primary_fixture = np.asarray(primary.predict(fixture), dtype=np.float64)
    second_fixture = np.asarray(independent.predict(fixture), dtype=np.float64)
    development_delta = float(np.max(np.abs(primary_development - second_development)))
    fixture_delta = float(np.max(np.abs(primary_fixture - second_fixture)))
    if development_delta != 0.0 or fixture_delta != 0.0:
        raise RF05ServingBundleError(
            "independent RF05 reconstructions are not prediction-identical"
        )
    return {
        "passed": True,
        "independent_fits": 2,
        "development_predictions_compared": len(development_features),
        "synthetic_predictions_compared": len(fixture),
        "maximum_development_prediction_difference_usd": development_delta,
        "maximum_synthetic_prediction_difference_usd": fixture_delta,
        "prediction_equivalence": "exact float64 equality",
    }


def _serialize_pipeline(pipeline: Pipeline) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(
        pipeline,
        buffer,
        compress=JOBLIB_COMPRESSION,
        protocol=PICKLE_PROTOCOL,
    )
    payload = buffer.getvalue()
    if not payload:
        raise RF05ServingBundleError("serialized RF05 model is empty")
    return payload


def _bundle_manifest(
    *,
    development: AuthorizedDevelopmentData,
    model_sha256: str,
    upstream: Mapping[str, str],
) -> dict[str, object]:
    specification = {
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
    specification_sha256 = hashlib.sha256(
        _canonical_json(specification).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "artifact_type": BUNDLE_ARTIFACT_TYPE,
        "model_file": "model.joblib",
        "model_version": MODEL_VERSION,
        "candidate_id": RF05_CANDIDATE_ID,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "rf05_identity_sha256": active_rf05_identity().identity_sha256,
        "model_sha256": model_sha256,
        "reconstruction_policy_sha256": POLICY_SHA256,
        "specification": specification,
        "specification_sha256": specification_sha256,
        "training_data": {
            "source_id": "kaggle_us_sales_cars_v2",
            "target_semantics": RETAIL_TRACK.target_semantics,
            "development_rows": DEVELOPMENT_ROWS,
            "development_identity_algorithm": (
                "sha256 length-prefixed canonical ordered predictors+target v1"
            ),
            "development_identity_sha256": development.identity_sha256,
            "candidate_sha256": upstream[
                "data/processed/kaggle_us_sales_cars_v2/asking_price_candidate.csv"
            ],
            "split_assignments_sha256": upstream[
                "data/processed/kaggle_us_sales_cars_v2/split/split_assignments.csv"
            ],
            "calibration_assignment_sha256": development.calibration_assignment_sha256,
            "calibration_rows_used_for_fit": False,
            "final_holdout_rows_used_for_fit_or_verification": False,
        },
        "calibration_binding": {
            "version": CALIBRATION_VERSION,
            "artifact_sha256": CALIBRATION_SHA256,
            "rf05_identity_sha256": active_rf05_identity().identity_sha256,
        },
        "serialization": {
            "format": "joblib",
            "pickle_protocol": PICKLE_PROTOCOL,
            "compression": JOBLIB_COMPRESSION,
            "trusted_local_only": True,
        },
        "runtime": current_runtime_versions(),
        "contents": ["manifest.json", "model.joblib"],
        "publication": "deployment_private_binary",
    }


def _golden_input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year": 2019,
                "make": "Toyota",
                "model": "Camry",
                "vehicle_status": "used",
                "mileage": 64_000,
            },
            {
                "year": 2022,
                "make": "Ford",
                "model": "F-150",
                "vehicle_status": "certified",
                "mileage": 21_500,
            },
            {
                "year": 2023,
                "make": "Honda",
                "model": "CR-V",
                "vehicle_status": "new",
                "mileage": np.nan,
            },
            {
                "year": 2015,
                "make": "BMW",
                "model": "X5",
                "vehicle_status": "used",
                "mileage": 108_000,
            },
            {
                "year": 2008,
                "make": "Chevrolet",
                "model": "Silverado 1500",
                "vehicle_status": "used",
                "mileage": 189_000,
            },
        ],
        columns=list(RETAIL_TRACK.input_features),
    )


def _golden_fixture_payload(
    pipeline: Pipeline,
    *,
    calibration: RetailCalibrationArtifact,
    model_sha256: str,
    manifest_sha256: str,
) -> dict[str, object]:
    frame = _golden_input_frame()
    predictions = np.asarray(pipeline.predict(frame), dtype=np.float64)
    fixtures: list[dict[str, object]] = []
    for position, row in enumerate(frame.to_dict(orient="records"), start=1):
        request = {
            "year": int(cast(int, row["year"])),
            "make": cast(str, row["make"]),
            "model": cast(str, row["model"]),
            "vehicle_status": cast(str, row["vehicle_status"]),
            "mileage": None if pd.isna(row["mileage"]) else int(cast(float, row["mileage"])),
            "interval_coverage": 0.9,
        }
        vehicle_status = cast(str, request["vehicle_status"])
        valuation = calibrated_valuation(
            point_prediction=float(predictions[position - 1]),
            vehicle_status=vehicle_status,
            coverage=0.9,
            artifact=calibration,
            data_quality=PredictionDataQuality(
                mileage_missing=request["mileage"] is None,
                rare_or_unseen_category=False,
                unsupported_feature_combination=False,
            ),
        )
        lower = round(valuation.interval_lower, 2)
        upper = round(valuation.interval_upper, 2)
        fixtures.append(
            {
                "fixture_id": f"synthetic_vehicle_{position}",
                "input": request,
                "expected": {
                    "predicted_value": round(valuation.predicted_value, 2),
                    "interval_lower": lower,
                    "interval_upper": upper,
                    "interval_width": round(upper - lower, 2),
                    "interval_coverage": 0.9,
                    "calibration_version": CALIBRATION_VERSION,
                    "model_version": MODEL_VERSION,
                },
            }
        )
    return {
        "schema_version": 1,
        "fixture_type": "privacy_safe_synthetic_rf05_serving_regression",
        "model_sha256": model_sha256,
        "manifest_sha256": manifest_sha256,
        "calibration_sha256": CALIBRATION_SHA256,
        "source_rows_included": False,
        "fixtures": fixtures,
    }


def _reconstruction_report(
    *,
    policy: Mapping[str, object],
    upstream: Mapping[str, str],
    development: AuthorizedDevelopmentData,
    reproduction: Mapping[str, object],
    deterministic: Mapping[str, object],
    manifest: Mapping[str, object],
    model_bytes: bytes,
    model_sha256: str,
    manifest_sha256: str,
    bundle_sha256: str,
    golden_sha256: str,
    golden_count: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_type": "retail_rf05_serving_reconstruction",
        "generated_at": datetime.now(UTC).isoformat(),
        "decision": "passed_for_private_local_serving",
        "classification": policy["classification"],
        "reconstruction_policy_sha256": POLICY_SHA256,
        "model_definition": manifest["specification"],
        "training_boundary": {
            "source_id": "kaggle_us_sales_cars_v2",
            "phase3_train_rows": PHASE3_TRAIN_ROWS,
            "calibration_rows_excluded": CALIBRATION_ROWS,
            "development_rows_fitted": DEVELOPMENT_ROWS,
            "development_status_counts": dict(development.status_counts),
            "development_identity_sha256": development.identity_sha256,
            "calibration_rows_used_for_fit": False,
            "final_holdout_requested_loaded_scored_or_inspected": False,
            "external_sources_used": False,
        },
        "upstream_verification": {
            "all_passed": True,
            "artifact_count": len(upstream),
            "sha256": dict(upstream),
        },
        "development_oof_reproduction": dict(reproduction),
        "determinism": dict(deterministic),
        "serving_bundle": {
            "model_version": MODEL_VERSION,
            "contents": ["manifest.json", "model.joblib"],
            "model_size_bytes": len(model_bytes),
            "model_sha256": model_sha256,
            "manifest_sha256": manifest_sha256,
            "bundle_sha256": bundle_sha256,
            "bundle_sha256_algorithm": (
                "sha256 domain + uint64 manifest length + manifest bytes + model bytes"
            ),
            "raw_training_rows_included": False,
            "distribution_status": "deployment_private_pending_explicit_permission",
        },
        "calibration_binding": manifest["calibration_binding"],
        "runtime": manifest["runtime"],
        "golden_fixtures": {
            "count": golden_count,
            "sha256": golden_sha256,
            "privacy_safe_synthetic_inputs_only": True,
        },
        "governance": {
            "new_ml_experiment": False,
            "model_selection_or_tuning_performed": False,
            "feature_engineering_changed": False,
            "calibration_regenerated_or_changed": False,
            "final_holdout_accessed": False,
            "model_decision_changed": False,
            "public_binary_distribution_approved": False,
        },
    }


def _validated_root(value: Path) -> Path:
    try:
        root = value.expanduser().resolve(strict=True)
    except OSError as error:
        raise RF05ServingBundleError("project root is unavailable") from error
    if not root.is_dir() or root.is_symlink():
        raise RF05ServingBundleError("project root must be a real directory")
    if not (root / "pyproject.toml").is_file() or not (root / "ml").is_dir():
        raise RF05ServingBundleError("project root does not contain AutoValue")
    return root


def _trusted_file(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise RF05ServingBundleError("trusted input path is invalid")
    path = root.joinpath(*relative.parts)
    try:
        information = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RF05ServingBundleError(f"trusted input is unavailable: {relative}") from error
    if (
        not stat.S_ISREG(information.st_mode)
        or path.is_symlink()
        or not resolved.is_relative_to(root)
    ):
        raise RF05ServingBundleError(f"trusted input is not a safe regular file: {relative}")
    return resolved


def _validated_bundle_output(root: Path, value: Path, *, force: bool) -> Path:
    models_root = (root / "models").resolve(strict=True)
    output = value if value.is_absolute() else root / value
    output = output.absolute()
    if output.name != MODEL_VERSION or output.parent.resolve(strict=True) != models_root:
        raise RF05ServingBundleError("bundle must be models/retail-rf05-v1")
    _check_replaceable(output, force=force, directory=True)
    return output


def _validated_text_output(root: Path, value: Path, *, force: bool) -> Path:
    output = value if value.is_absolute() else root / value
    output = output.absolute()
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise RF05ServingBundleError("output parent is unavailable") from error
    if not parent.is_relative_to(root):
        raise RF05ServingBundleError("output must remain inside the project")
    _check_replaceable(output, force=force, directory=False)
    return output


def _check_replaceable(path: Path, *, force: bool, directory: bool) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise RF05ServingBundleError("output may not be a symbolic link")
    if not force:
        raise RF05ServingBundleError("output already exists; explicit force is required")
    if directory:
        if not path.is_dir():
            raise RF05ServingBundleError("bundle output is not a directory")
        allowed = {"manifest.json", "model.joblib"}
        if {child.name for child in path.iterdir()} - allowed:
            raise RF05ServingBundleError("existing bundle contains unexpected files")
    elif not path.is_file():
        raise RF05ServingBundleError("text output is not a regular file")


def _write_bundle(
    output: Path,
    *,
    manifest_bytes: bytes,
    model_bytes: bytes,
    force: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix=".rf05-bundle-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        (temporary_path / "manifest.json").write_bytes(manifest_bytes)
        (temporary_path / "model.joblib").write_bytes(model_bytes)
        if force and output.exists():
            for child in output.iterdir():
                if child.name not in {"manifest.json", "model.joblib"}:
                    raise RF05ServingBundleError("existing bundle contains unexpected files")
                child.unlink()
            output.rmdir()
        os.replace(temporary_path, output)


def _write_atomic_text(path: Path, payload: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise RF05ServingBundleError("output already exists; explicit force is required")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(manifest_bytes: bytes, model_bytes: bytes) -> str:
    if not manifest_bytes or not model_bytes:
        raise RF05ServingBundleError("bundle components must not be empty")
    digest = hashlib.sha256(b"autovalue-retail-rf05-serving-bundle-v1\x00")
    digest.update(len(manifest_bytes).to_bytes(8, byteorder="big", signed=False))
    digest.update(manifest_bytes)
    digest.update(model_bytes)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, object]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(_DIGEST_CHARACTERS)


__all__ = [
    "BUNDLE_ARTIFACT_TYPE",
    "BUNDLE_SCHEMA_VERSION",
    "CALIBRATION_SHA256",
    "MODEL_VERSION",
    "POLICY_SHA256",
    "AuthorizedDevelopmentData",
    "RF05ServingBundleError",
    "ReconstructedBundle",
    "bundle_sha256",
    "current_runtime_versions",
    "development_identity_sha256",
    "reconstruct_rf05_serving_bundle",
]
