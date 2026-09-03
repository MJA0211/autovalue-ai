"""Authenticated RF05 bundle loading, corruption, and calibrated inference tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
from autovalue_api.schemas import VehicleValuationRequest
from autovalue_api.services.valuation import (
    FrozenRF05Service,
    ValuationUnavailableError,
    _canonical_json,
    _expected_calibration_binding,
    _expected_specification,
    _expected_training_data,
    _parse_manifest,
)
from autovalue_ml.modeling.calibration_artifact import (
    FEATURE_CONTRACT_VERSION,
    RF05_CANDIDATE_ID,
    active_rf05_identity,
)
from autovalue_ml.modeling.candidates import make_random_forest_candidate
from autovalue_ml.modeling.rf05_serving_bundle import (
    BUNDLE_ARTIFACT_TYPE,
    BUNDLE_SCHEMA_VERSION,
    JOBLIB_COMPRESSION,
    MODEL_VERSION,
    PICKLE_PROTOCOL,
    POLICY_SHA256,
    current_runtime_versions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = PROJECT_ROOT / "docs/experiments/retail-rf05-calibration-v1.artifact.json"


def _write_valid_bundle(root: Path) -> tuple[Path, str, str]:
    bundle = root / MODEL_VERSION
    bundle.mkdir()
    frame = pd.DataFrame(
        {
            "year": [2018, 2019, 2020, 2021, 2022, 2017, 2016, 2015],
            "make": ["Toyota", "Honda", "Ford", "Toyota", "Honda", "Ford", "Toyota", "Ford"],
            "model": ["Camry", "Civic", "F-150", "Camry", "Civic", "F-150", "Camry", "F-150"],
            "vehicle_status": ["used"] * 8,
            "mileage": [80_000, 70_000, 60_000, 40_000, 20_000, 90_000, 110_000, 120_000],
        }
    )
    target = [17_000, 18_000, 27_000, 23_000, 25_000, 22_000, 14_000, 19_000]
    pipeline = make_random_forest_candidate("retail", 5, n_jobs=1)
    pipeline.fit(frame, target)
    model_path = bundle / "model.joblib"
    joblib.dump(pipeline, model_path, compress=0, protocol=5)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    specification = _expected_specification()
    manifest: dict[str, Any] = {
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
        "specification_sha256": hashlib.sha256(_canonical_json(specification)).hexdigest(),
        "training_data": _expected_training_data(),
        "calibration_binding": _expected_calibration_binding(),
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
    manifest_bytes = _canonical_json(manifest)
    (bundle / "manifest.json").write_bytes(manifest_bytes)
    return bundle, hashlib.sha256(manifest_bytes).hexdigest(), model_sha256


def _service(
    *,
    bundle: Path,
    models_root: Path,
    manifest_sha256: str,
    model_sha256: str,
    calibration_path: Path = CALIBRATION_PATH,
    calibration_root: Path = CALIBRATION_PATH.parent,
) -> FrozenRF05Service:
    return FrozenRF05Service(
        bundle_dir=bundle,
        calibration_path=calibration_path,
        trusted_models_root=models_root,
        trusted_calibration_root=calibration_root,
        expected_manifest_sha256=manifest_sha256,
        expected_model_sha256=model_sha256,
    )


def test_valid_authenticated_bundle_serves_calibrated_prediction(tmp_path: Path) -> None:
    bundle, manifest_sha256, model_sha256 = _write_valid_bundle(tmp_path)
    service = _service(
        bundle=bundle,
        models_root=tmp_path,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
    )

    response = service.predict(
        VehicleValuationRequest(
            year=2020,
            make="Toyota",
            model="Camry",
            vehicle_status="used",
            mileage=None,
        )
    )

    assert service.ready is True
    assert service.public_unavailable_reason is None
    assert response.predicted_value > 0
    assert response.interval_coverage == 0.9
    assert response.interval_lower is not None
    assert response.interval_upper is not None
    assert response.interval_lower <= response.predicted_value <= response.interval_upper
    assert "missing_mileage" in response.warnings
    assert response.model_information is not None
    assert response.model_information.model_version == MODEL_VERSION


def test_missing_bundle_fails_closed_without_deserialization(tmp_path: Path) -> None:
    service = FrozenRF05Service(
        bundle_dir=tmp_path / MODEL_VERSION,
        calibration_path=CALIBRATION_PATH,
        trusted_models_root=tmp_path,
        trusted_calibration_root=CALIBRATION_PATH.parent,
    )

    assert service.ready is False
    assert service.public_unavailable_reason == "trusted RF05 artifact unavailable"
    with pytest.raises(ValuationUnavailableError, match="trusted RF05 artifact unavailable"):
        service.predict(
            VehicleValuationRequest(year=2020, make="Toyota", model="Camry", vehicle_status="used")
        )


def test_corrupted_model_fails_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, manifest_sha256, model_sha256 = _write_valid_bundle(tmp_path)
    model_path = bundle / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"corruption")
    load_called = False

    def unexpected_load(_: object) -> object:
        nonlocal load_called
        load_called = True
        raise AssertionError("unauthenticated bytes were deserialized")

    monkeypatch.setattr("autovalue_api.services.valuation.joblib.load", unexpected_load)
    service = _service(
        bundle=bundle,
        models_root=tmp_path,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
    )

    assert service.ready is False
    assert load_called is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("calibration_binding", {"version": "wrong"}),
        ("runtime", {"python": "0.0.0"}),
        ("training_data", {"development_rows": 98_551}),
    ],
)
def test_authenticated_manifest_with_binding_mismatch_fails_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    bundle, _, model_sha256 = _write_valid_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest[field] = replacement
    manifest_bytes = _canonical_json(manifest)
    manifest_path.write_bytes(manifest_bytes)

    service = _service(
        bundle=bundle,
        models_root=tmp_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        model_sha256=model_sha256,
    )

    assert service.ready is False


def test_manifest_authentication_mismatch_fails_closed(tmp_path: Path) -> None:
    bundle, _, model_sha256 = _write_valid_bundle(tmp_path)
    service = _service(
        bundle=bundle,
        models_root=tmp_path,
        manifest_sha256="0" * 64,
        model_sha256=model_sha256,
    )

    assert service.ready is False


def test_unexpected_bundle_file_fails_closed(tmp_path: Path) -> None:
    bundle, manifest_sha256, model_sha256 = _write_valid_bundle(tmp_path)
    (bundle / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    service = _service(
        bundle=bundle,
        models_root=tmp_path,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
    )

    assert service.ready is False


def test_bundle_outside_trusted_root_fails_closed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle, manifest_sha256, model_sha256 = _write_valid_bundle(outside)
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    service = _service(
        bundle=bundle,
        models_root=trusted,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
    )

    assert service.ready is False


def test_calibration_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    bundle, manifest_sha256, model_sha256 = _write_valid_bundle(models)
    calibration_root = tmp_path / "calibration"
    calibration_root.mkdir()
    calibration_path = calibration_root / CALIBRATION_PATH.name
    calibration_path.write_bytes(CALIBRATION_PATH.read_bytes() + b"corruption")
    service = _service(
        bundle=bundle,
        models_root=models,
        manifest_sha256=manifest_sha256,
        model_sha256=model_sha256,
        calibration_path=calibration_path,
        calibration_root=calibration_root,
    )

    assert service.ready is False


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b"{}",
        b'{"schema_version":2,"schema_version":2}',
    ],
)
def test_invalid_manifest_is_rejected(payload: bytes) -> None:
    with pytest.raises(ValuationUnavailableError):
        _parse_manifest(payload)
