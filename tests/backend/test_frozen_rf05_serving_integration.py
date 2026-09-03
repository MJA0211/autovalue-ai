"""Golden end-to-end checks for the private reconstructed RF05 bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
from autovalue_api.main import create_app
from autovalue_api.services.history import SQLitePredictionHistory
from autovalue_api.services.valuation import (
    TRUSTED_MANIFEST_SHA256,
    TRUSTED_MODEL_SHA256,
    FrozenRF05Service,
)
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = PROJECT_ROOT / "models/retail-rf05-v1"
CALIBRATION_PATH = PROJECT_ROOT / "docs/experiments/retail-rf05-calibration-v1.artifact.json"
GOLDEN_PATH = PROJECT_ROOT / "tests/fixtures/retail-rf05-v1.golden.json"

pytestmark = pytest.mark.skipif(
    not (BUNDLE_DIR / "model.joblib").is_file(),
    reason="deployment-private RF05 bundle is not provisioned",
)


def test_authentic_bundle_hashes_and_golden_api_outputs(tmp_path: Path) -> None:
    manifest_bytes = (BUNDLE_DIR / "manifest.json").read_bytes()
    model_bytes = (BUNDLE_DIR / "model.joblib").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == TRUSTED_MANIFEST_SHA256
    assert hashlib.sha256(model_bytes).hexdigest() == TRUSTED_MODEL_SHA256

    service = FrozenRF05Service(
        bundle_dir=BUNDLE_DIR,
        calibration_path=CALIBRATION_PATH,
        trusted_models_root=BUNDLE_DIR.parent,
        trusted_calibration_root=CALIBRATION_PATH.parent,
    )
    client = TestClient(
        create_app(
            valuation_service=service,
            prediction_history=SQLitePredictionHistory(tmp_path / "history.sqlite3"),
        )
    )
    assert client.get("/api/v1/model").json()["can_predict"] is True
    golden = json.loads(GOLDEN_PATH.read_bytes())
    assert golden["source_rows_included"] is False

    for fixture in cast(list[dict[str, Any]], golden["fixtures"]):
        response = client.post("/api/v1/valuations", json=fixture["input"])
        assert response.status_code == 200
        payload = response.json()
        expected = fixture["expected"]
        for field in (
            "predicted_value",
            "interval_lower",
            "interval_upper",
            "interval_width",
            "interval_coverage",
            "calibration_version",
        ):
            assert payload[field] == expected[field]
        assert payload["model_information"]["model_version"] == expected["model_version"]
        assert payload["currency"] == "USD"
