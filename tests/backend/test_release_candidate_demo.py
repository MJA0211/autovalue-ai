"""Authentic release-candidate smoke checks with synthetic demo vehicles."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

import pytest
from autovalue_api.main import create_app
from autovalue_api.services.history import SQLitePredictionHistory
from autovalue_api.services.valuation import FrozenRF05Service
from fastapi.testclient import TestClient

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
BUNDLE_DIR: Final = PROJECT_ROOT / "models/retail-rf05-v1"
CALIBRATION_PATH: Final = PROJECT_ROOT / "docs/experiments/retail-rf05-calibration-v1.artifact.json"
CLIENT_ID: Final = "48da3937-2163-440b-a050-b7964d14f406"

DEMO_VEHICLES: Final = (
    {
        "year": 2018,
        "make": "Subaru",
        "model": "Outback",
        "vehicle_status": "used",
        "mileage": 78_000,
        "interval_coverage": 0.9,
    },
    {
        "year": 2020,
        "make": "Ram",
        "model": "1500",
        "vehicle_status": "used",
        "mileage": 52_000,
        "interval_coverage": 0.9,
    },
    {
        "year": 2006,
        "make": "Honda",
        "model": "Accord",
        "vehicle_status": "used",
        "mileage": 210_000,
        "interval_coverage": 0.9,
    },
    {
        "year": 2019,
        "make": "Lexus",
        "model": "RX 350",
        "vehicle_status": "certified",
        "mileage": 61_000,
        "interval_coverage": 0.9,
    },
    {
        "year": 2023,
        "make": "Toyota",
        "model": "RAV4",
        "vehicle_status": "new",
        "mileage": None,
        "interval_coverage": 0.9,
    },
)

pytestmark = pytest.mark.skipif(
    not (BUNDLE_DIR / "model.joblib").is_file(),
    reason="deployment-private RF05 bundle is not provisioned",
)


def test_authentic_demo_vehicles_and_recent_history(tmp_path: Path) -> None:
    service = FrozenRF05Service(
        bundle_dir=BUNDLE_DIR,
        calibration_path=CALIBRATION_PATH,
        trusted_models_root=BUNDLE_DIR.parent,
        trusted_calibration_root=CALIBRATION_PATH.parent,
    )
    client = TestClient(
        create_app(
            valuation_service=service,
            prediction_history=SQLitePredictionHistory(tmp_path / "release-history.sqlite3"),
        )
    )

    status = client.get("/api/v1/model")
    assert status.status_code == 200
    assert status.json()["status"] == "ready"
    assert status.json()["model_version"] == "retail-rf05-v1"

    responses = []
    for vehicle in DEMO_VEHICLES:
        response = client.post(
            "/api/v1/valuations",
            json=vehicle,
            headers={"X-AutoValue-Client": CLIENT_ID},
        )
        assert response.status_code == 200
        payload = response.json()
        point = payload["predicted_value"]
        lower = payload["interval_lower"]
        upper = payload["interval_upper"]
        assert all(math.isfinite(value) for value in (point, lower, upper))
        assert 0.0 <= lower <= point <= upper
        assert payload["currency"] == "USD"
        assert payload["interval_coverage"] == 0.9
        assert payload["calibration_version"] == "retail-rf05-split-conformal-v1"
        assert payload["model_information"]["model_version"] == "retail-rf05-v1"
        assert isinstance(payload["warnings"], list)
        responses.append(payload)

    assert "missing_mileage" in responses[-1]["warnings"]

    history = client.get(
        "/api/v1/predictions/recent",
        headers={"X-AutoValue-Client": CLIENT_ID},
    )
    assert history.status_code == 200
    assert len(history.json()["predictions"]) == 5
    assert history.json()["predictions"][0]["model"] == "RAV4"

    malformed = client.post(
        "/api/v1/valuations",
        json={**DEMO_VEHICLES[0], "dealer": "not accepted"},
    )
    assert malformed.status_code == 422
