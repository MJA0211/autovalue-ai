"""Public valuation-route behavior and sanitized failure tests."""

from pathlib import Path
from typing import Any

import pytest
from autovalue_api.main import create_app
from autovalue_api.schemas import PredictionResponse, VehicleValuationRequest
from autovalue_api.services.history import SQLitePredictionHistory
from autovalue_api.services.valuation import ValuationUnavailableError
from fastapi.testclient import TestClient


class StubEngine:
    def __init__(self, *, ready: bool) -> None:
        self.ready = ready
        self.received: VehicleValuationRequest | None = None

    def predict(self, request: VehicleValuationRequest) -> PredictionResponse:
        self.received = request
        if not self.ready:
            raise ValuationUnavailableError("private diagnostic detail")
        return PredictionResponse(
            predicted_value=24_500.0,
            interval_lower=10_594.0,
            interval_upper=38_406.0,
            interval_coverage=request.interval_coverage,
            interval_width=27_812.0,
            confidence_label="Moderate confidence",
            calibration_version="retail-rf05-split-conformal-v1",
        )


def _payload(**updates: Any) -> dict[str, object]:
    payload: dict[str, object] = {
        "year": 2020,
        "make": "Toyota",
        "model": "Camry",
        "vehicle_status": "used",
        "mileage": 48_000,
        "interval_coverage": 0.9,
    }
    payload.update(updates)
    return payload


def test_model_status_discloses_evidence_and_readiness() -> None:
    client = TestClient(create_app(valuation_service=StubEngine(ready=False)))

    response = client.get("/api/v1/model")

    assert response.status_code == 200
    assert response.json()["status"] == "artifact_required"
    assert response.json()["can_predict"] is False
    assert response.json()["metrics"] == {
        "holdout_rows": 27_589,
        "mae_usd": 10_575.36,
        "rmse_usd": 34_118.14,
        "r_squared": 0.4176,
        "median_absolute_error_usd": 6_678.93,
    }


def test_valid_request_reaches_ready_engine() -> None:
    engine = StubEngine(ready=True)
    client = TestClient(create_app(valuation_service=engine))

    response = client.post("/api/v1/valuations", json=_payload())

    assert response.status_code == 200
    assert response.json()["currency"] == "USD"
    assert response.json()["interval_coverage"] == 0.9
    assert engine.received is not None
    assert engine.received.make == "Toyota"


def test_successful_prediction_round_trips_only_for_same_browser(tmp_path: Path) -> None:
    history = SQLitePredictionHistory(tmp_path / "history.sqlite3")
    client = TestClient(
        create_app(valuation_service=StubEngine(ready=True), prediction_history=history)
    )
    first_id = "d8a1ac42-914e-486d-8791-962edfb0d14b"
    other_id = "8a126587-c13f-4d64-b7c5-754774486f99"

    created = client.post(
        "/api/v1/valuations",
        json=_payload(),
        headers={"X-AutoValue-Client": first_id},
    )
    same_browser = client.get(
        "/api/v1/predictions/recent", headers={"X-AutoValue-Client": first_id}
    )
    other_browser = client.get(
        "/api/v1/predictions/recent", headers={"X-AutoValue-Client": other_id}
    )

    assert created.status_code == 200
    assert same_browser.status_code == 200
    assert len(same_browser.json()["predictions"]) == 1
    assert same_browser.json()["predictions"][0]["make"] == "Toyota"
    assert other_browser.json() == {"predictions": []}


def test_invalid_browser_identifier_is_rejected() -> None:
    client = TestClient(create_app(valuation_service=StubEngine(ready=True)))

    response = client.get(
        "/api/v1/predictions/recent", headers={"X-AutoValue-Client": "not-a-uuid"}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_client_id"


def test_unavailable_engine_returns_sanitized_503() -> None:
    client = TestClient(create_app(valuation_service=StubEngine(ready=False)))

    response = client.post("/api/v1/valuations", json=_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "valuation_service_unavailable",
            "message": "The verified RF05 serving artifact is not available.",
        }
    }
    assert "private diagnostic detail" not in response.text


@pytest.mark.parametrize(
    ("updates", "invalid_field"),
    [
        ({"year": 2024}, "year"),
        ({"mileage": -1}, "mileage"),
        ({"mileage": 500_001}, "mileage"),
        ({"interval_coverage": 0.85}, "interval_coverage"),
        ({"make": "Ford\nInjected"}, "make"),
        ({"condition": "excellent"}, "condition"),
    ],
)
def test_invalid_or_unsupported_input_is_rejected(
    updates: dict[str, object], invalid_field: str
) -> None:
    client = TestClient(create_app(valuation_service=StubEngine(ready=True)))

    response = client.post("/api/v1/valuations", json=_payload(**updates))

    assert response.status_code == 422
    assert invalid_field in response.text


def test_malformed_json_does_not_expose_a_traceback() -> None:
    client = TestClient(create_app(valuation_service=StubEngine(ready=True)))

    response = client.post(
        "/api/v1/valuations",
        content=b'{"year":',
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "Traceback" not in response.text
