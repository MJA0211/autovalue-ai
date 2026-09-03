"""Tests for the liveness contract."""

from autovalue_api.main import create_app
from fastapi.testclient import TestClient


def test_liveness_returns_public_service_metadata() -> None:
    client = TestClient(create_app())

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "AutoValue AI API",
        "version": "0.1.0",
        "environment": "development",
    }


def test_openapi_document_initializes() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["info"]["title"] == "AutoValue AI API"


def test_loopback_frontend_origin_is_allowed() -> None:
    client = TestClient(create_app())

    response = client.options(
        "/health/live",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
