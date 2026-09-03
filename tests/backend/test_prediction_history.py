"""SQLite prediction-history privacy, validation, and retention tests."""

import sqlite3
from pathlib import Path

import pytest
from autovalue_api.schemas import PredictionResponse, VehicleValuationRequest
from autovalue_api.services.history import SQLitePredictionHistory

CLIENT_ID = "d8a1ac42-914e-486d-8791-962edfb0d14b"


def _request(year: int = 2020) -> VehicleValuationRequest:
    return VehicleValuationRequest(
        year=year,
        make="Toyota",
        model="Camry",
        vehicle_status="used",
        mileage=48_000,
    )


def _response(value: float = 24_500.0) -> PredictionResponse:
    return PredictionResponse(
        predicted_value=value,
        interval_lower=max(0.0, value - 10_000),
        interval_upper=value + 10_000,
        interval_coverage=0.9,
        interval_width=20_000,
        confidence_label="Moderate confidence",
        calibration_version="retail-rf05-split-conformal-v1",
    )


def test_history_is_lazy_bounded_and_hashes_browser_identity(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "history.sqlite3"
    history = SQLitePredictionHistory(database, maximum_per_client=5)
    assert not database.exists()

    for index in range(7):
        history.save(CLIENT_ID, _request(year=2017 + index), _response(20_000 + index))

    records = history.list_recent(CLIENT_ID, limit=5)
    with sqlite3.connect(database) as connection:
        client_hashes = connection.execute(
            "SELECT DISTINCT client_hash FROM prediction_history"
        ).fetchall()

    assert len(records) == 5
    assert {record.year for record in records} == {2019, 2020, 2021, 2022, 2023}
    assert client_hashes[0][0] != CLIENT_ID
    assert len(client_hashes[0][0]) == 64


def test_history_rejects_invalid_identity_limit_and_incomplete_result(tmp_path: Path) -> None:
    history = SQLitePredictionHistory(tmp_path / "history.sqlite3")

    with pytest.raises(ValueError, match="UUID"):
        history.list_recent("bad-id")
    with pytest.raises(ValueError, match="between 1 and 25"):
        history.list_recent(CLIENT_ID, limit=0)
    with pytest.raises(ValueError, match="complete calibrated"):
        history.save(CLIENT_ID, _request(), PredictionResponse(predicted_value=20_000))


@pytest.mark.parametrize("maximum", [4, 101])
def test_invalid_retention_limit_is_rejected(tmp_path: Path, maximum: int) -> None:
    with pytest.raises(ValueError, match="between 5 and 100"):
        SQLitePredictionHistory(tmp_path / "history.sqlite3", maximum_per_client=maximum)
