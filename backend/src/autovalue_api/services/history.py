"""Anonymous SQLite prediction history with browser-level isolation."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from autovalue_api.schemas import PredictionResponse, VehicleValuationRequest


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    id: str
    created_at: str
    year: int
    make: str
    model: str
    vehicle_status: str
    mileage: float | None
    predicted_value: float
    interval_lower: float
    interval_upper: float
    interval_coverage: float


class PredictionHistory(Protocol):
    def save(
        self,
        client_id: str,
        request: VehicleValuationRequest,
        response: PredictionResponse,
    ) -> HistoryRecord: ...

    def list_recent(self, client_id: str, *, limit: int = 5) -> tuple[HistoryRecord, ...]: ...


class SQLitePredictionHistory:
    """Persist bounded non-sensitive estimates without storing browser IDs in cleartext."""

    def __init__(self, database_path: Path, *, maximum_per_client: int = 25) -> None:
        if maximum_per_client < 5 or maximum_per_client > 100:
            raise ValueError("maximum_per_client must be between 5 and 100")
        self._database_path = database_path
        self._maximum_per_client = maximum_per_client

    def save(
        self,
        client_id: str,
        request: VehicleValuationRequest,
        response: PredictionResponse,
    ) -> HistoryRecord:
        _validate_client_id(client_id)
        if (
            response.interval_lower is None
            or response.interval_upper is None
            or response.interval_coverage is None
        ):
            raise ValueError("only complete calibrated valuations may be saved")
        record = HistoryRecord(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
            year=request.year,
            make=request.make,
            model=request.model,
            vehicle_status=request.vehicle_status,
            mileage=request.mileage,
            predicted_value=response.predicted_value,
            interval_lower=response.interval_lower,
            interval_upper=response.interval_upper,
            interval_coverage=response.interval_coverage,
        )
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO prediction_history (
                    id, client_hash, created_at, year, make, model, vehicle_status,
                    mileage, predicted_value, interval_lower, interval_upper, interval_coverage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    _client_hash(client_id),
                    record.created_at,
                    record.year,
                    record.make,
                    record.model,
                    record.vehicle_status,
                    record.mileage,
                    record.predicted_value,
                    record.interval_lower,
                    record.interval_upper,
                    record.interval_coverage,
                ),
            )
            connection.execute(
                """DELETE FROM prediction_history
                   WHERE client_hash = ? AND id NOT IN (
                     SELECT id FROM prediction_history WHERE client_hash = ?
                     ORDER BY created_at DESC, id DESC LIMIT ?
                   )""",
                (_client_hash(client_id), _client_hash(client_id), self._maximum_per_client),
            )
        return record

    def list_recent(self, client_id: str, *, limit: int = 5) -> tuple[HistoryRecord, ...]:
        _validate_client_id(client_id)
        if limit < 1 or limit > 25:
            raise ValueError("history limit must be between 1 and 25")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, created_at, year, make, model, vehicle_status, mileage,
                          predicted_value, interval_lower, interval_upper, interval_coverage
                   FROM prediction_history WHERE client_hash = ?
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (_client_hash(client_id), limit),
            ).fetchall()
        return tuple(HistoryRecord(**dict(row)) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS prediction_history (
                id TEXT PRIMARY KEY,
                client_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                year INTEGER NOT NULL,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                vehicle_status TEXT NOT NULL,
                mileage REAL,
                predicted_value REAL NOT NULL,
                interval_lower REAL NOT NULL,
                interval_upper REAL NOT NULL,
                interval_coverage REAL NOT NULL
            )"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS prediction_history_client_time
               ON prediction_history (client_hash, created_at DESC)"""
        )
        return connection


def _validate_client_id(client_id: str) -> None:
    try:
        parsed = uuid.UUID(client_id)
    except (AttributeError, ValueError) as error:
        raise ValueError("client identifier must be a UUID") from error
    if str(parsed) != client_id.lower():
        raise ValueError("client identifier must use canonical UUID format")


def _client_hash(client_id: str) -> str:
    return hashlib.sha256(client_id.lower().encode("ascii")).hexdigest()


__all__ = ["HistoryRecord", "PredictionHistory", "SQLitePredictionHistory"]
