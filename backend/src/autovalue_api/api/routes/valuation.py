"""Public RF05 valuation and model-readiness routes."""

import sqlite3
import uuid
from contextlib import suppress
from dataclasses import asdict
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from autovalue_api.schemas import PredictionResponse, VehicleValuationRequest
from autovalue_api.services.history import HistoryRecord, PredictionHistory
from autovalue_api.services.valuation import ValuationEngine, ValuationUnavailableError

router = APIRouter(prefix="/api/v1", tags=["valuation"])


class FinalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    holdout_rows: int = 27_589
    mae_usd: float = 10_575.36
    rmse_usd: float = 34_118.14
    r_squared: float = 0.4176
    median_absolute_error_usd: float = 6_678.93


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready", "artifact_required"]
    can_predict: bool
    model_version: Literal["retail-rf05-v1"] = "retail-rf05-v1"
    candidate_id: Literal["phase4-retail-random_forest-05"] = "phase4-retail-random_forest-05"
    target: Literal["historical U.S. advertised asking price (USD, 2023)"] = (
        "historical U.S. advertised asking price (USD, 2023)"
    )
    default_interval_coverage: float = 0.9
    final_evaluation: Literal["passed with material limitations"] = (
        "passed with material limitations"
    )
    metrics: FinalMetrics = FinalMetrics()
    message: str


class HistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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


class HistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predictions: tuple[HistoryItem, ...]


def _engine(request: Request) -> ValuationEngine:
    return cast(ValuationEngine, request.app.state.valuation_service)


def _history(request: Request) -> PredictionHistory:
    return cast(PredictionHistory, request.app.state.prediction_history)


def _validated_client_id(client_id: str) -> str:
    try:
        _history_record_probe(client_id)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"code": "invalid_client_id", "message": str(error)},
        ) from error
    return client_id


def _history_record_probe(client_id: str) -> None:
    parsed = uuid.UUID(client_id)
    if str(parsed) != client_id.lower():
        raise ValueError("client identifier must use canonical UUID format")


@router.get("/model", response_model=ModelStatusResponse)
def model_status(request: Request) -> ModelStatusResponse:
    engine = _engine(request)
    ready = engine.ready
    return ModelStatusResponse(
        status="ready" if ready else "artifact_required",
        can_predict=ready,
        message=(
            "Frozen RF05 and calibration artifacts are verified and ready."
            if ready
            else "A trusted persisted RF05 bundle is required before real valuations can run."
        ),
    )


@router.post("/valuations", response_model=PredictionResponse)
def create_valuation(
    payload: VehicleValuationRequest,
    request: Request,
    client_id: Annotated[str | None, Header(alias="X-AutoValue-Client")] = None,
) -> PredictionResponse:
    try:
        response = _engine(request).predict(payload)
    except ValuationUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "valuation_service_unavailable",
                "message": "The verified RF05 serving artifact is not available.",
            },
        ) from error
    if client_id is not None:
        with suppress(OSError, sqlite3.Error):
            _history(request).save(_validated_client_id(client_id), payload, response)
    return response


@router.get("/predictions/recent", response_model=HistoryResponse)
def recent_predictions(
    request: Request,
    client_id: Annotated[str, Header(alias="X-AutoValue-Client")],
) -> HistoryResponse:
    try:
        records: tuple[HistoryRecord, ...] = _history(request).list_recent(
            _validated_client_id(client_id), limit=5
        )
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "history_service_unavailable",
                "message": "Recent prediction history is temporarily unavailable.",
            },
        ) from error
    return HistoryResponse(
        predictions=tuple(HistoryItem.model_validate(asdict(record)) for record in records)
    )
