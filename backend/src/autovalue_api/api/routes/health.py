"""Process liveness endpoint."""

from typing import Literal, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel

from autovalue_api import __version__
from autovalue_api.core.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Public liveness response."""

    status: Literal["ok"]
    service: str
    version: str
    environment: str


@router.get("/health/live", response_model=HealthResponse)
def liveness(request: Request) -> HealthResponse:
    """Confirm that the API process can serve requests."""
    settings = cast(Settings, request.app.state.settings)
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.environment,
    )
