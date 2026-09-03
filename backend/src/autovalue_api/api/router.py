"""Top-level API router."""

from fastapi import APIRouter

from autovalue_api.api.routes.health import router as health_router
from autovalue_api.api.routes.valuation import router as valuation_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(valuation_router)
