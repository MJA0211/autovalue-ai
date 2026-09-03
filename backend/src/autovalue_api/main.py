"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autovalue_api import __version__
from autovalue_api.api.router import api_router
from autovalue_api.core.config import Settings, get_settings
from autovalue_api.services.history import PredictionHistory, SQLitePredictionHistory
from autovalue_api.services.valuation import FrozenRF05Service, ValuationEngine


def create_app(
    settings: Settings | None = None,
    *,
    valuation_service: ValuationEngine | None = None,
    prediction_history: PredictionHistory | None = None,
) -> FastAPI:
    """Create an application without model or database side effects."""
    active_settings = settings or get_settings()
    application = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        description="U.S. used-vehicle valuation API. All valuation targets are in USD.",
    )
    application.state.settings = active_settings
    application.state.valuation_service = valuation_service or FrozenRF05Service(
        bundle_dir=active_settings.model_bundle_dir,
        calibration_path=active_settings.calibration_artifact_path,
        trusted_models_root=active_settings.trusted_models_root,
        trusted_calibration_root=active_settings.trusted_calibration_root,
    )
    application.state.prediction_history = prediction_history or SQLitePredictionHistory(
        active_settings.prediction_history_path
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type", "X-AutoValue-Client"],
    )
    application.include_router(api_router)
    return application


app = create_app()
