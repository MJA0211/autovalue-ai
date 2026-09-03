"""The backend facade stays experimental and outside public routing."""

from autovalue_api.main import create_app
from autovalue_api.services.shadow_learning import ExperimentalShadowLearningInterface
from autovalue_ml.online.service import ShadowLearningService
from fastapi.testclient import TestClient


def test_shadow_interface_is_explicitly_non_user_facing() -> None:
    interface = ExperimentalShadowLearningInterface(ShadowLearningService())

    assert interface.mode == "shadow"
    assert interface.status == "experimental"
    assert interface.user_facing is False
    assert interface.get_model_state()["user_facing"] is False


def test_shadow_interface_is_not_registered_in_public_openapi() -> None:
    document = TestClient(create_app()).get("/openapi.json").json()

    assert set(document["paths"]) == {
        "/health/live",
        "/api/v1/model",
        "/api/v1/valuations",
        "/api/v1/predictions/recent",
    }
    assert not any("shadow" in path or "outcome" in path for path in document["paths"])
