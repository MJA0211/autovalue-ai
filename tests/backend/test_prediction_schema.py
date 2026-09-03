import pytest
from autovalue_api.schemas import ModelInformation, PredictionResponse
from pydantic import ValidationError


def test_point_only_response_remains_valid() -> None:
    response = PredictionResponse(predicted_value=24_500.0)

    assert response.model_dump() == {
        "predicted_value": 24_500.0,
        "currency": "USD",
        "interval_lower": None,
        "interval_upper": None,
        "interval_coverage": None,
        "interval_width": None,
        "confidence_label": None,
        "calibration_version": None,
        "warnings": [],
        "model_information": None,
    }


def test_complete_calibrated_response_is_valid() -> None:
    response = PredictionResponse(
        predicted_value=24_500.0,
        interval_lower=10_593.85,
        interval_upper=38_406.15,
        interval_coverage=0.9,
        interval_width=27_812.30,
        confidence_label="Moderate confidence",
        calibration_version="retail-rf05-split-conformal-v1",
        warnings=["missing_mileage"],
        model_information=ModelInformation(
            candidate_id="phase4-retail-random_forest-05",
            feature_contract_version="retail-historical-asking-price-v2",
        ),
    )

    assert response.currency == "USD"
    assert response.interval_coverage == 0.9
    assert response.model_information is not None
    assert response.model_information.model_family == "random_forest"


@pytest.mark.parametrize(
    "updates",
    [
        {"interval_lower": 10_000.0},
        {
            "interval_lower": 10_000.0,
            "interval_upper": 30_000.0,
            "interval_coverage": 0.9,
            "interval_width": 19_000.0,
            "confidence_label": "High confidence",
            "calibration_version": "v1",
        },
        {
            "interval_lower": 25_000.0,
            "interval_upper": 30_000.0,
            "interval_coverage": 0.9,
            "interval_width": 5_000.0,
            "confidence_label": "High confidence",
            "calibration_version": "v1",
        },
    ],
)
def test_incomplete_or_inconsistent_intervals_fail_closed(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PredictionResponse.model_validate({"predicted_value": 20_000.0, **updates})


def test_duplicate_warnings_fail_closed() -> None:
    with pytest.raises(ValidationError, match="warnings must be unique"):
        PredictionResponse(
            predicted_value=20_000.0,
            interval_lower=10_000.0,
            interval_upper=30_000.0,
            interval_coverage=0.9,
            interval_width=20_000.0,
            confidence_label="Low confidence",
            calibration_version="v1",
            warnings=["missing_mileage", "missing_mileage"],
        )
