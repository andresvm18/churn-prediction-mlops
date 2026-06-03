from types import SimpleNamespace
from unittest.mock import patch

from api import model_service
from api.model_service import (
    build_input_dataframe,
    get_recommendation,
    get_risk_level,
)


def test_high_risk_level():
    # Probability >= 0.7 should return "High"
    assert get_risk_level(0.8) == "High"


def test_medium_risk_level():
    # Probability between 0.4 and 0.7 should return "Medium"
    assert get_risk_level(0.5) == "Medium"


def test_low_risk_level():
    # Probability < 0.4 should return "Low"
    assert get_risk_level(0.2) == "Low"


def test_high_risk_recommendation():
    # High risk should mention "high risk" in recommendation
    recommendation = get_recommendation("High")
    assert "high risk" in recommendation.lower()


def test_medium_risk_recommendation():
    # Medium risk should mention "moderate" in recommendation
    recommendation = get_recommendation("Medium")
    assert "moderate" in recommendation.lower()


def test_low_risk_recommendation():
    # Low risk should mention "low churn risk" in recommendation
    recommendation = get_recommendation("Low")
    assert "low churn risk" in recommendation.lower()


def test_build_input_dataframe():
    # Create mock customer data
    customer = SimpleNamespace(
        gender="Female",
        senior_citizen=0,
        partner="Yes",
        dependents="No",
        tenure_months=12,
        phone_service="Yes",
        multiple_lines="No",
        internet_service="Fiber optic",
        online_security="No",
        online_backup="No",
        device_protection="No",
        tech_support="No",
        streaming_tv="Yes",
        streaming_movies="Yes",
        contract="Month-to-month",
        paperless_billing="Yes",
        payment_method="Electronic check",
        monthly_charges=95.5,
        total_charges=1100.0,
    )

    # Build dataframe from mock data
    df = build_input_dataframe(customer)

    # Verify dataframe shape and values
    assert df.shape == (1, 19)
    assert df["Gender"][0] == "Female"
    assert df["Contract"][0] == "Month-to-month"
    assert df["Monthly Charges"][0] == 95.5


def test_predict_customer_churn_without_model():
    # Simulate missing model files
    with patch.object(model_service, "churn_model", None):
        # Should raise an error when trying to predict
        try:
            model_service.predict_customer_churn(None)
        except RuntimeError as exc:
            assert "Model files not found" in str(exc)