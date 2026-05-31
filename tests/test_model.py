from api.model_service import (
    get_risk_level,
    get_recommendation,
)


def test_high_risk_level():
    # Check that 0.8 probability returns "High"
    assert get_risk_level(0.8) == "High"


def test_medium_risk_level():
    # Check that 0.5 probability returns "Medium"
    assert get_risk_level(0.5) == "Medium"


def test_low_risk_level():
    # Check that 0.2 probability returns "Low"
    assert get_risk_level(0.2) == "Low"


def test_high_risk_recommendation():
    # Check recommendation contains "high risk" for High level
    recommendation = get_recommendation("High")
    assert "high risk" in recommendation.lower()


def test_medium_risk_recommendation():
    # Check recommendation contains "moderate" for Medium level
    recommendation = get_recommendation("Medium")
    assert "moderate" in recommendation.lower()


def test_low_risk_recommendation():
    # Check recommendation contains "low churn risk" for Low level
    recommendation = get_recommendation("Low")
    assert "low churn risk" in recommendation.lower()
