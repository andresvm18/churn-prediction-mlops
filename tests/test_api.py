import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app

# Create test client for FastAPI app
client = TestClient(app)


# Create valid test payload for prediction endpoint
@pytest.fixture
def valid_payload():
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure_months": 12,
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",
        "monthly_charges": 95.5,
        "total_charges": 1146.0,
    }


# Create mock prediction response
@pytest.fixture
def mock_prediction():
    return {
        "prediction": 1,
        "prediction_label": "Churn",
        "churn_probability": 0.9137,
        "risk_level": "High",
        "recommendation": (
            "Customer is at high risk of churn. "
            "Consider retention actions."
        ),
        "top_factors": [
            "Contract: Month-to-month",
            "Monthly Charges: 95.5",
            "Internet Service: Fiber optic",
        ],
    }


# Tests for health check endpoints
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# Tests for successful prediction (happy path)
@patch("api.main.predict_customer_churn")
def test_predict_returns_200(mock_predict, valid_payload, mock_prediction):
    mock_predict.return_value = mock_prediction
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


@patch("api.main.predict_customer_churn")
def test_predict_response_has_all_fields(mock_predict, valid_payload, mock_prediction):
    mock_predict.return_value = mock_prediction
    data = client.post("/predict", json=valid_payload).json()
    expected_fields = [
        "prediction",
        "prediction_label",
        "churn_probability",
        "risk_level",
        "recommendation",
        "top_factors",
    ]
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"


@patch("api.main.predict_customer_churn")
def test_predict_top_factors_format(mock_predict, valid_payload, mock_prediction):
    mock_predict.return_value = mock_prediction
    data = client.post("/predict", json=valid_payload).json()
    for factor in data["top_factors"]:
        assert ": " in factor, f"Factor not in expected format: {factor}"


# Tests for validation errors (HTTP 422)
def test_invalid_gender_returns_422(valid_payload):
    valid_payload["gender"] = "Robot"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_invalid_contract_returns_422(valid_payload):
    valid_payload["contract"] = "Weekly"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_negative_monthly_charges_returns_422(valid_payload):
    valid_payload["monthly_charges"] = -10.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_negative_tenure_returns_422(valid_payload):
    valid_payload["tenure_months"] = -1
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_missing_required_field_returns_422(valid_payload):
    del valid_payload["contract"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_invalid_senior_citizen_returns_422(valid_payload):
    valid_payload["senior_citizen"] = 5
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_invalid_payment_method_returns_422(valid_payload):
    valid_payload["payment_method"] = "Bitcoin"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


# Tests for service layer errors
@patch("api.main.predict_customer_churn")
def test_model_not_loaded_returns_503(mock_predict, valid_payload):
    mock_predict.side_effect = RuntimeError("Model files not found.")
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503


@patch("api.main.predict_customer_churn")
def test_unseen_category_returns_422(mock_predict, valid_payload):
    mock_predict.side_effect = ValueError("not seen during training")
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


@patch("api.main.predict_customer_churn")
def test_unexpected_error_returns_500(mock_predict, valid_payload):
    mock_predict.side_effect = Exception("Something went wrong")
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 500
