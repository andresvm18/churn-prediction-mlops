from fastapi.testclient import TestClient
from api.main import app

# Create test client for FastAPI app
client = TestClient(app)


def test_root_endpoint():
    # Test health check endpoint
    response = client.get("/")

    # Verify request was successful
    assert response.status_code == 200


def test_predict_endpoint():
    # Sample customer data for testing
    payload = {
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
        "total_charges": 1100.0
    }

    # Send POST request to prediction endpoint
    response = client.post("/predict", json=payload)

    # Verify request succeeded
    assert response.status_code == 200

    # Parse response data
    data = response.json()

    # Check all expected fields are present
    assert "prediction" in data
    assert "prediction_label" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "recommendation" in data