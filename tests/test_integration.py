import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from api.database import init_db

from api.main import app

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DB_PATH"] = TEST_DB

# Initialize test client
client = TestClient(app)

# Check if model artifacts exist
MODEL_AVAILABLE = os.path.exists("src/models/churn_model.pkl") and os.path.exists(
    "src/models/label_encoders.pkl"
)

# Decorator to skip tests when model files are missing
skip_if_no_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Model artifacts not found — run scripts/train_model.py first",
)


# Fixtures
@pytest.fixture(autouse=True, scope="session")
def setup_database():
    from api.database import init_db

    init_db()
    yield
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture
def high_risk_customer():
    return {
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "tenure_months": 1,  # Short tenure - high risk
        "phone_service": "Yes",
        "multiple_lines": "No",
        "internet_service": "Fiber optic",  # Fiber optic has higher churn
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",  # Month-to-month has highest churn
        "paperless_billing": "Yes",
        "payment_method": "Electronic check",  # Electronic check has higher churn
        "monthly_charges": 70.0,
        "total_charges": 70.0,
    }


@pytest.fixture
def low_risk_customer():
    return {
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "tenure_months": 60,  # Long tenure - low risk
        "phone_service": "Yes",
        "multiple_lines": "Yes",
        "internet_service": "DSL",  # DSL has lower churn
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",  # Long-term contract - low risk
        "paperless_billing": "No",
        "payment_method": "Bank transfer (automatic)",  # Automatic payment - low risk
        "monthly_charges": 65.0,
        "total_charges": 3900.0,
    }


# Pipeline tests
class TestFullPredictionPipeline:
    @skip_if_no_model
    def test_high_risk_customer_predicts_churn(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert data["churn_probability"] > 0.5  # Should be > 50% probability
        assert data["risk_level"] in ("High", "Medium")  # Should be high or medium risk

    @skip_if_no_model
    def test_low_risk_customer_does_not_churn(self, low_risk_customer):
        resp = client.post("/predict", json=low_risk_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert data["churn_probability"] < 0.5  # Should be < 50% probability
        assert data["risk_level"] == "Low"  # Should be low risk

    @skip_if_no_model
    def test_response_has_all_required_fields(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        data = resp.json()
        required_fields = [
            "prediction",
            "prediction_label",
            "churn_probability",
            "risk_level",
            "recommendation",
            "top_factors",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    @skip_if_no_model
    def test_churn_probability_is_between_0_and_1(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        prob = resp.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    @skip_if_no_model
    def test_top_factors_are_column_value_format(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        for factor in resp.json()["top_factors"]:
            assert ": " in factor, f"Factor not in 'Column: value' format: {factor}"

    @skip_if_no_model
    def test_top_factors_reference_real_columns(self, high_risk_customer):
        known_columns = {
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Tenure Months",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Contract",
            "Paperless Billing",
            "Payment Method",
            "Monthly Charges",
            "Total Charges",
        }
        resp = client.post("/predict", json=high_risk_customer)
        for factor in resp.json()["top_factors"]:
            col = factor.split(": ")[0]
            assert col in known_columns, f"Unexpected column in top_factors: {col}"

    @skip_if_no_model
    def test_prediction_label_matches_prediction_int(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        data = resp.json()
        if data["prediction"] == 1:
            assert data["prediction_label"] == "Churn"
        else:
            assert data["prediction_label"] == "No Churn"

    @skip_if_no_model
    def test_risk_level_consistent_with_probability(self, high_risk_customer):
        resp = client.post("/predict", json=high_risk_customer)
        data = resp.json()
        prob = data["churn_probability"]
        risk = data["risk_level"]

        if prob >= 0.7:
            assert risk == "High"
        elif prob >= 0.4:
            assert risk == "Medium"
        else:
            assert risk == "Low"

    @skip_if_no_model
    def test_consecutive_predictions_are_deterministic(self, high_risk_customer):
        resp1 = client.post("/predict", json=high_risk_customer)
        resp2 = client.post("/predict", json=high_risk_customer)
        # Compare probabilities and predictions
        assert resp1.json()["churn_probability"] == resp2.json()["churn_probability"]
        assert resp1.json()["prediction"] == resp2.json()["prediction"]


# History integration
class TestHistoryIntegration:

    @skip_if_no_model
    def test_prediction_is_saved_to_history(self, high_risk_customer):
        client.post("/predict", json=high_risk_customer)
        resp = client.get("/history?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert "rows" in data
        assert "total" in data
        assert len(data["rows"]) >= 1

    @skip_if_no_model
    def test_history_stats_returns_correct_structure(self, high_risk_customer):
        client.post("/predict", json=high_risk_customer)
        resp = client.get("/history/stats")
        assert resp.status_code == 200
        data = resp.json()
        for key in [
            "total",
            "churn_count",
            "churn_rate",
            "avg_probability",
            "high_risk",
            "medium_risk",
            "low_risk",
        ]:
            assert key in data, f"Missing stats key: {key}"

    @skip_if_no_model
    def test_history_risk_filter_works(self, high_risk_customer):
        client.post("/predict", json=high_risk_customer)
        resp = client.get("/history?risk_level=High&limit=50")
        assert resp.status_code == 200
        for row in resp.json()["rows"]:
            assert row["risk_level"] == "High"

    @skip_if_no_model
    def test_history_prediction_filter_works(self, low_risk_customer):
        client.post("/predict", json=low_risk_customer)
        resp = client.get("/history?prediction=0&limit=50")
        assert resp.status_code == 200
        for row in resp.json()["rows"]:
            assert row["prediction"] == 0

    @skip_if_no_model
    def test_history_pagination_offset_works(self, high_risk_customer):
        for _ in range(3):
            client.post("/predict", json=high_risk_customer)
        resp_page1 = client.get("/history?limit=2&offset=0")
        resp_page2 = client.get("/history?limit=2&offset=2")
        assert resp_page1.status_code == 200
        assert resp_page2.status_code == 200
        data1 = resp_page1.json()
        data2 = resp_page2.json()
        assert "total" in data1
        assert data1["total"] >= 3
        if data2["rows"]:
            assert data1["rows"][0]["id"] != data2["rows"][0]["id"]


# Batch integration
class TestBatchIntegration:
    @skip_if_no_model
    def test_batch_endpoint_processes_valid_csv(self):

        csv_content = (
            "gender,senior_citizen,partner,dependents,tenure_months,"
            "phone_service,multiple_lines,internet_service,online_security,"
            "online_backup,device_protection,tech_support,streaming_tv,"
            "streaming_movies,contract,paperless_billing,payment_method,"
            "monthly_charges,total_charges\n"
            "Female,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,"
            "Month-to-month,Yes,Electronic check,95.5,1146.0\n"
            "Male,0,No,No,48,Yes,Yes,DSL,Yes,Yes,No,Yes,No,No,"
            "One year,No,Bank transfer (automatic),65.0,3120.0\n"
        )
        resp = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verify both rows were processed successfully
        assert data["total_rows"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0

    @skip_if_no_model
    def test_batch_download_returns_csv(self):
        csv_content = (
            "gender,senior_citizen,partner,dependents,tenure_months,"
            "phone_service,multiple_lines,internet_service,online_security,"
            "online_backup,device_protection,tech_support,streaming_tv,"
            "streaming_movies,contract,paperless_billing,payment_method,"
            "monthly_charges,total_charges\n"
            "Female,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,"
            "Month-to-month,Yes,Electronic check,95.5,1146.0\n"
        )
        resp = client.post(
            "/predict/batch/download",
            files={"file": ("test.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]  # Should be CSV format

        # Parse CSV and verify columns
        import io
        import pandas as pd

        result_df = pd.read_csv(io.StringIO(resp.text))
        # Check prediction columns exist
        assert "prediction_label" in result_df.columns
        assert "churn_probability" in result_df.columns
        assert "risk_level" in result_df.columns


# Database coverage


class TestDatabaseCoverage:

    @skip_if_no_model
    def test_clear_history_deletes_records(self, high_risk_customer):
        # Ensure at least one record exists in the database
        client.post("/predict", json=high_risk_customer)

        # Delete all history records
        resp = client.delete("/history")
        assert resp.status_code == 200
        data = resp.json()

        # Verify response contains deletion count
        assert "deleted" in data
        assert isinstance(data["deleted"], int)

    @skip_if_no_model
    def test_history_empty_after_clear(self, high_risk_customer):
        # Create a prediction record
        client.post("/predict", json=high_risk_customer)

        # Clear all history
        client.delete("/history")

        # Verify history is now empty
        resp = client.get("/history?limit=100")
        assert resp.status_code == 200
        assert resp.json() == []  # Should return empty list

    @skip_if_no_model
    def test_save_prediction_failure_is_silent(self, high_risk_customer):
        from unittest.mock import patch

        # Mock database connection to raise an error
        with patch("api.database.get_connection", side_effect=Exception("DB error")):
            # Prediction should still succeed despite DB error
            resp = client.post("/predict", json=high_risk_customer)

        # Verify prediction still returns successfully
        assert resp.status_code == 200
        assert "churn_probability" in resp.json()

    @skip_if_no_model
    def test_prediction_is_saved_to_history(self, high_risk_customer):
        client.post("/predict", json=high_risk_customer)
        resp = client.get("/history?limit=1")
        assert resp.status_code == 200
        assert len(resp.json()["rows"]) >= 1

    @skip_if_no_model
    def test_history_empty_after_clear(self, high_risk_customer):
        client.post("/predict", json=high_risk_customer)
        client.delete("/history")
        resp = client.get("/history?limit=100")
        assert resp.status_code == 200
        assert resp.json()["rows"] == []


# Helper function to create a CSV with the specified number of rows.
def _create_large_csv(num_rows: int = 1001) -> str:
    header = (
        "gender,senior_citizen,partner,dependents,tenure_months,"
        "phone_service,multiple_lines,internet_service,online_security,"
        "online_backup,device_protection,tech_support,streaming_tv,"
        "streaming_movies,contract,paperless_billing,payment_method,"
        "monthly_charges,total_charges\n"
    )
    row = (
        "Female,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,"
        "Month-to-month,Yes,Electronic check,95.5,1146.0\n"
    )
    return header + row * num_rows


# Batch validation coverage
class TestBatchValidationCoverage:

    def test_batch_empty_csv_returns_400(self):
        # An empty CSV file must return 400.
        csv_content = (
            "gender,senior_citizen,partner,dependents,tenure_months,"
            "phone_service,multiple_lines,internet_service,online_security,"
            "online_backup,device_protection,tech_support,streaming_tv,"
            "streaming_movies,contract,paperless_billing,payment_method,"
            "monthly_charges,total_charges\n"  # Headers only, no data rows
        )
        resp = client.post(
            "/predict/batch",
            files={"file": ("empty.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_batch_missing_columns_returns_422(self):
        # A CSV missing required columns must return 422.
        csv_content = "name,age\nJohn,30\n"  # Missing all required columns
        resp = client.post(
            "/predict/batch",
            files={"file": ("bad.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 422
        assert "missing" in resp.json()["detail"].lower()

    def test_batch_non_csv_returns_400(self):
        # A non-CSV file must return 400.
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.txt", b"some text", "text/plain")},
        )
        assert resp.status_code == 400

    def test_batch_download_empty_csv_returns_400(self):
        # Empty CSV to download endpoint must return 400.
        csv_content = (
            "gender,senior_citizen,partner,dependents,tenure_months,"
            "phone_service,multiple_lines,internet_service,online_security,"
            "online_backup,device_protection,tech_support,streaming_tv,"
            "streaming_movies,contract,paperless_billing,payment_method,"
            "monthly_charges,total_charges\n"  # Headers only
        )
        resp = client.post(
            "/predict/batch/download",
            files={"file": ("empty.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 400

    def test_batch_download_missing_columns_returns_422(self):
        # CSV missing columns to download endpoint must return 422.
        csv_content = "name,age\nJohn,30\n"  # Missing all required columns
        resp = client.post(
            "/predict/batch/download",
            files={"file": ("bad.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 422

    def test_batch_download_non_csv_returns_400(self):
        # Non-CSV file to download endpoint must return 400.
        resp = client.post(
            "/predict/batch/download",
            files={"file": ("data.txt", b"some text", "text/plain")},
        )
        assert resp.status_code == 400

    def test_batch_over_limit_returns_400(self):
        # A CSV with more than 1000 rows must return 400.
        csv_content = _create_large_csv(1001)  # 1 row over limit

        resp = client.post(
            "/predict/batch",
            files={"file": ("big.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "1000" in resp.json()["detail"]

    def test_batch_download_over_limit_returns_400(self):
        # A CSV with more than 1000 rows to download endpoint must return 400."""
        csv_content = _create_large_csv(1001)  # 1 row over limit

        resp = client.post(
            "/predict/batch/download",
            files={"file": ("big.csv", csv_content.encode(), "text/csv")},
        )
        assert resp.status_code == 400
        assert "1000" in resp.json()["detail"]
