import pytest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock
from sklearn.preprocessing import LabelEncoder

from api.model_service import (
    build_input_dataframe,
    encode_input_data,
    get_risk_level,
    get_recommendation,
    get_top_factors,
)
from api.config import CATEGORICAL_COLUMNS


# Create a valid customer object for testing
@pytest.fixture
def sample_customer():
    return SimpleNamespace(
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
        total_charges=1146.0,
    )


# Create label encoders for testing
@pytest.fixture
def sample_encoders():
    values_per_col = {
        "Gender":            ["Female", "Male"],
        "Partner":           ["No", "Yes"],
        "Dependents":        ["No", "Yes"],
        "Phone Service":     ["No", "Yes"],
        "Multiple Lines":    ["No", "No phone service", "Yes"],
        "Internet Service":  ["DSL", "Fiber optic", "No"],
        "Online Security":   ["No", "No internet service", "Yes"],
        "Online Backup":     ["No", "No internet service", "Yes"],
        "Device Protection": ["No", "No internet service", "Yes"],
        "Tech Support":      ["No", "No internet service", "Yes"],
        "Streaming TV":      ["No", "No internet service", "Yes"],
        "Streaming Movies":  ["No", "No internet service", "Yes"],
        "Contract":          ["Month-to-month", "One year", "Two year"],
        "Paperless Billing": ["No", "Yes"],
        "Payment Method":    [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    }
    encoders = {}
    for col, values in values_per_col.items():
        enc = LabelEncoder()
        enc.fit(values)
        encoders[col] = enc
    return encoders


# Tests for build_input_dataframe function
class TestBuildInputDataframe:

    def test_returns_single_row_dataframe(self, sample_customer):
        df = build_input_dataframe(sample_customer)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 1

    def test_has_19_columns(self, sample_customer):
        df = build_input_dataframe(sample_customer)
        assert df.shape[1] == 19

    def test_column_values_match_input(self, sample_customer):
        df = build_input_dataframe(sample_customer)
        assert df["Gender"].iloc[0] == "Female"
        assert df["Contract"].iloc[0] == "Month-to-month"
        assert df["Monthly Charges"].iloc[0] == 95.5
        assert df["Senior Citizen"].iloc[0] == 0

    def test_all_categorical_columns_present(self, sample_customer):
        df = build_input_dataframe(sample_customer)
        for col in CATEGORICAL_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"


# Tests for encode_input_data function
class TestEncodeInputData:

    def test_output_is_numeric(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)
        for col in CATEGORICAL_COLUMNS:
            assert pd.api.types.is_numeric_dtype(encoded[col]), (
                f"Column '{col}' is not numeric after encoding"
            )

    def test_shape_preserved(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)
        assert encoded.shape == raw.shape

    def test_no_nulls_after_encoding(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)
        assert not encoded.isnull().any().any()

    def test_unseen_category_raises_value_error(self, sample_customer, sample_encoders):
        sample_customer.gender = "Unknown_gender"
        raw = build_input_dataframe(sample_customer)
        with pytest.raises(ValueError, match="not seen during training"):
            encode_input_data(raw, sample_encoders)


# Tests for get_risk_level function
class TestGetRiskLevel:

    def test_high_at_exact_threshold(self):
        assert get_risk_level(0.7) == "High"

    def test_high_above_threshold(self):
        assert get_risk_level(0.95) == "High"

    def test_medium_at_exact_threshold(self):
        assert get_risk_level(0.4) == "Medium"

    def test_medium_just_below_high(self):
        assert get_risk_level(0.699) == "Medium"

    def test_low_just_below_medium(self):
        assert get_risk_level(0.399) == "Low"

    def test_low_at_zero(self):
        assert get_risk_level(0.0) == "Low"

    def test_low_at_one(self):
        assert get_risk_level(1.0) == "High"


# Tests for get_recommendation function
class TestGetRecommendation:

    def test_high_risk_mentions_retention(self):
        rec = get_recommendation("High")
        assert "high risk" in rec.lower()
        assert "retention" in rec.lower()

    def test_medium_risk_mentions_moderate(self):
        rec = get_recommendation("Medium")
        assert "moderate" in rec.lower()

    def test_low_risk_mentions_low(self):
        rec = get_recommendation("Low")
        assert "low" in rec.lower()

    def test_unknown_risk_returns_fallback(self):
        rec = get_recommendation("Unknown")
        assert rec  # non-empty fallback


# Tests for get_top_factors function
class TestGetTopFactors:

    def test_returns_empty_list_when_no_explainer(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)
        result = get_top_factors(encoded, raw, explainer=None)
        assert result == []  # Empty list expected

    def test_returns_n_factors(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)

        # Create mock explainer that returns uniform SHAP values
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.array([0.1] * encoded.shape[1])  # numpy array, not list
        ]

        result = get_top_factors(encoded, raw, mock_explainer, n=3)
        assert len(result) == 3  # Verify exactly 3 factors returned

    def test_factors_are_column_value_format(self, sample_customer, sample_encoders):
        raw = build_input_dataframe(sample_customer)
        encoded = encode_input_data(raw, sample_encoders)

        # Create mock explainer with uniform values
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = [
            np.array([0.1] * encoded.shape[1])  # numpy array, not list
        ]

        result = get_top_factors(encoded, raw, mock_explainer, n=3)
        # Verify each factor contains the colon separator
        for factor in result:
            assert ": " in factor, f"Factor '{factor}' does not have 'Column: value' format"