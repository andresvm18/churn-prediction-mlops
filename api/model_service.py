import os
import logging

import joblib
import pandas as pd
import shap

# Import configuration constants
from api.config import (
    MODEL_PATH,
    ENCODER_PATH,
    CATEGORICAL_COLUMNS,
    CLASSIFICATION_THRESHOLD,
    RISK_THRESHOLD_HIGH,
    RISK_THRESHOLD_MEDIUM,
)

logger = logging.getLogger(__name__)

# Lazy-loaded singletons (loaded only once)
_churn_model = None
_categorical_encoders = None
_model_explainer = None
_artifacts_loaded = False


def _load_artifacts() -> None:
    global _churn_model, _categorical_encoders, _model_explainer, _artifacts_loaded

    # Skip if already loaded
    if _artifacts_loaded:
        return

    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at '{MODEL_PATH}'.")
    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder file not found at '{ENCODER_PATH}'.")

    # Load the model
    logger.info("Loading model from %s", MODEL_PATH)
    _churn_model = joblib.load(MODEL_PATH)

    # Load the encoders
    logger.info("Loading encoders from %s", ENCODER_PATH)
    _categorical_encoders = joblib.load(ENCODER_PATH)

    # Create SHAP explainer for the model
    logger.info("Initializing SHAP TreeExplainer")
    _model_explainer = shap.TreeExplainer(_churn_model)

    _artifacts_loaded = True
    logger.info("All artifacts loaded successfully.")


# Convert customer data to DataFrame
def build_input_dataframe(customer_data) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Gender": customer_data.gender,
                "Senior Citizen": customer_data.senior_citizen,
                "Partner": customer_data.partner,
                "Dependents": customer_data.dependents,
                "Tenure Months": customer_data.tenure_months,
                "Phone Service": customer_data.phone_service,
                "Multiple Lines": customer_data.multiple_lines,
                "Internet Service": customer_data.internet_service,
                "Online Security": customer_data.online_security,
                "Online Backup": customer_data.online_backup,
                "Device Protection": customer_data.device_protection,
                "Tech Support": customer_data.tech_support,
                "Streaming TV": customer_data.streaming_tv,
                "Streaming Movies": customer_data.streaming_movies,
                "Contract": customer_data.contract,
                "Paperless Billing": customer_data.paperless_billing,
                "Payment Method": customer_data.payment_method,
                "Monthly Charges": customer_data.monthly_charges,
                "Total Charges": customer_data.total_charges,
            }
        ]
    )


# Encode categorical columns using saved label encoders
def encode_input_data(
    input_data: pd.DataFrame,
    encoders: dict,
) -> pd.DataFrame:
    encoded = input_data.copy()

    # Encode each categorical column
    for col in CATEGORICAL_COLUMNS:
        encoder = encoders[col]
        value = encoded[col].astype(str)

        # Check for unseen values before transforming
        unseen = set(value) - set(encoder.classes_)
        if unseen:
            raise ValueError(
                f"Column '{col}' contains value(s) not seen during training: {unseen}. "
                f"Valid options are: {list(encoder.classes_)}"
            )

        encoded[col] = encoder.transform(value)

    # Convert all to numeric, fill NaN with 0
    encoded = encoded.apply(pd.to_numeric, errors="coerce").fillna(0)
    return encoded


# Determine risk level from probability
def get_risk_level(churn_probability: float) -> str:
    if churn_probability >= RISK_THRESHOLD_HIGH:
        return "High"
    if churn_probability >= RISK_THRESHOLD_MEDIUM:
        return "Medium"
    return "Low"


# Get recommendation based on risk level
def get_recommendation(risk_level: str) -> str:
    recommendations = {
        "High": "Customer is at high risk of churn. Consider retention actions.",
        "Medium": "Customer has moderate churn risk. Monitor behavior and engagement.",
        "Low": "Customer has low churn risk.",
    }
    return recommendations.get(risk_level, "Risk level unknown.")


# Get top influential features from SHAP
def get_top_factors(
    encoded_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    explainer,
    n: int = 3,
) -> list[str]:
    # Return empty list if no SHAP explainer is available
    if explainer is None:
        return []

    # Calculate SHAP values for the encoded input
    shap_values = explainer.shap_values(encoded_data)

    import numpy as np  # Ensures compatibility with both lists and arrays

    # Create DataFrame with feature names and their absolute SHAP impacts
    impacts = (
        pd.DataFrame(
            {
                "feature": encoded_data.columns,
                "impact": np.abs(shap_values[0]),  # np.abs works with lists AND arrays
            }
        )
        .sort_values("impact", ascending=False)
        .head(n)
    )  # Get top n features

    # Build human-readable strings with actual values
    result = []
    for feature in impacts["feature"]:
        # Get the original value from raw_data if it exists
        raw_value = raw_data[feature].iloc[0] if feature in raw_data.columns else ""
        result.append(f"{feature}: {raw_value}")

    return result


# Main prediction pipeline
def predict_customer_churn(customer_data) -> dict:
    # Ensure model and encoders are loaded
    _load_artifacts()

    # Build raw DataFrame
    raw_data = build_input_dataframe(customer_data)

    # Encode categorical columns
    encoded_data = encode_input_data(raw_data, _categorical_encoders)

    # Make prediction
    churn_probability = float(_churn_model.predict_proba(encoded_data)[0][1])
    prediction = int(churn_probability >= CLASSIFICATION_THRESHOLD)

    # Calculate risk and recommendation
    risk_level = get_risk_level(churn_probability)
    prediction_label = "Churn" if prediction == 1 else "No Churn"
    recommendation = get_recommendation(risk_level)

    # Get top factors influencing the prediction
    top_factors = get_top_factors(encoded_data, raw_data, _model_explainer)

    # Return formatted results
    return {
        "prediction": prediction,
        "prediction_label": prediction_label,
        "churn_probability": round(churn_probability, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "top_factors": top_factors,
    }
