import os
import joblib
import pandas as pd
import shap

MODEL_PATH = "src/models/churn_model.pkl"
ENCODER_PATH = "src/models/label_encoders.pkl"

# Load model, encoders, and SHAP explainer if files exist
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    churn_model = joblib.load(MODEL_PATH)
    categorical_encoders = joblib.load(ENCODER_PATH)
    model_explainer = shap.TreeExplainer(churn_model)
else:
    churn_model = None
    categorical_encoders = None
    model_explainer = None

# List of columns that need encoding (must match training)
CATEGORICAL_COLUMNS = [
    "Gender",
    "Partner",
    "Dependents",
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
]


def build_input_dataframe(customer_data):
    # Convert customer object to DataFrame with correct column names
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


def encode_input_data(input_data):
    # Apply saved encoders to transform text to numbers
    if categorical_encoders is None:
        raise RuntimeError("Label encoders not found. Prediction service unavailable.")

    encoded_data = input_data.copy()

    for column_name in CATEGORICAL_COLUMNS:
        encoder = categorical_encoders[column_name]
        encoded_data[column_name] = encoder.transform(
            encoded_data[column_name].astype(str)
        )

    # Ensure all data is numeric
    encoded_data = encoded_data.apply(pd.to_numeric, errors="coerce")

    encoded_data = encoded_data.fillna(0)  # Replace NaN with 0

    return encoded_data


def get_risk_level(churn_probability):
    # Determine risk level based on probability thresholds
    if churn_probability >= 0.7:
        return "High"
    if churn_probability >= 0.4:
        return "Medium"
    return "Low"


def get_recommendation(risk_level):
    # Generate business recommendation based on risk level
    if risk_level == "High":
        return "Customer is at high risk of churn. " "Consider retention actions."
    if risk_level == "Medium":
        return "Customer has moderate churn risk. " "Monitor behavior and engagement."
    return "Customer has low churn risk."


def get_top_factors(encoded_data):
    # Get top 3 most influential features for this prediction
    if model_explainer is None:
        return []

    shap_values = model_explainer.shap_values(encoded_data)

    feature_impacts = pd.DataFrame(
        {
            "feature": encoded_data.columns,
            "impact": abs(shap_values[0]),  # Absolute impact for class 1 (churn)
        }
    )

    top_features = feature_impacts.sort_values(by="impact", ascending=False).head(3)

    return top_features["feature"].tolist()


def predict_customer_churn(customer_data):
    # Main orchestrator: build, encode, predict, and return results
    if churn_model is None:
        raise RuntimeError("Model files not found. Prediction service unavailable.")

    input_data = build_input_dataframe(customer_data)  # Step 1: Build DataFrame
    encoded_data = encode_input_data(input_data)  # Step 2: Encode

    prediction = churn_model.predict(encoded_data)[0]  # Step 3: Get class (0/1)

    churn_probability = churn_model.predict_proba(encoded_data)[0][
        1
    ]  # Step 4: Get probability

    risk_level = get_risk_level(churn_probability)  # Step 5: Determine risk

    prediction_label = (
        "Churn" if prediction == 1 else "No Churn"  # Step 6: Human-readable label
    )

    recommendation = get_recommendation(risk_level)  # Step 7: Generate advice

    top_factors = get_top_factors(encoded_data)  # Step 8: SHAP analysis

    # Step 9: Return complete response
    return {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "churn_probability": round(float(churn_probability), 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "top_factors": top_factors,
    }
