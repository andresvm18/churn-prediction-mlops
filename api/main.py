import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI(title="Customer Churn Prediction API")

# Load trained model and encoders
churn_model = joblib.load("src/models/churn_model.pkl")
categorical_encoders = joblib.load("src/models/label_encoders.pkl")


# Define expected customer input
class CustomerData(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure_months: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float


# Health check endpoint
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
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
    }])

    # Define categorical columns
    categorical_columns = [
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
        "Payment Method"
    ]

    # Encode categorical columns using saved encoders
    for column_name in categorical_columns:
        encoder = categorical_encoders[column_name]

        input_data[column_name] = encoder.transform(
            input_data[column_name].astype(str)
        )

    # Convert all input features to numeric
    input_data = input_data.apply(pd.to_numeric, errors="coerce")
    input_data = input_data.fillna(0)

    # Generate prediction
    prediction = churn_model.predict(input_data)[0]

    # Generate churn probability
    prediction_probability = churn_model.predict_proba(input_data)[0][1]

    # Assign risk level
    risk_level = (
        "High"
        if prediction_probability >= 0.7
        else "Medium"
        if prediction_probability >= 0.4
        else "Low"
    )

    # Convert numeric prediction to human-readable label
    prediction_label = "Churn" if prediction == 1 else "No Churn"

    # Generate business recommendation based on risk level
    recommendation = (
        "Customer is at high risk of churn. Consider retention actions."
        if risk_level == "High"
        else "Customer has moderate churn risk. Monitor behavior and engagement."
        if risk_level == "Medium"
        else "Customer has low churn risk."
    )

    # Return enhanced response with label and recommendation
    return {
        "prediction": int(prediction),                      # Raw prediction (0 or 1)
        "prediction_label": prediction_label,              # Readable label ("Churn" or "No Churn")
        "churn_probability": round(float(prediction_probability), 4),  # Probability as percentage
        "risk_level": risk_level,                          # Categorical risk (High/Medium/Low)
        "recommendation": recommendation                   # Actionable business advice
    }