import joblib
import pandas as pd

# Load trained model and encoders from disk
churn_model = joblib.load("src/models/churn_model.pkl")
categorical_encoders = joblib.load("src/models/label_encoders.pkl")

# Define categorical columns that need encoding (consistent with training)
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
    "Payment Method"
]

# Convert customer data object to DataFrame with matching column names
def build_input_dataframe(customer_data):
    return pd.DataFrame([{
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

# Apply saved encoders to transform categorical text to numbers
def encode_input_data(input_data):
    encoded_data = input_data.copy()
    
    for column_name in CATEGORICAL_COLUMNS:
        encoder = categorical_encoders[column_name]  # Get saved encoder
        encoded_data[column_name] = encoder.transform(
            encoded_data[column_name].astype(str)  # Transform text to numbers
        )
    
    # Ensure all data is numeric
    encoded_data = encoded_data.apply(pd.to_numeric, errors="coerce")
    encoded_data = encoded_data.fillna(0)  # Replace any NaN with 0
    
    return encoded_data

# Determine risk level based on probability threshold
def get_risk_level(churn_probability):
    if churn_probability >= 0.7:
        return "High"
    if churn_probability >= 0.4:
        return "Medium"
    return "Low"

# Generate business recommendation based on risk level
def get_recommendation(risk_level):
    if risk_level == "High":
        return "Customer is at high risk of churn. Consider retention actions."
    if risk_level == "Medium":
        return "Customer has moderate churn risk. Monitor behavior and engagement."
    return "Customer has low churn risk."

# Main prediction function orchestrating all steps
def predict_customer_churn(customer_data):
    input_data = build_input_dataframe(customer_data)  # Step 1: Build DataFrame
    encoded_data = encode_input_data(input_data)       # Step 2: Encode categories
    
    prediction = churn_model.predict(encoded_data)[0]  # Step 3: Get class (0 or 1)
    churn_probability = churn_model.predict_proba(encoded_data)[0][1]  # Step 4: Get probability
    
    risk_level = get_risk_level(churn_probability)     # Step 5: Determine risk
    prediction_label = "Churn" if prediction == 1 else "No Churn"  # Step 6: Human-readable label
    recommendation = get_recommendation(risk_level)    # Step 7: Generate advice
    
    # Step 8: Return formatted response
    return {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "churn_probability": round(float(churn_probability), 4),
        "risk_level": risk_level,
        "recommendation": recommendation
    }