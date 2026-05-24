import requests
import streamlit as st

# FastAPI endpoint URL (local server)
API_URL = "http://api:8000/predict"

# Configure Streamlit page settings
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered"
)

# Display page header
st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn using a trained Machine Learning model.")

# --- Input fields for customer data ---
# Demographics
gender = st.selectbox("Gender", ["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

# Account information
tenure_months = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)

# Phone services
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

# Internet services
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# Contract and billing
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Charges
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=95.5)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1100.0)

# --- Prediction button ---
if st.button("Predict Churn"):
    # Prepare payload matching FastAPI schema
    customer_payload = {
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }
    
    # Send POST request to FastAPI
    response = requests.post(API_URL, json=customer_payload)
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()  # Parse JSON response
        
        # Display results
        st.subheader("Prediction Result")
        st.write(f"Prediction: **{result['prediction_label']}**")
        st.write(f"Churn Probability: **{result['churn_probability']}**")
        st.write(f"Risk Level: **{result['risk_level']}**")
        st.info(result["recommendation"])  # Show actionable recommendation
    else:
        st.error("Prediction failed. Make sure the FastAPI server is running.")