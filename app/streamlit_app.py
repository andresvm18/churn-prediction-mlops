import requests
import streamlit as st

from pathlib import Path
from textwrap import dedent
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


# Page config
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Load CSS file
css_path = Path(__file__).parent / "styles.css"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.html(f"<style>{css_file.read()}</style>")
else:
    st.error("No se encontró styles.css")


# SVG icon helper
def svg(path_d, size=14, stroke_w=1.75):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke_w}" '
        f'stroke-linecap="round" stroke-linejoin="round">{path_d}</svg>'
    )


# Icon paths
ICONS = {
    "trending-down": '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/><polyline points="17 18 23 18 23 12"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "wifi": '<path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>',
}


# Get icon by name
def ic(name, size=14):
    return svg(ICONS.get(name, ""), size)


# Header
st.html(
    f"""
    <div class="ci-header-wrap">

        <div class="ci-logo-mark">
            {ic("trending-down", 18)}
        </div>

        <div>
            <p class="ci-wordmark">Churn Intelligence</p>
            <p class="ci-eyebrow">Customer retention analytics</p>
        </div>

    </div>
    """
)


# Two columns layout
left_col, right_col = st.columns([2.5, 1], gap="large")


# Left column - input form
with left_col:

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Personal", "Services", "Billing"])

    # Tab 1 - Personal info
    with tab1:

        st.html(f'<div class="ci-section">{ic("user")} Basic information</div>')

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.radio("Senior citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x else "No",
                                      horizontal=True)

        with col2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.html(f'<div class="ci-section">{ic("clock")} Tenure</div>')

        tenure_months = st.slider("Months with company", 0, 72, 12)

    # Tab 2 - Services
    with tab2:

        st.html(f'<div class="ci-section">{ic("wifi")} Connectivity</div>')

        col1, col2 = st.columns(2)

        with col1:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            if phone_service == "Yes":
                multiple_lines = st.selectbox("Multiple lines", ["Yes", "No"])
            else:
                multiple_lines = "No phone service"

        with col2:
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

        # Show online services only if customer has internet
        if internet_service != "No":
            st.html(f'<div class="ci-section">{ic("shield")} Online services</div>')

            col1, col2, col3 = st.columns(3)

            with col1:
                online_security = st.selectbox("Online security", ["Yes", "No"])
                online_backup = st.selectbox("Online backup", ["Yes", "No"])

            with col2:
                device_protection = st.selectbox("Device protection", ["Yes", "No"])
                tech_support = st.selectbox("Tech support", ["Yes", "No"])

            with col3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
                streaming_movies = st.selectbox("Streaming movies", ["Yes", "No"])
        else:
            # Default values when no internet
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

    # Tab 3 - Billing
    with tab3:

        st.html(f'<div class="ci-section">{ic("file")} Contract</div>')

        col1, col2 = st.columns(2)

        with col1:
            contract = st.selectbox("Contract type",
                                   ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        with col2:
            payment_method = st.selectbox("Payment method",
                                         ["Electronic check", "Mailed check",
                                          "Bank transfer (automatic)", "Credit card (automatic)"])

        st.html(f'<div class="ci-section">{ic("dollar")} Charges</div>')

        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=95.5)

        with col2:
            total_charges = st.number_input("Total charges", min_value=0.0, value=1100.0)


# Right column - prediction panel
with right_col:

    # Risk factors info box
    st.html("""
        <div class="ci-risk-panel">
            <div class="ci-risk-head">Key risk factors</div>
            <div class="ci-risk-row">Month-to-month contract</div>
            <div class="ci-risk-row">Low tenure</div>
            <div class="ci-risk-row">High monthly charges</div>
            <div class="ci-risk-row">Electronic check payment</div>
        </div>
    """)

    # Predict button
    predict = st.button("Run prediction", use_container_width=True)

    # Make prediction when button is clicked
    if predict:

        # Prepare payload
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

        try:
            # Call API
            response = requests.post(API_URL, json=customer_payload, timeout=10)

            # Success
            if response.status_code == 200:
                result = response.json()
                probability = result["churn_probability"] * 100

                st.html('<hr class="ci-divider">')

                # Show results
                st.metric("Prediction", result["prediction_label"])
                st.metric("Risk Level", result["risk_level"])
                st.metric("Probability", f"{probability:.2f}%")
                st.progress(result["churn_probability"])
                st.info(result["recommendation"])

            else:
                st.error("Prediction failed")

        # Connection error
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure FastAPI is running.")

        # Timeout error
        except requests.exceptions.Timeout:
            st.error("API took too long to respond.")

        # Other errors
        except Exception as error:
            st.error(f"Error: {error}")


# Footer
st.html('<div class="ci-footer">Churn Intelligence — Powered by Machine Learning</div>')