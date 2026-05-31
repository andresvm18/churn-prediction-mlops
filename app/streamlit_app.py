import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Load CSS ─────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")
else:
    st.error("styles.css not found")


# ─── SVG helpers ──────────────────────────────────────────────────────────────
def svg(path_d: str, size: int = 14, stroke_w: float = 1.75) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke_w}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f"{path_d}</svg>"
    )


ICONS = {
    "trending-down": (
        '<polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>'
        '<polyline points="17 18 23 18 23 12"/>'
    ),
    "user": (
        '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>'
        '<circle cx="12" cy="7" r="4"/>'
    ),
    "wifi": (
        '<path d="M5 12.55a11 11 0 0 1 14.08 0"/>'
        '<path d="M1.42 9a16 16 0 0 1 21.16 0"/>'
    ),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "clock": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>'
    ),
    "dollar": (
        '<line x1="12" y1="1" x2="12" y2="23"/>'
        '<line x1="17" y1="5" x2="7" y2="19"/>'
    ),
    "file": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
    ),
    "alert-triangle": (
        '<path d="M12 9v4"/>'
        '<path d="M12 17h.01"/>'
        '<path d="M12 3L3 21h18L12 3z"/>'
    ),
    "bar-chart": (
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
    ),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
}


def ic(name: str, size: int = 14) -> str:
    return svg(ICONS.get(name, ""), size)


# ─── Session state ────────────────────────────────────────────────────────────
for key, default in [
    ("top_factors", []),
    ("has_prediction", False),
    ("last_prediction", None),
    ("is_loading", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Header ───────────────────────────────────────────────────────────────────
st.html(f"""
<div class="ci-header-wrap">
    <div class="ci-logo-mark">{ic("trending-down", 20)}</div>
    <div class="ci-header-text">
        <p class="ci-wordmark">Churn Intelligence</p>
        <p class="ci-eyebrow">Customer retention analytics</p>
    </div>
    <div class="ci-header-badge">● ML Model v2.1</div>
</div>
""")

# ─── Layout ───────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([2.6, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN — Input form
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    tab1, tab2, tab3 = st.tabs(["Personal", "Services", "Billing"])

    # ── Tab 1 · Personal ──────────────────────────────────────────────────────
    with tab1:
        st.html(f'<div class="ci-section">{ic("user", 13)} Basic information</div>')
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.radio(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x else "No",
                horizontal=True,
            )
        with c2:
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.html(f'<div class="ci-section">{ic("clock", 13)} Tenure</div>')
        tenure_months = st.slider("Months with company", 0, 72, 12)

    # ── Tab 2 · Services ─────────────────────────────────────────────────────
    with tab2:
        st.html(
            f'<div class="ci-section">{ic("wifi", 13)} Connectivity</div>'
        )
        c1, c2 = st.columns(2)
        with c1:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = (
                st.selectbox("Multiple lines", ["Yes", "No"])
                if phone_service == "Yes"
                else "No phone service"
            )
        with c2:
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )

        if internet_service != "No":
            st.html(
                f'<div class="ci-section">'
                f'{ic("shield", 13)} Online add-ons</div>'
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                online_security = st.selectbox("Online security", ["Yes", "No"])
                online_backup = st.selectbox("Online backup", ["Yes", "No"])
            with c2:
                device_protection = st.selectbox(
                    "Device protection", ["Yes", "No"]
                )
                tech_support = st.selectbox("Tech support", ["Yes", "No"])
            with c3:
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
                streaming_movies = st.selectbox("Streaming movies", ["Yes", "No"])
        else:
            online_security = online_backup = device_protection = tech_support = (
                streaming_tv
            ) = streaming_movies = "No internet service"

    # ── Tab 3 · Billing ───────────────────────────────────────────────────────
    with tab3:
        st.html(f'<div class="ci-section">{ic("file", 13)} Contract</div>')
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox(
                "Contract type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        with c2:
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.html(f'<div class="ci-section">{ic("dollar", 13)} Charges</div>')
        c1, c2 = st.columns(2)
        with c1:
            monthly_charges = st.number_input(
                "Monthly charges ($)", min_value=0.0, value=95.5, step=1.0
            )
        with c2:
            total_charges = st.number_input(
                "Total charges ($)", min_value=0.0, value=1100.0, step=10.0
            )

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN — Prediction panel
# ══════════════════════════════════════════════════════════════════════════════
with right_col:

    if (
        st.session_state.has_prediction
        and st.session_state.last_prediction
        and not st.session_state.is_loading
    ):
        result = st.session_state.last_prediction
        probability = result["churn_probability"] * 100
        risk_level = result["risk_level"]

        risk_color, risk_emoji = {
            "High": ("#c0392b", "🔴"),
            "Medium": ("#d68910", "🟡"),
            "Low": ("#1e8449", "🟢"),
        }.get(risk_level, ("#1a1714", "⚪"))

        # Risk level card
        st.markdown(
            f"""
        <div class="ci-risk-card">
            <div class="ci-risk-header">
                {ic("alert-triangle", 12)}&nbsp;Risk Level
            </div>
            <div class="ci-risk-value" style="color:{risk_color};">
                {risk_emoji} {risk_level}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Probability card
        st.markdown(
            f"""
        <div class="ci-prob-card">
            <div class="ci-prob-label">Churn Probability</div>
            <div class="ci-prob-value" style="color:{risk_color};">
                {probability:.1f}%
            </div>
            <div class="ci-prob-bar">
                <div class="ci-prob-bar-fill"
                     style="width:{probability}%; background:{risk_color};">
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Key risk factors
        if st.session_state.top_factors:
            st.markdown(
                '<div class="ci-risk-head">Key Risk Factors</div>',
                unsafe_allow_html=True,
            )
            badges = "".join(
                f'<span class="ci-badge">{f}</span>'
                for f in st.session_state.top_factors
            )
            st.markdown(
                f'<div class="ci-badges">{badges}</div>',
                unsafe_allow_html=True,
            )

        st.html('<hr class="ci-divider-light">')

        # Recommendation
        st.markdown(
            f"""
        <div class="ci-recommendation">
            <span class="ci-rec-icon">💡</span>
            <span>{result["recommendation"]}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    elif not st.session_state.is_loading:
        empty_state_html = f"""
            <div class="ci-empty-state">
                <div class="ci-empty-icon">{ic("bar-chart", 22)}</div>
                <p class="ci-empty-title">No prediction yet</p>
                <p class="ci-empty-sub">
                    Fill in the customer details<br>
                    and run the model to see results.
                </p>
            </div>
            """

        st.markdown(empty_state_html, unsafe_allow_html=True)

    # ── Button ────────────────────────────────────────────────────────────────
    btn_placeholder = st.empty()

    if st.session_state.is_loading:
        btn_placeholder.markdown(
            '<div class="ci-button-loading">'
            '<div class="ci-spinner"></div>'
            "<span>Analyzing customer data…</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        if btn_placeholder.button(
            "Run prediction",
            use_container_width=True,
            type="primary",
            key="predict_btn",
        ):
            st.session_state.is_loading = True
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  Prediction logic
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.is_loading:
    payload = {
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
        "total_charges": total_charges,
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            result = resp.json()
            st.session_state.top_factors = result.get("top_factors", [])
            st.session_state.has_prediction = True
            st.session_state.last_prediction = result
        else:
            st.error(f"Prediction failed — status {resp.status_code}")
            st.session_state.has_prediction = False

    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API. Is FastAPI running on port 8000?")
        st.session_state.has_prediction = False
    except requests.exceptions.Timeout:
        st.error("API timed out. Try again.")
        st.session_state.has_prediction = False
    except Exception as err:
        st.error(f"Unexpected error: {err}")
        st.session_state.has_prediction = False

    st.session_state.is_loading = False
    st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.html(
    '<div class="ci-footer">'
    "Churn Intelligence · Powered by Machine Learning · v2.1"
    "</div>"
)
