import io
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Load CSS
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")

# Header
st.html("""
<div class="ci-header-wrap">
    <div class="ci-logo-mark">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.75"
             stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 20h9"/>
            <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
        </svg>
    </div>
    <div class="ci-header-text">
        <p class="ci-wordmark">Prediction History</p>
        <p class="ci-eyebrow">All single predictions logged by the API</p>
    </div>
    <div class="ci-header-badge">● ML Model v2.3</div>
</div>
""")


# Fetch data
def fetch_stats() -> dict:
    try:
        resp = requests.get(f"{API_URL}/history/stats", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def fetch_history(limit: int, risk_level: str, pred_filter: str) -> list[dict]:
    params = {"limit": limit}
    if risk_level != "All":
        params["risk_level"] = risk_level
    if pred_filter == "Churn":
        params["prediction"] = 1
    elif pred_filter == "No Churn":
        params["prediction"] = 0
    try:
        resp = requests.get(f"{API_URL}/history", params=params, timeout=5)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []


# Summary stats

stats = fetch_stats()

if not stats or stats.get("total", 0) == 0:
    st.html(f"""
    <div class="ci-empty-state">
        <div class="ci-empty-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="1.75">
                <path d="M12 20h9"/>
                <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>
            </svg>
        </div>
        <p class="ci-empty-title">No predictions yet</p>
        <p class="ci-empty-sub">
            Run a prediction in Single Prediction<br>
            and it will appear here automatically.
        </p>
    </div>
    """)
    st.stop()

st.html('<div class="ci-section">📊 Summary</div>')
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions", stats["total"])
c2.metric("Predicted churn", f"{stats['churn_count']} ({stats['churn_rate']}%)")
c3.metric("Avg probability", f"{stats['avg_probability']}%")
c4.metric("High risk", stats["high_risk"])

st.html('<hr class="ci-divider-light">')

# Risk breakdown
st.html('<div class="ci-section">🎯 Risk breakdown</div>')
rb1, rb2, rb3 = st.columns(3)
rb1.metric("🔴 High risk",   stats["high_risk"])
rb2.metric("🟡 Medium risk", stats["medium_risk"])
rb3.metric("🟢 Low risk",    stats["low_risk"])

st.html('<hr class="ci-divider-light">')

# Filters
st.html('<div class="ci-section">🔍 Filter & export</div>')
f1, f2, f3 = st.columns(3)
with f1:
    risk_filter = st.selectbox("Risk level", ["All", "High", "Medium", "Low"])
with f2:
    pred_filter = st.selectbox("Prediction", ["All", "Churn", "No Churn"])
with f3:
    limit = st.selectbox("Show last", [25, 50, 100, 200], index=1)

# History table
rows = fetch_history(limit, risk_filter, pred_filter)

if not rows:
    st.info("No predictions match the selected filters.")
    st.stop()

df = pd.DataFrame(rows)

# Parse top_factors from JSON string back to readable format
if "top_factors" in df.columns:
    df["top_factors"] = df["top_factors"].apply(
        lambda x: " | ".join(json.loads(x)) if x else ""
    )

# Format timestamp
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

# Select and rename display columns
display_cols = {
    "created_at":        "Date",
    "gender":            "Gender",
    "tenure_months":     "Tenure",
    "contract":          "Contract",
    "internet_service":  "Internet",
    "monthly_charges":   "Monthly $",
    "prediction_label":  "Prediction",
    "churn_probability": "Probability",
    "risk_level":        "Risk",
    "top_factors":       "Top Factors",
}
available = {k: v for k, v in display_cols.items() if k in df.columns}
display_df = df[list(available.keys())].rename(columns=available)

st.dataframe(display_df, use_container_width=True, hide_index=True)

st.html('<hr class="ci-divider-light">')

# Export
col_dl, col_clear, _ = st.columns([1, 1, 2])

with col_dl:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

with col_clear:
    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state["confirm_clear"] = True

if st.session_state.get("confirm_clear"):
    st.warning("Are you sure? This will delete all prediction history permanently.")
    yes, no, _ = st.columns([1, 1, 3])
    with yes:
        if st.button("Yes, delete all", type="primary", use_container_width=True):
            try:
                resp = requests.delete(f"{API_URL}/history", timeout=5)
                if resp.status_code == 200:
                    st.success(f"Deleted {resp.json()['deleted']} predictions.")
                    st.session_state["confirm_clear"] = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    with no:
        if st.button("Cancel", use_container_width=True):
            st.session_state["confirm_clear"] = False
            st.rerun()

# Footer
st.html(
    '<div class="ci-footer">'
    "Churn Intelligence · Powered by Machine Learning · v2.3"
    "</div>"
)