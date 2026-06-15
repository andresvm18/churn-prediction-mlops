import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BATCH_URL = os.getenv("API_BATCH_URL", "http://127.0.0.1:8000/predict/batch/download")

# ─── Load CSS ─────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")

# ─── Header ───────────────────────────────────────────────────────────────────
st.html("""
<div class="ci-header-wrap">
    <div class="ci-logo-mark">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="1.75"
             stroke-linecap="round" stroke-linejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
    </div>
    <div class="ci-header-text">
        <p class="ci-wordmark">Batch Prediction</p>
        <p class="ci-eyebrow">Process multiple customers at once</p>
    </div>
    <div class="ci-header-badge">● ML Model v2.2</div>
</div>
""")

# ─── Template columns ──────────────────────────────────────────────────────────
TEMPLATE_COLS = [
    "gender", "senior_citizen", "partner", "dependents", "tenure_months",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract", "paperless_billing", "payment_method",
    "monthly_charges", "total_charges",
]
SAMPLE_ROWS = [
    ["Female", 0, "Yes", "No", 12, "Yes", "No", "Fiber optic",
     "No", "No", "No", "No", "Yes", "Yes",
     "Month-to-month", "Yes", "Electronic check", 95.5, 1146.0],
    ["Male", 0, "No", "No", 24, "Yes", "Yes", "DSL",
     "Yes", "No", "Yes", "No", "No", "No",
     "One year", "No", "Bank transfer (automatic)", 55.0, 1320.0],
]

# ─── How it works card ─────────────────────────────────────────────────────────
st.html("""
<div class="ci-snapshot-card">
  <div class="ci-snap-header">📋&nbsp; How it works</div>
  <div style="padding: 0.5rem 0; font-size: 0.85rem; color: var(--ci-text-muted, #6b6560); line-height: 1.8;">
    <b>1.</b> Download the CSV template below<br>
    <b>2.</b> Fill in your customer data (up to 1 000 rows)<br>
    <b>3.</b> Upload the file and click <em>Run batch prediction</em><br>
    <b>4.</b> Download the enriched CSV with predictions
  </div>
</div>
""")

st.html('<hr class="ci-divider-light">')

# ─── Template download ─────────────────────────────────────────────────────────
template_csv = pd.DataFrame(SAMPLE_ROWS, columns=TEMPLATE_COLS).to_csv(index=False)

# Constrain button width to left third so it doesn't stretch full page
col_dl, _, _ = st.columns(3)
with col_dl:
    st.download_button(
        label="⬇ Download CSV template",
        data=template_csv,
        file_name="churn_batch_template.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )

st.html('<hr class="ci-divider-light">')

# ─── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Maximum 1 000 rows. Columns must match the template.",
)

if uploaded_file is not None:
    try:
        preview_df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    # ── File info + preview ────────────────────────────────────────────────────
    st.html(f"""
    <div class="ci-snapshot-card">
      <div class="ci-snap-header">📄&nbsp; File preview</div>
      <div style="font-size:0.8rem;color:var(--ci-text-muted,#6b6560);padding-top:0.3rem;">
        <b>{len(preview_df)}</b> rows &nbsp;·&nbsp;
        <b>{len(preview_df.columns)}</b> columns detected
      </div>
    </div>
    """)
    st.dataframe(preview_df.head(5), use_container_width=True)

    st.html('<hr class="ci-divider-light">')

    if st.button(
        "🚀 Run batch prediction",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Running predictions — this may take a few seconds..."):
            try:
                resp = requests.post(
                    API_BATCH_URL,
                    files={"file": (uploaded_file.name, uploaded_file, "text/csv")},
                    timeout=120,
                )

                if resp.status_code == 200:
                    result_csv = resp.content
                    result_df  = pd.read_csv(io.StringIO(result_csv.decode("utf-8")))

                    total    = len(result_df)
                    churners = (result_df["prediction"] == 1).sum()
                    safe     = (result_df["prediction"] == 0).sum()
                    errors   = (result_df["status"] == "error").sum()

                    # ── Summary metrics ────────────────────────────────────────
                    st.html('<div class="ci-section">📊 Summary</div>')
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total customers",   total)
                    c2.metric("🔴 Predicted churn", f"{churners} ({churners/total*100:.1f}%)")
                    c3.metric("🟢 Low risk",         f"{safe} ({safe/total*100:.1f}%)")
                    c4.metric("⚠️ Errors",           errors)

                    st.html('<hr class="ci-divider-light">')

                    # ── Risk breakdown ─────────────────────────────────────────
                    if "risk_level" in result_df.columns:
                        st.html('<div class="ci-section">🎯 Risk breakdown</div>')
                        risk_counts = result_df["risk_level"].value_counts()
                        rb1, rb2, rb3 = st.columns(3)
                        rb1.metric("🔴 High risk",   risk_counts.get("High",   0))
                        rb2.metric("🟡 Medium risk", risk_counts.get("Medium", 0))
                        rb3.metric("🟢 Low risk",    risk_counts.get("Low",    0))

                        st.html('<hr class="ci-divider-light">')

                    # ── Results table ──────────────────────────────────────────
                    st.html('<div class="ci-section">📋 Results</div>')
                    display_cols = [
                        "prediction_label", "churn_probability",
                        "risk_level", "top_factors", "recommendation", "status",
                    ]
                    available = [c for c in display_cols if c in result_df.columns]
                    st.dataframe(result_df[available], use_container_width=True)

                    st.html('<hr class="ci-divider-light">')

                    # ── Download results ───────────────────────────────────────
                    col_res, _, _ = st.columns(3)
                    with col_res:
                        st.download_button(
                            label="⬇ Download results CSV",
                            data=result_csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True,
                        )

                elif resp.status_code == 422:
                    st.error(f"Validation error: {resp.json().get('detail', 'Invalid file.')}")
                elif resp.status_code == 503:
                    st.error("Prediction service unavailable. Is the model loaded?")
                else:
                    st.error(f"API error — status {resp.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is FastAPI running on port 8000?")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try with a smaller file.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.html(
    '<div class="ci-footer">'
    "Churn Intelligence · Powered by Machine Learning · v2.2"
    "</div>"
)