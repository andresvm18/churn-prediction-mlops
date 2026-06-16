import io
import json
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
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

# Fetch helpers
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

def fetch_all_for_charts() -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/history?limit=1000", timeout=5)
        return resp.json() if resp.status_code == 200 else []
    except Exception:
        return []

# Shared chart layout
CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#1a1714", family="Outfit, sans-serif", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    height=240,
    xaxis=dict(
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        tickfont=dict(color="#5c574f"),
    ),
    yaxis=dict(
        gridcolor="#e2ddd6",
        linecolor="#e2ddd6",
        tickfont=dict(color="#5c574f"),
    ),
)

# Summary stats
stats = fetch_stats()
if not stats or stats.get("total", 0) == 0:
    st.html("""
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
c2.metric("Predicted churn",   f"{stats['churn_count']} ({stats['churn_rate']}%)")
c3.metric("Avg probability",   f"{stats['avg_probability']}%")
c4.metric("High risk",         stats["high_risk"])
st.html('<hr class="ci-divider-light">')

# Risk breakdown
st.html('<div class="ci-section">🎯 Risk breakdown</div>')
rb1, rb2, rb3 = st.columns(3)
rb1.metric("🔴 High risk",   stats["high_risk"])
rb2.metric("🟡 Medium risk", stats["medium_risk"])
rb3.metric("🟢 Low risk",    stats["low_risk"])
st.html('<hr class="ci-divider-light">')

# Charts
st.html('<div class="ci-section">📈 Analytics</div>')
chart_data = fetch_all_for_charts()
if chart_data:
    chart_df = pd.DataFrame(chart_data)
    chart_df["created_at"] = pd.to_datetime(chart_df["created_at"])
    chart_df["churn_probability"] = pd.to_numeric(
        chart_df["churn_probability"], errors="coerce"
    )

    col_left, col_right = st.columns(2)

    # Left: Temporal evolution
    with col_left:
        st.markdown("**Churn rate over time**")
        st.caption("Daily average churn probability of analyzed customers.")
        temporal = (
            chart_df.set_index("created_at")
            .resample("D")["churn_probability"]
            .agg(["mean", "count"])
            .reset_index()
        )
        temporal.columns = ["date", "avg_probability", "count"]
        temporal["avg_probability"] = (temporal["avg_probability"] * 100).round(1)
        temporal = temporal[temporal["count"] > 0]

        if len(temporal) >= 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=temporal["date"],
                y=temporal["avg_probability"],
                mode="lines+markers",
                line=dict(color="#c0392b", width=2),
                marker=dict(color="#c0392b", size=6),
                name="Avg probability",
                hovertemplate="%{x|%b %d}<br>%{y:.1f}%<extra></extra>",
            ))
            fig.update_layout(
                **CHART_LAYOUT,
                yaxis_title="Avg churn probability (%)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif len(temporal) == 1:
            st.metric(
                "Today's avg probability",
                f"{temporal['avg_probability'].iloc[0]:.1f}%",
            )
            st.caption("Make predictions on multiple days to see the trend.")
        else:
            st.info("Not enough data to render the chart yet.")

    # Right: Probability distribution
    with col_right:
        st.markdown("**Probability distribution**")
        st.caption("How predictions are distributed across risk ranges.")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = [
            "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
            "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
        ]
        chart_df["bucket"] = pd.cut(
            chart_df["churn_probability"],
            bins=bins,
            labels=labels,
            right=False,
        )
        dist = (
            chart_df["bucket"]
            .value_counts()
            .reindex(labels, fill_value=0)
            .reset_index()
        )
        dist.columns = ["range", "count"]
        bar_colors = [
            "#1e8449" if i < 4 else "#d68910" if i < 7 else "#c0392b"
            for i in range(len(dist))
        ]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=dist["range"],
            y=dist["count"],
            marker_color=bar_colors,
            hovertemplate="%{x}<br>%{y} predictions<extra></extra>",
        ))
        fig2.update_layout(
            **CHART_LAYOUT,
            yaxis_title="Number of predictions",
            showlegend=False,
            bargap=0.15,
        )
        st.plotly_chart(fig2, use_container_width=True)

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
if "top_factors" in df.columns:
    df["top_factors"] = df["top_factors"].apply(
        lambda x: " | ".join(json.loads(x)) if x else ""
    )
if "created_at" in df.columns:
    df["created_at"] = (
        pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
    )

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

st.html('<div class="ci-dataframe-wrap">')
st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Probability": st.column_config.NumberColumn(format="%.4f"),
        "Monthly $":   st.column_config.NumberColumn(format="$%.0f"),
    },
)
st.html('</div>')

st.html('<hr class="ci-divider-light">')

# Export & clear
st.html('<hr class="ci-divider-light">')
btn_col1, btn_col2, _ = st.columns([1, 1, 2])

with btn_col1:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export CSV",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="export_csv_btn",
    )

with btn_col2:
    if st.button(
        "🗑 Clear history",
        use_container_width=True,
        key="clear_history_btn",
    ):
        st.session_state["confirm_clear"] = True

if st.session_state.get("confirm_clear"):
    st.warning("Are you sure? This will delete all prediction history permanently.")
    confirm_col1, confirm_col2, _ = st.columns([1, 1, 3])
    with confirm_col1:
        if st.button(
            "Yes, delete all",
            type="primary",
            use_container_width=True,
            key="confirm_delete_btn",
        ):
            try:
                resp = requests.delete(f"{API_URL}/history", timeout=5)
                if resp.status_code == 200:
                    st.success(f"Deleted {resp.json()['deleted']} predictions.")
                    st.session_state["confirm_clear"] = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    with confirm_col2:
        if st.button(
            "Cancel",
            use_container_width=True,
            key="cancel_delete_btn",
        ):
            st.session_state["confirm_clear"] = False
            st.rerun()

# Footer
st.html(
    '<div class="ci-footer">'
    "Churn Intelligence · Powered by Machine Learning · v2.3"
    "</div>"
)