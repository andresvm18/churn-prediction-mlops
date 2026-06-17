import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

css_path = Path(__file__).parent.parent / "styles.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.html(f"<style>{f.read()}</style>")

st.html(
    """
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
"""
)


def fetch_stats() -> dict:
    try:
        resp = requests.get(f"{API_URL}/history/stats", timeout=5)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:
        return {}


def fetch_history(
    limit: int,
    offset: int,
    risk_level: str,
    pred_filter: str,
) -> tuple[list[dict], int]:
    params = {"limit": limit, "offset": offset}
    if risk_level != "All":
        params["risk_level"] = risk_level
    if pred_filter == "Churn":
        params["prediction"] = 1
    elif pred_filter == "No Churn":
        params["prediction"] = 0
    try:
        resp = requests.get(f"{API_URL}/history", params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data["rows"], data["total"]
        return [], 0
    except Exception:
        return [], 0


def fetch_all_for_charts() -> list[dict]:
    try:
        resp = requests.get(f"{API_URL}/history?limit=1000", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("rows", []) if isinstance(data, dict) else data
        return []
    except Exception:
        return []


def build_table(rows: list[dict]) -> str:
    def pred_badge(label: str) -> str:
        if label == "Churn":
            return (
                '<span style="background:#fdf0ef;color:#c0392b;border:1px solid #f5c6c2;'
                'border-radius:999px;padding:2px 10px;font-size:0.7rem;font-weight:600;">'
                "Churn</span>"
            )
        return (
            '<span style="background:#edf7f0;color:#1e8449;border:1px solid #b7dfc5;'
            'border-radius:999px;padding:2px 10px;font-size:0.7rem;font-weight:600;">'
            "No Churn</span>"
        )

    def risk_badge(level: str) -> str:
        styles = {
            "High": "background:#fdf0ef;color:#c0392b;border:1px solid #f5c6c2;",
            "Medium": "background:#fef9ec;color:#d68910;border:1px solid #f5e0a0;",
            "Low": "background:#edf7f0;color:#1e8449;border:1px solid #b7dfc5;",
        }
        s = styles.get(
            level, "background:#f0ece5;color:#5c574f;border:1px solid #e2ddd6;"
        )
        return (
            f'<span style="{s}border-radius:999px;padding:2px 8px;'
            f'font-size:0.68rem;font-weight:600;">{level}</span>'
        )

    def fmt_factors(raw: str) -> str:
        try:
            factors = json.loads(raw) if raw else []
            return " · ".join(factors)
        except Exception:
            return raw or ""

    def fmt_date(raw: str) -> str:
        try:
            return pd.to_datetime(raw).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return raw

    th = (
        "padding:0.6rem 0.85rem;text-align:left;font-size:0.68rem;"
        "font-family:'DM Mono',monospace;text-transform:uppercase;"
        "letter-spacing:0.08em;color:#9c968d;font-weight:500;"
        "white-space:nowrap;border-bottom:2px solid #e2ddd6;background:#f8f6f2;"
    )
    td = (
        "padding:0.6rem 0.85rem;color:#1a1714;font-size:0.82rem;"
        "border-bottom:1px solid #f0ece5;vertical-align:middle;"
    )
    td_f = (
        "padding:0.6rem 0.85rem;color:#5c574f;font-size:0.75rem;"
        "border-bottom:1px solid #f0ece5;vertical-align:middle;"
        "max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
    )

    headers = [
        "Date",
        "Gender",
        "Tenure",
        "Contract",
        "Internet",
        "Monthly $",
        "Prediction",
        "Probability",
        "Risk",
        "Top Factors",
    ]
    header_html = "".join(f'<th style="{th}">{h}</th>' for h in headers)

    rows_html = ""
    for i, row in enumerate(rows):
        bg = "#ffffff" if i % 2 == 0 else "#fafaf9"
        prob = row.get("churn_probability", 0)
        rows_html += (
            f'<tr style="background:{bg};" '
            f"onmouseover=\"this.style.background='#f8f6f2'\" "
            f"onmouseout=\"this.style.background='{bg}'\">"
            f'<td style="{td}">{fmt_date(row.get("created_at", ""))}</td>'
            f'<td style="{td}">{row.get("gender", "")}</td>'
            f'<td style="{td}">{row.get("tenure_months", "")}</td>'
            f'<td style="{td}">{row.get("contract", "")}</td>'
            f'<td style="{td}">{row.get("internet_service", "")}</td>'
            f'<td style="{td}">${row.get("monthly_charges", 0):.0f}</td>'
            f'<td style="{td}">{pred_badge(row.get("prediction_label", ""))}</td>'
            f'<td style="{td}">{float(prob):.4f}</td>'
            f'<td style="{td}">{risk_badge(row.get("risk_level", ""))}</td>'
            f'<td style="{td_f}">{fmt_factors(row.get("top_factors", ""))}</td>'
            f"</tr>"
        )

    return (
        '<div style="overflow-x:auto;border:1px solid #e2ddd6;border-radius:12px;'
        'overflow:hidden;background:white;margin-bottom:0.5rem;">'
        '<table style="width:100%;border-collapse:collapse;background:white;">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )


CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#1a1714", family="Outfit, sans-serif", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    height=240,
    xaxis=dict(
        gridcolor="#e2ddd6", linecolor="#e2ddd6", tickfont=dict(color="#5c574f")
    ),
    yaxis=dict(
        gridcolor="#e2ddd6", linecolor="#e2ddd6", tickfont=dict(color="#5c574f")
    ),
)

stats = fetch_stats()

if not stats or stats.get("total", 0) == 0:
    st.html(
        """
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
    """
    )
    st.stop()

st.html('<div class="ci-section">📊 Summary</div>')
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total predictions", stats["total"])
c2.metric("Predicted churn", f"{stats['churn_count']} ({stats['churn_rate']}%)")
c3.metric("Avg probability", f"{stats['avg_probability']}%")
c4.metric("High risk", stats["high_risk"])
st.html('<hr class="ci-divider-light">')

st.html('<div class="ci-section">🎯 Risk breakdown</div>')
rb1, rb2, rb3 = st.columns(3)
rb1.metric("🔴 High risk", stats["high_risk"])
rb2.metric("🟡 Medium risk", stats["medium_risk"])
rb3.metric("🟢 Low risk", stats["low_risk"])
st.html('<hr class="ci-divider-light">')

st.html('<div class="ci-section">📈 Analytics</div>')
chart_data = fetch_all_for_charts()

if chart_data:
    chart_df = pd.DataFrame(chart_data)
    chart_df["created_at"] = pd.to_datetime(chart_df["created_at"])
    chart_df["churn_probability"] = pd.to_numeric(
        chart_df["churn_probability"], errors="coerce"
    )

    col_left, col_right = st.columns(2)

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
            fig.add_trace(
                go.Scatter(
                    x=temporal["date"],
                    y=temporal["avg_probability"],
                    mode="lines+markers",
                    line=dict(color="#c0392b", width=2),
                    marker=dict(color="#c0392b", size=6),
                    hovertemplate="%{x|%b %d}<br>%{y:.1f}%<extra></extra>",
                )
            )
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

    with col_right:
        st.markdown("**Probability distribution**")
        st.caption("How predictions are distributed across risk ranges.")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = [
            "0-10%",
            "10-20%",
            "20-30%",
            "30-40%",
            "40-50%",
            "50-60%",
            "60-70%",
            "70-80%",
            "80-90%",
            "90-100%",
        ]
        chart_df["bucket"] = pd.cut(
            chart_df["churn_probability"], bins=bins, labels=labels, right=False
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
        fig2.add_trace(
            go.Bar(
                x=dist["range"],
                y=dist["count"],
                marker_color=bar_colors,
                hovertemplate="%{x}<br>%{y} predictions<extra></extra>",
            )
        )
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
f1, f2 = st.columns(2)
with f1:
    risk_filter = st.selectbox("Risk level", ["All", "High", "Medium", "Low"])
with f2:
    pred_filter = st.selectbox("Prediction", ["All", "Churn", "No Churn"])

PAGE_SIZE = 25

if "history_page" not in st.session_state:
    st.session_state["history_page"] = 0

if (
    st.session_state.get("last_risk_filter") != risk_filter
    or st.session_state.get("last_pred_filter") != pred_filter
):
    st.session_state["history_page"] = 0
    st.session_state["last_risk_filter"] = risk_filter
    st.session_state["last_pred_filter"] = pred_filter

current_page = st.session_state["history_page"]
offset = current_page * PAGE_SIZE

rows, total = fetch_history(PAGE_SIZE, offset, risk_filter, pred_filter)

if not rows and total == 0:
    st.info("No predictions match the selected filters.")
    st.stop()

total_pages = max(1, -(-total // PAGE_SIZE))

st.caption(
    f"Showing {offset + 1}–{min(offset + PAGE_SIZE, total)} of {total} predictions"
    f" · Page {current_page + 1} of {total_pages}"
)

st.html(build_table(rows))
st.html('<hr class="ci-divider-light">')

# Pagination controls
nav1, nav2, _, nav3 = st.columns([1, 1, 3, 1])

with nav1:
    if st.button(
        "← Previous",
        use_container_width=True,
        key="prev_page",
        disabled=current_page == 0,
    ):
        st.session_state["history_page"] -= 1
        st.rerun()

with nav2:
    if st.button(
        "Next →",
        use_container_width=True,
        key="next_page",
        disabled=current_page >= total_pages - 1,
    ):
        st.session_state["history_page"] += 1
        st.rerun()

with nav3:
    if st.button(
        "↑ First",
        use_container_width=True,
        key="first_page",
        disabled=current_page == 0,
    ):
        st.session_state["history_page"] = 0
        st.rerun()

st.html('<hr class="ci-divider-light">')

# Export & clear
df = pd.DataFrame(rows)
btn_col1, btn_col2, _ = st.columns([1, 1, 2])

with btn_col1:
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Export page",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="export_csv_btn",
    )

with btn_col2:
    if st.button(
        "Clear history",
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
                    st.session_state["history_page"] = 0
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

st.html(
    '<div class="ci-footer">'
    "Churn Intelligence · Powered by Machine Learning · v2.3"
    "</div>"
)
