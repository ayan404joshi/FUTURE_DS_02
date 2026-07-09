"""Interactive Streamlit dashboard for churn analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ASSETS_DIR = PROJECT_ROOT / "assets"


st.set_page_config(page_title="Customer Retention & Churn Dashboard", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(15,108,189,0.12), transparent 35%),
                    linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        color: #10233f;
        font-family: "Baskerville", "Georgia", serif;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(15,108,189,0.14), rgba(255,255,255,0.88));
        border: 1px solid rgba(16,35,63,0.08);
        box-shadow: 0 20px 50px rgba(15,108,189,0.08);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(16,35,63,0.08);
        box-shadow: 0 10px 28px rgba(16,35,63,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_data() -> pd.DataFrame:
    """Load the cleaned dataset for dashboarding."""
    return pd.read_csv(PROCESSED_DIR / "subscription_customers_clean.csv", parse_dates=["signup_date", "churn_date", "last_login_date", "cohort_month", "churn_month"])


@st.cache_data

def _load_tables() -> dict[str, pd.DataFrame]:
    """Load analysis tables only once per session."""
    tables = {
        "trend": pd.read_csv(PROCESSED_DIR / "monthly_churn_trend.csv", parse_dates=["signup_date"]),
        "plan": pd.read_csv(PROCESSED_DIR / "churn_by_plan.csv"),
        "region": pd.read_csv(PROCESSED_DIR / "churn_by_region.csv"),
        "channel": pd.read_csv(PROCESSED_DIR / "churn_by_channel.csv"),
        "payment": pd.read_csv(PROCESSED_DIR / "churn_by_payment.csv"),
        "reasons": pd.read_csv(PROCESSED_DIR / "churn_reason_pareto.csv"),
        "clv": pd.read_csv(PROCESSED_DIR / "clv_tier_churn.csv"),
        "cohort": pd.read_csv(PROCESSED_DIR / "cohort_retention_table.csv", index_col=0),
        "importance": pd.read_csv(PROCESSED_DIR / "feature_importance.csv"),
        "summary": pd.read_json(PROCESSED_DIR / "analysis_summary.json", typ="series"),
    }
    return tables


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _kpi_card(label: str, value: str, delta: str | None = None) -> None:
    """Render a metric card in Streamlit."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:#54708f;">{label}</div>
            <div style="font-size:2rem; font-weight:700; margin-top:0.2rem; color:#10233f;">{value}</div>
            <div style="font-size:0.9rem; color:#6b7d95;">{delta or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1 style="margin:0; font-size:2.2rem;">Customer Retention & Churn Dashboard</h1>
        <p style="margin:0.4rem 0 0 0; font-size:1rem; color:#47607d;">Interactive view of churn patterns, retention cohorts, CLV risk, and model-derived churn drivers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

raw_df = _load_data()
tables = _load_tables()
summary = tables["summary"]

min_date = raw_df["signup_date"].min().date()
max_date = raw_df["signup_date"].max().date()

with st.sidebar:
    st.header("Filters")
    date_range = st.date_input("Signup date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    selected_plans = st.multiselect("Plan type", sorted(raw_df["plan_type"].dropna().unique()), default=sorted(raw_df["plan_type"].dropna().unique()))
    selected_regions = st.multiselect("Region", sorted(raw_df["region"].dropna().unique()), default=sorted(raw_df["region"].dropna().unique()))
    selected_channels = st.multiselect("Acquisition channel", sorted(raw_df["acquisition_channel"].dropna().unique()), default=sorted(raw_df["acquisition_channel"].dropna().unique()))

start_date, end_date = date_range if isinstance(date_range, tuple) else (min_date, max_date)
filtered = raw_df[
    (raw_df["signup_date"].dt.date >= start_date)
    & (raw_df["signup_date"].dt.date <= end_date)
    & (raw_df["plan_type"].isin(selected_plans))
    & (raw_df["region"].isin(selected_regions))
    & (raw_df["acquisition_channel"].isin(selected_channels))
].copy()

kpi_cols = st.columns(4)
kpi_cols[0].markdown("<div class='metric-card'>", unsafe_allow_html=True)
with kpi_cols[0]:
    _kpi_card("Total Customers", f"{len(filtered):,}", f"Filtered from {len(raw_df):,}")
with kpi_cols[1]:
    _kpi_card("Churn Rate", _format_pct(filtered["churn_flag"].mean()), "Current filtered cohort")
with kpi_cols[2]:
    _kpi_card("Average CLV", f"${filtered['historical_clv'].mean():,.0f}")
with kpi_cols[3]:
    _kpi_card("Average Tenure", f"{filtered['tenure_months'].mean():.1f} months")

st.divider()

left, right = st.columns([1.25, 1])
with left:
    trend = filtered.groupby(filtered["signup_date"].dt.to_period("M").dt.to_timestamp()).agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean")).reset_index()
    fig = px.line(trend, x="signup_date", y="churn_rate", markers=True, title="Churn Rate Trend by Signup Month", labels={"signup_date": "Signup Month", "churn_rate": "Churn Rate"})
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(tables["plan"], x="plan_type", y="churn_rate", title="Churn by Plan Type", color="churn_rate", color_continuous_scale="Blues")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.bar(tables["region"], x="region", y="churn_rate", title="Churn by Region", color="churn_rate", color_continuous_scale="Blues")
    fig.update_layout(height=380, template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.bar(tables["channel"], x="acquisition_channel", y="churn_rate", title="Churn by Acquisition Channel", color="churn_rate", color_continuous_scale="Blues")
    fig.update_layout(height=380, template="plotly_white", coloraxis_showscale=False, xaxis_title="Acquisition Channel")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    cohort = tables["cohort"].apply(pd.to_numeric, errors="coerce")
    heat = go.Figure(data=go.Heatmap(z=cohort.values, x=[str(c) for c in cohort.columns], y=[str(i) for i in cohort.index], colorscale="Blues", colorbar=dict(title="Retention")))
    heat.update_layout(title="Cohort Retention Heatmap", height=480, template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(heat, use_container_width=True)
with right:
    fig = px.bar(tables["reasons"].head(6), x="churn_reason", y="count", title="Churn Reason Breakdown", color="count", color_continuous_scale="Reds")
    fig.update_layout(height=480, template="plotly_white", coloraxis_showscale=False, xaxis_title="Reason")
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    fig = px.bar(tables["clv"], x="clv_tier", y="churn_rate", title="CLV Tier vs Churn Rate", color="churn_rate", color_continuous_scale="Viridis")
    fig.update_layout(height=360, template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    importance = tables["importance"].head(12).sort_values("importance", ascending=True)
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Top Churn Drivers", color="importance", color_continuous_scale="Blues")
    fig.update_layout(height=360, template="plotly_white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Filtered Data Preview")
st.dataframe(filtered.head(20), use_container_width=True)

st.caption("Dashboard built from the processed churn analysis artifacts. Refresh after running the pipeline to update metrics and charts.")
