"""Run EDA, cohort analysis, and CLV cross-tabs for the churn project."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import ASSETS_DIR, ANALYSIS_SUMMARY_FILE, CLEAN_DATA_FILE, PROCESSED_DIR, ensure_directories, write_json

sns.set_theme(style="whitegrid", context="talk")


def _save_figure(path: Path, tight: bool = True) -> None:
    """Persist the active matplotlib figure with consistent styling."""
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _monthly_churn_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate churn by signup cohort month."""
    monthly = (
        df.groupby(pd.Grouper(key="signup_date", freq="MS"))
        .agg(customers=("customer_id", "count"), churned=("churn_flag", "sum"))
        .reset_index()
    )
    monthly["churn_rate"] = monthly["churned"] / monthly["customers"]
    return monthly


def _cohort_retention_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a retention matrix by signup cohort and months since signup."""
    working = df.copy()
    working["signup_month"] = working["signup_date"].dt.to_period("M")
    observation_end = working["churn_date"].fillna(working["last_login_date"])
    observation_end = observation_end.fillna(working["signup_date"])
    working["months_since_signup"] = (observation_end.dt.year - working["signup_date"].dt.year) * 12 + (
        observation_end.dt.month - working["signup_date"].dt.month
    )
    working["months_since_signup"] = working["months_since_signup"].clip(lower=0)

    cohort_counts = working.groupby(["signup_month", "months_since_signup"])["customer_id"].count().unstack(fill_value=0)
    cohort_sizes = working.groupby("signup_month")["customer_id"].count()
    retention = cohort_counts.div(cohort_sizes, axis=0)
    return retention


def _pareto_churn_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate churn reason counts and cumulative share."""
    reasons = (
        df.loc[df["churn_flag"] == 1, "churn_reason"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("churn_reason")
        .reset_index(name="count")
    )
    reasons["share"] = reasons["count"] / reasons["count"].sum()
    reasons["cumulative_share"] = reasons["share"].cumsum()
    return reasons


def run_analysis(clean_path: Path = CLEAN_DATA_FILE) -> dict[str, Any]:
    """Generate the core analytical outputs for the project."""
    ensure_directories()
    df = pd.read_csv(clean_path, parse_dates=["signup_date", "churn_date", "last_login_date", "cohort_month", "churn_month"])

    summary: dict[str, Any] = {}
    summary["total_customers"] = int(len(df))
    summary["churn_rate"] = float(df["churn_flag"].mean())
    summary["avg_clv"] = float(df["historical_clv"].mean())
    summary["avg_tenure"] = float(df["tenure_months"].mean())
    summary["avg_monthly_charge"] = float(df["monthly_charge"].mean())

    churn_by_plan = df.groupby("plan_type").agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean")).reset_index()
    churn_by_region = df.groupby("region").agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean")).reset_index()
    churn_by_channel = df.groupby("acquisition_channel").agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean")).reset_index()
    churn_by_payment = df.groupby("payment_method").agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean")).reset_index()

    monthly_trend = _monthly_churn_trend(df)
    reasons = _pareto_churn_reasons(df)
    retention = _cohort_retention_table(df)
    clv_churn = df.groupby("clv_tier").agg(customers=("customer_id", "count"), churn_rate=("churn_flag", "mean"), avg_clv=("historical_clv", "mean")).reset_index()

    usage_corr = float(df[["usage_frequency", "churn_flag"]].corr(method="spearman").iloc[0, 1])
    support_corr = float(df[["support_tickets_raised", "churn_flag"]].corr(method="spearman").iloc[0, 1])
    summary["usage_churn_spearman"] = usage_corr
    summary["support_churn_spearman"] = support_corr
    summary["top_reason_1"] = str(reasons.iloc[0]["churn_reason"])
    summary["top_reason_1_share"] = float(reasons.iloc[0]["share"])
    summary["top_reason_2"] = str(reasons.iloc[1]["churn_reason"])
    summary["top_reason_2_share"] = float(reasons.iloc[1]["share"])
    summary["top_two_reason_share"] = float(reasons.head(2)["share"].sum())
    summary["high_value_churn_rate"] = float(clv_churn.loc[clv_churn["clv_tier"] == "High", "churn_rate"].iloc[0])
    summary["high_value_share"] = float((df["clv_tier"] == "High").mean())

    # Overall churn trend by signup month.
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_trend, x="signup_date", y="churn_rate", marker="o", color="#0F6CBD")
    plt.title("Monthly Churn Rate by Signup Cohort")
    plt.xlabel("Signup Month")
    plt.ylabel("Churn Rate")
    plt.ylim(0, max(0.6, monthly_trend["churn_rate"].max() * 1.15))
    plt.xticks(rotation=45)
    _save_figure(ASSETS_DIR / "churn_trend.png")

    # Churn by segment.
    segment_specs = [
        (churn_by_plan, "plan_type", "Churn Rate by Plan Type", "plan_churn.png"),
        (churn_by_region, "region", "Churn Rate by Region", "region_churn.png"),
        (churn_by_channel, "acquisition_channel", "Churn Rate by Acquisition Channel", "channel_churn.png"),
        (churn_by_payment, "payment_method", "Churn Rate by Payment Method", "payment_churn.png"),
    ]
    for segment_df, segment_col, title, filename in segment_specs:
        plt.figure(figsize=(11, 6))
        ordered = segment_df.sort_values("churn_rate", ascending=False)
        sns.barplot(data=ordered, x=segment_col, y="churn_rate", color="#0F6CBD")
        plt.title(title)
        plt.xlabel(segment_col.replace("_", " ").title())
        plt.ylabel("Churn Rate")
        plt.xticks(rotation=25, ha="right")
        _save_figure(ASSETS_DIR / filename)

    # Pareto churn reasons.
    plt.figure(figsize=(11, 6))
    ax1 = plt.gca()
    sns.barplot(data=reasons, x="churn_reason", y="count", color="#8c2d04", ax=ax1)
    ax1.set_ylabel("Churn Count")
    ax1.set_xlabel("Churn Reason")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(reasons["churn_reason"], reasons["cumulative_share"], color="#0F6CBD", marker="o", linewidth=2.5)
    ax2.set_ylabel("Cumulative Share")
    ax2.axhline(0.8, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 1.05)
    plt.title("Pareto Analysis of Churn Reasons")
    _save_figure(ASSETS_DIR / "churn_reason_pareto.png")

    # Usage and support relationship to churn.
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="churn_flag", y="usage_frequency", color="#2a9d8f")
    plt.title("Usage Frequency vs Churn Status")
    plt.xlabel("Churn Flag")
    plt.ylabel("Usage Frequency")
    _save_figure(ASSETS_DIR / "usage_vs_churn.png")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="churn_flag", y="support_tickets_raised", color="#e76f51")
    plt.title("Support Tickets vs Churn Status")
    plt.xlabel("Churn Flag")
    plt.ylabel("Support Tickets Raised")
    _save_figure(ASSETS_DIR / "support_vs_churn.png")

    # Tenure distribution.
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="tenure_months", hue="churn_flag", bins=24, kde=True, element="step", stat="density", common_norm=False)
    plt.title("Tenure Distribution for Churned vs Retained Customers")
    plt.xlabel("Tenure Months")
    plt.ylabel("Density")
    _save_figure(ASSETS_DIR / "tenure_distribution.png")

    # Cohort heatmap.
    plt.figure(figsize=(14, 8))
    sns.heatmap(retention, cmap="Blues", annot=False, cbar_kws={"label": "Retention Rate"})
    plt.title("Monthly Cohort Retention Heatmap")
    plt.xlabel("Months Since Signup")
    plt.ylabel("Signup Cohort Month")
    _save_figure(ASSETS_DIR / "cohort_retention_heatmap.png")

    # CLV tier view.
    plt.figure(figsize=(11, 6))
    ordered_clv = clv_churn.sort_values("churn_rate", ascending=False)
    sns.barplot(data=ordered_clv, x="clv_tier", y="churn_rate", color="#264653")
    plt.title("Churn Rate by CLV Tier")
    plt.xlabel("CLV Tier")
    plt.ylabel("Churn Rate")
    _save_figure(ASSETS_DIR / "clv_tier_churn.png")

    # Save analytical tables for dashboard reuse.
    monthly_trend.to_csv(PROCESSED_DIR / "monthly_churn_trend.csv", index=False)
    churn_by_plan.to_csv(PROCESSED_DIR / "churn_by_plan.csv", index=False)
    churn_by_region.to_csv(PROCESSED_DIR / "churn_by_region.csv", index=False)
    churn_by_channel.to_csv(PROCESSED_DIR / "churn_by_channel.csv", index=False)
    churn_by_payment.to_csv(PROCESSED_DIR / "churn_by_payment.csv", index=False)
    reasons.to_csv(PROCESSED_DIR / "churn_reason_pareto.csv", index=False)
    clv_churn.to_csv(PROCESSED_DIR / "clv_tier_churn.csv", index=False)
    retention.to_csv(PROCESSED_DIR / "cohort_retention_table.csv")

    summary.update(
        {
            "churn_by_plan": churn_by_plan.to_dict(orient="records"),
            "churn_by_region": churn_by_region.to_dict(orient="records"),
            "churn_by_channel": churn_by_channel.to_dict(orient="records"),
            "churn_by_payment": churn_by_payment.to_dict(orient="records"),
            "monthly_trend_last_value": float(monthly_trend["churn_rate"].iloc[-1]),
            "usage_churn_spearman": usage_corr,
            "support_churn_spearman": support_corr,
            "pareto_reasons": reasons.to_dict(orient="records"),
            "cohort_shape": list(retention.shape),
            "clv_tier_churn": clv_churn.to_dict(orient="records"),
        }
    )

    write_json(ANALYSIS_SUMMARY_FILE, summary)
    return summary


if __name__ == "__main__":
    result = run_analysis()
    print(f"Analysis summary written to {ANALYSIS_SUMMARY_FILE}")
