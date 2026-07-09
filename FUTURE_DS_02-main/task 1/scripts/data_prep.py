"""Clean and prepare the synthetic subscription dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser as date_parser

from common import CLEANING_LOG_FILE, CLEAN_DATA_FILE, RAW_DATA_FILE, ensure_directories, write_json


@dataclass(frozen=True)
class CleaningSummary:
    """Track key data quality actions for the cleaning log."""

    raw_rows: int
    raw_columns: int
    duplicate_rows_removed: int
    missing_values_imputed: dict[str, int]
    outliers_capped: dict[str, int]
    cleaned_rows: int
    cleaned_columns: int


DATE_COLUMNS = ["signup_date", "churn_date", "last_login_date"]
CATEGORICAL_COLUMNS = [
    "plan_type",
    "churn_reason",
    "payment_method",
    "region",
    "age_group",
    "acquisition_channel",
]
NUMERIC_COLUMNS = [
    "monthly_charge",
    "tenure_months",
    "usage_frequency",
    "support_tickets_raised",
    "upsell_count",
    "downsell_count",
    "discount_applied",
    "discount_pct",
]


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse date strings that may use multiple raw-export formats."""
    def _parse_one(value: Any) -> pd.Timestamp:
        if pd.isna(value):
            return pd.NaT
        text = str(value).strip()
        for dayfirst in (True, False):
            try:
                return pd.Timestamp(date_parser.parse(text, dayfirst=dayfirst, fuzzy=True))
            except (ValueError, TypeError, OverflowError):
                continue
        return pd.NaT

    return series.apply(_parse_one)


def _cap_outliers(series: pd.Series) -> tuple[pd.Series, int]:
    """Winsorize extreme numeric values using the 1.5 IQR rule."""
    if series.dropna().empty:
        return series, 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    capped = series.clip(lower=lower, upper=upper)
    changed = int((series != capped).sum())
    return capped, changed


def clean_dataset(raw_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Clean raw data and create analysis-ready derived fields."""
    ensure_directories()
    raw = pd.read_csv(raw_path)
    raw_rows, raw_columns = raw.shape

    cleaned = raw.copy()
    duplicate_rows_removed = int(cleaned.duplicated(subset=["customer_id"], keep="first").sum())
    cleaned = cleaned.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)

    for column in DATE_COLUMNS:
        cleaned[column] = _parse_mixed_dates(cleaned[column])

    cleaned["churn_flag"] = cleaned["churn_flag"].fillna(0).astype(int)
    cleaned["discount_applied"] = cleaned["discount_applied"].fillna(0).astype(int)
    cleaned["discount_pct"] = cleaned["discount_pct"].fillna(0).astype(float)

    missing_values_imputed: dict[str, int] = {}
    for column in CATEGORICAL_COLUMNS:
        mode_value = cleaned[column].mode(dropna=True)
        fill_value = str(mode_value.iloc[0]) if not mode_value.empty else "Unknown"
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            cleaned[column] = cleaned[column].fillna(fill_value)
        missing_values_imputed[column] = missing_count

    for column in ["support_tickets_raised", "usage_frequency", "tenure_months", "upsell_count", "downsell_count"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        missing_count = int(cleaned[column].isna().sum())
        if missing_count:
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())
        missing_values_imputed[column] = missing_count

    cleaned["monthly_charge"] = pd.to_numeric(cleaned["monthly_charge"], errors="coerce")
    monthly_missing = int(cleaned["monthly_charge"].isna().sum())
    if monthly_missing:
        cleaned["monthly_charge"] = cleaned["monthly_charge"].fillna(cleaned.groupby("plan_type")["monthly_charge"].transform("median"))
        cleaned["monthly_charge"] = cleaned["monthly_charge"].fillna(cleaned["monthly_charge"].median())
    missing_values_imputed["monthly_charge"] = monthly_missing

    missing_values_imputed["last_login_date"] = int(cleaned["last_login_date"].isna().sum())
    missing_login_mask = cleaned["last_login_date"].isna()
    cleaned.loc[missing_login_mask, "last_login_date"] = cleaned.loc[missing_login_mask, "signup_date"] + pd.to_timedelta(
        np.maximum(1, cleaned.loc[missing_login_mask, "tenure_months"] * 10), unit="D"
    )
    cleaned["last_login_date"] = cleaned["last_login_date"].fillna(cleaned["signup_date"])

    cleaned["signup_date"] = cleaned["signup_date"].fillna(cleaned["signup_date"].mode().iloc[0])
    cleaned["churn_date"] = cleaned["churn_date"].fillna(pd.NaT)
    cleaned["tenure_months"] = cleaned["tenure_months"].round().clip(lower=1).astype(int)
    cleaned["usage_frequency"] = cleaned["usage_frequency"].round().clip(lower=0).astype(int)
    cleaned["support_tickets_raised"] = cleaned["support_tickets_raised"].round().clip(lower=0).astype(int)
    cleaned["upsell_count"] = cleaned["upsell_count"].round().clip(lower=0).astype(int)
    cleaned["downsell_count"] = cleaned["downsell_count"].round().clip(lower=0).astype(int)

    outliers_capped: dict[str, int] = {}
    for column in ["usage_frequency", "support_tickets_raised", "monthly_charge", "discount_pct"]:
        cleaned[column], changed = _cap_outliers(cleaned[column].astype(float))
        outliers_capped[column] = changed

    cleaned["discount_pct"] = cleaned["discount_pct"].clip(lower=0, upper=0.5).fillna(0)
    cleaned["monthly_charge"] = cleaned["monthly_charge"].clip(lower=5)

    observation_end = cleaned["churn_date"].fillna(cleaned["last_login_date"]).max()
    if pd.isna(observation_end):
        observation_end = cleaned["signup_date"].max()

    cleaned["cohort_month"] = cleaned["signup_date"].dt.to_period("M").dt.to_timestamp()
    cleaned["tenure_bucket"] = pd.cut(
        cleaned["tenure_months"],
        bins=[0, 3, 6, 12, 24, 60, np.inf],
        labels=["0-3", "4-6", "7-12", "13-24", "25-60", "60+"],
        include_lowest=True,
        right=True,
    ).astype(str)
    recency_days = (observation_end - cleaned["last_login_date"]).dt.days
    cleaned["recency_days"] = recency_days.fillna(0).clip(lower=0).astype(int)
    cleaned["frequency_per_month"] = (cleaned["usage_frequency"] / cleaned["tenure_months"].replace(0, np.nan)).fillna(0)
    cleaned["historical_clv"] = (cleaned["monthly_charge"] * (1 - cleaned["discount_pct"]) * cleaned["tenure_months"]).round(2)

    cleaned["clv_tier"] = pd.qcut(
        cleaned["historical_clv"],
        q=3,
        labels=["Low", "Medium", "High"],
        duplicates="drop",
    )
    cleaned["clv_tier"] = cleaned["clv_tier"].astype(str)

    cleaned["signup_year_month"] = cleaned["signup_date"].dt.strftime("%Y-%m")
    cleaned["churn_month"] = cleaned["churn_date"].dt.to_period("M").dt.to_timestamp()
    cleaned["churn_month"] = cleaned["churn_month"].astype("datetime64[ns]")

    cleaned = cleaned.sort_values(["signup_date", "customer_id"]).reset_index(drop=True)
    cleaned.to_csv(CLEAN_DATA_FILE, index=False)

    summary = CleaningSummary(
        raw_rows=raw_rows,
        raw_columns=raw_columns,
        duplicate_rows_removed=duplicate_rows_removed,
        missing_values_imputed=missing_values_imputed,
        outliers_capped=outliers_capped,
        cleaned_rows=len(cleaned),
        cleaned_columns=cleaned.shape[1],
    )
    _write_cleaning_log(summary)
    write_json(
        CLEANING_LOG_FILE.with_suffix(".json"),
        {
            "raw_rows": summary.raw_rows,
            "raw_columns": summary.raw_columns,
            "duplicate_rows_removed": summary.duplicate_rows_removed,
            "missing_values_imputed": summary.missing_values_imputed,
            "outliers_capped": summary.outliers_capped,
            "cleaned_rows": summary.cleaned_rows,
            "cleaned_columns": summary.cleaned_columns,
        },
    )
    return cleaned


def _write_cleaning_log(summary: CleaningSummary) -> None:
    """Write a human-readable markdown log of cleaning decisions."""
    lines = [
        "# Data Cleaning Log",
        "",
        f"- Raw rows: {summary.raw_rows:,}",
        f"- Raw columns: {summary.raw_columns}",
        f"- Duplicate customer rows removed: {summary.duplicate_rows_removed:,}",
        f"- Final cleaned rows: {summary.cleaned_rows:,}",
        f"- Final cleaned columns: {summary.cleaned_columns}",
        "",
        "## Missing Value Strategy",
        "",
        "- Categorical fields were imputed with the modal category to preserve segment structure.",
        "- Numeric fields were imputed with the median because the synthetic subscription data is intentionally skewed.",
        "- Missing `last_login_date` values were reconstructed from `signup_date` and tenure to preserve recency metrics.",
        "",
        "## Outlier Handling",
        "",
        "- `usage_frequency`, `support_tickets_raised`, `monthly_charge`, and `discount_pct` were winsorized using the 1.5 IQR rule.",
        "- `discount_pct` was clipped to a business-plausible range between 0% and 50%.",
        "",
        "## Derived Fields",
        "",
        "- `cohort_month`: signup month used for retention cohort analysis.",
        "- `tenure_bucket`: coarse tenure band for retention segmentation.",
        "- `recency_days`: days since the most recent login.",
        "- `frequency_per_month`: normalized usage intensity.",
        "- `historical_clv`: discounted lifetime revenue proxy.",
        "- `clv_tier`: low, medium, and high value customer bands.",
        "",
        "## Notes",
        "",
        "- The raw file intentionally included mixed date formats, duplicates, and small missingness patterns so the cleaning step could demonstrate production-style data preparation.",
    ]
    CLEANING_LOG_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    dataset = clean_dataset()
    print(f"Cleaned dataset written to {CLEAN_DATA_FILE} with {len(dataset):,} rows")
