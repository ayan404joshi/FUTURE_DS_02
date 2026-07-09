"""Generate a realistic synthetic subscription customer dataset."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import RAW_DATA_FILE, ensure_directories


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for the synthetic dataset generator."""

    n_customers: int = 7500
    start_date: str = "2023-01-01"
    end_date: str = "2025-06-30"
    random_seed: int = 42


PLAN_TYPES = ["Basic", "Standard", "Premium"]
REGIONS = ["North America", "Europe", "Latin America", "APAC"]
CHANNELS = ["Organic Search", "Paid Ads", "Referral", "Direct", "Partnerships", "Email"]
PAYMENT_METHODS = ["Card", "ACH", "PayPal", "Wallet"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
CHURN_REASONS = [
    "billing",
    "competitor",
    "product dissatisfaction",
    "no usage",
    "price sensitivity",
    "support issues",
]


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return whole months between two dates."""
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def _random_dates(rng: np.random.Generator, config: GenerationConfig) -> pd.DatetimeIndex:
    """Sample signup dates uniformly across the configured window."""
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    span_days = (end - start).days
    offsets = rng.integers(0, span_days + 1, size=config.n_customers)
    return pd.to_datetime(start + pd.to_timedelta(offsets, unit="D"))


def _format_mixed_date(value: pd.Timestamp, rng: np.random.Generator) -> str:
    """Render dates using several formats to mimic messy raw exports."""
    styles = ["iso", "slash", "month_name", "day_first"]
    style = rng.choice(styles, p=[0.45, 0.2, 0.2, 0.15])
    if style == "iso":
        return value.strftime("%Y-%m-%d")
    if style == "slash":
        return value.strftime("%m/%d/%Y")
    if style == "month_name":
        return value.strftime("%b %d, %Y")
    return value.strftime("%d-%m-%Y")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Stable sigmoid helper."""
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def _build_usage_and_tickets(
    rng: np.random.Generator,
    plan_type: str,
    channel: str,
    age_group: str,
    region: str,
) -> tuple[int, int]:
    """Create usage and support activity with realistic business relationships."""
    plan_usage = {"Basic": 6.0, "Standard": 10.0, "Premium": 14.0}[plan_type]
    channel_adjustment = {
        "Organic Search": 0.5,
        "Paid Ads": -0.5,
        "Referral": 1.0,
        "Direct": 0.8,
        "Partnerships": 1.1,
        "Email": 0.2,
    }[channel]
    age_adjustment = {
        "18-24": 1.2,
        "25-34": 1.0,
        "35-44": 0.6,
        "45-54": -0.2,
        "55+": -0.8,
    }[age_group]
    region_adjustment = {"North America": 0.8, "Europe": 0.4, "Latin America": -0.6, "APAC": 0.2}[region]
    usage_base = plan_usage + channel_adjustment + age_adjustment + region_adjustment
    usage = int(np.clip(rng.normal(loc=usage_base, scale=2.2), 0, 25))
    tickets_lambda = max(0.2, 2.4 - 0.12 * usage + (0.3 if plan_type == "Basic" else 0) - (0.15 if plan_type == "Premium" else 0))
    tickets = int(np.clip(rng.poisson(lam=tickets_lambda), 0, 12))
    return usage, tickets


def _build_churn_probability(
    usage_frequency: np.ndarray,
    support_tickets: np.ndarray,
    monthly_charge: np.ndarray,
    discount_pct: np.ndarray,
    plan_type: Iterable[str],
    channel: Iterable[str],
    tenure_proxy: np.ndarray,
) -> np.ndarray:
    """Construct a churn probability with interpretable business drivers."""
    plan_flag = np.array([1.0 if plan == "Basic" else 0.0 for plan in plan_type])
    premium_flag = np.array([1.0 if plan == "Premium" else 0.0 for plan in plan_type])
    paid_ads_flag = np.array([1.0 if ch == "Paid Ads" else 0.0 for ch in channel])
    referral_flag = np.array([1.0 if ch == "Referral" else 0.0 for ch in channel])

    score = (
        0.7
        - 0.22 * usage_frequency
        + 0.46 * support_tickets
        + 0.025 * monthly_charge
        + 2.0 * discount_pct
        + 0.85 * plan_flag
        - 0.5 * premium_flag
        + 0.4 * paid_ads_flag
        - 0.25 * referral_flag
        - 0.03 * tenure_proxy
    )
    return _sigmoid(score)


def _assign_churn_reason(
    rng: np.random.Generator,
    usage_frequency: int,
    support_tickets: int,
    monthly_charge: float,
    discount_pct: float,
    plan_type: str,
) -> str:
    """Assign a churn reason using the strongest driver for each customer."""
    reason_weights = np.array([
        0.9 + 0.12 * monthly_charge,
        0.5 + (1.2 if plan_type == "Basic" else 0.2),
        0.8 + (1.5 if support_tickets >= 4 else 0.0),
        1.4 + (1.8 if usage_frequency <= 3 else 0.0),
        1.0 + (1.5 if discount_pct <= 0.05 else 0.0),
        0.7 + (1.2 if support_tickets >= 3 else 0.0),
    ])
    reason_weights = reason_weights / reason_weights.sum()
    return str(rng.choice(CHURN_REASONS, p=reason_weights))


def generate_synthetic_dataset(config: GenerationConfig | None = None) -> pd.DataFrame:
    """Generate the raw synthetic dataset with intentional imperfections."""
    ensure_directories()
    config = config or GenerationConfig()
    rng = np.random.default_rng(config.random_seed)

    signup_dates = _random_dates(rng, config)
    plan_type = rng.choice(PLAN_TYPES, size=config.n_customers, p=[0.42, 0.38, 0.20])
    region = rng.choice(REGIONS, size=config.n_customers, p=[0.38, 0.24, 0.18, 0.20])
    acquisition_channel = rng.choice(CHANNELS, size=config.n_customers, p=[0.25, 0.18, 0.16, 0.18, 0.11, 0.12])
    age_group = rng.choice(AGE_GROUPS, size=config.n_customers, p=[0.16, 0.32, 0.24, 0.16, 0.12])
    payment_method = rng.choice(PAYMENT_METHODS, size=config.n_customers, p=[0.64, 0.12, 0.16, 0.08])

    base_charge = {"Basic": 19.0, "Standard": 39.0, "Premium": 79.0}
    monthly_charge = np.array([
        np.round(rng.normal(loc=base_charge[plan], scale={"Basic": 4.0, "Standard": 6.0, "Premium": 10.0}[plan]), 2)
        for plan in plan_type
    ])
    monthly_charge = np.clip(monthly_charge, 9.99, 129.0)

    discount_pct = np.where(
        rng.random(config.n_customers) < np.where(plan_type == "Premium", 0.38, np.where(plan_type == "Standard", 0.28, 0.18)),
        rng.uniform(0.05, 0.30, size=config.n_customers),
        rng.uniform(0.00, 0.08, size=config.n_customers),
    )
    discount_applied = (discount_pct > 0.0).astype(int)

    usage_frequency = np.zeros(config.n_customers, dtype=int)
    support_tickets_raised = np.zeros(config.n_customers, dtype=int)
    for idx in range(config.n_customers):
        usage_frequency[idx], support_tickets_raised[idx] = _build_usage_and_tickets(
            rng, plan_type[idx], acquisition_channel[idx], age_group[idx], region[idx]
        )

    observation_end = pd.Timestamp(config.end_date)
    tenure_proxy_months = np.array([
        _months_between(pd.Timestamp(date), observation_end) for date in signup_dates
    ])
    churn_probability = _build_churn_probability(
        usage_frequency=usage_frequency,
        support_tickets=support_tickets_raised,
        monthly_charge=monthly_charge,
        discount_pct=discount_pct,
        plan_type=plan_type,
        channel=acquisition_channel,
        tenure_proxy=tenure_proxy_months,
    )

    churn_flag = (rng.random(config.n_customers) < churn_probability).astype(int)

    tenure_months = np.where(
        churn_flag == 1,
        np.maximum(1, np.round(np.minimum(tenure_proxy_months, rng.normal(loc=tenure_proxy_months * 0.58 + 2.0, scale=3.2)))).astype(int),
        np.maximum(1, tenure_proxy_months),
    )

    churn_date = []
    last_login_date = []
    churn_reason = []
    upsell_count = []
    downsell_count = []

    for idx in range(config.n_customers):
        signup = pd.Timestamp(signup_dates[idx])
        tenure_days = int(np.clip(tenure_months[idx] * 30.4, 30, 900))
        end_of_tenure = min(signup + pd.Timedelta(days=tenure_days), observation_end)
        if churn_flag[idx] == 1:
            churn_offset = rng.integers(low=max(14, int(tenure_days * 0.35)), high=max(15, tenure_days))
            churn_dt = min(signup + pd.Timedelta(days=int(churn_offset)), observation_end)
            churn_date.append(churn_dt)
            churn_reason.append(
                _assign_churn_reason(
                    rng,
                    usage_frequency[idx],
                    support_tickets_raised[idx],
                    monthly_charge[idx],
                    discount_pct[idx],
                    plan_type[idx],
                )
            )
            last_login_offset = int(np.clip(rng.normal(loc=22 + 4 * support_tickets_raised[idx], scale=9), 0, 120))
            last_login_date.append(max(signup, churn_dt - pd.Timedelta(days=last_login_offset)))
        else:
            churn_date.append(pd.NaT)
            churn_reason.append(None)
            last_login_gap = int(np.clip(rng.normal(loc=7 + 10 / (usage_frequency[idx] + 1), scale=5), 0, 40))
            last_login_date.append(max(signup, end_of_tenure - pd.Timedelta(days=last_login_gap)))

        upsell_base = max(0, tenure_months[idx] // 8 + (1 if plan_type[idx] == "Premium" else 0))
        downsell_base = max(0, 1 if churn_flag[idx] == 1 and discount_pct[idx] < 0.1 else 0)
        upsell_count.append(int(np.clip(rng.poisson(lam=max(0.2, upsell_base / 2 + 0.25 * usage_frequency[idx])), 0, 5)))
        downsell_count.append(int(np.clip(rng.poisson(lam=max(0.1, downsell_base + 0.18 * (4 - usage_frequency[idx]))), 0, 4)))

    df = pd.DataFrame(
        {
            "customer_id": [f"C{idx:05d}" for idx in range(1, config.n_customers + 1)],
            "signup_date": [_format_mixed_date(date, rng) for date in signup_dates],
            "plan_type": plan_type,
            "monthly_charge": monthly_charge,
            "tenure_months": tenure_months,
            "churn_flag": churn_flag,
            "churn_date": [
                _format_mixed_date(date, rng) if pd.notna(date) else None for date in churn_date
            ],
            "churn_reason": churn_reason,
            "usage_frequency": usage_frequency,
            "support_tickets_raised": support_tickets_raised,
            "last_login_date": [_format_mixed_date(date, rng) for date in last_login_date],
            "payment_method": payment_method,
            "region": region,
            "age_group": age_group,
            "acquisition_channel": acquisition_channel,
            "upsell_count": upsell_count,
            "downsell_count": downsell_count,
            "discount_applied": discount_applied,
            "discount_pct": np.round(discount_pct, 3),
        }
    )

    # Introduce a small amount of realistic raw-data messiness.
    missing_map = {
        "payment_method": 0.012,
        "churn_reason": 0.04,
        "support_tickets_raised": 0.015,
        "last_login_date": 0.012,
        "discount_pct": 0.02,
    }
    for column, fraction in missing_map.items():
        missing_index = rng.choice(df.index, size=max(1, int(len(df) * fraction)), replace=False)
        df.loc[missing_index, column] = np.nan

    duplicate_rows = df.sample(frac=0.02, random_state=config.random_seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    RAW_DATA_FILE.write_text(df.to_csv(index=False), encoding="utf-8")
    return df


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    print(f"Generated raw dataset with {len(dataset):,} rows at {RAW_DATA_FILE}")
