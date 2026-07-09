# Data Cleaning Log

- Raw rows: 7,650
- Raw columns: 19
- Duplicate customer rows removed: 150
- Final cleaned rows: 7,500
- Final cleaned columns: 27

## Missing Value Strategy

- Categorical fields were imputed with the modal category to preserve segment structure.
- Numeric fields were imputed with the median because the synthetic subscription data is intentionally skewed.
- Missing `last_login_date` values were reconstructed from `signup_date` and tenure to preserve recency metrics.

## Outlier Handling

- `usage_frequency`, `support_tickets_raised`, `monthly_charge`, and `discount_pct` were winsorized using the 1.5 IQR rule.
- `discount_pct` was clipped to a business-plausible range between 0% and 50%.

## Derived Fields

- `cohort_month`: signup month used for retention cohort analysis.
- `tenure_bucket`: coarse tenure band for retention segmentation.
- `recency_days`: days since the most recent login.
- `frequency_per_month`: normalized usage intensity.
- `historical_clv`: discounted lifetime revenue proxy.
- `clv_tier`: low, medium, and high value customer bands.

## Notes

- The raw file intentionally included mixed date formats, duplicates, and small missingness patterns so the cleaning step could demonstrate production-style data preparation.