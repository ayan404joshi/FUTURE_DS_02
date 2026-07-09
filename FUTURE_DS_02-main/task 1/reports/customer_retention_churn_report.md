# Customer Retention & Churn Analysis Report

## Executive Summary

The business analyzed 7,500 subscription customers and observed an overall churn rate of 50.7%. The strongest churn drivers were the top two churn reasons, which together explain 59.0% of churn events, while high-value customers churned at 22.3%, above the overall baseline.
The churn model selected random forest as the best performer with ROC-AUC 0.980 and F1 0.935. The dominant quantitative drivers align with the EDA: low usage, elevated support contacts, and price/discount sensitivity.

## Business Problem & Objective

The objective was to identify churn patterns, isolate the retention levers with the highest business value, quantify customer lifetime value, and build a churn prediction model that can support targeted retention actions.

## Data Overview & Methodology

A realistic synthetic subscription dataset was generated with 7,500 customers, including sign-up date, plan, usage, support, payment, region, age group, acquisition channel, discount, upsell, downsell, tenure, churn label, and churn reason. The raw data intentionally contained mixed date formats, duplicates, missing values, and outliers so the cleaning step could demonstrate production-style preparation.
The analysis used pandas for data preparation, seaborn/matplotlib for charts, scikit-learn for modeling, and a class-weight strategy to address the imbalanced churn target without fabricating synthetic category combinations.

## Key Findings

- Overall churn rate: 50.7%
- Average CLV: 452.59
- Average tenure: 12.44 months
- Usage-frequency vs churn Spearman correlation: -0.45
- Support-ticket vs churn Spearman correlation: 0.37

### Churn Trend

Monthly churn trend shows how retention moved across signup cohorts. The latest measured monthly churn rate was 62.1%.

### Segment Differences

Churn varies materially across plan, region, acquisition channel, and payment method. The dashboard and charts in the appendix provide the exact segment ranking used in the recommendation design.

### Cohort Retention

Cohort heatmaps reveal the earliest months after signup as the highest-risk window, which is consistent with onboarding friction and value-realization lag.

### CLV Insights

High-value customers account for 33.3% of the base but churn at 22.3%, which creates a clear revenue risk signal.

## Churn Prediction Model Summary & Top Drivers

The best model was random forest with ROC-AUC 0.980, precision 0.938, recall 0.931, and F1 0.935.
Top drivers were extracted from the fitted model and align with the EDA patterns shown below.

## Top Drivers

- recency_days: 0.5054
- historical_clv: 0.0795
- tenure_months: 0.0759
- usage_frequency: 0.0731
- clv_tier_High: 0.0413
- support_tickets_raised: 0.0335
- monthly_charge: 0.0312
- frequency_per_month: 0.0259
- plan_type_Basic: 0.0253
- tenure_bucket_25-60: 0.0197

## Actionable Recommendations

### Recommendation 1: Attack the top churn reasons first
- Evidence: The top two churn reasons account for 59.0% of all churn events.
- Action: Create a billing-and-product retention playbook for customers citing billing, no usage, price sensitivity, and support issues.
- Estimated impact: If targeted interventions reduce churn among those reasons by 20%, estimated overall churn could drop by about 6.0%.

### Recommendation 2: Protect high-value customers
- Evidence: High CLV customers churn at 22.3% versus 50.7% overall.
- Action: Launch a high-value save desk for premium and high-CLV accounts with proactive renewal outreach and priority support.
- Estimated impact: Closing half of the high-value churn gap could reduce total churn by roughly 0.0% and protect disproportionate revenue.

### Recommendation 3: Intervene earlier on low-usage / high-support accounts
- Evidence: Usage and support show meaningful churn separation (Spearman -0.45 and 0.37).
- Action: Trigger lifecycle nudges, in-app education, and support escalation when engagement drops and ticket volume rises.
- Estimated impact: A 10% reduction in churn among the highest-risk engagement segments could lower churn by about 4.1%.

## Limitations & Future Work

The dataset is synthetic, so the numerical results demonstrate methodology rather than live business performance. A real engagement would add product telemetry, billing history, contract terms, and customer success touches, then validate the model on a time-based holdout set.
Future work should include calibrated probability thresholds, uplift modeling, and retention experiment tracking.

## Appendix

- [assets/churn_trend.png](../assets/churn_trend.png)
- [assets/plan_churn.png](../assets/plan_churn.png)
- [assets/region_churn.png](../assets/region_churn.png)
- [assets/channel_churn.png](../assets/channel_churn.png)
- [assets/payment_churn.png](../assets/payment_churn.png)
- [assets/churn_reason_pareto.png](../assets/churn_reason_pareto.png)
- [assets/usage_vs_churn.png](../assets/usage_vs_churn.png)
- [assets/support_vs_churn.png](../assets/support_vs_churn.png)
- [assets/tenure_distribution.png](../assets/tenure_distribution.png)
- [assets/cohort_retention_heatmap.png](../assets/cohort_retention_heatmap.png)
- [assets/clv_tier_churn.png](../assets/clv_tier_churn.png)
- [processed/top_churn_drivers.png](../data/processed/top_churn_drivers.png)