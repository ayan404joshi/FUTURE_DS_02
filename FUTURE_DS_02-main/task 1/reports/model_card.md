# Churn Prediction Model Card

- Training target: `churn_flag`.
- Best model selected by ROC-AUC: random_forest.
- Imbalance strategy: class weights rather than SMOTE, because the feature set contains mixed categorical and numeric variables and the priority was to preserve business interpretability without generating synthetic category combinations.
- Evaluation split: 75/25 stratified train-test split with random state 42.

## Performance

- Logistic Regression ROC-AUC: 0.962
- Random Forest ROC-AUC: 0.980
- Best model F1: 0.935
- Best model precision: 0.938
- Best model recall: 0.931

## Key Features

- Usage frequency and support tickets are expected to dominate the top rankings.
- Price and discount sensitivity are represented through monthly charge, discount rate, and plan type.
- Recency and tenure capture customer engagement decay over time.

## Limitations

- The model is trained on synthetic data and should be treated as a methodology demonstration rather than a production-ready classifier.
- Churn labels are event-defined within the generated dataset and may not capture every business nuance found in a live billing system.
- The current workflow optimizes explanation and reproducibility over hyperparameter search depth.