"""Train churn prediction models and export evaluation artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import FEATURES_FILE, MODEL_FILE, MODEL_CARD_FILE, PREPROCESSOR_FILE, PROCESSED_DIR, ensure_directories, write_json


NUMERIC_FEATURES = [
    "monthly_charge",
    "tenure_months",
    "usage_frequency",
    "support_tickets_raised",
    "upsell_count",
    "downsell_count",
    "discount_applied",
    "discount_pct",
    "recency_days",
    "frequency_per_month",
    "historical_clV",
]

CATEGORICAL_FEATURES = [
    "plan_type",
    "payment_method",
    "region",
    "age_group",
    "acquisition_channel",
    "tenure_bucket",
    "clv_tier",
]

# The feature name uses a typo-safe alias to keep the list readable in downstream tables.
NUMERIC_FEATURES = [feature.replace("historical_clV", "historical_clv") for feature in NUMERIC_FEATURES]
TARGET = "churn_flag"


def _build_preprocessor() -> ColumnTransformer:
    """Create the feature engineering pipeline used by both models."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract transformed feature names for downstream importance reporting."""
    num_names = NUMERIC_FEATURES
    cat_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return list(num_names) + cat_names


def train_models(clean_path: Path = PROCESSED_DIR / "subscription_customers_clean.csv") -> dict[str, Any]:
    """Train logistic regression and random forest churn models."""
    ensure_directories()
    df = pd.read_csv(clean_path)

    feature_frame = df[
        [
            "monthly_charge",
            "tenure_months",
            "usage_frequency",
            "support_tickets_raised",
            "upsell_count",
            "downsell_count",
            "discount_applied",
            "discount_pct",
            "recency_days",
            "frequency_per_month",
            "historical_clv",
            "plan_type",
            "payment_method",
            "region",
            "age_group",
            "acquisition_channel",
            "tenure_bucket",
            "clv_tier",
        ]
    ].copy()
    target = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    preprocessor = _build_preprocessor()

    logistic_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)),
        ]
    )
    random_forest_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced_subsample",
                    max_depth=10,
                    min_samples_leaf=8,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    logistic_proba = logistic_model.predict_proba(X_test)[:, 1]
    rf_proba = random_forest_model.predict_proba(X_test)[:, 1]

    logistic_pred = (logistic_proba >= 0.5).astype(int)
    rf_pred = (rf_proba >= 0.5).astype(int)

    metrics = {
        "logistic_regression": {
            "accuracy": float(accuracy_score(y_test, logistic_pred)),
            "precision": float(precision_score(y_test, logistic_pred, zero_division=0)),
            "recall": float(recall_score(y_test, logistic_pred, zero_division=0)),
            "f1": float(f1_score(y_test, logistic_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, logistic_proba)),
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
            "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
            "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, rf_proba)),
        },
    }

    best_name = max(metrics, key=lambda key: metrics[key]["roc_auc"])
    best_model = random_forest_model if best_name == "random_forest" else logistic_model

    dump(best_model, MODEL_FILE)
    dump(best_model.named_steps["preprocessor"], PREPROCESSOR_FILE)

    feature_names = _get_feature_names(best_model.named_steps["preprocessor"])
    write_json(FEATURES_FILE, feature_names)

    if best_name == "random_forest":
        importances = best_model.named_steps["classifier"].feature_importances_
    else:
        importances = np.abs(best_model.named_steps["classifier"].coef_[0])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    importance_df.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    top_importance = importance_df.head(12)
    plt.figure(figsize=(12, 7))
    sns_bar = plt.barh(top_importance["feature"], top_importance["importance"], color="#0F6CBD")
    plt.gca().invert_yaxis()
    plt.title(f"Top Churn Drivers: {best_name.replace('_', ' ').title()}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "top_churn_drivers.png", dpi=220, bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, rf_pred if best_name == "random_forest" else logistic_pred)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    for (row, col), value in np.ndenumerate(cm):
        plt.text(col, row, int(value), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()

    write_json(
        PROCESSED_DIR / "model_metrics.json",
        {
            "best_model": best_name,
            "metrics": metrics,
        },
    )

    model_card = [
        "# Churn Prediction Model Card",
        "",
        f"- Training target: `churn_flag`.",
        f"- Best model selected by ROC-AUC: {best_name}.",
        "- Imbalance strategy: class weights rather than SMOTE, because the feature set contains mixed categorical and numeric variables and the priority was to preserve business interpretability without generating synthetic category combinations.",
        "- Evaluation split: 75/25 stratified train-test split with random state 42.",
        "",
        "## Performance",
        "",
        f"- Logistic Regression ROC-AUC: {metrics['logistic_regression']['roc_auc']:.3f}",
        f"- Random Forest ROC-AUC: {metrics['random_forest']['roc_auc']:.3f}",
        f"- Best model F1: {metrics[best_name]['f1']:.3f}",
        f"- Best model precision: {metrics[best_name]['precision']:.3f}",
        f"- Best model recall: {metrics[best_name]['recall']:.3f}",
        "",
        "## Key Features",
        "",
        "- Usage frequency and support tickets are expected to dominate the top rankings.",
        "- Price and discount sensitivity are represented through monthly charge, discount rate, and plan type.",
        "- Recency and tenure capture customer engagement decay over time.",
        "",
        "## Limitations",
        "",
        "- The model is trained on synthetic data and should be treated as a methodology demonstration rather than a production-ready classifier.",
        "- Churn labels are event-defined within the generated dataset and may not capture every business nuance found in a live billing system.",
        "- The current workflow optimizes explanation and reproducibility over hyperparameter search depth.",
    ]
    MODEL_CARD_FILE.write_text("\n".join(model_card), encoding="utf-8")

    return {
        "metrics": metrics,
        "best_model": best_name,
        "top_features": top_importance.to_dict(orient="records"),
        "classification_report": classification_report(y_test, rf_pred if best_name == "random_forest" else logistic_pred, output_dict=True),
    }


if __name__ == "__main__":
    results = train_models()
    print(f"Model metrics written to {PROCESSED_DIR / 'model_metrics.json'}")
