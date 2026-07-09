"""Shared helpers for the retention and churn analysis project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

RAW_DATA_FILE = RAW_DIR / "subscription_customers_raw.csv"
CLEAN_DATA_FILE = PROCESSED_DIR / "subscription_customers_clean.csv"
ANALYSIS_SUMMARY_FILE = PROCESSED_DIR / "analysis_summary.json"
CLEANING_LOG_FILE = REPORTS_DIR / "data_cleaning_log.md"
MODEL_CARD_FILE = REPORTS_DIR / "model_card.md"
REPORT_MD_FILE = REPORTS_DIR / "customer_retention_churn_report.md"
REPORT_DOCX_FILE = REPORTS_DIR / "customer_retention_churn_report.docx"
MODEL_FILE = PROCESSED_DIR / "churn_model.joblib"
PREPROCESSOR_FILE = PROCESSED_DIR / "churn_preprocessor.joblib"
FEATURES_FILE = PROCESSED_DIR / "model_feature_names.json"


def ensure_directories() -> None:
    """Create the standard project folders if they do not already exist."""
    for directory in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        NOTEBOOKS_DIR,
        SCRIPTS_DIR,
        DASHBOARD_DIR,
        REPORTS_DIR,
        ASSETS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """Serialize a Python object as pretty-printed JSON."""
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def safe_rate(numerator: float, denominator: float) -> float:
    """Compute a rate while guarding against divide-by-zero."""
    return float(numerator) / float(denominator) if denominator else 0.0
