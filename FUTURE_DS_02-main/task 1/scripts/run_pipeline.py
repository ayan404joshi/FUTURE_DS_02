"""Run the complete customer retention and churn analysis pipeline."""
from __future__ import annotations

from data_generation import GenerationConfig, generate_synthetic_dataset
from data_prep import clean_dataset
from analysis import run_analysis
from modeling import train_models
from report_builder import build_report


def main() -> None:
    """Execute all phases sequentially and print a concise summary."""
    generate_synthetic_dataset(GenerationConfig(n_customers=7500, random_seed=42))
    clean_dataset()
    analysis_summary = run_analysis()
    model_summary = train_models()
    report_summary = build_report()

    print("Pipeline complete.")
    print(f"Customers analyzed: {analysis_summary['total_customers']:,}")
    print(f"Overall churn rate: {analysis_summary['churn_rate']:.2%}")
    print(f"Best churn model: {model_summary['best_model']}")
    print(f"Report DOCX: {report_summary['docx']}")


if __name__ == "__main__":
    main()
