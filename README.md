# FUTURE_DS_02
Customer Retention & Churn Analysis
End-to-end customer retention and churn analysis for a subscription business. The project includes synthetic data generation, cleaning, EDA, cohort analysis, CLV analysis, churn modeling, an interactive dashboard, and a formal Word report.

Key Results
Customers analyzed: 7,500
Overall churn rate: 50.7%
Average CLV: $452.59
Average tenure: 12.4 months
Best churn model: Random Forest
Best model ROC-AUC: 0.980
Top churn reason: billing (43.5% of churn)
Top two churn reasons: 59.0% of churn events
Tech Stack
Python, pandas, numpy
matplotlib, seaborn, plotly
scikit-learn, imbalanced-learn
Streamlit for the dashboard
python-docx for the Word report
Folder Structure
data/raw - raw synthetic dataset
data/processed - cleaned dataset, model outputs, and analytical tables
notebooks - notebook summary of the project
scripts - pipeline code for generation, cleaning, analysis, modeling, and reporting
dashboard - Streamlit app
reports - cleaning log, model card, and final report
assets - exported charts used by the report and dashboard
How to Run
Install dependencies.
pip install -r requirements.txt
Run the full pipeline.
python scripts/run_pipeline.py
Launch the dashboard.
streamlit run dashboard/app.py
Outputs
data/raw/subscription_customers_raw.csv
data/processed/subscription_customers_clean.csv
data/processed/analysis_summary.json
data/processed/model_metrics.json
data/processed/feature_importance.csv
data/processed/churn_model.joblib
reports/data_cleaning_log.md
reports/model_card.md
reports/customer_retention_churn_report.docx
reports/customer_retention_churn_report.md
assets/*.png
Dashboard Preview
A representative preview is available in assets/churn_trend.png and the dashboard is fully interactive in Streamlit.

Notes
The dataset is synthetic but intentionally structured to reflect realistic churn relationships.
Class imbalance is handled with class weights rather than SMOTE to preserve interpretability across mixed feature types.
Every report claim is backed by a chart or a number exported by the pipeline.
