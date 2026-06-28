## Overview
This project implements a production-grade credit scoring engine built on the HMEQ (Home Equity Mortgage Questionnaire) dataset. It estimates the Probability of Default (PD) for home equity loan applications and delivers a real-time credit decision through a Flask REST API and browser dashboard.
-Data ingestion from CSV or MySQL
-Feature engineering — six domain-driven financial ratios computed at both training and inference time
-Multiple model families — Logistic Regression, Voting Ensemble, and XGBoost Stacking — all tracked via MLflow
-Real-time scoring through a Flask REST API that returns a probability, a simulated credit score, a risk band, and an automated decision on every call
-Explainability via SHAP reason codes that identify the top risk drivers and mitigating factors for each individual prediction
-Post-deployment monitoring using Population Stability Index (PSI) and KS statistics to detect data and score drift over time
-A browser dashboard with panels for real-time scoring, model monitoring, and EDA analytics
## Dashboard preview

