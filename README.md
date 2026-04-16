# Telco Customer Churn Prediction

This project is an end-to-end telco churn prediction workflow built to identify customers at risk of leaving, explain the main drivers behind that risk and translate model outputs into practical retention recommendations. It combines exploratory analysis, supervised machine learning, SHAP-based interpretability, a saved prediction pipeline and a small FastAPI serving layer so the work reads as both a data science case study and a deployable decision-support prototype.

## Business Problem

Customer churn is a core issue for subscription-based telecom businesses. When customers leave, the company loses recurring revenue and may fail to recover acquisition and onboarding costs. A useful churn model should therefore do more than predict a binary outcome: it should help retention teams prioritize outreach, understand why a customer appears risky and decide which actions are worth reviewing.

## Project Goals

- Identify customers with high churn risk using a reproducible machine learning pipeline
- Analyze churn patterns across tenure, contract type, services, billing behavior and customer profile
- Explain model predictions globally and at the customer level using SHAP
- Convert actionable churn drivers into retention recommendation ideas
- Expose the saved churn pipeline through a lightweight FastAPI API

## Workflow

### 1. Exploratory Data Analysis

The EDA notebook assesses data quality, missing values, target imbalance and churn patterns across key business dimensions. It highlights that churn is concentrated in identifiable customer profiles, especially early-tenure customers, month-to-month contracts, fiber optic users and electronic payment behavior.

### 2. Modeling and Evaluation

The modeling notebook builds a reusable preprocessing and modeling workflow using project modules from `src/telco_churn/`. It compares Random Forest and XGBoost pipelines, tunes the selected XGBoost model, optimizes a decision threshold and evaluates the final model on a test set.

Notebook 2 reports strong ranking performance for the tuned XGBoost model, with ROC AUC around 0.85 and churner recall around 74% at the selected threshold. These results are notebook-reported project results rather than a formal production benchmark.

### 3. Interpretability and Recommendation Logic

The interpretability notebook uses SHAP to explain both global churn patterns and individual customer predictions. Churn probabilities are translated into operational risk segments:

- `Low risk`
- `Moderate risk`
- `High risk`

The recommendation logic keeps the top positive SHAP contributors that are also treated as actionable business levers, then maps up to two of those drivers to predefined retention ideas. If SHAP explanations are unavailable, the API falls back to risk-segment-based recommendations.

### 4. FastAPI Serving Layer

The project includes a minimal FastAPI layer with:

- `GET /health` to check service status and whether the model is loaded
- `POST /predict` to score one customer and return a business-oriented churn response

## Dataset

This project uses the **Telco Customer Churn** dataset available on Kaggle:

[Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Before running the notebooks or the API, download the dataset and place the CSV file at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project Structure

```text
.
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/          # generated locally, not versioned
├── notebooks/
│   ├── 01_eda_and_business_insight.ipynb
│   ├── 02_model_training_evaluation.ipynb
│   └── 03_model_interpretability.ipynb
├── outputs/
│   └── figures/
├── src/
│   └── telco_churn/
│       ├── api/
│       │   ├── main.py
│       │   ├── prediction.py
│       │   └── schemas.py
│       ├── config.py
│       ├── dashboard.py
│       ├── evaluate.py
│       ├── preprocess.py
│       ├── recommend.py
│       ├── train.py
│       └── visualization.py
├── tests/
│   └── test_api.py
├── poetry.lock
├── pyproject.toml
└── README.md
```

Key current artifacts include generated analysis figures in `outputs/figures/`.
The trained model pipeline (`models/churn_model.pkl`) is generated locally and is not versioned in the repository.

## Technologies Used

- **Python** for the project codebase
- **pandas** for data loading and preparation
- **scikit-learn** for preprocessing, pipelines, model evaluation and threshold analysis
- **XGBoost** for the final churn model
- **SHAP** for model interpretability
- **matplotlib** and **seaborn** for visual analysis
- **FastAPI**, **Pydantic** and **Uvicorn** for the API layer
- **Streamlit** for the interactive retention dashboard
- **pytest** and **httpx** for API tests
- **Poetry** for dependency management

## Run Locally

Install dependencies:

```bash
poetry install
```
Download the dataset from Kaggle and place it at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Start Jupyter to explore the notebooks:

```bash
poetry run jupyter notebook
```

Run the test suite:

```bash
poetry run pytest
```

## Run the FastAPI App

To run the API, you must first generate the trained model artifact locally so that the following file exists:

```text
models/churn_model.pkl
```

Start the API locally:

```bash
poetry run uvicorn src.telco_churn.api.main:app --reload
```

Then open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The health endpoint is available at:

```text
GET http://127.0.0.1:8000/health
```

It returns the API status and whether the saved model was loaded successfully.

## Run the Streamlit Dashboard

The project also includes a Streamlit dashboard that turns the churn model into an interactive retention-support tool. It relies on the FastAPI backend for service health checks and predictions, so the trained model must exist at:

```text
models/churn_model.pkl
```

Start the API in one terminal:

```bash
poetry run uvicorn src.telco_churn.api.main:app --reload
```

Then start the dashboard in a second terminal:

```bash
poetry run streamlit run src/telco_churn/dashboard.py
```

The dashboard allows users to review customer scenarios, adjust profile information, assess churn risk, and explore the main risk factors and recommended retention actions. Example customer profiles are included to make the demo easier to use. Technical request and response details are available through an optional debug mode.

## API Response

The prediction endpoint accepts one customer profile and returns a churn-oriented decision-support response:

```text
POST /predict
```

Example response shape:

```json
{
  "churn_probability": 0.72,
  "risk_segment": "High risk",
  "top_churn_drivers": [
    "Contract_Month-to-month",
    "OnlineSecurity_No"
  ],
  "recommendation": [
    "Offer a discount on a one-year or two-year contract.",
    "Bundle online security for three free months."
  ]
}
```

The returned fields are:

- `churn_probability`: estimated probability that the customer will churn.
- `risk_segment`: operational risk band based on the predicted probability.
- `top_churn_drivers`: up to two actionable SHAP drivers that increase churn risk.
- `recommendation`: retention ideas mapped from those drivers or a fallback recommendation when driver-level explanations are unavailable.

## Current Limitations and Next Steps

This project is currently a portfolio churn modeling and API prototype, not a fully deployed production system.

Planned improvements include:

- Add stricter input validation for inconsistent service combinations
- Build a Streamlit UI for interactive exploration and demos
- Add Dockerization for portable local and cloud deployment
- Expand API and model behavior tests beyond the current minimal test coverage
- Add monitoring, calibration checks and deployment documentation
