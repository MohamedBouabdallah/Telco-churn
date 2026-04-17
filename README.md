# Telco Customer Churn Prediction

An end-to-end churn prediction project designed to identify at-risk telecom customers, explain the main drivers of churn and translate model outputs into practical retention recommendations. The project combines exploratory analysis, supervised machine learning, SHAP-based interpretability, a FastAPI backend, Dockerized deployment and a Streamlit dashboard for business-facing interaction.

## Live Demo

- [**Dashboard (Streamlit)**](https://telco-customer-retention-dashboard.streamlit.app/)
- [**API documentation (FastAPI)**](https://telco-churn-api-ea9o.onrender.com/docs)
- [**API health endpoint**](https://telco-churn-api-ea9o.onrender.com/health)

## How to Use the Demo

1. Open the Streamlit dashboard  
2. Load an example customer profile or adjust the form manually  
3. Click **Assess churn risk**  
4. Review:
   - churn probability
   - risk segment
   - main risk factors
   - recommended retention actions

> **Note:** The API is hosted on a free Render instance. The first request may take a little longer if the backend is waking up from inactivity.

## Business Problem

Customer churn is a core issue for subscription-based telecom businesses. When customers leave, the company loses recurring revenue and may fail to recover acquisition and onboarding costs. A useful churn model should therefore do more than predict a binary outcome: it should help retention teams prioritize outreach, understand why a customer appears risky and decide which actions are worth reviewing.

## Project Goals

- Identify customers with high churn risk using a reproducible machine learning pipeline
- Analyze churn patterns across tenure, contract type, services, billing behavior and customer profile
- Explain model predictions globally and at the customer level using SHAP
- Convert actionable churn drivers into retention recommendation ideas
- Expose the saved churn pipeline through a FastAPI API and an interactive Streamlit dashboard

## Workflow

### 1. Exploratory Data Analysis

The EDA notebook assesses data quality, missing values, target imbalance and churn patterns across key business dimensions. It highlights that churn is concentrated in identifiable customer profiles, especially early-tenure customers, month-to-month contracts, fiber optic users and customers using electronic check payment.

### 2. Modeling and Evaluation

The modeling notebook builds a reusable preprocessing and modeling workflow using project modules from `src/telco_churn/`. It compares Random Forest and XGBoost pipelines, tunes the selected XGBoost model, optimizes a decision threshold and evaluates the final model on a test set.

The final XGBoost model reaches:
- **ROC AUC:** around **0.85**
- **Recall on churners:** around **74%** at the selected threshold

These are project-level notebook results rather than formal production benchmarks.

### 3. Interpretability and Recommendation Logic

The interpretability notebook uses SHAP to explain both global churn patterns and individual customer predictions. Churn probabilities are translated into operational risk segments:

- `Low risk`
- `Moderate risk`
- `High risk`

The recommendation logic keeps the top positive SHAP contributors that are also treated as actionable business levers, then maps up to two of those drivers to predefined retention ideas. If SHAP explanations are unavailable, the API falls back to risk-segment-based recommendations.

### 4. API and Dashboard

The project includes:
- a **FastAPI backend** with:
  - `GET /health`
  - `POST /predict`
- a **Streamlit dashboard** for interactive customer risk assessment and retention support

## Dataset

This project uses the **Telco Customer Churn** dataset available on Kaggle:

[Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

For deployment simplicity and reproducibility, the public CSV used in the project is included directly in the repository at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project Structure

```text
.
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl
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
├── Dockerfile
├── .dockerignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

Key current artifacts include generated analysis figures in `outputs/figures/`.

For deployment simplicity and reproducibility, the repository includes:
- the public demo dataset at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- the trained model artifact at `models/churn_model.pkl`

In a production setting, these artifacts would typically be stored outside the code repository.

## Technologies Used

- **Python** for the project codebase
- **pandas** for data loading and preparation
- **scikit-learn** for preprocessing, pipelines, model evaluation and threshold analysis
- **XGBoost** for the final churn model
- **SHAP** for model interpretability
- **matplotlib** and **seaborn** for visual analysis
- **FastAPI**, **Pydantic** and **Uvicorn** for the API layer
- **Streamlit** for the interactive retention dashboard
- **Docker** for containerizing the FastAPI backend
- **Render** for API deployment
- **Streamlit Community Cloud** for dashboard deployment
- **pytest** and **httpx** for API tests
- **Poetry** for dependency management

## Run Locally

Install dependencies:

```bash
poetry install
```

Start Jupyter to explore the notebooks:

```bash
poetry run jupyter notebook
```

Run the test suite:

```bash
poetry run pytest
```

The repository already includes the public demo dataset used by the notebooks and API:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Run the FastAPI App

To run the API, the trained model artifact is expected at:

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

Useful endpoints:

```text
GET http://127.0.0.1:8000/health
POST http://127.0.0.1:8000/predict
```

The health endpoint returns the API status and whether the saved model was loaded successfully.

## Run the Streamlit Dashboard

Start the API in one terminal:

```bash
poetry run uvicorn src.telco_churn.api.main:app --reload
```

Then start the dashboard in a second terminal:

```bash
poetry run streamlit run src/telco_churn/dashboard.py
```

By default, the dashboard targets the local API at:

```text
http://127.0.0.1:8000
````

The dashboard allows users to:

- load example customer profiles
- adjust customer attributes manually
- assess churn risk
- review main risk factors
- review recommended retention actions

An optional debug mode is also available to inspect API status, request payloads, and raw responses.

## Run the FastAPI API with Docker

Build the image from the repository root:

```bash
docker build -t telco-churn-api .
````

Run the container locally:

```bash
docker run --rm -p 8000:8000 telco-churn-api
````

Check the health endpoint:

```bash
curl http://localhost:8000/health
````

The Docker setup is used for the deployed FastAPI backend and keeps the API portable across local and cloud environments.

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

This project is deployed as a portfolio decision-support prototype rather than a fully productionized retention system.

Possible next steps include:

- Add stricter input validation for inconsistent service combinations
- Expand API and model behavior tests beyond the current minimal coverage
- Add monitoring and calibration diagnostics
- Improve deployment documentation and configuration management
- Refine the dashboard UX and recommendation logic further
