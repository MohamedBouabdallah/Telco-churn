import joblib
import pandas as pd

from src.telco_churn.config import DATA_PATH, MODEL_PATH
from src.telco_churn.recommend import (
    assign_segment,
    get_top_actionable_drivers,
    get_transform_feature_names,
    recommend_interventions,
    shap,
)

NUMERICAL_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

BACKGROUND_SAMPLE_SIZE = 200

FALLBACK_RECOMMENDATIONS = {
    "High risk": "Prioritize immediate retention outreach with a personalized offer.",
    "Moderate risk": "Monitor this customer and offer a targeted loyalty incentive.",
    "Low risk": "Maintain regular service quality and standard customer engagement.",
}


def load_model():
    """Load the trained preprocessing + model pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found")
    return joblib.load(MODEL_PATH)


def load_background_data(sample_size = BACKGROUND_SAMPLE_SIZE):
    """Load a small reference sample used as SHAP background."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found")

    df = pd.read_csv(DATA_PATH)

    # Reproduce the minimal cleaning needed for TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    background_df = df[FEATURE_COLUMNS].sample(
        n=min(sample_size, len(df)),
        random_state=42,
    )

    return background_df.reset_index(drop=True)


MODEL = load_model()
PREPROCESSOR = MODEL.named_steps["preprocessor"]
XGB_MODEL = MODEL.named_steps["model"]


def build_shap_explainer():
    """Build a reusable SHAP explainer from a stable background sample."""
    if shap is None:
        return None, None

    try:
        feature_names = get_transform_feature_names(
            preprocessor=PREPROCESSOR,
            categorical_cols=CATEGORICAL_COLUMNS,
            numerical_cols=NUMERICAL_COLUMNS,
        )

        background_df = load_background_data()
        background_transformed = PREPROCESSOR.transform(background_df)

        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()

        background_transformed_df = pd.DataFrame(
            background_transformed,
            columns=feature_names,
        )

        explainer = shap.Explainer(
            XGB_MODEL,
            background_transformed_df,
            algorithm="tree",
        )

        return explainer, feature_names

    except Exception:
        return None, None


EXPLAINER, FEATURE_NAMES = build_shap_explainer()


def predict_customer(customer):
    """Return business-oriented churn risk output for one customer."""
    customer_df = pd.DataFrame([customer], columns=FEATURE_COLUMNS)
    churn_probability = float(MODEL.predict_proba(customer_df)[0, 1])
    risk_segment = assign_segment(churn_probability)

    top_drivers, recommendation = explain_customer(customer_df, risk_segment)

    return {
        "churn_probability": churn_probability,
        "risk_segment": risk_segment,
        "top_churn_drivers": top_drivers,
        "recommendation": recommendation,
    }


def explain_customer(customer_df, risk_segment):
    """Compute SHAP drivers when possible, otherwise use risk-based advice."""
    if shap is None or EXPLAINER is None or FEATURE_NAMES is None:
        return [], fallback_recommendation(risk_segment)

    try:
        transformed = PREPROCESSOR.transform(customer_df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        transformed_df = pd.DataFrame(transformed, columns=FEATURE_NAMES)

        shap_values = EXPLAINER(transformed_df)
        row_values = shap_values.values[0]

        top_drivers = get_top_actionable_drivers(
            shap_values_for_row=row_values,
            feature_names=FEATURE_NAMES,
            top_n=2,
        )

        if not top_drivers:
            return [], fallback_recommendation(risk_segment)

        recommendation = recommend_interventions(
            shap_values_for_row=row_values,
            feature_names=FEATURE_NAMES,
            top_n=2,
        )

        return top_drivers, recommendation

    except Exception:
        return [], fallback_recommendation(risk_segment)


def fallback_recommendation(risk_segment):
    """Provide API-local guidance when actionable SHAP drivers are unavailable."""
    return [
        FALLBACK_RECOMMENDATIONS.get(
            risk_segment,
            "Review this customer with the retention team.",
        )
    ]
