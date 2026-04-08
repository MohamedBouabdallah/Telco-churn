import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.telco_churn.config import CV_FOLDS, MODEL_PATH, RANDOM_STATE
from src.telco_churn.preprocess import build_preprocessor

def compute_scale_pos_weight(y_train):
    """Compute the ratio to help the model handle imbalanced classes"""
    negatives = (y_train == 0).sum()
    positives = (y_train == 1).sum()
    return negatives / positives

def build_model_pipelines(X_train, y_train):
    """Create pipelines combining preprocessing and model itself"""
    preprocessor = build_preprocessor(X_train)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    rf_model = RandomForestClassifier(n_estimators = 100, random_state = RANDOM_STATE)

    xgb_model = XGBClassifier(
        random_state = RANDOM_STATE,
        eval_metric = "logloss",
        scale_pos_weight = scale_pos_weight
    )

    return {
        "Random Forest": Pipeline(steps = [
            ("preprocessing", preprocessor),
            ("model", rf_model)
        ]),
        "XGBoost": Pipeline(steps = [
            ("preprocessor", preprocessor),
            ("model", xgb_model)
        ])
    }

def compare_models(models, X_train, y_train):
    """Evaluate multiple models using cross-validation"""
    scoring = ["roc_auc", "recall", "average_precision"]
    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}:")
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv = CV_FOLDS,
            scoring = scoring
        )
        results[name] = {
            metric: cv_results[f"test_{metric}"].mean() for metric in scoring
        }
    
    return pd.DataFrame(results).T.sort_values("roc_auc", ascending = False)

if __name__ == "__main__":
    from src.telco_churn.preprocess import load_telco_data, split_features_target, split_train_test

    # Data preparation
    df = load_telco_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Model Comparison
    print("--- Model Comparison ---")
    pipelines = build_model_pipelines(X_train, y_train)
    comparison = compare_models(pipelines, X_train, y_train)
    print(comparison)