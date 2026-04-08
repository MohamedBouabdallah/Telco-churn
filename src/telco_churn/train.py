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

def tune_xgboost(xgbpipeline, X_train, y_train, n_iter = 40):
    """Find the best hyperparameters for XGBoost"""
    param_grid = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
    }

    search = RandomizedSearchCV(
        estimator = xgbpipeline,
        param_distributions = param_grid,
        n_iter = n_iter,
        scoring = "roc_auc",
        cv = CV_FOLDS,
        random_state = RANDOM_STATE,
        n_jobs = 1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_

def save_model(model, path = MODEL_PATH):
    """Save the final model to a file"""
    path.parent.mkdir(parents = True, exist_ok = True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

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

    # XGBoost optimisation
    print("--- Tuning XGBoost ----")
    best_xgb = tune_xgboost(pipelines["XGBoost"], X_train, y_train)

    # Save the model
    save_model(best_xgb)