import pandas as pd
from src.telco_churn.config import ACTIONABLE_INTERVENTIONS, HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD

try:
    import shap
except:
    shap = None

def assign_segment(probability):
    """Assign a risk category based on probability thresholds"""
    if probability >= HIGH_RISK_THRESHOLD:
        return "High risk"
    if probability >= LOW_RISK_THRESHOLD:
        return "Moderate risk"
    return "Low risk"

def build_results_frame(X_test, y_proba, y_pred):
    """Combine feature with predictions and risk segment"""
    results = X_test.copy().reset_index(drop = True)
    results["churn_proba"] = y_proba
    results["churn_prediction"] = y_pred
    results["risk_segment"] = results["churn_proba"].apply(assign_segment)
    return results

def get_transform_feature_names(preprocessor, categorical_cols, numerical_cols):
    """Recover feature names after one-hot encoding"""
    cat_feature_names = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_cols)
        .tolist()
    )
    return numerical_cols + cat_feature_names

def compute_shap_values(model, X_test, categorical_cols, numerical_cols):
    """Calculate SHAP values for model interpretability"""
    if shap is None:
        raise ImportError("shap is required to compute explanations")
    
    preprocessor = model.named_steps["preprocessor"]
    xgb_model = model.named_steps["model"]

    # Transform data to the appropriate format for xgboost
    transformed = preprocessor.transform(X_test)
    feature_names = get_transform_feature_names(preprocessor, categorical_cols, numerical_cols)
    transformed_df = pd.DataFrame(transformed, columns = feature_names)

    # Compute the SHAP values
    explainer = shap.Explainer(xgb_model, transformed_df, algorithm = "tree")
    shap_values = explainer(transformed_df)

    return shap_values, transformed_df, feature_names

def get_top_actionable_drivers(shap_values_for_row, feature_names, interventions=ACTIONABLE_INTERVENTIONS, top_n=2):
    """Return the top actionable features that positively drive churn"""
    shap_dict = dict(zip(feature_names, shap_values_for_row))

    actionable = {
        feature: shap_value
        for feature, shap_value in shap_dict.items()
        if feature in interventions and shap_value > 0
    }

    if not actionable:
        return []

    return sorted(actionable, key=actionable.get, reverse=True)[:top_n]

def recommend_interventions(shap_values_for_row, feature_names, interventions=ACTIONABLE_INTERVENTIONS, top_n=2):
    """Suggest interventions from the top actionable churn drivers"""
    top_features = get_top_actionable_drivers(
        shap_values_for_row=shap_values_for_row,
        feature_names=feature_names,
        interventions=interventions,
        top_n=top_n,
    )

    if not top_features:
        return ["No specific interventions identified."]

    return [interventions[feature] for feature in top_features]

def attach_recommendations(results_df, shap_values, feature_names, interventions=ACTIONABLE_INTERVENTIONS, top_n=2):
    """Add actionable drivers and interventions to the results dataframe"""
    enriched = results_df.copy()
    shap_array = shap_values.values

    enriched["actionable_drivers"] = [
        get_top_actionable_drivers(
            shap_values_for_row=shap_array[index],
            feature_names=feature_names,
            interventions=interventions,
            top_n=top_n,
        )
        for index in range(len(enriched))
    ]

    enriched["interventions"] = [
        [interventions[feature] for feature in drivers]
        if drivers else ["No specific interventions identified."]
        for drivers in enriched["actionable_drivers"]
    ]

    return enriched

if __name__ == "__main__":
    import joblib
    from src.telco_churn.config import (
        MODEL_PATH, 
        OUTPUT_DIR
    )
    from src.telco_churn.preprocess import (
        load_telco_data, 
        split_features_target, 
        split_train_test
    )
    from src.telco_churn.evaluate import find_best_threshold

    # Load data
    df = load_telco_data()
    X, y = split_features_target(df)
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Load model
    model = joblib.load(MODEL_PATH)
    print("Model loaded for interpretation")

    # Optimal threshold logic
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Finding optimal threshold based on training data...")
    best_threshold = find_best_threshold(model, X_train, y_train)
    y_pred_custom = (y_proba >= best_threshold).astype(int)

    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values, transformed_df, feature_names = compute_shap_values(
        model, X_test, categorical_cols, numerical_cols
    )

    # Build results
    print("Generating business recommendations...")
    results_frame = build_results_frame(X_test, y_proba, y_pred_custom)
    final_report = attach_recommendations(results_frame, shap_values, feature_names)

    # Save
    output_path = OUTPUT_DIR / "client_risk_recommendations.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_report.to_csv(output_path, index=False)
    
    print(f"Report saved to: {output_path}")