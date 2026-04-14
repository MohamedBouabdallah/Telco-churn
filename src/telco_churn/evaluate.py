import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score
)

def find_best_threshold(model, X_train, y_train):
    """Find the optimal probability threshold to maximize F1-score"""
    y_proba_train = model.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_proba_train)

    # Compute F1-score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_threshold = thresholds[f1_scores.argmax()]

    return float(best_threshold)

def evaluate_classifier(model, X_test, y_test, threshold):
    """Compute final metrics using a specific decision threshold"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "average_precision": average_precision_score(y_test, y_proba),
        "threshold": threshold,
        "classification_report": classification_report(
            y_test, y_pred, target_names = ["No churn", "Churn"]
        )
    }

    predictions = pd.DataFrame({
        "y_true": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred
    })

    return metrics, predictions

def plot_confusion_matrix(y_test, y_pred, save_path = None):
    """Visualize true vs predicted classifications"""
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels = ["No churn", "Churn"],
        cmap = "Blues"
    )
    plt.title("Confusion Matrix - Tuned XGBoost")

    if save_path:
        save_path.parent.mkdir(parents = True, exist_ok = True)
        plt.savefig(save_path)
        print(f"Confusion matrix saved.")

    plt.close()

if __name__ == "__main__":
    import joblib
    from src.telco_churn.config import MODEL_PATH, FIGURES_DIR
    from src.telco_churn.preprocess import load_telco_data, split_features_target, split_train_test

    # Load data and split
    df = load_telco_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Load model
    if not MODEL_PATH.exists():
        print(f"Error : Model not found at this path.")
    else:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded.")

        # Search for the optimal threshold on the training set
        print("Optimisation of the decision threshold...")
        best_threshold = find_best_threshold(model, X_train, y_train)

        # Final evaluation on the test set
        metrics, predictions = evaluate_classifier(model, X_test, y_test, best_threshold)

        # Print the results
        print("\n" + "="*30)
        print(f"FINAL RESULTS (Threshold: {metrics['threshold']:.2f})")
        print("="*30)
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print("\nDetailed Report:")
        print(metrics['classification_report'])

        # Visualisation
        print("\nConfusion matrix generation..")
        plot_confusion_matrix(y_test, predictions['y_pred'], save_path = FIGURES_DIR / "confusion_matrix_tuned_xgb.png")