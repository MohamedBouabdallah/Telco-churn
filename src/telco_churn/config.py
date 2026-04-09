from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


# Project constants
TARGET_COL = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 5

# Customer segmentation parameters
LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.60

# Recommendation system parameters
ACTIONABLE_INTERVENTIONS = {
    "Contract_Month-to-month": "Offer a discount on a one-year or two-year contract.",
    "OnlineSecurity_No": "Bundle online security for three free months.",
    "TechSupport_No": "Offer discounted technical support.",
    "MonthlyCharges": "Propose a cheaper plan or bundle.",
    "InternetService_Fiber optic": "Check service quality and perceived value of fiber.",
    "PaperlessBilling_No": "Promote paperless billing with an incentive.",
    "OnlineBackup_No": "Offer online backup at a discounted price.",
}