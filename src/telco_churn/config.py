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