import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.telco_churn.config import DATA_PATH, ID_COL, RANDOM_STATE, TARGET_COL, TEST_SIZE

def load_telco_data(path = DATA_PATH):
    """Load the csv file from the data folder"""
    # load the csv file from the data folder
    return pd.read_csv(path)

def clean_telco_data(df):
    """Fix types and drop missing targets"""
    cleaned = df.copy()
    # Convert TotalCharges to numeric (fixes the empty string issue)
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors= 'coerce')
    # remove rows where Churn is missing
    cleaned = cleaned.dropna(axis = 0, subset = [TARGET_COL])
    return cleaned

def split_features_target(df):
    """Separate X and y and converts target to 0/1"""
    cleaned = clean_telco_data(df)
    y = (cleaned[TARGET_COL] == "Yes").astype(int)
    # Drop target and ID from the features
    X = cleaned.drop(columns = [TARGET_COL, ID_COL], errors = 'ignore')
    return X, y

def build_preprocessor(X):
    """Create the transformation pipeline for numeric and categorical variables"""
    # Identify automatically numerical and categorical variables
    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Define transformers
    num_transformer = SimpleImputer(strategy= "median")
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

    # Combine both
    return ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

def split_train_test(X, y):
    """Standard train/test split with stratification"""
    return train_test_split(
        X, y,
        test_size = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify = y
    )

if __name__ == "__main__":
    # 1. Load data
    df = load_telco_data()
    print(f"Data loaded: {df.shape}")

    # 2. Process features and target
    X, y = split_features_target(df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # 3. Check types (TotalCharges should be float now)
    print(f"TotalCharges type: {X['TotalCharges'].dtype}")

    # 4. Test the preprocessor (ColumnTransformer)
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    print(f"Matrix shape after OneHot/Scaling: {X_processed.shape}")