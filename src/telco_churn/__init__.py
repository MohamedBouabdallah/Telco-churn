"""Utilities for the Telco churn portfolio project"""

from .config import DATA_PATH, MODEL_PATH, OUTPUT_DIR
from .preprocess import load_telco_data, split_features_target, split_train_test

__all__ = [
    "DATA_PATH",
    "MODEL_PATH",
    "OUTPUT_DIR",
    "load_telco_data",
    "split_features_target",
    "split_train_test",
]
