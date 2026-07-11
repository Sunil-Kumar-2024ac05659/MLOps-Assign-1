"""
Data loading, cleaning, and preprocessing pipeline for the Heart Disease dataset.
Provides a reusable sklearn Pipeline + ColumnTransformer for training and inference.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple

# Feature schema derived from UCI Heart Disease variable list
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET_COL = "target"


def load_raw_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Replace '?' strings (present in raw UCI files) with NaN
    df.replace("?", np.nan, inplace=True)
    return df


def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure target is binary (0 = no disease, 1 = disease)."""
    if df[TARGET_COL].dtype == object or df[TARGET_COL].max() > 1:
        df[TARGET_COL] = (df[TARGET_COL].astype(float) > 0).astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> Tuple[list, list]:
    """Return (numerical_cols, categorical_cols) that actually exist in df."""
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_cols, cat_cols


def build_preprocessing_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    """Build a ColumnTransformer with imputation + scaling/encoding."""
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Full data prep: load → clean → split. Returns X_train, X_test, y_train, y_test."""
    df = load_raw_data(csv_path)
    df = binarise_target(df)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    available_features = [c for c in feature_cols if c in df.columns]

    X = df[available_features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = os.path.join("data", "raw", "heart_disease.csv")
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Class distribution (train):\n{y_train.value_counts()}")
