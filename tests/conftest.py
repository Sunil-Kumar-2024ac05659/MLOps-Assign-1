"""
Shared fixtures for the test suite.
Uses in-memory synthetic data so no dataset download is required in CI.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import os
import tempfile

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data_processing import (
    build_preprocessing_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


@pytest.fixture(scope="session")
def sample_df():
    """Synthetic heart disease dataframe (50 rows)."""
    rng = np.random.default_rng(42)
    n = 50
    data = {
        "age":      rng.integers(30, 75, n).astype(float),
        "sex":      rng.integers(0, 2, n).astype(float),
        "cp":       rng.integers(0, 4, n).astype(float),
        "trestbps": rng.integers(90, 180, n).astype(float),
        "chol":     rng.integers(150, 350, n).astype(float),
        "fbs":      rng.integers(0, 2, n).astype(float),
        "restecg":  rng.integers(0, 3, n).astype(float),
        "thalach":  rng.integers(80, 200, n).astype(float),
        "exang":    rng.integers(0, 2, n).astype(float),
        "oldpeak":  rng.uniform(0, 6, n).round(1),
        "slope":    rng.integers(0, 3, n).astype(float),
        "ca":       rng.integers(0, 4, n).astype(float),
        "thal":     rng.integers(0, 4, n).astype(float),
        TARGET_COL: rng.integers(0, 2, n),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_X_y(sample_df):
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = sample_df[[c for c in feature_cols if c in sample_df.columns]]
    y = sample_df[TARGET_COL]
    return X, y


@pytest.fixture(scope="session")
def trained_pipeline(sample_X_y):
    """A minimal trained pipeline for inference tests."""
    X, y = sample_X_y
    num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="session")
def saved_model_path(trained_pipeline, tmp_path_factory):
    """Save pipeline to a temp file and return the path."""
    tmp = tmp_path_factory.mktemp("models")
    path = str(tmp / "test_model.joblib")
    joblib.dump(trained_pipeline, path)
    return path


@pytest.fixture
def sample_patient_dict():
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }
