"""Unit tests for data_processing.py"""

import pytest
import numpy as np
import pandas as pd

from src.data_processing import (
    load_raw_data,
    binarise_target,
    get_feature_columns,
    build_preprocessing_pipeline,
    prepare_data,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
)


class TestLoadRawData:
    def test_replaces_question_marks_with_nan(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("age,sex,target\n63,1,0\n45,?,1\n")
        df = load_raw_data(str(csv))
        assert pd.isna(df.loc[1, "sex"])

    def test_returns_dataframe(self, tmp_path):
        csv = tmp_path / "test.csv"
        csv.write_text("age,sex,target\n63,1,0\n")
        df = load_raw_data(str(csv))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestBinariseTarget:
    def test_values_already_binary(self, sample_df):
        df = sample_df.copy()
        result = binarise_target(df)
        assert set(result[TARGET_COL].unique()).issubset({0, 1})

    def test_multi_class_becomes_binary(self):
        df = pd.DataFrame({"target": [0, 1, 2, 3, 4]})
        result = binarise_target(df)
        assert set(result["target"].unique()).issubset({0, 1})
        # 0 stays 0; 1,2,3,4 become 1
        assert result["target"].iloc[0] == 0
        assert result["target"].iloc[2] == 1


class TestGetFeatureColumns:
    def test_returns_only_existing_columns(self, sample_df):
        X = sample_df.drop(columns=[TARGET_COL])
        num_cols, cat_cols = get_feature_columns(X)
        for c in num_cols:
            assert c in X.columns
        for c in cat_cols:
            assert c in X.columns

    def test_all_known_features_present(self, sample_df):
        X = sample_df.drop(columns=[TARGET_COL])
        num_cols, cat_cols = get_feature_columns(X)
        assert len(num_cols) == len(NUMERICAL_FEATURES)
        assert len(cat_cols) == len(CATEGORICAL_FEATURES)


class TestBuildPreprocessingPipeline:
    def test_pipeline_fits_and_transforms(self, sample_X_y):
        X, _ = sample_X_y
        num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.ndim == 2

    def test_no_nan_after_transform(self, sample_X_y):
        X, _ = sample_X_y
        num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X)
        assert not np.isnan(X_transformed).any()

    def test_pipeline_handles_missing_values(self, sample_X_y):
        X, _ = sample_X_y
        X_missing = X.copy()
        X_missing.iloc[0, 0] = np.nan
        num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]
        preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
        X_transformed = preprocessor.fit_transform(X_missing)
        assert not np.isnan(X_transformed).any()


class TestPrepareData:
    def test_returns_four_splits(self, tmp_path, sample_df):
        csv = tmp_path / "heart.csv"
        sample_df.to_csv(csv, index=False)
        result = prepare_data(str(csv))
        assert len(result) == 4

    def test_correct_split_sizes(self, tmp_path, sample_df):
        csv = tmp_path / "heart.csv"
        sample_df.to_csv(csv, index=False)
        X_train, X_test, y_train, y_test = prepare_data(str(csv), test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(sample_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_no_target_leakage(self, tmp_path, sample_df):
        csv = tmp_path / "heart.csv"
        sample_df.to_csv(csv, index=False)
        X_train, X_test, y_train, y_test = prepare_data(str(csv))
        assert TARGET_COL not in X_train.columns
        assert TARGET_COL not in X_test.columns
