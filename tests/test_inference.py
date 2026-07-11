"""Unit tests for inference.py"""

import pytest
import numpy as np

from src.inference import predict, load_model


class TestPredict:
    def test_returns_required_keys(self, trained_pipeline, sample_patient_dict):
        result = predict(trained_pipeline, sample_patient_dict)
        assert "prediction" in result
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_prediction_is_binary(self, trained_pipeline, sample_patient_dict):
        result = predict(trained_pipeline, sample_patient_dict)
        assert result["prediction"] in {0, 1}

    def test_confidence_in_range(self, trained_pipeline, sample_patient_dict):
        result = predict(trained_pipeline, sample_patient_dict)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, trained_pipeline, sample_patient_dict):
        result = predict(trained_pipeline, sample_patient_dict)
        total = result["probabilities"]["no_disease"] + result["probabilities"]["disease"]
        assert abs(total - 1.0) < 1e-4

    def test_label_matches_prediction(self, trained_pipeline, sample_patient_dict):
        result = predict(trained_pipeline, sample_patient_dict)
        if result["prediction"] == 1:
            assert result["label"] == "Heart Disease"
        else:
            assert result["label"] == "No Heart Disease"

    def test_accepts_dataframe(self, trained_pipeline, sample_patient_dict):
        import pandas as pd
        df = pd.DataFrame([sample_patient_dict])
        result = predict(trained_pipeline, df)
        assert "prediction" in result


class TestLoadModel:
    def test_loads_from_valid_path(self, saved_model_path):
        pipeline = load_model(saved_model_path)
        assert pipeline is not None

    def test_raises_on_missing_path(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path/model.joblib")
