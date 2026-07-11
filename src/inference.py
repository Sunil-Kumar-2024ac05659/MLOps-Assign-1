"""Inference utilities: load the saved pipeline and run predictions."""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Union

MODEL_PATH = os.path.join("models", "best_model.joblib")


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
    return joblib.load(model_path)


def predict(
    pipeline,
    data: Union[dict, pd.DataFrame],
) -> dict:
    """Run inference on a single record (dict) or a DataFrame."""
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    prediction = int(pipeline.predict(df)[0])
    proba = pipeline.predict_proba(df)[0]
    confidence = float(proba[prediction])

    return {
        "prediction": prediction,
        "label": "Heart Disease" if prediction == 1 else "No Heart Disease",
        "confidence": round(confidence, 4),
        "probabilities": {
            "no_disease": round(float(proba[0]), 4),
            "disease":    round(float(proba[1]), 4),
        },
    }
