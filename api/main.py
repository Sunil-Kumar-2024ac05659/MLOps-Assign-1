"""
FastAPI application for Heart Disease prediction.

Endpoints:
    GET  /health      – liveness check
    GET  /metrics     – Prometheus metrics
    POST /predict     – single-record inference
    POST /predict/batch – batch inference
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.inference import load_model, predict  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("heart-disease-api")

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["label"]
)

# ── Model lifecycle ────────────────────────────────────────────────────────────
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    model_path = os.getenv("MODEL_PATH", os.path.join("models", "best_model.joblib"))
    logger.info(f"Loading model from {model_path}")
    try:
        model_pipeline = load_model(model_path)
        logger.info("Model loaded successfully")
    except FileNotFoundError as exc:
        logger.warning(f"Model not found at startup: {exc}. /predict will return 503.")
    yield
    logger.info("Shutting down API")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment 01 – BITS Pilani",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    age: float = Field(..., example=63, description="Age in years")
    sex: float = Field(..., example=1, description="Sex (1=male, 0=female)")
    cp: float = Field(..., example=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., example=145, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., example=233, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(..., example=1, description="Fasting blood sugar > 120 mg/dl (1=true)")
    restecg: float = Field(..., example=0, description="Resting ECG results (0-2)")
    thalach: float = Field(..., example=150, description="Maximum heart rate achieved")
    exang: float = Field(..., example=0, description="Exercise induced angina (1=yes)")
    oldpeak: float = Field(..., example=2.3, description="ST depression induced by exercise")
    slope: Optional[float] = Field(None, example=0, description="Slope of peak exercise ST segment")
    ca: Optional[float] = Field(None, example=0, description="Number of major vessels (0-3)")
    thal: Optional[float] = Field(None, example=1, description="Thalassemia type")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probabilities: dict


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    status = "ready" if model_pipeline is not None else "model_not_loaded"
    return {"status": status, "service": "heart-disease-api"}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(patient: PatientFeatures):
    if model_pipeline is None:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    start = time.time()
    try:
        data = patient.model_dump()
        result = predict(model_pipeline, data)

        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
        PREDICTION_COUNT.labels(label=result["label"]).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

        logger.info(f"Prediction: {result['label']}  confidence={result['confidence']}")
        return result

    except Exception as exc:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(patients: List[PatientFeatures]):
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.time()
    results = []
    for patient in patients:
        data = patient.model_dump()
        result = predict(model_pipeline, data)
        results.append(result)
        PREDICTION_COUNT.labels(label=result["label"]).inc()

    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.time() - start)
    REQUEST_COUNT.labels(endpoint="/predict/batch", method="POST", status="200").inc()
    logger.info(f"Batch prediction: {len(results)} records")

    return {"results": results, "total": len(results)}


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }
