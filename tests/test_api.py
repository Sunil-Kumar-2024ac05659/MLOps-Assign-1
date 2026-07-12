"""Integration tests for the FastAPI application."""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(saved_model_path):
    """Create test client with model loaded from saved_model_path fixture."""
    os.environ["MODEL_PATH"] = saved_model_path
    # Import app after env var is set
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        response = client.get("/health")
        assert "status" in response.json()


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200


class TestPredictEndpoint:
    VALID_PAYLOAD = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
    }

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        body = response.json()
        assert "prediction" in body
        assert "label" in body
        assert "confidence" in body
        assert "probabilities" in body

    def test_predict_binary_output(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        assert response.json()["prediction"] in [0, 1]

    def test_predict_confidence_range(self, client):
        response = client.post("/predict", json=self.VALID_PAYLOAD)
        conf = response.json()["confidence"]
        assert 0.0 <= conf <= 1.0

    def test_predict_missing_required_field_returns_422(self, client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "age"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_type_returns_422(self, client):
        payload = {**self.VALID_PAYLOAD, "age": "not_a_number"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    VALID_PAYLOAD = [
        {"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
         "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
         "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1},
        {"age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200,
         "fbs": 0, "restecg": 1, "thalach": 170, "exang": 0,
         "oldpeak": 0.5, "slope": 1, "ca": 0, "thal": 2},
    ]

    def test_batch_returns_200(self, client):
        response = client.post("/predict/batch", json=self.VALID_PAYLOAD)
        assert response.status_code == 200

    def test_batch_correct_count(self, client):
        response = client.post("/predict/batch", json=self.VALID_PAYLOAD)
        body = response.json()
        assert body["total"] == 2
        assert len(body["results"]) == 2


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]
