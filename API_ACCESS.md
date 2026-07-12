# API Access Instructions

The Heart Disease Prediction API is not publicly hosted. Use the instructions below to run it locally or via Docker.

---

## Option 1 – Run Locally (Python)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset & train model (generates models/best_model.joblib)
python data/download_data.py
python src/train.py

# 3. Start the API
uvicorn api.main:app --reload --port 8000
```

- Swagger UI:  http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Predict endpoint: `POST http://localhost:8000/predict`

---

## Option 2 – Run with Docker

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest
```

---

## Option 3 – Full Stack (API + Prometheus + Grafana)

```bash
docker-compose up --build
```

| Service    | URL                         |
|------------|-----------------------------|
| API        | http://localhost:8000       |
| Swagger UI | http://localhost:8000/docs  |
| Prometheus | http://localhost:9090       |
| Grafana    | http://localhost:3000       |

---

## Sample Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Expected Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.8712,
  "probabilities": {
    "no_disease": 0.1288,
    "disease": 0.8712
  }
}
```
