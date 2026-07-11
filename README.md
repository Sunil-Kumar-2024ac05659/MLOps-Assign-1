# Heart Disease Prediction – MLOps Pipeline

**MLOps Assignment 01**

End-to-end ML solution: data → EDA → training → MLflow tracking → FastAPI → Docker → Kubernetes → CI/CD → Prometheus/Grafana monitoring.

---

## Project Structure

```
MLOps-Assignment-1/
├── data/
│   ├── download_data.py        # UCI dataset download script
│   ├── raw/                    # Raw CSV (git-ignored)
│   └── processed/
├── src/
│   ├── data_processing.py      # Preprocessing pipeline
│   ├── train.py                # Model training + MLflow
│   └── inference.py            # Inference utilities
├── api/
│   └── main.py                 # FastAPI application
├── tests/
│   ├── conftest.py
│   ├── test_data_processing.py
│   ├── test_inference.py
│   └── test_api.py
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   └── 02_model_training.ipynb # Training & experiment tracking
├── models/                     # Saved model artifacts (git-ignored)
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   └── prometheus.yml
├── screenshots/                # Report screenshots
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- (Optional) kubectl + Minikube for Kubernetes deployment

### 1 – Clone and install dependencies

```bash
git clone <repo-url>
cd MLOps-Assignment-1
pip install -r requirements.txt
```

### 2 – Download the dataset

```bash
python data/download_data.py
```

### 3 – Train models (with MLflow tracking)

```bash
python src/train.py
```

### 4 – View MLflow UI

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5 – Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 6 – Run tests

```bash
pytest tests/ -v --cov=src --cov=api
```

---

## Docker

### Build & run

```bash
# Train first to generate models/best_model.joblib
python src/train.py

# Build
docker build -t heart-disease-api:latest .

# Run
docker run -p 8000:8000 heart-disease-api:latest
```

### Full stack with Prometheus + Grafana

```bash
docker-compose up --build
```

| Service     | URL                        |
|-------------|----------------------------|
| API         | http://localhost:8000      |
| Swagger UI  | http://localhost:8000/docs |
| Prometheus  | http://localhost:9090      |
| Grafana     | http://localhost:3000      |

---

## API Usage

### Health check

```bash
curl http://localhost:8000/health
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease",
  "confidence": 0.8712,
  "probabilities": {"no_disease": 0.1288, "disease": 0.8712}
}
```

### Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

---

## Kubernetes Deployment (Minikube)

```bash
# Start Minikube
minikube start

# Build image inside Minikube
eval $(minikube docker-env)
docker build -t heart-disease-api:latest .

# Deploy
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Get service URL
minikube service heart-disease-api-service --url
```

---

## CI/CD Pipeline (GitHub Actions)

The pipeline at [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) runs on every push/PR:

| Stage         | What it does                                  |
|---------------|-----------------------------------------------|
| **Lint**      | flake8 on src/, api/, tests/                  |
| **Test**      | pytest with coverage report                   |
| **Train**     | Downloads data, trains all models, MLflow log |
| **Docker**    | Builds image, runs container, tests /predict  |

---

## Models Trained

| Model               | Notes                                    |
|---------------------|------------------------------------------|
| Logistic Regression | Baseline, interpretable, fast            |
| Random Forest       | Ensemble, robust to outliers             |
| XGBoost             | Gradient boosting, typically best AUC    |

All three are tuned with `GridSearchCV` (5-fold stratified CV) and compared by ROC-AUC. Best model is saved to `models/best_model.joblib`.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC-AUC
- Confusion matrix
- 5-fold stratified cross-validation

---

## Dataset

**Heart Disease UCI Dataset** – UCI Machine Learning Repository (ID: 45)

- 303 patient records
- 13 features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Binary target: 0 = no heart disease, 1 = heart disease

---

## Tech Stack

| Category        | Tools                              |
|-----------------|------------------------------------|
| Language        | Python 3.11                        |
| ML              | scikit-learn, XGBoost              |
| Tracking        | MLflow                             |
| API             | FastAPI + Uvicorn                  |
| Testing         | Pytest + pytest-cov                |
| Containers      | Docker, Docker Compose             |
| Orchestration   | Kubernetes (Minikube / cloud K8s)  |
| Monitoring      | Prometheus + Grafana               |
| CI/CD           | GitHub Actions                     |
| Linting         | flake8                             |
