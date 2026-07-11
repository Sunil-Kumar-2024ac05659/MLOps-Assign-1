"""
Model training with MLflow experiment tracking.

Trains two classifiers (Logistic Regression + Random Forest / XGBoost),
performs hyperparameter tuning, logs everything to MLflow, and saves the
best pipeline to models/.

Usage:
    python src/train.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_processing import (
    prepare_data, build_preprocessing_pipeline, get_feature_columns,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
)

DATA_PATH = os.path.join("data", "raw", "heart_disease.csv")
MODEL_DIR = "models"
EXPERIMENT_NAME = "heart-disease-classifier"
CV_FOLDS = 5
RANDOM_STATE = 42


# ── helper utilities ──────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def save_confusion_matrix_plot(y_true, y_pred, title: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_prob, title: str, path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_feature_importance_plot(model, feature_names: list, path: str):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    top_n = min(20, len(feature_names))
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), importances[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title("Feature Importance (top 20)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ── training runs ─────────────────────────────────────────────────────────────

def train_model(
    name: str,
    classifier,
    param_grid: dict,
    X_train, X_test, y_train, y_test,
    preprocessor,
    tmp_dir: str,
) -> dict:
    """Train one classifier, tune, log to MLflow, return metrics + pipeline."""

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_name", name)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

        # Rename param_grid keys to target the pipeline step
        prefixed_grid = {f"classifier__{k}": v for k, v in param_grid.items()}

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        grid_search = GridSearchCV(
            pipeline, prefixed_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = {k.replace("classifier__", ""): v
                       for k, v in grid_search.best_params_.items()}

        # Log best hyperparameters
        mlflow.log_params(best_params)

        # CV scores
        cv_results = cross_validate(
            best_pipeline, X_train, y_train,
            cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            return_train_score=False,
        )
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mlflow.log_metric(f"cv_{metric}_mean", cv_results[f"test_{metric}"].mean())
            mlflow.log_metric(f"cv_{metric}_std",  cv_results[f"test_{metric}"].std())

        # Test-set evaluation
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_prob)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"Best params: {best_params}")
        print(f"Test metrics: {json.dumps({k: round(v, 4) for k, v in test_metrics.items()}, indent=2)}")
        print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

        # Artifacts
        os.makedirs(tmp_dir, exist_ok=True)

        cm_path = os.path.join(tmp_dir, f"{name}_confusion_matrix.png")
        save_confusion_matrix_plot(y_test, y_pred, f"{name} – Confusion Matrix", cm_path)
        mlflow.log_artifact(cm_path)

        roc_path = os.path.join(tmp_dir, f"{name}_roc_curve.png")
        save_roc_curve_plot(y_test, y_prob, f"{name} – ROC Curve", roc_path)
        mlflow.log_artifact(roc_path)

        # Feature importance from inner classifier
        inner_clf = best_pipeline.named_steps["classifier"]
        fi_path = os.path.join(tmp_dir, f"{name}_feature_importance.png")
        try:
            feature_names = list(
                best_pipeline.named_steps["preprocessor"].get_feature_names_out()
            )
        except Exception:
            feature_names = [f"f{i}" for i in range(100)]
        save_feature_importance_plot(inner_clf, feature_names, fi_path)
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path)

        # Save pipeline as MLflow model artifact
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        return {"name": name, "pipeline": best_pipeline, "metrics": test_metrics}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp_dir = os.path.join(MODEL_DIR, "tmp_plots")

    # Data
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    num_cols, cat_cols = get_feature_columns(X_train)

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Class balance    : {y_train.value_counts().to_dict()}")
    print(f"Numerical features  : {num_cols}")
    print(f"Categorical features: {cat_cols}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Model 1: Logistic Regression ──────────────────────────────────────────
    preprocessor_lr = build_preprocessing_pipeline(num_cols, cat_cols)
    lr_result = train_model(
        name="LogisticRegression",
        classifier=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid={"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]},
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        preprocessor=preprocessor_lr,
        tmp_dir=tmp_dir,
    )

    # ── Model 2: Random Forest ────────────────────────────────────────────────
    preprocessor_rf = build_preprocessing_pipeline(num_cols, cat_cols)
    rf_result = train_model(
        name="RandomForest",
        classifier=RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        preprocessor=preprocessor_rf,
        tmp_dir=tmp_dir,
    )

    # ── Model 3: XGBoost ──────────────────────────────────────────────────────
    preprocessor_xgb = build_preprocessing_pipeline(num_cols, cat_cols)
    xgb_result = train_model(
        name="XGBoost",
        classifier=XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        ),
        param_grid={
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        preprocessor=preprocessor_xgb,
        tmp_dir=tmp_dir,
    )

    # ── Select best model by ROC-AUC and save ────────────────────────────────
    results = [lr_result, rf_result, xgb_result]
    best = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\nBest model: {best['name']}  (ROC-AUC = {best['metrics']['roc_auc']:.4f})")

    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(best["pipeline"], model_path)
    print(f"Best model saved to {model_path}")

    # Save model metadata
    meta = {
        "model_name": best["name"],
        "metrics": {k: round(v, 4) for k, v in best["metrics"].items()},
        "num_features": num_cols,
        "cat_features": cat_cols,
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return best


if __name__ == "__main__":
    main()
