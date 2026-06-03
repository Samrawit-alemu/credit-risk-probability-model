"""
src/train.py
============

Task 5 - Model Training and Tracking.

Trains and compares at least two classifiers (Logistic Regression and Random
Forest, plus XGBoost when available) to predict the ``is_high_risk`` proxy
target produced by :mod:`src.data_processing`. Each model is hyper-parameter
tuned with ``GridSearchCV`` and every run is logged to **MLflow** (parameters,
metrics and the fitted model artifact). The best model by ROC-AUC is registered
in the MLflow Model Registry and also dumped to ``models/best_model.pkl`` for
the API.

Note on target leakage: the raw RFM columns (Recency/Frequency/Monetary) are
*excluded* from the feature set because the proxy target is derived directly
from them; using them would make the task circular. The remaining behavioural
aggregates are still highly predictive (the proxy is a deterministic function of
behaviour), so high scores here are expected and documented.

Usage:
    python -m src.train
"""

from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

TARGET = "is_high_risk"
EXPERIMENT_NAME = "credit-risk-proxy"
REGISTERED_MODEL_NAME = "credit-risk-model"
PROCESSED_PATH = "data/processed/processed_data.csv"
BEST_MODEL_PATH = "models/best_model.pkl"

# Project root (two levels up from this file) used to pin a deterministic,
# local MLflow tracking store. sqlite fully supports the Model Registry.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRACKING_URI = "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db").replace(
    "\\", "/"
)


def load_dataset(path: str = PROCESSED_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.data_processing` first."
        )
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric aggregates and one-hot encode dominant categories."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def get_model_specs() -> dict:
    """Return ``{name: (estimator, param_grid)}`` for the models to compare.

    Grids are intentionally small so training stays fast and reproducible.
    """
    specs = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 10, None],
            },
        ),
    }

    # XGBoost is optional; include it only if installed.
    try:
        from xgboost import XGBClassifier

        specs["xgboost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.1, 0.3],
            },
        )
    except ImportError:
        pass

    return specs


def evaluate(model, X_test, y_test) -> dict:
    """Compute the five required classification metrics."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_and_track() -> dict:
    """Run the full training + MLflow tracking workflow.

    Returns a summary dict with the best model's name, metrics and run id.
    """
    df = load_dataset()
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Use an explicit local tracking URI unless the user configured one.
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[train] MLflow tracking URI: {mlflow.get_tracking_uri()}")
    preprocessor = build_preprocessor()

    results = []
    for name, (estimator, grid) in get_model_specs().items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline(
                steps=[("preprocess", preprocessor), ("model", estimator)]
            )
            search = GridSearchCV(
                pipeline, grid, cv=5, scoring="roc_auc", n_jobs=1
            )
            search.fit(X_train, y_train)

            best = search.best_estimator_
            metrics = evaluate(best, X_test, y_test)

            mlflow.log_param("model_type", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_best_roc_auc", float(search.best_score_))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best, name=name, input_example=X_test.head(3))

            print(f"[train] {name:20s} roc_auc={metrics['roc_auc']:.4f} "
                  f"f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}")

            results.append(
                {
                    "name": name,
                    "estimator": best,
                    "metrics": metrics,
                    "run_id": run.info.run_id,
                }
            )

    # Pick and register the best model by ROC-AUC.
    best_result = max(results, key=lambda r: r["metrics"]["roc_auc"])
    print(f"\n[train] best model: {best_result['name']} "
          f"(roc_auc={best_result['metrics']['roc_auc']:.4f})")

    _register_best(best_result)

    os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
    joblib.dump(best_result["estimator"], BEST_MODEL_PATH)
    print(f"[train] best model dumped -> {BEST_MODEL_PATH}")

    return {
        "best_model": best_result["name"],
        "metrics": best_result["metrics"],
        "run_id": best_result["run_id"],
    }


def _register_best(best_result: dict) -> None:
    """Register the best run's model in the MLflow Model Registry."""
    model_uri = f"runs:/{best_result['run_id']}/{best_result['name']}"
    try:
        mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
        print(f"[train] registered '{REGISTERED_MODEL_NAME}' from {model_uri}")
    except Exception as exc:  # noqa: BLE001 - registry can fail on some backends
        print(f"[train] WARNING: model registration failed ({exc}). "
              f"Model still logged + dumped to {BEST_MODEL_PATH}.")


if __name__ == "__main__":
    train_and_track()
