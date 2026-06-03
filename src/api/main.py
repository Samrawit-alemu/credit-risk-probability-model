"""
src/api/main.py
===============

FastAPI service exposing the credit-risk model (Task 6).

The model is loaded once at startup. It is loaded from the MLflow Model Registry
when available, otherwise it falls back to the local ``models/best_model.pkl``
produced by :mod:`src.train` (handy inside containers without the tracking DB).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import CustomerFeatures, PredictionResponse

REGISTERED_MODEL_NAME = "credit-risk-model"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

app = FastAPI(
    title="Bati Bank Credit Risk API",
    description="Predicts the probability that a customer is a high-risk credit proxy.",
    version="1.0.0",
)

# Model handle populated at startup.
_model = None
_model_source = "uninitialised"


def load_model():
    """Load the best model: try the MLflow registry, then the local pickle."""
    global _model_source

    # 1) MLflow Model Registry (preferred).
    try:
        import mlflow
        import mlflow.sklearn

        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI",
            "sqlite:///" + os.path.join(PROJECT_ROOT, "mlflow.db").replace("\\", "/"),
        )
        mlflow.set_tracking_uri(tracking_uri)
        # Load as a native sklearn estimator so predict_proba is available
        # (pyfunc would only expose predict, i.e. hard labels).
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/latest")
        _model_source = f"mlflow:{tracking_uri}"
        return model
    except Exception as exc:  # noqa: BLE001 - fall back to local pickle
        print(f"[api] MLflow registry load failed ({exc}); trying local pickle.")

    # 2) Local pickle fallback.
    import joblib

    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(
            f"No model available: MLflow registry failed and "
            f"{LOCAL_MODEL_PATH} does not exist. Run `python -m src.train` first."
        )
    model = joblib.load(LOCAL_MODEL_PATH)
    _model_source = f"local:{LOCAL_MODEL_PATH}"
    return model


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = load_model()
    print(f"[api] model loaded from {_model_source}")


@app.get("/")
def read_root() -> dict:
    return {
        "status": "Credit Scoring API is online",
        "model_source": _model_source,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures) -> PredictionResponse:
    """Return the high-risk probability for one customer."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = features.to_frame()

    try:
        proba = _predict_proba(input_df)
    except Exception as exc:  # noqa: BLE001 - surface inference errors clearly
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}")

    is_high_risk = int(proba >= 0.5)
    return PredictionResponse(
        is_high_risk=is_high_risk,
        risk_probability=round(float(proba), 4),
        recommendation="Reject" if is_high_risk else "Approve",
    )


def _predict_proba(input_df) -> float:
    """Extract the positive-class probability across sklearn / pyfunc models."""
    # sklearn estimator/pipeline.
    if hasattr(_model, "predict_proba"):
        return _model.predict_proba(input_df)[0][1]

    # MLflow pyfunc wrapper.
    result = _model.predict(input_df)
    try:
        # DataFrame with probability columns.
        return float(result.iloc[0, -1])
    except AttributeError:
        # Array-like of labels/probabilities.
        return float(result[0])
