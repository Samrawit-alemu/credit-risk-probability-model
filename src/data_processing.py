"""
src/data_processing.py
======================

End-to-end, reproducible data-processing for the Bati Bank credit-risk model.

This module covers two deliverables:

* **Task 3 - Feature Engineering**: a single, fitted ``sklearn.pipeline.Pipeline``
  that turns the *raw* transaction-level data into a model-ready, customer-level
  feature matrix (aggregate features, datetime features, categorical encoding,
  missing-value imputation and scaling). Weight-of-Evidence (WoE) / Information
  Value (IV) helpers are provided for interpretable feature analysis.

* **Task 4 - Proxy Target Engineering**: RFM metrics + K-Means clustering to
  derive a binary ``is_high_risk`` proxy label, which is merged back into the
  processed dataset.

Run as a script to produce the processed dataset and the fitted pipeline:

    python -m src.data_processing \
        --input data/raw/data.csv \
        --output data/processed/processed_data.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reproducibility: a single random seed used everywhere in this module.
RANDOM_STATE = 42

# Columns that identify *who* and *when* — kept out of the model feature matrix.
CUSTOMER_KEY = "CustomerId"
TIME_COL = "TransactionStartTime"

# Final customer-level feature groups (defined after aggregation).
NUMERIC_FEATURES = [
    "TotalAmount",
    "AvgAmount",
    "StdAmount",
    "MinAmount",
    "MaxAmount",
    "TransactionCount",
    "TotalValue",
    "AvgValue",
    "AvgHour",
    "AvgDay",
    "AvgMonth",
    "UniqueProducts",
    "FraudCount",
]
CATEGORICAL_FEATURES = [
    "TopProductCategory",
    "TopChannelId",
    "TopProviderId",
]


# ---------------------------------------------------------------------------
# Task 3 (1 & 2): Aggregate + datetime feature engineering
# ---------------------------------------------------------------------------
class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Collapse raw *transaction-level* rows into one row per customer.

    Produces aggregate features (sum / mean / std / count of transaction
    amounts), datetime features (average hour / day / month of activity) and the
    dominant category per customer. This is the first step of the pipeline, so
    the whole ``Pipeline`` consumes raw data and emits a model-ready frame.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - sklearn API
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

        # --- datetime features extracted before aggregation -----------------
        df["_hour"] = df[TIME_COL].dt.hour
        df["_day"] = df[TIME_COL].dt.day
        df["_month"] = df[TIME_COL].dt.month

        grouped = df.groupby(CUSTOMER_KEY)

        agg = grouped.agg(
            TotalAmount=("Amount", "sum"),
            AvgAmount=("Amount", "mean"),
            StdAmount=("Amount", "std"),
            MinAmount=("Amount", "min"),
            MaxAmount=("Amount", "max"),
            TransactionCount=("TransactionId", "count"),
            TotalValue=("Value", "sum"),
            AvgValue=("Value", "mean"),
            AvgHour=("_hour", "mean"),
            AvgDay=("_day", "mean"),
            AvgMonth=("_month", "mean"),
            UniqueProducts=("ProductId", "nunique"),
            FraudCount=("FraudResult", "sum"),
        )

        # Std is NaN for customers with a single transaction -> 0 variability.
        agg["StdAmount"] = agg["StdAmount"].fillna(0.0)

        # Dominant (most frequent) category per customer.
        agg["TopProductCategory"] = _safe_mode(grouped["ProductCategory"])
        agg["TopChannelId"] = _safe_mode(grouped["ChannelId"])
        agg["TopProviderId"] = _safe_mode(grouped["ProviderId"])

        return agg.reset_index()


def _safe_mode(grouped_series) -> pd.Series:
    """Most frequent value per group, robust to empty/NaN groups."""
    return grouped_series.agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"
    )


def build_feature_pipeline() -> Pipeline:
    """Return the **single fitted-able** sklearn Pipeline (Task 3 deliverable).

    Steps:
        1. ``CustomerAggregator`` - raw transactions -> customer-level frame.
        2. ``ColumnTransformer``:
             * numeric  -> median impute -> standard scale
             * category -> one-hot encode (unknown-safe)

    The fitted pipeline transforms raw input into a model-ready numeric matrix.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("aggregate", CustomerAggregator()),
            ("preprocess", preprocessor),
        ]
    )


# ---------------------------------------------------------------------------
# Task 3 (6): Weight of Evidence (WoE) & Information Value (IV)
# ---------------------------------------------------------------------------
def calculate_woe_iv(
    df: pd.DataFrame, feature: str, target: str, bins: int = 10
) -> tuple[pd.DataFrame, float]:
    """Compute the WoE table and total IV of ``feature`` against ``target``.

    Numeric features are bucketed into quantile ``bins``; categorical features
    use their raw categories. Returns ``(woe_table, information_value)``.

    WoE  = ln(%non-events / %events)
    IV   = Σ (%non-events - %events) * WoE
    """
    data = df[[feature, target]].copy()

    if pd.api.types.is_numeric_dtype(data[feature]):
        data["bucket"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bucket"] = data[feature].astype(str)

    grouped = data.groupby("bucket", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["total", "events"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    # Laplace smoothing to avoid divide-by-zero in empty buckets.
    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()
    grouped["pct_events"] = (grouped["events"] + 0.5) / (total_events + 0.5)
    grouped["pct_non_events"] = (grouped["non_events"] + 0.5) / (total_non_events + 0.5)

    grouped["woe"] = np.log(grouped["pct_non_events"] / grouped["pct_events"])
    grouped["iv"] = (grouped["pct_non_events"] - grouped["pct_events"]) * grouped["woe"]

    information_value = grouped["iv"].sum()
    return grouped.reset_index(), float(information_value)


def information_value_report(
    df: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
    """Rank features by predictive power (Information Value)."""
    rows = []
    for feat in features:
        try:
            _, iv = calculate_woe_iv(df, feat, target)
            rows.append({"feature": feat, "iv": iv, "strength": _iv_strength(iv)})
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append({"feature": feat, "iv": np.nan, "strength": f"error: {exc}"})
    return pd.DataFrame(rows).sort_values("iv", ascending=False, na_position="last")


def _iv_strength(iv: float) -> str:
    """Standard IV interpretation bands used in credit scoring."""
    if iv < 0.02:
        return "not predictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (too strong)"


# ---------------------------------------------------------------------------
# Task 4: RFM + K-Means proxy target
# ---------------------------------------------------------------------------
def compute_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute Recency / Frequency / Monetary metrics per customer.

    ``snapshot_date`` anchors Recency. If not supplied, it defaults to one day
    after the latest transaction so the most recent customer still has R >= 1.
    """
    data = df.copy()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL], errors="coerce")

    if snapshot_date is None:
        snapshot_date = data[TIME_COL].max() + pd.Timedelta(days=1)

    rfm = data.groupby(CUSTOMER_KEY).agg(
        Recency=(TIME_COL, lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum"),
    )
    return rfm.reset_index()


def assign_high_risk_label(
    rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Cluster customers on scaled RFM and label the high-risk segment.

    The high-risk cluster is chosen **programmatically** (not hard-coded): after
    clustering, each cluster's mean RFM profile is scored as
    ``recency_rank + low_frequency_rank + low_monetary_rank``; the cluster with
    the worst (highest) score - high recency, low frequency, low monetary - is
    flagged ``is_high_risk = 1``.
    """
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    # Mean RFM profile per cluster, in original units.
    profile = rfm.groupby("Cluster")[features].mean()

    # Higher recency = worse; lower frequency/monetary = worse. Rank so that the
    # worst value gets the highest score, then sum the three ranks.
    risk_score = (
        profile["Recency"].rank(ascending=True)        # high recency -> high rank
        + profile["Frequency"].rank(ascending=False)   # low frequency -> high rank
        + profile["Monetary"].rank(ascending=False)    # low monetary -> high rank
    )
    high_risk_cluster = int(risk_score.idxmax())

    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)
    return rfm


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_processed_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Produce the model-ready, customer-level dataset *with* the proxy target.

    Returns interpretable (un-scaled) aggregate features plus ``is_high_risk``.
    Scaling/encoding is applied later inside the training pipeline to avoid
    train/test leakage.
    """
    # Customer-level aggregate + datetime + dominant-category features.
    customer_features = CustomerAggregator().fit_transform(raw)

    # Proxy target from RFM clustering.
    rfm = compute_rfm(raw)
    rfm_labeled = assign_high_risk_label(rfm)

    processed = customer_features.merge(
        rfm_labeled[[CUSTOMER_KEY, "Recency", "Frequency", "Monetary", "is_high_risk"]],
        on=CUSTOMER_KEY,
        how="left",
    )
    return processed


def process_data(
    input_path: str, output_path: str, pipeline_path: str | None = None
) -> pd.DataFrame:
    """Load raw data, build the processed dataset, persist it, and fit/save the pipeline."""
    raw = pd.read_csv(input_path)
    processed = build_processed_dataset(raw)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed.to_csv(output_path, index=False)
    print(f"[data_processing] processed dataset -> {output_path}  shape={processed.shape}")

    target_dist = processed["is_high_risk"].value_counts().to_dict()
    print(f"[data_processing] is_high_risk distribution: {target_dist}")

    # Fit and persist the single Pipeline object (Task 3 deliverable).
    if pipeline_path:
        import joblib

        pipeline = build_feature_pipeline()
        pipeline.fit(raw)
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(pipeline, pipeline_path)
        print(f"[data_processing] fitted pipeline -> {pipeline_path}")

    # Information Value report for interpretable feature ranking (Task 3.6).
    iv_features = [c for c in NUMERIC_FEATURES if c in processed.columns]
    iv_report = information_value_report(processed, iv_features, "is_high_risk")
    print("[data_processing] Information Value report:")
    print(iv_report.to_string(index=False))

    return processed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bati Bank data processing pipeline")
    parser.add_argument("--input", default="data/raw/data.csv")
    parser.add_argument("--output", default="data/processed/processed_data.csv")
    parser.add_argument("--pipeline", default="models/processing_pipeline.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_data(args.input, args.output, args.pipeline)
