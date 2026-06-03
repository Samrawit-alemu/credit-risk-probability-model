"""
tests/test_data_processing.py
=============================

Unit tests for helper functions in :mod:`src.data_processing` (Task 5.6).

The tests use a small, hand-built transaction frame so they run fast and do not
depend on the full raw dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    CustomerAggregator,
    assign_high_risk_label,
    build_processed_dataset,
    calculate_woe_iv,
    compute_rfm,
)


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """A tiny, deterministic transaction-level dataset (3 customers)."""
    return pd.DataFrame(
        {
            "TransactionId": [f"T{i}" for i in range(8)],
            "CustomerId": ["C1", "C1", "C1", "C2", "C2", "C3", "C3", "C3"],
            "Amount": [100.0, 200.0, 50.0, 1000.0, 1500.0, 10.0, 20.0, 5.0],
            "Value": [100.0, 200.0, 50.0, 1000.0, 1500.0, 10.0, 20.0, 5.0],
            "ProductId": ["P1", "P2", "P1", "P3", "P3", "P1", "P1", "P2"],
            "ProductCategory": [
                "airtime", "data", "airtime",
                "financial_services", "financial_services",
                "airtime", "airtime", "data",
            ],
            "ChannelId": [
                "ChannelId_3", "ChannelId_3", "ChannelId_2",
                "ChannelId_3", "ChannelId_3",
                "ChannelId_2", "ChannelId_2", "ChannelId_2",
            ],
            "ProviderId": [
                "ProviderId_1", "ProviderId_1", "ProviderId_2",
                "ProviderId_4", "ProviderId_4",
                "ProviderId_6", "ProviderId_6", "ProviderId_6",
            ],
            "FraudResult": [0, 0, 0, 0, 1, 0, 0, 0],
            "TransactionStartTime": pd.to_datetime(
                [
                    "2018-11-01T08:00:00Z", "2018-11-05T09:00:00Z",
                    "2018-11-10T10:00:00Z", "2018-11-02T11:00:00Z",
                    "2018-11-15T12:00:00Z", "2018-11-03T13:00:00Z",
                    "2018-11-04T14:00:00Z", "2018-11-06T15:00:00Z",
                ]
            ),
        }
    )


def test_aggregator_returns_one_row_per_customer(sample_transactions):
    """CustomerAggregator should collapse transactions to one row per customer
    and expose every engineered feature column."""
    out = CustomerAggregator().fit_transform(sample_transactions)

    # One row per unique customer.
    assert len(out) == sample_transactions["CustomerId"].nunique() == 3

    # All declared model feature columns are produced.
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        assert col in out.columns, f"missing engineered column: {col}"

    # Aggregates are correct for customer C1 (amounts 100, 200, 50).
    c1 = out.loc[out["CustomerId"] == "C1"].iloc[0]
    assert c1["TotalAmount"] == pytest.approx(350.0)
    assert c1["TransactionCount"] == 3
    assert c1["AvgAmount"] == pytest.approx(350.0 / 3)


def test_compute_rfm_shape_and_values(sample_transactions):
    """compute_rfm should return Recency/Frequency/Monetary for each customer."""
    rfm = compute_rfm(sample_transactions)

    assert set(["Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert len(rfm) == 3

    # Frequency equals the transaction count per customer.
    c3 = rfm.loc[rfm["CustomerId"] == "C3"].iloc[0]
    assert c3["Frequency"] == 3
    assert c3["Monetary"] == pytest.approx(35.0)  # 10 + 20 + 5

    # Recency is non-negative for every customer.
    assert (rfm["Recency"] >= 0).all()


def test_assign_high_risk_label_is_binary(sample_transactions):
    """The proxy label must be a binary 0/1 column covering all customers."""
    rfm = compute_rfm(sample_transactions)
    labeled = assign_high_risk_label(rfm, n_clusters=3)

    assert "is_high_risk" in labeled.columns
    assert set(labeled["is_high_risk"].unique()).issubset({0, 1})
    assert labeled["is_high_risk"].notna().all()


def test_build_processed_dataset_has_target_and_no_nulls(sample_transactions):
    """The integrated processed dataset must contain is_high_risk with no nulls."""
    processed = build_processed_dataset(sample_transactions)

    assert "is_high_risk" in processed.columns
    assert processed["is_high_risk"].isnull().sum() == 0
    assert len(processed) == 3


def test_calculate_woe_iv_returns_non_negative_iv():
    """Information Value is non-negative and the WoE table aligns with input."""
    df = pd.DataFrame(
        {
            "feature": np.r_[np.zeros(50), np.ones(50)],
            "target": np.r_[np.zeros(40), np.ones(10), np.zeros(10), np.ones(40)],
        }
    )
    woe_table, iv = calculate_woe_iv(df, "feature", "target", bins=2)

    assert iv >= 0
    assert "woe" in woe_table.columns
    assert len(woe_table) >= 1
