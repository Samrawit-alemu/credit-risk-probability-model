"""
src/api/pydantic_models.py
==========================

Request/response schemas for the credit-risk API (Task 6).

The request fields mirror the customer-level features the model was trained on
(see ``NUMERIC_FEATURES`` and ``CATEGORICAL_FEATURES`` in
:mod:`src.data_processing`). ``CustomerFeatures.to_frame`` converts a validated
request into the single-row DataFrame the model pipeline expects.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Aggregated, customer-level features for a single prediction request."""

    # Numeric aggregate features.
    TotalAmount: float = Field(..., description="Sum of transaction amounts")
    AvgAmount: float = Field(..., description="Mean transaction amount")
    StdAmount: float = Field(..., description="Std dev of transaction amounts")
    MinAmount: float = Field(..., description="Minimum transaction amount")
    MaxAmount: float = Field(..., description="Maximum transaction amount")
    TransactionCount: int = Field(..., ge=0, description="Number of transactions")
    TotalValue: float = Field(..., description="Sum of absolute transaction values")
    AvgValue: float = Field(..., description="Mean absolute transaction value")
    AvgHour: float = Field(..., ge=0, le=23, description="Average transaction hour")
    AvgDay: float = Field(..., ge=1, le=31, description="Average day of month")
    AvgMonth: float = Field(..., ge=1, le=12, description="Average month")
    UniqueProducts: int = Field(..., ge=0, description="Distinct products purchased")
    FraudCount: int = Field(..., ge=0, description="Number of flagged-fraud txns")

    # Categorical (dominant value per customer).
    TopProductCategory: str = Field(..., description="Most frequent product category")
    TopChannelId: str = Field(..., description="Most frequent channel")
    TopProviderId: str = Field(..., description="Most frequent provider")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TotalAmount": 350000.0,
                "AvgAmount": 5000.0,
                "StdAmount": 2500.0,
                "MinAmount": 100.0,
                "MaxAmount": 50000.0,
                "TransactionCount": 70,
                "TotalValue": 360000.0,
                "AvgValue": 5142.0,
                "AvgHour": 12.5,
                "AvgDay": 15.0,
                "AvgMonth": 12.0,
                "UniqueProducts": 8,
                "FraudCount": 0,
                "TopProductCategory": "airtime",
                "TopChannelId": "ChannelId_3",
                "TopProviderId": "ProviderId_6",
            }
        }
    }

    def to_frame(self) -> pd.DataFrame:
        """Return a single-row DataFrame matching the model's feature columns."""
        return pd.DataFrame([self.model_dump()])


class PredictionResponse(BaseModel):
    """Risk prediction returned by the ``/predict`` endpoint."""

    is_high_risk: int = Field(..., description="1 = high risk, 0 = low risk")
    risk_probability: float = Field(..., description="Probability of high risk [0-1]")
    recommendation: str = Field(..., description="Approve or Reject")
