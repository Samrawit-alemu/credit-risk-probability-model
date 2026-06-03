# Credit Risk Probability Model for Alternative Data

## Project Overview

This project builds a Credit Scoring Model for **Bati Bank** in partnership with
an eCommerce platform. Since the platform lacks historical "default" labels, we
engineer a **proxy for credit risk** from customer behavioral data (RFM
analysis + K-Means clustering) and build an end-to-end machine-learning pipeline
to predict creditworthiness, served through a containerized REST API.

## Project Structure

```
credit-risk-probability-model/
├── .github/workflows/ci.yml      # CI: flake8 + pytest on push/PR to main
├── data/
│   ├── raw/                      # Raw Xente transaction data
│   └── processed/                # Model-ready customer-level dataset (generated)
├── models/                       # Fitted pipeline + best model (generated)
├── notebooks/eda.ipynb           # Task 2 exploratory analysis
├── src/
│   ├── data_processing.py        # Task 3 + 4: feature pipeline + proxy target
│   ├── train.py                  # Task 5: training, MLflow tracking, registry
│   └── api/
│       ├── main.py               # Task 6: FastAPI service
│       └── pydantic_models.py    # Request/response schemas
├── tests/test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the processed dataset + proxy target, and fit the feature pipeline
python -m src.data_processing

# 3. Train models, log to MLflow, register the best model
python -m src.train
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: inspect runs

# 4. Run the unit tests and linter (same as CI)
flake8 src tests
pytest -v

# 5. Serve the API
uvicorn src.api.main:app --reload          # then open http://localhost:8000/docs
# or, with Docker:
docker-compose up --build
```

Example request to `POST /predict`:

```json
{
  "TotalAmount": 350000.0, "AvgAmount": 5000.0, "StdAmount": 2500.0,
  "MinAmount": 100.0, "MaxAmount": 50000.0, "TransactionCount": 70,
  "TotalValue": 360000.0, "AvgValue": 5142.0, "AvgHour": 12.5,
  "AvgDay": 15.0, "AvgMonth": 12.0, "UniqueProducts": 8, "FraudCount": 0,
  "TopProductCategory": "airtime", "TopChannelId": "ChannelId_3",
  "TopProviderId": "ProviderId_6"
}
```

---

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

**Why is it important for a bank's model to be "interpretable" rather than a
"black box"?**

Under the Basel II Capital Accord, financial institutions are required to
maintain a transparent and rigorous process for risk measurement.
Interpretability is crucial for three reasons:

- **Regulatory Compliance:** Regulators require banks to explain why a specific
  credit decision was made. A "black box" model that cannot justify an approval
  or rejection is legally indefensible.
- **Fairness and Bias:** Interpretability lets us ensure the model isn't using
  proxy features that discriminate against protected groups (e.g., gender,
  ethnicity, or location).
- **Risk Management:** If a model predicts high risk, the bank needs to
  understand which behaviors drive that risk to adjust interest rates or loan
  terms effectively.

### 2. The Use of Proxy Variables (RFM Patterns)

**Why are we using RFM (Recency, Frequency, Monetary) patterns to estimate
"High Risk," and what are the dangers?**

This dataset has no "default" label (ground truth), so we use RFM analysis as a
proxy:

- **Why RFM?** Customers with high Recency (haven't bought in a long time), low
  Frequency (rarely use the service), and low Monetary value (low spend) are
  statistically more likely to be disengaged. In a credit context, disengaged
  users are treated as higher risk because they lack a proven track record of
  consistent platform usage.
- **The Dangers:** A proxy is an assumption, not a fact. The main risk is
  **misclassification**. A customer might transact infrequently simply because
  they are new or use other platforms — not because they are financially
  unstable. This can lead the bank to reject "good" customers (false positives),
  resulting in lost revenue and poor customer experience, or to approve genuinely
  risky customers whose behavior happens to mimic the "safe" cluster.

### 3. Weight of Evidence (WoE) vs. Complex Models

**Why might a bank prefer a simple Logistic Regression with WoE over a
high-performance model like Gradient Boosting?**

In a regulated context, the trade-off is between predictive power and
auditability:

- **Scorecard Format:** Logistic Regression combined with WoE allows a "credit
  scorecard" where each feature is assigned a transparent number of points —
  easy for non-technical staff and customers to understand.
- **Stability:** Simple models are often more stable under economic shifts.
  Complex models like Gradient Boosting can overfit noise, making them less
  predictable in real-world distribution changes.
- **Linearity:** WoE transforms non-linear relationships into a monotonic linear
  form, making it easy to see exactly how a change in behavior affects the score.
- **The counter-argument:** Gradient Boosting typically delivers higher
  discriminative power (ROC-AUC). The defensible choice depends on whether the
  marginal accuracy justifies the added cost of explainability tooling (e.g.,
  SHAP) and the regulatory burden of validating a more complex model.
