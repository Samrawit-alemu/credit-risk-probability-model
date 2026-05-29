Credit Risk Probability Model for Alternative Data

Project Overview

This project aims to build a Credit Scoring Model for Bati Bank in partnership
with an eCommerce platform. Since the platform lacks historical "default"
labels, we are engineering a proxy for credit risk using customer behavioral
data (RFM analysis) and building an end-to-end machine learning pipeline to
predict creditworthiness.

Credit Scoring Business Understanding

1. Basel II Accord and Model Interpretability

Question: Why is it important for a bank's model to be "interpretable" rather
than a "black box"?

Answer: Under the Basel II Capital Accord, financial institutions are required
to maintain a transparent and rigorous process for risk measurement.
Interpretability is crucial for three reasons:

- Regulatory Compliance: Regulators require banks to explain why a specific
  credit decision was made. A "black box" model that cannot justify an
  approval or rejection is legally indefensible.
- Fairness and Bias: Interpretability allows us to ensure the model isn't
  using "proxy" features that discriminate against protected groups (e.g.,
  gender, ethnicity, or location).
- Risk Management: If a model predicts high risk, the bank needs to understand
  which behaviors are driving that risk to adjust interest rates or loan terms
  effectively.

2. The Use of Proxy Variables (RFM Patterns)

Question: Why are we using RFM (Recency, Frequency, Monetary) patterns to guess
"High Risk," and what are the dangers?

Answer: In this dataset, we do not have a "default" label (ground truth).
Therefore, we use RFM analysis as a proxy:

- Why RFM? Customers with high Recency (haven't bought in a long time), low
  Frequency (rarely use the service), and low Monetary value (low spend) are
  statistically more likely to be "disengaged." In a credit context,
  disengaged users are treated as higher risk because they lack a proven track
  record of consistent platform usage.
- The Dangers: Using a proxy is an assumption, not a fact. The main risk is
  Misclassification. A customer might have low frequency simply because they
  are new or use other platforms, not because they are financially unstable.
  This can lead to the bank rejecting "Good" customers (False Positives),
  resulting in lost revenue and poor customer experience.

3. Weight of Evidence (WoE) vs. Complex Models

Question: Why would a bank prefer a simple Logistic Regression with WoE over a
high-performance model like XGBoost?

Answer: In a regulated financial context, the trade-off is often between
predictive power and auditability:

- Scorecard Format: Logistic Regression combined with WoE (Weight of Evidence)
  allows the creation of a "Credit Scorecard." Each feature (e.g., Age,
  Transaction Amount) is assigned a specific number of points. This is easy
  for non-technical bank staff and customers to understand.
- Stability: Simple models are often more stable when the economy changes.
  Complex models like XGBoost or Gradient Boosting can "overfit" to specific
  noise in the data, making them unpredictable in real-world shifts.
- Linearity: WoE transforms non-linear relationships into linear ones, making
  it easier to see exactly how a change in a customer's behavior will affect
  their credit score.
