# Future Disengagement Prediction — COFINFAD

Predicting which customers of a Colombian fintech will show **transactional
disengagement** (silence or sharp decline) in Q4 2023, using only data
observable through September 2023.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Temporal Design](#temporal-design)
4. [Target Definition](#target-definition)
5. [Feature Engineering](#feature-engineering)
6. [Three Feature Setups](#three-feature-setups)
7. [Modeling](#modeling)
8. [Leakage Controls](#leakage-controls)
9. [Evaluation Protocol](#evaluation-protocol)
10. [Results](#results)
11. [Interactive Dashboard](#interactive-dashboard)
12. [Project Structure](#project-structure)
13. [How to Run](#how-to-run)
14. [Requirements](#requirements)
15. [Limitations](#limitations)

---

## Overview

No contractual churn label exists in the COFINFAD dataset. The
pre-computed `churn_probability` column is a continuous score (0.1–0.5),
not a ground-truth event — using it as a feature or target would create
circularity. We therefore frame the task as **future transactional
disengagement prediction**: can we identify, from Jan–Sep behaviour,
which customers will become silent or sharply decline in Oct–Dec?

**Key results (test set):**

| Setup | Best Model | ROC AUC | PR AUC | Lift@10% |
|-------|-----------|---------|--------|----------|
| Full | XGBoost | 0.795 | 0.059 | 2.59× |
| Tx-strict | RF | 0.786 | 0.067 | 2.46× |
| Cust-only | XGBoost | 0.495 | 0.024 | 0.92× |

The **Tx-strict** setup — using only transaction-derived features — retains
nearly all predictive signal, confirming that temporal transaction patterns
drive the prediction, not customer-level attributes.

---

## Dataset

**COFINFAD** — Colombian Fintech Financial Analytics Dataset.

| Item | Detail |
|------|--------|
| Customers | 48,723 |
| Transactions | 3,159,157 |
| Period | January – December 2023 |
| Source files | `customer_data.csv` (54 columns), `transactions_data.csv` (4 columns) |

Data is **not included** in this repository (exceeds GitHub size limits).
Place the CSV files in `data/COFINFAD Colombian Fintech Financial Analytics Dat/`.

---

## Temporal Design

```
┌──────────────────────────────────────┐  ┌─────────────────────┐
│    OBSERVATION WINDOW                │  │  PREDICTION WINDOW  │
│    Jan – Sep 2023                    │  │  Oct – Dec 2023     │
│                                      │  │                     │
│  Features built here:                │  │  Target defined     │
│  • RFM aggregates                    │  │  here:              │
│  • Transaction type shares           │  │  future_disengaged  │
│  • Trends, gaps, weekend ratio       │  │  (0 or 1)           │
│  • Customer attributes (snapshot)    │  │                     │
└──────────────────────────────────────┘  └─────────────────────┘
                        │ ← No information crosses Sep 30 →
```

---

## Target Definition

`future_disengaged = 1` when a customer has **zero Q4 transactions** or
a Q4 monthly transaction rate **below 20%** of their Jan–Sep rate.

This is a **behavioural proxy** for disengagement, not contractual churn.
In production it would be replaced by actual churn events (account closures,
cancellations) or business-defined inactivity rules.

| Class | Count | Rate |
|-------|-------|------|
| Active (0) | 47,581 | 97.7% |
| Disengaged (1) | 1,142 | 2.3% |

---

## Feature Engineering

All transaction features are built from raw `transactions_data.csv`
restricted to the observation window (Jan–Sep 2023).

### Transaction-Derived (20 features)

- **RFM:** count, total, mean, median, std, CV, min, max, recency, span, frequency
- **Type shares:** deposit, payment, transfer, withdrawal
- **Behavioural:** weekend ratio, count trend, amount trend, max gap, late ratio

### Customer Attributes (24 features)

Demographics, product holdings, digital engagement, satisfaction, NPS, support
interactions, tenure. These are **not timestamped** — treated as observation-cutoff
snapshots with a documented assumption (see §2 of the notebook).

---

## Three Feature Setups

| Setup | Features | Purpose |
|-------|----------|---------|
| **Full** | Customer attributes + tx features (44 raw → 47 encoded) | Best available information |
| **Tx-strict** | Transaction features only (20) | Ultra-strict benchmark — zero customer-level variables |
| **Cust-only** | Customer attributes only (24 raw → 27 encoded) | Baseline — demographics + product flags |

---

## Modeling

Three algorithms trained per setup (9 models total):

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| **Logistic Regression** | C=1.0, balanced class weights |
| **Random Forest** | 300 trees, max_depth=8, balanced weights |
| **XGBoost** | 300 rounds, max_depth=4, lr=0.05, scale_pos_weight |

All preprocessing (imputation, scaling, encoding) is handled by
`ColumnTransformer` pipelines **fit only on the training split**.

---

## Leakage Controls

1. **Pre-computed aggregates** in `customer_data` spanning the full year
   are excluded (25 columns). All transaction features are rebuilt from raw data.
2. **`churn_probability`, `customer_lifetime_value`, `clv_segment`** are
   never used as features.
3. **`customer_segment`** excluded — may incorporate future-window behaviour.
4. **Preprocessing** fit on train only — no distribution leakage.
5. **Customer-level variables** documented as assumed snapshots; Tx-strict
   benchmark verifies signal without them.

---

## Evaluation Protocol

| Split | Size | Role |
|-------|------|------|
| Train | 29,233 (60%) | Model fitting |
| Validation | 9,745 (20%) | Threshold tuning, early stopping |
| Test | 9,745 (20%) | Final held-out evaluation only |

**Threshold** is tuned on validation (maximise F1). **Test set is never
used** for model selection or threshold tuning.

Metrics: ROC AUC, PR AUC, Brier score, F1, precision, recall,
precision@10%, recall@10%, lift at top decile, calibration curves.

---

## Results

### All Models (Test Set)

| Setup | Model | ROC AUC | PR AUC | Brier | F1 | Lift@D1 |
|-------|-------|---------|--------|-------|-----|---------|
| Full | XGB | 0.795 | 0.059 | 0.129 | 0.110 | 2.59 |
| Full | RF | 0.785 | 0.058 | 0.154 | 0.097 | 2.50 |
| Full | LR | 0.611 | 0.030 | 0.236 | 0.058 | 1.10 |
| Tx-strict | RF | 0.786 | 0.067 | 0.164 | 0.098 | 2.46 |
| Tx-strict | XGB | 0.784 | 0.056 | 0.136 | 0.100 | 2.59 |
| Tx-strict | LR | 0.618 | 0.030 | 0.237 | 0.062 | 1.01 |
| Cust-only | XGB | 0.495 | 0.024 | 0.175 | 0.043 | 0.92 |
| Cust-only | RF | 0.489 | 0.023 | 0.188 | 0.029 | 0.83 |
| Cust-only | LR | 0.485 | 0.023 | 0.249 | 0.043 | 0.97 |

### Key Findings

- **Transaction features carry all the signal.** Tx-strict matches Full
  within 1% ROC AUC; Cust-only is near random (0.495).
- **Tree-based models dominate.** XGBoost and RF significantly outperform
  Logistic Regression.
- **Severe class imbalance** (2.3%) makes PR AUC the more informative
  metric. Absolute F1 is low but lift at top decile is strong (2.5×).
- **`churn_probability` sanity check:** Pearson ρ ≈ 0.01 — our model
  captures a different construct than the dataset's pre-computed score.

---

## Interactive Dashboard

```bash
streamlit run app.py
```

Six pages:

| Page | Content |
|------|---------|
| **Overview** | Key metrics, temporal design, dataset summary |
| **Data Explorer** | Target distribution, feature histograms by class |
| **Model Performance** | Full metrics table, ROC/PR/calibration curves, confusion matrices |
| **Setup Comparison** | Full vs Tx-strict vs Cust-only bar charts, ROC overlays, lift |
| **Threshold Analysis** | Validation sweep, interactive threshold slider, test report |
| **Explainability** | XGBoost importance, LR coefficients, SHAP analysis |

---

## Project Structure

```
├── README.md
├── requirements.txt
├── app.py                          # Streamlit dashboard
├── .gitignore
├── data/                           # (not tracked — place CSVs here)
│   └── COFINFAD .../
│       ├── customer_data.csv
│       └── transactions_data.csv
├── notebooks/
│   ├── churn_modeling.py           # Full pipeline (percent-script)
│   └── churn_modeling.ipynb        # Same pipeline as Jupyter notebook
└── outputs/                        # (not tracked — generated by pipeline)
    └── churn_artifacts.pkl         # Pickled models + predictions for dashboard
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place COFINFAD data in data/ directory

# 3. Run the pipeline (generates models + artifacts)
python3 notebooks/churn_modeling.py

# 4. Launch the dashboard
streamlit run app.py
```

---

## Requirements

- Python ≥ 3.9
- numpy, pandas, scikit-learn, xgboost, shap, matplotlib, streamlit

See [requirements.txt](requirements.txt) for pinned versions.

---

## Limitations

1. **No contractual churn event** — the target is a behavioural proxy.
2. **Single temporal split** — one observation/prediction window pair;
   no walk-forward validation.
3. **Customer attributes assumed static** — untimestamped variables may
   incorporate future information. The Tx-strict benchmark mitigates this.
4. **12-month dataset** — limits generalisability to seasonal patterns.
5. **Class imbalance** — ~2% disengagement rate makes precision on the
   minority class challenging.