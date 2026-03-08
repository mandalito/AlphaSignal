# AlphaSignal — Client Intelligence & Distribution Analytics Platform

A **three-signal client intelligence system** that transforms raw fintech
transaction data into calibrated, risk-adjusted commercial opportunity
scores — enabling relationship managers to prioritise upsell, retention,
and monitoring actions with quantitative precision.

Built on the **COFINFAD** Colombian Fintech Financial Analytics Dataset
(48,723 clients · 3.2 M transactions · 2023).

*Prepared for Pictet Asset Management — EPFL Machine Learning in Finance.*

**Live demo:** [alphasignal.streamlit.app](https://alphasignal.streamlit.app/)

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Dataset](#dataset)
4. [Temporal Design](#temporal-design)
5. [Target Definitions](#target-definitions)
6. [Feature Engineering](#feature-engineering)
7. [Three Feature Setups](#three-feature-setups)
8. [Modeling](#modeling)
9. [Leakage Controls](#leakage-controls)
10. [Evaluation Protocol](#evaluation-protocol)
11. [Walk-Forward Temporal Validation](#walk-forward-temporal-validation)
12. [Probability Calibration](#probability-calibration)
13. [Results — Redemption Risk](#results--redemption-risk)
14. [Multi-Signal Architecture](#multi-signal-architecture)
15. [Master Commercial Signal](#master-commercial-signal)
16. [Opportunity Frontier](#opportunity-frontier)
17. [Client Prioritisation](#client-prioritisation)
18. [Strategic Report](#strategic-report)
19. [Project Structure](#project-structure)
20. [How to Run](#how-to-run)
21. [Requirements](#requirements)
22. [Limitations](#limitations)

---

## Overview

The project implements a **client intelligence and distribution analytics
platform** built on the COFINFAD dataset. No contractual churn label exists
— the pre-computed `churn_probability` column is a continuous score
(0.1–0.5), not a ground-truth event. We therefore construct three
behavioral signals from transaction data:

```
transactions data
       ↓
feature engineering (Jan–Sep 2023)
       ↓
ML models + rules
       ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Redemption Risk  │  │ Buy Propensity  │  │ Engagement Score│
│ P(disengagement) │  │ P(growth)       │  │ behavioral health│
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         └──────────┬─────────┘──────────┬─────────┘
                    ↓                    ↓
           Master Commercial Signal
                    ↓
        Commercial Opportunity Ranking
```

**Key results (test set):**

| Signal | Best Model | ROC AUC |
|--------|-----------|---------|
| Redemption Risk (Full) | XGBoost | 0.795 |
| Redemption Risk (Tx-strict) | RF | 0.786 |
| Buy Propensity (Tx-strict) | RF | 0.779 |

**Master Signal outputs:** 8,130 Upsell / 20,136 Retention / 20,457 Monitor

---

## System Architecture

The system transforms raw behavioral transaction data into a multi-signal
intelligence framework for client engagement prioritisation. The
architecture follows a four-stage analytical pipeline used in asset
management distribution:

1. **Behavioural Prediction** — ML models estimate disengagement risk and growth propensity
2. **Signal Architecture** — Calibrated probabilities combined with engagement intensity
3. **Economic Translation** — Signals converted into Expected Client Value
4. **Distribution Intelligence** — Risk-adjusted scores allocate relationship management effort

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA                                 │
│   customer_data.csv (48,723 clients × 54 features)              │
│   transactions_data.csv (3.2M transactions, 2023)               │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│              BEHAVIOURAL FEATURE ENGINEERING                     │
│   Observation window: Jan – Sep 2023                            │
│   • RFM aggregates (count, total, mean, recency, frequency)    │
│   • Transaction-type shares (deposit, payment, transfer, …)    │
│   • Trends (count slope, amount slope, late-ratio)              │
│   • Behavioural indicators (max gap, weekend ratio, CV)         │
│   • Customer attributes (demographics, products, satisfaction)  │
└──────────────────────────┬──────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SIGNAL LAYER                               │
│                                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Redemption Risk  │ │ Buy Propensity  │ │ Engagement Score│   │
│  │ P(disengagement) │ │ P(growth)       │ │ behavioural idx │   │
│  │ XGBoost (calib.) │ │ RF (calibrated) │ │ rule-based      │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           └──────────┬────────┘───────────┬───────┘             │
│                      ↓                    ↓                     │
│         Isotonic Probability Calibration                        │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│               MASTER COMMERCIAL SIGNAL                          │
│   BuyPropensity × (1 − RedemptionRisk) × EngagementScore       │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │  Expected Client Value = Master × tx    │                   │
│   └─────────────────────────────────────────┘                   │
│   ┌─────────────────────────────────────────┐                   │
│   │  Opportunity Frontier Score = ECV/Risk  │                   │
│   └─────────────────────────────────────────┘                   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│        COMMERCIAL & DISTRIBUTION INTELLIGENCE                   │
│   12-section strategic report: executive summary, data          │
│   description, temporal strategy, features, models,             │
│   calibration, signal architecture, expected client value,      │
│   opportunity frontier, commercial intelligence,                │
│   distribution intelligence & strategic interpretation          │
└─────────────────────────────────────────────────────────────────┘
```

**Key architectural principles:**
- **Temporal integrity:** strict observation/prediction window separation
- **Calibrated probabilities:** isotonic calibration ensures signal
  reliability for expected-value calculations
- **Walk-forward validation:** expanding-window validation demonstrates
  temporal stability of behavioral signals
- **Risk-aware prioritisation:** Opportunity Frontier Score provides a
  Sharpe-ratio-like metric for client ranking

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

## Target Definitions

### Redemption Risk — `future_disengaged`

`future_disengaged = 1` when a customer has **zero Q4 transactions** or
a Q4 monthly transaction rate **below 20%** of their Jan–Sep rate.

| Class | Count | Rate |
|-------|-------|------|
| Active (0) | 47,581 | 97.7% |
| Disengaged (1) | 1,142 | 2.3% |

### Buy Propensity — `future_growth`

`future_growth = 1` when a customer's Q4 monthly transaction rate reaches
**at least 1.5×** their Jan–Sep monthly rate.

| Class | Count | Rate |
|-------|-------|------|
| No growth (0) | 39,134 | 80.3% |
| Growth (1) | 9,589 | 19.7% |

Both targets are **behavioural proxies** — not contractual events. In
production they would be replaced by business-defined labels.

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

## Walk-Forward Temporal Validation

To demonstrate model stability across time, we implement expanding-window
walk-forward validation. Each fold rebuilds RFM features and the
disengagement target from a different temporal cut:

| Fold | Observation Window | Prediction Window |
|------|-------------------|-------------------|
| 1 | Jan – May 2023 | Jun – Jul 2023 |
| 2 | Jan – Jun 2023 | Jul – Aug 2023 |
| 3 | Jan – Jul 2023 | Aug – Sep 2023 |
| 4 | Jan – Aug 2023 | Sep – Oct 2023 |
| 5 | Jan – Sep 2023 | Oct – Nov 2023 |

For each fold, an XGBoost model is trained and evaluated using ROC AUC,
PR AUC, and Brier score. The procedure demonstrates that the behavioral
disengagement signal is not dependent on a single temporal split — it
remains stable as the training window expands.

**Conclusion:** Low standard deviation in ROC AUC across folds confirms
that the transaction-derived behavioral signals provide consistent
predictive value regardless of the observation window cutoff.

---

## Probability Calibration

Raw model probabilities may not correspond to true event frequencies.
We apply **isotonic calibration** on the validation set to both the
Redemption Risk and Buy Propensity models.

**Why calibration matters:**
- The Master Signal formula is multiplicative:
  `BuyPropensity × (1 − RedemptionRisk) × Engagement`
- This product is meaningful only if probabilities are well-calibrated
- A prediction of 0.7 should correspond to ~70% observed event frequency
- The Expected Client Value metric inherits any probability distortion

**Evaluation:**
- Calibration curves (reliability diagrams) before vs after calibration
- Brier score comparison (lower = more reliable probabilities)
- Isotonic calibration is fit on the validation set (never test)

---

## Results — Redemption Risk

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

## Multi-Signal Architecture

Three behavioral signals are derived from the same observation-window
features:

### Signal 1 — Redemption Risk (ML)

`P(future_disengaged)` — probability the customer becomes inactive.
Uses the Full-setup XGBoost model (best ROC AUC).

### Signal 2 — Buy Propensity (ML)

`P(future_growth)` — probability the customer increases activity ≥ 1.5×.
Trained on Tx-strict features using the same pipeline (LR, RF, XGBoost).

| Model | ROC AUC |
|-------|--------|
| RF | 0.779 |
| XGBoost | 0.777 |
| LR | 0.756 |

### Signal 3 — Engagement Score (Rule-Based)

Weighted composite of MinMax-normalised behavioral features:

```
engagement = 0.35 × tx_freq_norm
           + 0.25 × tx_count_trend_norm
           + 0.20 × (1 − tx_recency_norm)
           + 0.20 × (1 − tx_max_gap_norm)
```

Interpretation: 0 = disengaged, 1 = highly active.

---

## Master Commercial Signal

All three signals are normalised to [0, 1] and combined into an expected
commercial opportunity score:

```
CommercialOpportunity = BuyPropensity × (1 − RedemptionRisk) × EngagementScore
```

The result is normalised to [0, 1] via MinMaxScaler to produce the Master Signal.

### Commercial Interpretation

The Master Signal approximates expected commercial opportunity:

```
CommercialOpportunity ≈ P(growth) × Engagement × (1 − DisengagementRisk)
```

This formulation mirrors expected-value scoring frameworks used in client
intelligence systems in asset management distribution. Rather than predicting
a single behavioral outcome, the signal integrates:

- **Growth potential** — probability of future activity increase
- **Client engagement strength** — behavioural health indicator
- **Downside risk** — probability of disengagement

to produce a unified opportunity score that can guide commercial prioritisation.

### Expected Client Value

A monetised prioritisation metric combining opportunity with historical
activity:

```
ExpectedClientValue = MasterSignal × tx_total
```

This highlights clients who combine **high opportunity** with **high
potential business value**, enabling dollar-weighted commercial prioritisation.

---

## Opportunity Frontier

### Opportunity Frontier Visualization

A scatter chart displaying:
- **X-axis:** Redemption Risk (probability of disengagement)
- **Y-axis:** Buy Propensity (probability of growth)
- **Color/Size:** Expected Client Value

This chart helps identify three client clusters:
- **High opportunity / low risk** (upper-left) — ideal upsell candidates
- **High opportunity / high risk** (upper-right) — retention priority
- **Low opportunity** (bottom) — monitoring only

The visualization translates behavioral signals into actionable client
prioritisation insights for distribution teams.

### Opportunity Frontier Score

A quantitative metric inspired by return-risk tradeoffs in portfolio theory:

```
OpportunityFrontierScore = ExpectedClientValue / RedemptionRisk
```

Conceptually analogous to a Sharpe ratio: clients with a high ratio of
expected opportunity to risk rank highest. This provides a complementary
prioritisation dimension alongside the Master Signal:

| Metric | What it measures |
|--------|------------------|
| **Master Signal** | Absolute opportunity (high value, any risk) |
| **Frontier Score** | Risk-adjusted opportunity (high value per unit of risk) |

Clients appearing at the top of the Frontier ranking but not the Master
Signal ranking represent **hidden opportunities** — moderate absolute
value but very favorable risk profiles.

---

## Client Prioritisation

Each customer receives a recommended action:

| Action | Rule |
|--------|------|
| **Upsell** | BuyPropensity > 0.6 AND RedemptionRisk < 0.3 |
| **Retention** | RedemptionRisk > 0.6 |
| **Monitor** | All other clients |

Clients are ranked by descending **Expected Client Value**. The top 100
form the **commercial opportunity list**.

| Action | Count | % |
|--------|-------|---|
| Upsell | 8,130 | 16.7% |
| Retention | 20,136 | 41.3% |
| Monitor | 20,457 | 42.0% |

---

## Strategic Report

**Live demo:** [alphasignal.streamlit.app](https://alphasignal.streamlit.app/)

```bash
# Or run locally:
streamlit run app.py
```

The application is structured as a **12-section strategic narrative report**
with a visual pipeline progress indicator tracking advancement through the
analytical stages: Data → Modeling → Signals → Value → Distribution.

| § | Section | Pipeline Stage |
|---|---------|----------------|
| 1 | **Executive Summary** | Overview |
| 2 | **Problem Definition** | Data |
| 3 | **Dataset Description** | Data |
| 4 | **Temporal Modeling Strategy** | Modeling |
| 5 | **Feature Engineering & Explainability** | Modeling |
| 6 | **Machine Learning Models** | Modeling |
| 7 | **Probability Calibration** | Modeling |
| 8 | **Signal Architecture** | Signals |
| 9 | **Expected Client Value** | Value |
| 10 | **Opportunity Frontier** | Value |
| 11 | **Commercial Intelligence** | Distribution |
| 12 | **Distribution Intelligence** | Distribution |

---

## Project Structure

```
├── README.md
├── requirements.txt
├── app.py                          # Streamlit strategic report (12 sections)
├── .gitignore
├── data/                           # (not tracked — place CSVs here)
│   └── COFINFAD .../
│       ├── customer_data.csv
│       └── transactions_data.csv
├── notebooks/
│   ├── churn_modeling.py           # Full pipeline (percent-script)
│   └── churn_modeling.ipynb        # Same pipeline as Jupyter notebook
└── outputs/                        # (generated by pipeline)
    ├── churn_artifacts.pkl         # Pickled models + predictions for report
    ├── evaluation_results.csv
    ├── figures/
    └── models/
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

1. **No contractual churn event** — targets are behavioural proxies.
2. **Customer attributes assumed static** — untimestamped variables may
   incorporate future information. The Tx-strict benchmark mitigates this.
3. **12-month dataset** — limits generalisability to seasonal patterns.
4. **Class imbalance** — 2.3% disengagement rate makes precision on the
   minority class challenging.
5. **Multiplicative signal formulation** — the commercial opportunity
   score uses a multiplicative decomposition (P(growth) × P(retention) ×
   engagement); in production the functional form and any weights would
   be validated against downstream conversion or revenue KPIs.
6. **Walk-forward uses simplified features** — the temporal validation
   rebuilds RFM features for each fold but does not replicate the full
   feature engineering pipeline (e.g., monthly trends). This is
   sufficient to demonstrate signal stability but is not identical to
   the primary evaluation.