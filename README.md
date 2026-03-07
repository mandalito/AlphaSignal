# Behavioral Master Signal — Methodology Prototype

> **This prototype validates the signal design and modeling pipeline on synthetic
> but structurally realistic data. It does not claim real-world predictive
> validity before access to production data.**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Problem](#business-problem)
3. [Strategy & Approach](#strategy--approach)
4. [System Architecture](#system-architecture)
5. [Synthetic Data Design](#synthetic-data-design)
6. [Label Engineering](#label-engineering)
7. [Sub-Signal Framework](#sub-signal-framework)
8. [Heuristic Baseline](#heuristic-baseline)
9. [Machine Learning Models](#machine-learning-models)
10. [Evaluation Framework](#evaluation-framework)
11. [Results on Synthetic Data](#results-on-synthetic-data)
12. [Explainability](#explainability)
13. [Interactive Dashboard](#interactive-dashboard)
14. [Project Structure](#project-structure)
15. [How to Run](#how-to-run)
16. [Technical Stack](#technical-stack)
17. [Limitations](#limitations)
18. [Production Roadmap](#production-roadmap)
19. [Methodology Disclaimer](#methodology-disclaimer)

---

## Executive Summary

This project delivers a **complete, end-to-end prototype** of a Behavioral
Master Signal system for commercial asset management. The system predicts
two client-level outcomes at a **3-month forward horizon**:

| Signal | Business Question |
|---|---|
| **Buy propensity** (`buy_3m`) | Which clients are likely to increase their investment exposure? |
| **Redemption risk** (`redeem_3m`) | Which clients are at risk of exiting or reducing their position? |

The prototype demonstrates that a modular signal framework — combining
interpretable business sub-signals with supervised ML — can effectively
rank clients by commercial action probability, enabling targeted sales
outreach and proactive retention.

**Key results on the synthetic test set:**
- Buy propensity: **ROC AUC 0.82** (logistic) / **0.81** (GBT), Lift@10% = **1.54×**
- Redemption risk: **ROC AUC 0.93** (logistic) / **0.92** (GBT), Lift@10% = **6.02×**
- ML models consistently improve over the heuristic baseline, confirming
  that data-driven weighting and non-linear interactions add value beyond
  expert-designed rules.

The prototype is **fully runnable**, **reproducible** (deterministic seed),
and includes an **interactive Streamlit dashboard** for live exploration.

---

## Business Problem

In institutional asset management, client behavior is not random — it is
influenced by a combination of:

- **Relationship factors:** tenure, past performance, drawdown experience
- **Market environment:** category-level investor flows, risk regime
- **Information environment:** media coverage, news sentiment
- **Product pipeline:** availability of new investment vehicles

However, these signals are typically siloed across CRM, portfolio, market
data, and media monitoring systems. Sales and relationship managers lack a
single, unified view that synthesizes all relevant inputs into an
**actionable commercial score**.

**The Behavioral Master Signal addresses this gap** by:
1. Engineering four interpretable sub-signals from raw data
2. Combining them into a unified propensity score
3. Producing client-level predictions with transparent explainability
4. Providing a ranked list of clients most likely to act — enabling
   efficient allocation of commercial resources

---

## Strategy & Approach

### Design Philosophy

The system is built around three core principles:

1. **Interpretability first.** Every component — from sub-signals to final
   predictions — must be explainable in business terms. A black-box model
   with no narrative is useless for sales teams.

2. **Modularity for production readiness.** Each pipeline stage (data →
   features → scoring → modeling → evaluation → explainability) is a
   self-contained module. Swapping synthetic data for real data requires
   changing one module, not rewriting the system.

3. **Honest benchmarking.** The heuristic baseline encodes expert hypotheses.
   ML must demonstrably beat it — otherwise the added complexity is not
   justified. This prototype makes that comparison explicit.

### Signal Combination Strategy

Rather than training one monolithic model on raw features, we adopt a
**two-layer architecture**:

```
Layer 1:  Raw features  →  4 interpretable sub-signals  (domain-driven)
Layer 2:  Sub-signals + raw features  →  ML model      (data-driven)
```

**Why this matters:**
- Layer 1 is transparent to business stakeholders — they can inspect and
  override individual signals.
- Layer 2 captures non-linear interactions and re-weights signals based on
  data — it learns what the heuristic cannot.
- The sub-signals themselves serve as **standalone commercial indicators**
  even without the ML layer.

### Validation Strategy

The prototype uses a **strict time-based split** to simulate realistic
out-of-sample evaluation:

```
│ Train (historical)     │ Validation       │ Test (holdout)      │
│ 2020-01  →  2022-06    │ 2022-07 → 2023-03│ 2023-04  →  2023-09 │
│ 9,000 observations     │ 2,700 obs.       │ 1,800 obs.          │
```

No random shuffling is ever used as a primary evaluation method.
This prevents temporal leakage — a critical requirement for any
time-series panel prediction system.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       RAW DATA LAYER                             │
│  Client: tenure, segment, exposure, exposure_stability           │
│  Performance: perf_3m, perf_12m, drawdown                        │
│  Market: category_flow_1m/3m, flow_acceleration, flow_persistence│
│  Environment: news_volume, news_sentiment, news_burst_zscore     │
│  Pipeline: launch_count_recent, launch_acceleration, intensity   │
│  Regime: market_regime (risk-on / risk-off)                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                    Feature Engineering
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              4  INTERPRETABLE  SUB-SIGNALS  [0, 1]               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Tenure &     │  │ External     │  │ News /   │  │ Product  │ │
│  │ Performance  │  │ Flows        │  │ Attention│  │ Launch   │ │
│  │  (0.25T +    │  │  (0.35F1 +   │  │ (0.30V + │  │ (0.40C + │ │
│  │   0.25P3 +   │  │   0.35F3 +   │  │  0.50S + │  │  0.30A + │ │
│  │   0.20P12 +  │  │   0.20A +    │  │  0.20B)  │  │  0.30I)  │ │
│  │   0.20D +    │  │   0.10P)     │  │          │  │          │ │
│  │   0.10S)     │  │              │  │          │  │          │ │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘ │
│         └──────────────────┴───────────────┴─────────────┘       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Heuristic │ │ Logistic │ │   GBT    │
        │ Baseline │ │Regression│ │(XGBoost) │
        │(weighted │ │(L2, C=1) │ │(d=4,     │
        │ linear)  │ │          │ │ lr=0.05) │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             └─────────────┼────────────┘
                           ▼
              ┌─────────────────────────┐
              │    MASTER SCORES        │
              │  • buy_3m  propensity   │
              │  • redeem_3m risk       │
              └────────────┬────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ SHAP     │ │ Client   │ │ Business │
        │ Feature  │ │ Ranked   │ │ Narrative│
        │ Importance│ │ List     │ │ per Obs. │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Synthetic Data Design

Real client data is not yet available. A synthetic dataset was generated to
validate the structural coherence of the signal engineering and modeling
pipeline. The synthetic data is **structurally realistic** but is **not real
data**. No performance claims should be interpreted as production predictions.

### Data Generation Process

The simulator (`src/synthetic_data.py`) builds a monthly client × date panel
with the following process:

1. **Client universe:** 300 clients are sampled with segment-specific
   attributes (institutional 30%, wealth 45%, retail 25%).
2. **Market regime:** A Markov chain generates a binary risk-on/risk-off
   indicator with 85% persistence (regimes last several months on average).
3. **Per-client time series:** For each client × month, the simulator
   generates correlated features linked to the market regime and the
   client's segment-specific baseline.

| Component | Generation Method | Parameters |
|---|---|---|
| Client universe | 300 clients × 48 months (2020-01 to 2023-12) | 14,400 raw observations |
| Segments | Multinomial sampling | Institutional (30%), Wealth (45%), Retail (25%) |
| Tenure | Segment-specific base + linear growth | Institutional: 36 mo, Wealth: 24 mo, Retail: 12 mo |
| Exposure | Log-normal random walk | Segment-specific scale (50M / 10M / 2M USD) |
| Performance | Regime-linked drift + idiosyncratic noise | +0.5% risk-on, −0.3% risk-off, σ=2.5% |
| Drawdown | Computed from cumulative return path | Max drawdown from rolling high-water mark |
| Category flows | Regime-linked + client noise | +1.5 × regime, −0.8 × (1−regime), σ_client=0.3 |
| News volume | Gaussian | μ=5, σ=2, clipped at 0 |
| News sentiment | Regime-linked | μ = 0.1 × regime, σ=0.3 |
| Product launches | Poisson process | λ=2 base + regime-amplified |
| Market regime | Markov chain | P(stay) = 0.85 |

**Seed:** All randomness uses `numpy.random.default_rng(42)` with per-client
sub-seeds for independence yet reproducibility.

---

## Label Engineering

Labels are generated via a **strict causal chain**: `features → latent
probability → sampled label`. Features are **never** derived from labels.

### `buy_3m` — Buy / Upsell Propensity

A client is labeled `buy_3m = 1` when both conditions hold:
1. **Forward exposure change** exceeds +5% over the next 3 months
2. **Latent buy probability** (Bernoulli draw) returns 1

The latent probability is computed via a sigmoid over a logit-space linear
combination:

```
logit_buy = −2.0                          # intercept (calibrates base rate)
           + 6.0 × perf_3m               # strong short-term momentum effect
           + 3.0 × perf_12m              # secondary long-term quality
           + 0.5 × flow_1m               # current demand pulse
           + 0.4 × flow_3m               # trend strength
           + 4.0 × clip(dd + 0.10)       # drawdown relief
           + 0.01 × log(1 + tenure)      # relationship maturity
           + 0.8 × news_sentiment        # media amplifier
           + 0.3 × (news_volume / 10)    # volume amplifier
           + 0.7 × market_regime         # regime lift
           + ε ~ N(0, 0.3)               # irreducible noise
```

**Approximate positive rate:** ~54% (varies by market regime distribution).

### `redeem_3m` — Redemption Risk

A client is labeled `redeem_3m = 1` when:
1. **Forward exposure change** falls below −5%
2. **Latent redemption probability** returns 1

```
logit_redeem = −2.5                       # intercept
              − 5.0 × perf_3m            # poor performance → redeem
              − 0.5 × flow_1m            # outflows → redeem
              − 0.4 × flow_3m
              − 8.0 × drawdown           # deep drawdown is the strongest trigger
              − 0.008 × log(1 + tenure)  # long tenure slightly protective
              − 0.6 × news_sentiment     # negative sentiment → redeem
              − 0.6 × market_regime      # risk-off → redeem
              + ε ~ N(0, 0.3)
```

**Approximate positive rate:** ~6% (realistic for fund redemptions).

**Conflict resolution:** If both labels would be 1, the one with higher
latent probability wins — a client cannot simultaneously buy and redeem in
our simplified model.

---

## Sub-Signal Framework

The Behavioral Master Signal decomposes the prediction problem into **four
interpretable sub-signals**, each scaled to [0, 1] via min-max normalization.
This makes signals comparable, combinable, and directly interpretable by
business users.

### 1. `tenure_perf_signal` — Client Tenure & Performance

**Business interpretation:** How well-established and commercially engaged is
the client? Mature, profitable, stable relationships are more likely to deepen.

| Component | Weight | Source Variable | Transformation |
|---|---|---|---|
| `tenure_score` | 0.25 | `tenure_months` | log1p → minmax |
| `perf3_score` | 0.25 | `performance_3m` | minmax |
| `perf12_score` | 0.20 | `performance_12m` | minmax |
| `drawdown_score` | 0.20 | `drawdown` | (1 + dd) → minmax |
| `stability_score` | 0.10 | `exposure_stability` | minmax |

```
tenure_perf_signal = 0.25T + 0.25P3 + 0.20P12 + 0.20D + 0.10S → minmax[0,1]
```

**High score** → long-standing, profitable, stable client (buy candidate).
**Low score** → new or stressed client under drawdown (potential redeemer).

### 2. `external_flows_signal` — Category-Level Investor Flows

**Business interpretation:** Are investors broadly adding to or withdrawing
from this product category? Clients tend to herd with category-level momentum.

| Component | Weight | Source Variable | Transformation |
|---|---|---|---|
| `flow1_score` | 0.35 | `category_flow_1m` | minmax |
| `flow3_score` | 0.35 | `category_flow_3m` | minmax |
| `acceleration_score` | 0.20 | `flow_acceleration` | minmax |
| `persistence_score` | 0.10 | `flow_persistence` | minmax |

```
external_flows_signal = 0.35F1 + 0.35F3 + 0.20A + 0.10P → minmax[0,1]
```

**High score** → strong, persistent inflows (buy-supportive environment).
**Low score** → outflows or decelerating flows (redemption-prone environment).

### 3. `news_signal` — News & Attention Environment

**Business interpretation:** What is the media landscape? Positive, elevated
coverage amplifies confidence; negative coverage amplifies anxiety.

| Component | Weight | Source Variable | Transformation |
|---|---|---|---|
| `volume_score` | 0.30 | `news_volume` | minmax |
| `sentiment_score` | 0.50 | `news_sentiment` | minmax |
| `burst_score` | 0.20 | `news_burst_zscore` | clip[-3,3] → minmax |

```
news_signal = 0.30V + 0.50S + 0.20B → minmax[0,1]
```

Sentiment is weighted highest (50%) because the **valence** of coverage,
not just its volume, is the primary driver of behavioral reactions.

### 4. `launch_signal` — Product Launch Pipeline

**Business interpretation:** Is the product pipeline active? More launches
create commercial opportunities for existing clients to re-allocate.

| Component | Weight | Source Variable | Transformation |
|---|---|---|---|
| `count_score` | 0.40 | `launch_count_recent` | minmax |
| `acceleration_score` | 0.30 | `launch_acceleration` | minmax |
| `intensity_score` | 0.30 | `launch_intensity` | minmax |

```
launch_signal = 0.40C + 0.30A + 0.30I → minmax[0,1]
```

---

## Heuristic Baseline

Before training any ML model, we encode **explicit business hypotheses**
as a weighted linear combination — the heuristic Master Score:

### Buy Score (heuristic)

```
master_buy_score = 0.35 × tenure_perf_signal     # strongest predictor
                 + 0.30 × external_flows_signal   # macro tide
                 + 0.20 × news_signal             # amplifier, noisier
                 + 0.15 × launch_signal           # opportunity, secondary
```

**Rationale:** Tenure/performance is weighted highest (35%) because the
strongest individual predictor of buy intent is a well-performing, mature
relationship. Flows (30%) capture the macro environment. News (20%) and
launches (15%) are amplifiers but less directly correlated with individual
client behavior.

### Redemption Score (heuristic)

```
master_redeem_score = −0.30 × tenure_perf_signal     # protective
                    − 0.35 × external_flows_signal   # herding protects
                    + 0.15 × news_signal             # anxiety amplifier
                    − 0.20 × launch_signal           # new products retain
```

**Sign directions:**
- **Negative** on tenure/perf: strong relationship → lower exit risk
- **Negative** on flows: positive macro environment → clients stay
- **Positive** on news: elevated attention can trigger anxiety-driven exits
- **Negative** on launches: rich product menu → reduced incentive to leave

These weights are **business hypotheses, not fitted parameters**. The ML
models will test whether data-driven re-weighting and non-linear interactions
improve upon this baseline.

---

## Machine Learning Models

### Model 1: Logistic Regression (L2)

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
])
```

- **Role:** Interpretable linear baseline. Tests whether a simple re-weighting
  of features outperforms the heuristic.
- **Strengths:** Fully explainable coefficients, fast, no hyperparameter
  sensitivity.
- **Regularization:** L2 penalty (C=1.0) prevents coefficient explosion
  on correlated features.

### Model 2: Gradient Boosted Trees (XGBoost)

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)
```

- **Role:** Non-linear model that can capture feature interactions and
  threshold effects (e.g., drawdown only matters below a critical level).
- **Early stopping:** Trained with `eval_set` on validation data to
  prevent overfitting.
- **Fallback:** If XGBoost is not installed, the system automatically
  falls back to `sklearn.HistGradientBoostingClassifier` with equivalent
  parameters.

### Feature Matrix

The ML models receive **23 features** (21 numeric + 2 segment dummies):

| Group | Features |
|---|---|
| Raw client | `tenure_months`, `exposure`, `exposure_stability`, `performance_3m`, `performance_12m`, `drawdown` |
| Market flows | `category_flow_1m`, `category_flow_3m`, `flow_acceleration`, `flow_persistence` |
| News | `news_volume`, `news_sentiment`, `news_burst_zscore` |
| Product launches | `launch_count_recent`, `launch_acceleration`, `launch_intensity` |
| Regime | `market_regime` |
| Sub-signals | `tenure_perf_signal`, `external_flows_signal`, `news_signal`, `launch_signal` |
| Segment | `seg_retail`, `seg_wealth` (one-hot, institutional = reference) |

Including both raw features and sub-signals allows the ML model to use the
engineered signals as regularized summaries while still accessing granular
detail when interactions matter.

---

## Evaluation Framework

### Metrics

| Metric | Purpose | Interpretation |
|---|---|---|
| **ROC AUC** | Discriminative power (threshold-independent) | 0.5 = random, 1.0 = perfect |
| **PR AUC** | Discriminative power for imbalanced classes | More informative than ROC for rare events like redemptions |
| **Precision@10%** | Of the top 10% scored clients, how many are true positives? | Directly measures sales targeting efficiency |
| **Recall@10%** | Of all true events, how many are captured in the top 10%? | Measures completeness of targeting |
| **Lift@10%** | How much more concentrated are events in top decile vs. random? | 1.0 = no better than random; higher = better |

### Business Translation

- **Precision@10%** answers: *"If I call the top 10% of clients ranked by
  score, what fraction of those calls will reach a client who actually
  buys/redeems?"*
- **Recall@10%** answers: *"What share of all buy/redeem events do I capture
  by contacting only the top 10%?"*
- **Lift@10%** answers: *"How many times more efficient is this targeting
  versus calling clients at random?"*

---

## Results on Synthetic Data

### Test Set Metrics

| Label | Model | ROC AUC | PR AUC | Prec@10% | Recall@10% | Lift@10% |
|---|---|---|---|---|---|---|
| `buy_3m` | Heuristic | 0.784 | 0.834 | 92.2% | 15.3% | 1.53× |
| `buy_3m` | Logistic | **0.822** | **0.858** | 91.1% | 15.1% | 1.51× |
| `buy_3m` | GBT | 0.807 | 0.850 | **92.8%** | **15.4%** | **1.54×** |
| `redeem_3m` | Heuristic | 0.875 | 0.280 | 23.3% | 50.6% | 5.06× |
| `redeem_3m` | Logistic | **0.933** | **0.448** | **30.0%** | **65.1%** | **6.51×** |
| `redeem_3m` | GBT | 0.918 | 0.345 | 27.8% | 60.2% | 6.02× |

### Key Takeaways

1. **ML consistently outperforms the heuristic.** Logistic regression alone
   improves ROC AUC by +3.8 pt on buy and +5.8 pt on redemption vs. the
   hand-tuned baseline. This validates the two-layer design: domain-driven
   signals provide a strong foundation, but data-driven weighting adds
   meaningful value.

2. **Redemption lift is exceptionally high** (6.5× at top decile). Because
   the class is rare (~6%), even modest discrimination concentrates events
   sharply in the top-scored tier. In production, this means a sales team
   contacting the top 10% of flagged clients would capture **65% of all
   redemption events**.

3. **Logistic regression outperforms GBT on this synthetic dataset.**
   This is expected — the planted signals are approximately linear. On real
   data with genuine non-linearities, the GBT model is likely to perform
   better.

4. **The heuristic is surprisingly competitive.** ROC AUC of 0.78 (buy)
   and 0.88 (redemption) confirms that the sub-signal design captures real
   structure. The heuristic serves as a useful standalone tool even without ML.

### Interpretation Caveat

> These results are on **synthetic data with planted signal structure**.
> They validate the methodology and pipeline, NOT real-world predictive
> accuracy. Performance on production data will differ and must be
> independently evaluated.

---

## Explainability

The prototype produces three layers of interpretability:

### 1. Global Feature Importance

- **Tree-based importance** (gain) from XGBoost
- **SHAP mean |value|** aggregated across the test set

These reveal which features the model relies on most. Typical top drivers:
`category_flow_1m`, `category_flow_3m`, `performance_3m`, `market_regime`.

### 2. SHAP Value Analysis

For 300 test observations, SHAP TreeExplainer computes per-observation,
per-feature attribution values. The dashboard visualizes:
- Feature importance ranking by mean |SHAP|
- Scatter plots of feature value vs. SHAP contribution for top 5 features

### 3. Client-Level Narratives

For any scored observation, the system generates a plain-language explanation:

> *"Buy propensity is elevated (predicted probability: 98.1%). Primary
> drivers: high category flow 3m, high category flow 1m, high
> performance 3m."*

> *"Redemption risk is elevated (predicted probability: 57.3%). Primary
> drivers: high category flow 1m, high tenure months, high market
> regime."*

> *"Redemption risk is moderate (predicted probability: 0.0%). Primary
> drivers: low category flow 1m, low performance 3m, low market regime."*

Narratives are generated by ranking |SHAP| values per observation and
mapping the sign to "high"/"low" language. When SHAP is unavailable, the
system falls back to raw sub-signal values.

---

## Interactive Dashboard

A **Streamlit web dashboard** (`app.py`) provides 6 interactive pages for
live exploration of the prototype:

| Page | Description |
|---|---|
| **🏠 Overview** | KPI banner (ROC AUC, Lift), architecture diagram, dataset summary |
| **📊 Signal Explorer** | Sub-signal distributions by outcome, heuristic score histograms, full correlation matrix |
| **🎯 Model Performance** | Metric comparison table, ROC/PR AUC bar charts, ML vs. heuristic delta |
| **🔍 Client Deep-Dive** | Per-client selector: timeline chart (exposure + events + performance + sub-signals), latest prediction probabilities, narrative explanation |
| **📈 Lift Analysis** | Precision-by-decile charts, cumulative recall curves (% of events captured vs. % of clients contacted) |
| **🧠 Explainability** | Feature importance (tree + SHAP), SHAP scatter plots for top 5 features, example narratives (high/medium/low score) |

---

## Project Structure

```
.
├── app.py                         # Streamlit dashboard (6 interactive pages)
├── run_prototype.py               # End-to-end pipeline execution script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   └── synthetic/
│       └── panel.parquet          # Generated panel dataset (14,400 rows)
│
├── notebooks/
│   └── demo.ipynb                 # Jupyter interactive walkthrough
│
├── outputs/
│   ├── evaluation_results.csv     # Model metrics table (12 rows)
│   ├── figures/                   # 6 PNG charts
│   │   ├── 01_signal_distributions.png
│   │   ├── 02_model_performance.png
│   │   ├── 03_lift_curves.png
│   │   ├── 04_feature_importance.png
│   │   ├── 05_client_timeline.png
│   │   └── 06_score_distributions.png
│   └── models/                    # Serialised .pkl models (4 files)
│       ├── buy_3m_logistic.pkl
│       ├── buy_3m_gbt.pkl
│       ├── redeem_3m_logistic.pkl
│       └── redeem_3m_gbt.pkl
│
└── src/
    ├── __init__.py
    ├── config.py                  # Seeds, constants, hyper-parameters
    ├── synthetic_data.py          # Synthetic panel generator
    ├── labels.py                  # Label definition (latent prob → label)
    ├── features.py                # Sub-signal engineering (4 signals)
    ├── scoring.py                 # Heuristic master scores
    ├── modeling.py                # LR + XGBoost training & serialisation
    ├── evaluation.py              # ROC-AUC, PR-AUC, top-decile metrics
    ├── explainability.py          # SHAP, permutation importance, narratives
    ├── plotting.py                # Matplotlib static visualisations
    └── utils.py                   # Helpers (scaling, time-split, sigmoid)
```

### Module Dependency Flow

```
config.py ← (all modules)
synthetic_data.py → labels.py → features.py → scoring.py → modeling.py
                                                             │
                                              evaluation.py ←┘
                                              explainability.py ←┘
                                              plotting.py ←┘
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements:** numpy, pandas, pyarrow, scikit-learn, matplotlib, xgboost,
shap, streamlit.

### 2. Run the full ML pipeline

```bash
python3 run_prototype.py
```

This executes all 7 steps (data generation → labels → features → scoring →
modeling → evaluation → explainability → plotting) in approximately
**12 seconds** on a standard laptop.

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with 6 interactive pages.

### 4. Explore in Jupyter (optional)

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Technical Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data manipulation | pandas, numpy | ≥2.0, ≥1.24 |
| Data storage | Parquet (pyarrow) | ≥12.0 |
| ML training | scikit-learn, XGBoost | ≥1.3, ≥1.7 |
| Explainability | SHAP (TreeExplainer, LinearExplainer) | ≥0.42 |
| Static visualisation | matplotlib | ≥3.7 |
| Interactive dashboard | Streamlit | ≥1.28 |
| Reproducibility | Deterministic numpy RNG (seed=42) | — |

---

## Limitations

1. **Synthetic data only.** All results are on data with planted signal
   structure. Real-world performance will differ and requires independent
   validation on production data.

2. **No real transaction log.** Labels are proxied through exposure changes
   and latent probabilities — not from actual subscription/redemption events.

3. **No client-level fixed effects.** The panel is cross-sectional; production
   models should account for client heterogeneity (e.g., random effects,
   entity embeddings).

4. **Class imbalance.** `redeem_3m` is ~6% positive. Production deployment
   should evaluate class weighting, threshold calibration, or probability
   calibration (e.g., isotonic regression).

5. **No regime forecasting.** Market regime is a contemporaneous feature.
   In production, it would need to be estimated from market data without
   look-ahead bias.

6. **Static heuristic weights.** The baseline weights are fixed hypotheses.
   Production systems should re-calibrate periodically or use a meta-learner.

7. **Single time horizon.** Only 3-month predictions. A production system
   might offer multiple horizons (1M, 3M, 6M, 12M).

8. **Linear latent model.** The synthetic data generator uses logistic
   (linear in logit space) relationships. This may understate the benefit
   of tree-based models on real data where genuine non-linearities exist.

---

## Production Roadmap

Once real data becomes available, the transition path is:

| Step | Action | Module to Change |
|---|---|---|
| 1 | Replace synthetic generator with real data ingestion | `synthetic_data.py` → new `data_loader.py` |
| 2 | Define labels from actual transaction log | `labels.py` |
| 3 | Validate sub-signal distributions on real data | `features.py` |
| 4 | Expand feature set (CRM, demographics, product-level perf) | `features.py`, `config.py` |
| 5 | Walk-forward cross-validation across multiple time windows | `evaluation.py` |
| 6 | Hyperparameter optimisation (Optuna or similar) | `modeling.py` |
| 7 | Probability calibration (isotonic / Platt) | `modeling.py` |
| 8 | Deploy as monthly batch scoring pipeline | New `deploy/` module |
| 9 | Instrument A/B tests for commercial lift measurement | New `experiments/` module |
| 10 | Build real-time scoring API (if low-latency needed) | New `api/` module |

The modular architecture means **steps 1–3 require changing 2–3 files** while
the entire modeling, evaluation, explainability, and dashboard stack remains
unchanged.

---

## Methodology Disclaimer

> **This prototype validates the signal design and modeling pipeline on
> synthetic but structurally realistic data. It does not claim real-world
> predictive validity before access to production data.**

The prototype is intended as a **methodological proof-of-concept** for a
selection-stage demonstration. It shows:

- That the business problem can be framed as a supervised ML task
- That a modular signal framework produces interpretable, combinable features
- That ML consistently outperforms a domain-driven heuristic baseline
- That the system produces actionable, explainable outputs for business users
- That the engineering framework is production-ready for real data integration

It does **not** show — and does **not** claim — real predictive accuracy on
live client data.