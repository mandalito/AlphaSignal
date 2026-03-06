# Behavioral Master Signal — Prototype

> **This prototype validates the signal design and modeling pipeline on synthetic
> but structurally realistic data. It does not claim real-world predictive
> validity before access to production data.**

---

## Project Objective

Build a credible, business-oriented prototype of a **Behavioral Master Signal**
system that predicts two commercial outcomes for investment fund clients:

| Label | Definition |
|---|---|
| `buy_3m` | Client increases exposure in the next 3 months |
| `redeem_3m` | Client redeems / exits in the next 3 months |

The prototype demonstrates:
- Problem framing and signal design
- A modular, reproducible ML pipeline
- Rigorous time-aware evaluation
- Interpretable, business-facing explainability
- A clear framework ready to plug into real data

---

## Why Synthetic Data?

Real client data is not yet available.  A synthetic dataset was generated to:

1. Validate the structural coherence of the feature engineering pipeline.
2. Demonstrate the modeling and evaluation framework.
3. Show that the signal design encodes plausible business logic.

The synthetic data is **structurally realistic** (client segments, tenure
dynamics, market-regime-linked flows, etc.) but it is **not real data**.

No performance claims made on this prototype should be interpreted as
predictions for a production deployment.

---

## What Is Simulated

| Component | Assumption |
|---|---|
| Client universe | 300 clients × 48 months (2020–2023), monthly panel |
| Client segments | Institutional (30%), Wealth (45%), Retail (25%) |
| Tenure | Segment-specific starting tenure + linear growth |
| Exposure | Log-normal random walk per client |
| Performance | Markov-regime-linked monthly returns + idiosyncratic noise |
| Drawdown | Computed from cumulative return path |
| Category flows | Market-regime-linked with client-specific noise |
| News volume/sentiment | Regime-linked with random variation |
| Product launches | Poisson process, amplified in risk-on regime |
| Market regime | Markov chain: risk-on / risk-off (persistence = 85%) |

**Label generation:**
Labels are derived from latent probabilities (features → prob → label).
Features are **never** built from labels.
A 5% exposure change threshold is applied alongside latent probability sampling.

Approximate positive rates on the synthetic data (may vary by run):
- `buy_3m`: ~50–55% (relatively balanced; depends on market regime distribution)
- `redeem_3m`: ~5–8% (minority class, realistic for fund redemptions)

---

## Project Structure

```
.
├── data/
│   └── synthetic/
│       └── panel.parquet          # Generated panel dataset
├── notebooks/
│   └── demo.ipynb                 # Interactive walkthrough
├── outputs/
│   ├── evaluation_results.csv     # Model metrics table
│   ├── figures/                   # 6 PNG charts
│   └── models/                    # Serialised .pkl models
├── src/
│   ├── config.py                  # Seeds, constants, hyper-params
│   ├── synthetic_data.py          # Data generation
│   ├── labels.py                  # Label definition & attachment
│   ├── features.py                # Sub-signal engineering
│   ├── scoring.py                 # Heuristic master scores
│   ├── modeling.py                # LR + XGBoost training
│   ├── evaluation.py              # ROC-AUC, PR-AUC, lift metrics
│   ├── explainability.py          # SHAP / permutation importance
│   ├── plotting.py                # Visualisations
│   └── utils.py                   # Helpers (scaling, time-split)
├── run_prototype.py               # End-to-end execution script
├── requirements.txt
└── README.md
```

---

## The Four Sub-Signals

The master score is built from four interpretable sub-signals, each scaled
to **[0, 1]**:

### 1. `tenure_perf_signal` — Client Tenure & Performance
Captures relationship maturity and investment return quality.

| Input | Role |
|---|---|
| `tenure_months` | Relationship length proxy |
| `performance_3m` | Short-term return momentum |
| `performance_12m` | Long-term return quality |
| `drawdown` | Stress indicator |
| `exposure_stability` | Commitment / stickiness |

**High score** → long-standing, profitable, stable client.
**Low score** → new or stressed client under drawdown.

---

### 2. `external_flows_signal` — Category Flows
Captures the macro momentum of money flows into the product category.

| Input | Role |
|---|---|
| `category_flow_1m` | Current month demand pulse |
| `category_flow_3m` | Medium-term trend |
| `flow_acceleration` | Momentum change |
| `flow_persistence` | Directional consistency |

**High score** → sustained inflows → supportive of buy decisions.
**Low score** → outflows → redemption trigger.

---

### 3. `news_signal` — News / Attention
Captures the media attention environment.

| Input | Role |
|---|---|
| `news_volume` | Attention level |
| `news_sentiment` | Coverage valence (positive/negative) |
| `news_burst_zscore` | Abnormal media spike |

**High score** → positive, elevated coverage → can amplify buy confidence.
**Low score** → negative or absent coverage → can amplify anxiety.

---

### 4. `launch_signal` — Product Launch / Strategy
Captures commercial pipeline intensity.

| Input | Role |
|---|---|
| `launch_count_recent` | Active pipeline |
| `launch_acceleration` | Speed of pipeline growth |
| `launch_intensity` | Relative pace vs. history |

**High score** → active, accelerating launches → commercial momentum.
**Low score** → stagnant pipeline → reduced allocation incentive.

---

## Heuristic Master Score

Before ML, a simple weighted linear combination provides a baseline:

```
master_buy_score    = 0.35 × tenure_perf  + 0.30 × ext_flows
                    + 0.20 × news         + 0.15 × launch

master_redeem_score = −0.30 × tenure_perf − 0.35 × ext_flows
                    + 0.15 × news         − 0.20 × launch
```

These weights are **business hypotheses**, not fitted parameters.  The ML
models test whether data-driven weighting or non-linear interactions
improve upon them.

---

## Modeling

Two models are trained per label using a **strict time-based split**:

| Period | Dates | N rows (approx.) |
|---|---|---|
| Train | up to 2022-06 | 9,000 |
| Validation | 2022-07 → 2023-03 | 2,700 |
| Test | 2023-04 → 2023-09 | 1,800 |

**Models:**
- `logistic` — L2 logistic regression with StandardScaler
- `gbt` — XGBoost (falls back to HistGradientBoostingClassifier if unavailable)

---

## Results (Synthetic Data)

Sample metrics on the test set (values will vary slightly if re-run):

| Label | Model | ROC AUC | PR AUC | Lift@10% |
|---|---|---|---|---|
| buy_3m | Heuristic | 0.78 | 0.83 | 1.53 |
| buy_3m | Logistic | 0.82 | 0.86 | 1.51 |
| buy_3m | GBT | 0.81 | 0.85 | 1.54 |
| redeem_3m | Heuristic | 0.88 | 0.28 | 5.06 |
| redeem_3m | Logistic | 0.93 | 0.45 | 6.51 |
| redeem_3m | GBT | 0.92 | 0.37 | 5.78 |

**What these numbers mean:**
- On synthetic data with planted signal, the models recover the structure.
- They do NOT indicate expected performance on real production data.
- The redemption model shows high lift because the class is rare (~6%) and
  concentrated signal is learnable.
- The heuristic baseline performs respectably, validating the signal design.

---

## Explainability Outputs

For each model, the prototype produces:
- **SHAP feature importance** (mean absolute SHAP value per feature)
- **Example client narratives** in plain business language:

> *"Buy propensity is elevated (predicted probability: 97.8%). Primary
> drivers: high category flow 3m, high external flows signal, high
> category flow 1m."*

> *"Redemption risk is elevated (predicted probability: 63.8%). Primary
> drivers: high category flow 1m, high tenure months, high market regime."*

---

## Generated Figures

| File | Description |
|---|---|
| `01_signal_distributions.png` | Sub-signal violin plots by outcome label |
| `02_model_performance.png` | ROC / PR AUC bar chart |
| `03_lift_curves.png` | Precision-by-decile lift chart |
| `04_feature_importance.png` | SHAP feature importance (top 15) |
| `05_client_timeline.png` | One client's exposure, performance, signals over time |
| `06_score_distributions.png` | Heuristic score distributions by label |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_prototype.py
```

The script completes in under 30 seconds on a standard laptop.

### 3. Explore interactively

```bash
jupyter notebook notebooks/demo.ipynb
```

---

## Limitations

1. **Synthetic data only.** All results are on planted-signal synthetic data.
   Real-world performance will differ and must be validated on production data.

2. **No real transaction log.** Labels are proxied through exposure changes
   and latent probabilities — not from actual subscription / redemption events.

3. **No client-level fixed effects.** The panel is cross-sectional; advanced
   production models should account for client heterogeneity.

4. **Class imbalance.** The `redeem_3m` label is imbalanced (~6% positive).
   In production, SMOTE, class weighting, or threshold calibration may be
   needed.

5. **No regime forecasting.** The market regime is treated as a contemporaneous
   feature.  In production, the regime itself would need to be predicted or
   estimated from market data without look-ahead.

6. **Static weights.** Heuristic weights are fixed hypotheses.  In production,
   they should be re-calibrated periodically.

---

## Next Steps Once Real Data Is Available

1. Replace `synthetic_data.py` with a real data ingestion pipeline.
2. Define labels from the actual transaction log (subscriptions, redemptions).
3. Re-run `features.py` to verify signal quality on real distributions.
4. Expand the feature set: product-level performance, client demographics,
   CRM interaction history, regulatory filings.
5. Add cross-validation across multiple time windows (walk-forward).
6. Deploy the scoring pipeline as a monthly batch job.
7. Instrument A/B tests to measure commercial lift in sales campaigns.

---

## Methodology Disclaimer

> This prototype validates the signal design and modeling pipeline on synthetic
> but structurally realistic data. It does not claim real-world predictive
> validity before access to production data.

The prototype is intended as a **methodological proof-of-concept** for a
selection-stage demonstration, not as a production system.