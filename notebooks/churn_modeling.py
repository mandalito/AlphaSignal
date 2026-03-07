# %% [markdown]
# # Future-Disengagement Prediction — COFINFAD Dataset
#
# **Objective:** Predict which fintech customers will disengage in the **next
# quarter** using only information available **before** the prediction window.
#
# ### Temporal Design (key methodological choice)
#
# ```
# ┌────────────────────────────────────┐  ┌──────────────────────┐
# │  OBSERVATION WINDOW                │  │  PREDICTION WINDOW   │
# │  Jan 2023 → Sep 30 2023           │  │  Oct 1 → Dec 29 2023 │
# │  ► features built here             │  │  ► target built here  │
# └────────────────────────────────────┘  └──────────────────────┘
#                 9 months                        3 months
# ```
#
# All transaction features are computed **exclusively** from the observation
# window. The target (`future_disengaged`) is derived **exclusively** from
# the prediction window. No information leaks across the boundary.
#
# ### Pipeline Steps
# 1. Load and validate `customer_data.csv` and `transactions_data.csv`
# 2. Identify and exclude leakage-prone columns
# 3. Construct `future_disengaged` target from Q4 transaction activity
# 4. Engineer behavioral features from Jan–Sep transactions only
# 5. Compare three feature setups: Full, Tx-strict, Customer-only
# 6. Train Logistic Regression, Random Forest, XGBoost (preprocessing fit on train only)
# 7. Evaluate with ROC-AUC, PR-AUC, F1, calibration, Brier, precision/recall @k, lift
# 8. Tune decision threshold on validation split (never on test)
# 9. Sanity-check predictions against `churn_probability` (post hoc only)
# 10. Interpret via feature importance, coefficients, and SHAP
#
# ---
#
# > **Disclaimer:** This is a methodology prototype on a public dataset.
# > No contractual churn label exists — the target is a future-disengagement
# > proxy. Results demonstrate pipeline rigor, not production-validated
# > predictions.

# %% [markdown]
# ## 0. Setup & Imports

# %%
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})

SEED = 42
DATA_DIR = "data/COFINFAD Colombian Fintech Financial Analytics Dat"

# ── Temporal boundaries ──────────────────────────────────────────
OBS_END   = pd.Timestamp("2023-09-30")   # last day of observation window
PRED_START = pd.Timestamp("2023-10-01")  # first day of prediction window

print(f"XGBoost: {HAS_XGB} | SHAP: {HAS_SHAP}")
print(f"Observation window: … → {OBS_END.date()}")
print(f"Prediction  window: {PRED_START.date()} → …")

# %% [markdown]
# ---
# ## 1. Data Loading & Validation

# %%
cust = pd.read_csv(os.path.join(DATA_DIR, "customer_data.csv"))
txns = pd.read_csv(os.path.join(DATA_DIR, "transactions_data.csv"),
                    parse_dates=["date"])
print(f"Customers:    {cust.shape[0]:,} × {cust.shape[1]}")
print(f"Transactions: {txns.shape[0]:,} × {txns.shape[1]}")
print(f"Tx date range: {txns['date'].min().date()} → {txns['date'].max().date()}")

# %%
# Integrity checks
print(f"Unique customers (cust): {cust['customer_id'].nunique():,}")
print(f"Unique customers (txns): {txns['customer_id'].nunique():,}")
print(f"Duplicated IDs: {cust['customer_id'].duplicated().sum()}")
nulls = cust.isnull().sum()
print(f"\nNon-zero null counts:")
print(nulls[nulls > 0].to_frame("n_null").assign(
    pct=lambda d: (d["n_null"] / len(cust) * 100).round(1)))
cust_ids, txn_ids = set(cust["customer_id"]), set(txns["customer_id"])
print(f"\nOrphans  cust→txn: {len(cust_ids - txn_ids)}  "
      f"txn→cust: {len(txn_ids - cust_ids)}")

# %% [markdown]
# ---
# ## 2. Leakage Risk Assessment
#
# ### Leakage-Risk Checklist
#
# | Column(s) | Risk | Decision | Reason |
# |-----------|------|----------|--------|
# | `churn_probability` | 🔴 CRITICAL | **Exclude** (used *only* for post-hoc sanity check §10) | Pre-computed target score |
# | `customer_lifetime_value` | 🔴 CRITICAL | **Exclude** | CLV embeds churn assumptions |
# | `clv_segment` | 🟡 HIGH | **Exclude** | Derived from CLV |
# | Pre-computed tx aggregates (`tx_count`, `avg_tx_value`, …, `avg_daily_transactions`) | 🟡 HIGH | **Exclude** | Computed over *full* year incl. future window; we rebuild from obs. window |
# | `preferred_transaction_type`, `weekend_transaction_ratio` | 🟡 HIGH | **Exclude** | Same reason |
# | Date columns (`first_tx`, `last_tx`, …) | 🟢 LOW | **Exclude** | We derive recency from raw txns |
# | `base/tx/product_satisfaction` | 🟡 MED | **Exclude** | Sub-components of `satisfaction_score` |
# | `last_survey_date`, `app_store_rating` | 🟢 LOW | **Exclude** | Non-behavioral metadata |
# | `feature_requests`, `complaint_topics` | 🟢 LOW | **Exclude** | Free text, not tabular-ready |
# | `customer_segment` | 🟡 MED | **Exclude** | Likely derived from full-year behaviour (incl. future window) |
#
# **Retained customer-level columns** (demographics, product holdings, digital
# engagement, satisfaction aggregate, support, tenure) are treated as static
# attributes known at the observation cutoff.

# %%
LEAKAGE_EXCLUDE = [
    # ── Critical leakage ──
    "churn_probability", "customer_lifetime_value", "clv_segment",
    # ── Pre-computed tx aggregates (span full year) ──
    "tx_count", "avg_tx_value", "total_tx_volume",
    "monthly_transaction_count", "average_transaction_value",
    "total_transaction_volume", "transaction_frequency",
    "avg_daily_transactions", "weekend_transaction_ratio",
    "preferred_transaction_type",
    # ── Date metadata ──
    "first_tx", "last_tx", "first_transaction_date", "last_transaction_date",
    # ── Satisfaction sub-components ──
    "base_satisfaction", "tx_satisfaction", "product_satisfaction",
    # ── Admin / non-behavioral ──
    "last_survey_date", "app_store_rating",
    "feature_requests", "complaint_topics",
    # ── Segment derived from full-year behaviour ──
    "customer_segment",
]
print(f"Excluding {len(LEAKAGE_EXCLUDE)} columns from features")

# %% [markdown]
# ### Assumption: Customer-Level Variables as Observation-Cutoff Snapshots
#
# Several retained customer-level variables are **not timestamped**:
# `app_logins_frequency`, `feature_usage_diversity`, `support_tickets_count`,
# `resolved_tickets_ratio`, `satisfaction_score`, `nps_score`,
# `international_transactions`, `failed_transactions`, `customer_tenure`.
#
# **Because no temporal metadata is available**, we treat these as
# **point-in-time snapshots known at the observation cutoff (Sep 30, 2023)**.
# This is a necessary modelling assumption — we cannot verify from the
# dataset alone whether these values incorporate Q4 information.
#
# **Mitigation.** The **Tx-strict** benchmark (§5) uses *only*
# transaction-derived features from the observation window, with zero
# reliance on customer-level columns. Comparing Tx-strict against the
# Full setup isolates the contribution — and potential contamination —
# of customer-level variables.

# %% [markdown]
# ---
# ## 3. Target Construction — Future Window Only
#
# ### What the target represents
#
# `future_disengaged` is a **behavioural proxy for future transactional
# disengagement**. It is **not** contractual churn (account closure), since
# no such label exists in the COFINFAD dataset. The goal is to detect
# customers who show a **strong activity decline** in the prediction window
# relative to their observation-window baseline.
#
# In a production setting, this proxy would be replaced by actual churn
# events (account closures, subscription cancellations) or business-defined
# inactivity rules validated by domain experts.
#
# ### Definition
#
# A customer is labelled `future_disengaged = 1` when they show
# **transactional silence or sharp decline** in Q4 2023 (Oct–Dec),
# using *only* data from the prediction window:
#
# | Condition | Threshold | Rationale |
# |-----------|-----------|-----------|
# | Zero transactions in Q4 | `q4_count == 0` | Complete silence |
# | Activity collapse | Q4 monthly rate < 20 % of obs-window monthly rate | Sharp decline from baseline |
#
# The target is intentionally **conservative**: a customer must show strong
# disengagement evidence, not merely a minor dip.

# %%
# ── Split transactions by temporal boundary ──
txns_obs  = txns[txns["date"] <= OBS_END].copy()
txns_fut  = txns[txns["date"] >= PRED_START].copy()
print(f"Observation-window txns: {len(txns_obs):,}")
print(f"Future-window txns:      {len(txns_fut):,}")

# %%
# ── Per-customer counts in each window ──
obs_counts = txns_obs.groupby("customer_id").size().rename("obs_count")
fut_counts = txns_fut.groupby("customer_id").size().rename("fut_count")

target_df = (
    pd.DataFrame({"customer_id": cust["customer_id"]})
    .merge(obs_counts, on="customer_id", how="left")
    .merge(fut_counts, on="customer_id", how="left")
)
target_df["obs_count"] = target_df["obs_count"].fillna(0).astype(int)
target_df["fut_count"] = target_df["fut_count"].fillna(0).astype(int)

# Monthly rates
OBS_MONTHS = 9
FUT_MONTHS = 3
target_df["obs_monthly"] = target_df["obs_count"] / OBS_MONTHS
target_df["fut_monthly"] = target_df["fut_count"] / FUT_MONTHS

# Activity collapse ratio
target_df["activity_ratio"] = np.where(
    target_df["obs_monthly"] > 0,
    target_df["fut_monthly"] / target_df["obs_monthly"],
    np.where(target_df["fut_count"] > 0, 1.0, 0.0),
)

# ── Target label ──
target_df["future_disengaged"] = (
    (target_df["fut_count"] == 0) |
    (target_df["activity_ratio"] < 0.20)
).astype(int)

print(f"\nTarget distribution:")
dist = target_df["future_disengaged"].value_counts().to_frame("n")
dist["pct"] = (dist["n"] / dist["n"].sum() * 100).round(1)
print(dist)

# %% [markdown]
# ---
# ## 4. Feature Engineering — Observation Window Only
#
# All transaction features below are computed **exclusively** from
# `txns_obs` (Jan–Sep 2023). No future-window data touches the feature
# matrix.

# %%
# ── 4a. Basic RFM aggregates ──
tx_agg = txns_obs.groupby("customer_id").agg(
    tx_count=("amount", "count"),
    tx_total=("amount", "sum"),
    tx_mean=("amount", "mean"),
    tx_median=("amount", "median"),
    tx_std=("amount", "std"),
    tx_min=("amount", "min"),
    tx_max=("amount", "max"),
    tx_first=("date", "min"),
    tx_last=("date", "max"),
).reset_index()

tx_agg["tx_std"] = tx_agg["tx_std"].fillna(0)
tx_agg["tx_cv"] = tx_agg["tx_std"] / tx_agg["tx_mean"].clip(lower=1)
tx_agg["tx_recency"] = (OBS_END - tx_agg["tx_last"]).dt.days
tx_agg["tx_span"] = (tx_agg["tx_last"] - tx_agg["tx_first"]).dt.days.clip(lower=1)
tx_agg["tx_freq"] = tx_agg["tx_count"] / tx_agg["tx_span"]
tx_agg = tx_agg.drop(columns=["tx_first", "tx_last"])

print(f"RFM features: {tx_agg.shape}")

# %%
# ── 4b. Transaction-type shares ──
type_ct = txns_obs.groupby(["customer_id", "type"]).size().unstack(fill_value=0)
type_sh = type_ct.div(type_ct.sum(axis=1), axis=0)
type_sh.columns = [f"sh_{c.lower()}" for c in type_sh.columns]
type_sh = type_sh.reset_index()

# ── 4c. Weekend ratio ──
txns_obs["is_wknd"] = txns_obs["date"].dt.dayofweek >= 5
wknd = txns_obs.groupby("customer_id")["is_wknd"].mean().reset_index(name="tx_wknd_ratio")

# %%
# ── 4d. Monthly count trend (linear slope) ──
txns_obs["month"] = txns_obs["date"].dt.to_period("M").dt.to_timestamp()
mc = txns_obs.groupby(["customer_id", "month"]).size().reset_index(name="mc")

def _slope(g):
    if len(g) < 2:
        return 0.0
    x = np.arange(len(g), dtype=float)
    y = g["mc"].values.astype(float)
    return np.polyfit(x, y, 1)[0] if x.std() > 0 else 0.0

count_trend = mc.groupby("customer_id").apply(_slope).reset_index(name="tx_count_trend")

# ── 4e. Monthly amount trend (relative slope) ──
ma = txns_obs.groupby(["customer_id", "month"])["amount"].sum().reset_index()

def _amt_slope(g):
    if len(g) < 2:
        return 0.0
    x = np.arange(len(g), dtype=float)
    y = g["amount"].values.astype(float)
    if x.std() == 0 or y.mean() == 0:
        return 0.0
    return np.polyfit(x, y, 1)[0] / y.mean()

amt_trend = ma.groupby("customer_id").apply(_amt_slope).reset_index(name="tx_amt_trend")

# %%
# ── 4f. Max inactivity gap ──
def _max_gap(g):
    d = g["date"].sort_values()
    if len(d) < 2:
        return (OBS_END - d.iloc[0]).days
    return d.diff().dt.days.dropna().max()

max_gap = txns_obs.groupby("customer_id").apply(_max_gap).reset_index(name="tx_max_gap")

# ── 4g. Last-quarter-of-obs vs first-two-quarters ratio ──
OBS_Q3_START = pd.Timestamp("2023-07-01")
early = txns_obs[txns_obs["date"] < OBS_Q3_START].groupby("customer_id").size()
late  = txns_obs[txns_obs["date"] >= OBS_Q3_START].groupby("customer_id").size()
q_ratio = pd.DataFrame({"customer_id": cust["customer_id"]})
q_ratio = q_ratio.merge(early.rename("obs_early").reset_index(), how="left")
q_ratio = q_ratio.merge(late.rename("obs_late").reset_index(), how="left")
q_ratio["obs_early"] = q_ratio["obs_early"].fillna(0)
q_ratio["obs_late"]  = q_ratio["obs_late"].fillna(0)
q_ratio["tx_late_ratio"] = np.where(
    q_ratio["obs_early"] > 0,
    (q_ratio["obs_late"] / 3) / (q_ratio["obs_early"] / 6),
    np.where(q_ratio["obs_late"] > 0, 2.0, 0.0),
)
q_ratio = q_ratio[["customer_id", "tx_late_ratio"]]

print("Observation-window feature engineering complete")

# %% [markdown]
# ---
# ## 5. Build Modeling DataFrames
#
# We assemble three feature setups as **raw DataFrames** (no imputation or
# encoding yet). All preprocessing is deferred to `ColumnTransformer`
# pipelines **fit only on the training split** (§6), preventing any
# distribution leakage from validation or test data.
#
# | Setup | Features | Purpose |
# |-------|----------|---------|
# | **Full** | Customer attributes + obs-window tx features | Best available information |
# | **Tx-strict** | Obs-window tx features only | Ultra-strict benchmark — no customer-level variables |
# | **Cust-only** | Customer attributes only (no tx features) | Baseline — demographics + product + satisfaction |
#
# The **Tx-strict** setup is the key robustness check: it uses *only*
# transaction-derived features from Jan–Sep 2023 and proves whether
# predictive signal exists without relying on potentially leaked
# customer-level aggregates.

# %%
# ── 5a. Prepare customer attributes (raw — no encoding/imputation) ──
cust_clean = cust.drop(columns=[c for c in LEAKAGE_EXCLUDE if c in cust.columns])
sanity_churn = cust[["customer_id", "churn_probability"]].copy()

# Cast booleans to int (type cast, not statistical — safe before split)
for c in cust_clean.select_dtypes(include="bool").columns:
    cust_clean[c] = cust_clean[c].astype(int)

# Drop high-cardinality / non-useful string columns
drop_str = ["location", "occupation", "education_level", "marital_status"]
cust_clean = cust_clean.drop(columns=[c for c in drop_str if c in cust_clean.columns])

# Column groups for ColumnTransformer
NUM_CUST = [c for c in cust_clean.columns
            if c != "customer_id"
            and cust_clean[c].dtype in ("int64", "float64", "int32")]
ORD_CUST = ["income_bracket", "feedback_sentiment"]
CAT_CUST = ["gender", "acquisition_channel"]
CUST_ALL = NUM_CUST + ORD_CUST + CAT_CUST

print(f"Customer features — numeric: {len(NUM_CUST)}  ordinal: {len(ORD_CUST)}  "
      f"nominal: {len(CAT_CUST)}  total: {len(CUST_ALL)}")

# %%
# ── 5b. Merge all tx features ──
tx_feats = (
    pd.DataFrame({"customer_id": cust["customer_id"]})
    .merge(tx_agg, on="customer_id", how="left")
    .merge(type_sh, on="customer_id", how="left")
    .merge(wknd, on="customer_id", how="left")
    .merge(count_trend, on="customer_id", how="left")
    .merge(amt_trend, on="customer_id", how="left")
    .merge(max_gap, on="customer_id", how="left")
    .merge(q_ratio, on="customer_id", how="left")
)
TX_FEAT = [c for c in tx_feats.columns if c != "customer_id"]

# Customers with zero obs-window transactions → structural zero (not imputation)
for c in TX_FEAT:
    tx_feats[c] = tx_feats[c].fillna(0)

FULL_FEAT = CUST_ALL + TX_FEAT

# %%
# ── 5c. Assemble DataFrames (raw, unprocessed) ──
base = pd.DataFrame({"customer_id": cust["customer_id"]})
base = base.merge(target_df[["customer_id", "future_disengaged"]], on="customer_id")

full_df = base.merge(cust_clean, on="customer_id").merge(tx_feats, on="customer_id")
tx_df   = base.merge(tx_feats, on="customer_id")
cu_df   = base.merge(cust_clean, on="customer_id")

y_all = full_df["future_disengaged"].values

print(f"Samples: {len(y_all):,}  |  Disengaged: {y_all.sum():,} ({y_all.mean():.1%})")
print(f"Full features:      {len(FULL_FEAT)}")
print(f"Tx-strict features: {len(TX_FEAT)}")
print(f"Cust-only features: {len(CUST_ALL)}")

# %% [markdown]
# ---
# ## 6. Train / Validation / Test Split (60 / 20 / 20)
#
# | Split | % | Role |
# |-------|---|------|
# | **Train** | 60 % | Model fitting |
# | **Validation** | 20 % | Threshold tuning, early-stopping |
# | **Test** | 20 % | Final held-out evaluation *only* |
#
# The **test set is never used** for model fitting or threshold selection.

# %%
idx_all = np.arange(len(y_all))

# Step 1: 80 % trainval / 20 % test
idx_tv, idx_test = train_test_split(
    idx_all, test_size=0.20, random_state=SEED, stratify=y_all)

# Step 2: 75 % of trainval → train (= 60 % overall), 25 % → val (= 20 %)
idx_train, idx_val = train_test_split(
    idx_tv, test_size=0.25, random_state=SEED, stratify=y_all[idx_tv])

print(f"Train:      {len(idx_train):,}  ({y_all[idx_train].mean():.1%} disengaged)")
print(f"Validation: {len(idx_val):,}  ({y_all[idx_val].mean():.1%} disengaged)")
print(f"Test:       {len(idx_test):,}  ({y_all[idx_test].mean():.1%} disengaged)")

# %% [markdown]
# ---
# ## 6b. Preprocessing Pipelines (fit on train only)
#
# All imputation, scaling, and encoding live inside `ColumnTransformer`
# objects that are **fit exclusively on the training rows**. Validation
# and test data are only *transformed*.
#
# ```
# ColumnTransformer
#     numerical   → SimpleImputer(median) + StandardScaler
#     ordinal     → OrdinalEncoder (with known categories)
#     nominal     → SimpleImputer(most_frequent) + OneHotEncoder
# ```

# %%
def make_preprocessor(num_cols=None, tx_cols=None,
                      ord_cols=None, cat_cols=None):
    """Build a ColumnTransformer for the given column groups."""
    transformers = []
    all_num = list(num_cols or []) + list(tx_cols or [])
    if all_num:
        transformers.append((
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]),
            all_num))
    if ord_cols:
        transformers.append((
            "ord",
            OrdinalEncoder(
                categories=[["Low", "Medium", "High", "Very High"],
                            ["Negative", "Neutral", "Positive"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1),
            ord_cols))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(drop="first",
                                            sparse_output=False,
                                            handle_unknown="infrequent_if_exist"))]),
            cat_cols))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


# One preprocessor per setup
pre_full = make_preprocessor(NUM_CUST, TX_FEAT, ORD_CUST, CAT_CUST)
pre_tx   = make_preprocessor(tx_cols=TX_FEAT)
pre_cu   = make_preprocessor(NUM_CUST, ord_cols=ORD_CUST, cat_cols=CAT_CUST)

# %%
# ── Fit on train, transform all splits ──
def prepare(df, feat_cols, pre):
    """Fit preprocessor on train only, transform train/val/test."""
    X = df[feat_cols]
    Xtr = pre.fit_transform(X.iloc[idx_train])
    Xva = pre.transform(X.iloc[idx_val])
    Xte = pre.transform(X.iloc[idx_test])
    y_tr = y_all[idx_train]
    y_va = y_all[idx_val]
    y_te = y_all[idx_test]
    fnames = list(pre.get_feature_names_out())
    return Xtr, Xva, Xte, y_tr, y_va, y_te, fnames


d_full = prepare(full_df, FULL_FEAT, pre_full)
d_tx   = prepare(tx_df,   TX_FEAT,   pre_tx)
d_cu   = prepare(cu_df,   CUST_ALL,  pre_cu)

for label, d in [("Full", d_full), ("Tx-strict", d_tx), ("Cust-only", d_cu)]:
    print(f"  {label:12s}: {d[0].shape[1]} features  "
          f"(train {d[0].shape[0]}, val {d[1].shape[0]}, test {d[2].shape[0]})")
print(f"\nPreprocessors fit on train ({len(idx_train):,} rows) only ✓")

# %% [markdown]
# ---
# ## 7. Model Training — Three Setups × Three Algorithms

# %%
def make_models(y_tr):
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    m = {
        "LR": LogisticRegression(C=1.0, max_iter=1000,
                  class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8,
                  min_samples_leaf=20, class_weight="balanced",
                  random_state=SEED, n_jobs=-1),
    }
    if HAS_XGB:
        m["XGB"] = XGBClassifier(n_estimators=300, max_depth=4,
                  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=spw, eval_metric="logloss",
                  random_state=SEED, n_jobs=-1)
    else:
        m["XGB"] = HistGradientBoostingClassifier(
                  max_iter=300, max_depth=4, learning_rate=0.05,
                  random_state=SEED)
    return m

# %%
setups = {
    "Full":      d_full,
    "Tx-strict": d_tx,
    "Cust-only": d_cu,
}

trained    = {}   # {(setup, algo): fitted model}
preds_val  = {}   # {(setup, algo): y_prob on validation}
preds_test = {}   # {(setup, algo): y_prob on test}

for sname, (Xtr, Xva, Xte, ytr, yva, yte, _fc) in setups.items():
    for algo, mdl in make_models(ytr).items():
        if algo == "XGB" and HAS_XGB:
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        else:
            mdl.fit(Xtr, ytr)
        preds_val[(sname, algo)]  = mdl.predict_proba(Xva)[:, 1]
        preds_test[(sname, algo)] = mdl.predict_proba(Xte)[:, 1]
        trained[(sname, algo)]    = mdl
        roc = roc_auc_score(yte, preds_test[(sname, algo)])
        pr  = average_precision_score(yte, preds_test[(sname, algo)])
        print(f"  {sname:12s} | {algo:3s} | ROC {roc:.4f} | PR {pr:.4f}")

# %% [markdown]
# ---
# ## 8. Comprehensive Evaluation (Test Set)

# %%
def precision_at_k(y_true, y_prob, k):
    idx = np.argsort(y_prob)[::-1][:k]
    return y_true[idx].mean()

def recall_at_k(y_true, y_prob, k):
    idx = np.argsort(y_prob)[::-1][:k]
    return y_true[idx].sum() / max(y_true.sum(), 1)

def lift_top_decile(y_true, y_prob):
    n10 = max(len(y_true) // 10, 1)
    top = y_true[np.argsort(y_prob)[::-1][:n10]]
    return top.mean() / max(y_true.mean(), 1e-9)

y_te = y_all[idx_test]

rows = []
for (sname, algo), yp in preds_test.items():
    yhat = (yp >= 0.5).astype(int)
    k10 = max(len(y_te) // 10, 1)
    rows.append({
        "Setup": sname, "Model": algo,
        "ROC_AUC": roc_auc_score(y_te, yp),
        "PR_AUC": average_precision_score(y_te, yp),
        "Brier": brier_score_loss(y_te, yp),
        "F1": f1_score(y_te, yhat),
        "Precision": precision_score(y_te, yhat, zero_division=0),
        "Recall": recall_score(y_te, yhat),
        "P@10%": precision_at_k(y_te, yp, k10),
        "R@10%": recall_at_k(y_te, yp, k10),
        "Lift@D1": lift_top_decile(y_te, yp),
    })

eval_df = pd.DataFrame(rows).sort_values(["Setup", "ROC_AUC"], ascending=[True, False])
print(eval_df.to_string(index=False, float_format="{:.4f}".format))

# %%
# ── ROC / PR / Calibration curves (Full setup, test set) ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
base_rate = y_te.mean()

for algo in ["LR", "RF", "XGB"]:
    yp = preds_test[("Full", algo)]
    fpr, tpr, _ = roc_curve(y_te, yp)
    axes[0].plot(fpr, tpr, label=f"{algo} ({roc_auc_score(y_te,yp):.3f})", lw=1.5)
    prec_a, rec_a, _ = precision_recall_curve(y_te, yp)
    axes[1].plot(rec_a, prec_a, label=f"{algo} ({average_precision_score(y_te,yp):.3f})", lw=1.5)
    pt, pp = calibration_curve(y_te, yp, n_bins=10, strategy="uniform")
    axes[2].plot(pp, pt, "o-", ms=4, label=algo, lw=1.5)

axes[0].plot([0,1],[0,1],"k--",lw=.7,alpha=.5); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC", fontweight="bold"); axes[0].legend(fontsize=7)
axes[1].axhline(base_rate, color="gray", ls="--", lw=.7, label=f"base {base_rate:.2f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall", fontweight="bold"); axes[1].legend(fontsize=7)
axes[2].plot([0,1],[0,1],"k--",lw=.7,alpha=.5); axes[2].set_xlabel("Mean predicted prob")
axes[2].set_ylabel("Fraction positive"); axes[2].set_title("Calibration", fontweight="bold")
axes[2].legend(fontsize=7)
fig.suptitle("Full-Setup Evaluation — Future Disengagement (test set)\n"
             "Features from obs. window only · preprocessing fit on train only", y=1.03)
fig.tight_layout(); plt.show()

# %%
# ── Confusion matrices (Full setup, default threshold 0.5, test set) ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, algo in zip(axes, ["LR", "RF", "XGB"]):
    yp = preds_test[("Full", algo)]
    cm = confusion_matrix(y_te, (yp >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Active", "Disengaged"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Full / {algo}", fontsize=10, fontweight="bold")
fig.suptitle("Confusion Matrices — threshold 0.5 (test set)", fontsize=11)
fig.tight_layout(); plt.show()

# %% [markdown]
# ---
# ## 9. Threshold Tuning (Validation Set Only)
#
# The decision threshold is selected by sweeping values on the
# **validation split** and choosing the value that maximises F1.
# The test set is **never** used for threshold selection.

# %%
y_va = y_all[idx_val]

# Identify best Full-setup model by validation ROC AUC
val_rocs = {a: roc_auc_score(y_va, preds_val[("Full", a)])
            for a in ["LR", "RF", "XGB"]}
best_algo = max(val_rocs, key=val_rocs.get)
yp_val = preds_val[("Full", best_algo)]
yp_te  = preds_test[("Full", best_algo)]

thresholds = np.arange(0.05, 0.96, 0.01)
th_rows = []
for t in thresholds:
    yhat = (yp_val >= t).astype(int)
    th_rows.append({
        "threshold": t,
        "f1": f1_score(y_va, yhat, zero_division=0),
        "precision": precision_score(y_va, yhat, zero_division=0),
        "recall": recall_score(y_va, yhat, zero_division=0),
    })

th_df = pd.DataFrame(th_rows)
best_t = th_df.loc[th_df["f1"].idxmax(), "threshold"]
best_f1_val = th_df["f1"].max()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(th_df["threshold"], th_df["f1"], label="F1", lw=1.5, color="#1976D2")
ax.plot(th_df["threshold"], th_df["precision"], label="Precision", lw=1, ls="--", color="#388E3C")
ax.plot(th_df["threshold"], th_df["recall"], label="Recall", lw=1, ls="--", color="#D32F2F")
ax.axvline(best_t, color="black", lw=.8, ls=":",
           label=f"Best F1 = {best_f1_val:.4f} at t = {best_t:.2f}")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
ax.set_title(f"Threshold Tuning on Validation — Full / {best_algo}",
             fontweight="bold")
ax.legend(); fig.tight_layout(); plt.show()

# %%
# ── Final evaluation on TEST with frozen threshold ──
yhat_tuned = (yp_te >= best_t).astype(int)

print(f"=== Final: Full / {best_algo} | threshold {best_t:.2f} (tuned on validation) ===")
print(classification_report(y_te, yhat_tuned, target_names=["Active", "Disengaged"]))
print(f"Brier score: {brier_score_loss(y_te, yp_te):.4f}")
print(f"ROC AUC:     {roc_auc_score(y_te, yp_te):.4f}")

# %% [markdown]
# ---
# ## 10. Post-Hoc Sanity Check — `churn_probability`
#
# The dataset ships a pre-computed `churn_probability` column. We **never**
# use it as a training feature. Instead we correlate our model's predicted
# scores with it to check directional consistency.

# %%
test_ids = full_df.iloc[idx_test]["customer_id"].values
sanity = pd.DataFrame({"customer_id": test_ids, "model_score": yp_te})
sanity = sanity.merge(sanity_churn, on="customer_id")
sanity["true_label"] = y_te

corr = sanity["model_score"].corr(sanity["churn_probability"])
auc_cp = roc_auc_score(sanity["true_label"], sanity["churn_probability"])
auc_us = roc_auc_score(sanity["true_label"], sanity["model_score"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hexbin(sanity["churn_probability"], sanity["model_score"],
               gridsize=40, cmap="YlOrRd", mincnt=1)
axes[0].set_xlabel("churn_probability (dataset)"); axes[0].set_ylabel("Our model score")
axes[0].set_title(f"Score Correlation (ρ = {corr:.3f})", fontweight="bold")
plt.colorbar(axes[0].collections[0], ax=axes[0], label="count")

for label, score, ls in [("Our model", sanity["model_score"], "-"),
                          ("churn_probability", sanity["churn_probability"], "--")]:
    fpr, tpr, _ = roc_curve(sanity["true_label"], score)
    axes[1].plot(fpr, tpr, ls=ls, lw=1.5,
                 label=f"{label} (AUC {roc_auc_score(sanity['true_label'], score):.3f})")
axes[1].plot([0,1],[0,1],"k--",lw=.6,alpha=.5)
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].set_title("ROC: Our Model vs churn_probability", fontweight="bold")
axes[1].legend(); fig.tight_layout(); plt.show()

print(f"Our model ROC AUC for future_disengaged: {auc_us:.4f}")
print(f"churn_probability ROC AUC for same target: {auc_cp:.4f}")

# %% [markdown]
# ---
# ## 11. Interpretation
#
# ### 11a. Feature Importance (XGBoost Gain)
# ### 11b. Logistic Regression Coefficients
# ### 11c. SHAP (if available)

# %%
# ── 11a. XGBoost feature importance ──
xgb_full = trained[("Full", "XGB")]
feat_names_full = d_full[6]

imp = pd.DataFrame({"feature": feat_names_full,
                     "importance": xgb_full.feature_importances_}
                    ).sort_values("importance", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8, 6))
imp_p = imp.sort_values("importance")
ax.barh(imp_p["feature"], imp_p["importance"], color="#1976D2", alpha=.85)
ax.set_xlabel("Importance (Gain)")
ax.set_title("Top 20 Features — XGBoost (Full Setup)\nFuture Disengagement",
             fontweight="bold")
ax.tick_params(axis="y", labelsize=8); fig.tight_layout(); plt.show()

# %%
# ── 11b. Logistic Regression coefficients ──
lr_full = trained[("Full", "LR")]
coefs = lr_full.coef_[0]
lr_df = pd.DataFrame({"feature": feat_names_full, "coef": coefs})
lr_df["abs"] = lr_df["coef"].abs()
lr_top = lr_df.sort_values("abs", ascending=False).head(20).sort_values("coef")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#D32F2F" if v > 0 else "#388E3C" for v in lr_top["coef"]]
ax.barh(lr_top["feature"], lr_top["coef"], color=colors, alpha=.85)
ax.axvline(0, color="k", lw=.5)
ax.set_xlabel("Coefficient (standardised)")
ax.set_title("Top 20 — Logistic Regression (Full Setup)\nRed ↑ risk  Green ↓ risk",
             fontsize=10, fontweight="bold")
ax.tick_params(axis="y", labelsize=8); fig.tight_layout(); plt.show()

# %%
# ── 11c. SHAP (TreeExplainer on XGBoost) ──
if HAS_SHAP:
    explainer = shap.TreeExplainer(xgb_full)
    Xf_te = d_full[2]  # preprocessed test features
    n_samp = min(2000, len(Xf_te))
    sample_idx = np.random.RandomState(SEED).choice(
        len(Xf_te), size=n_samp, replace=False)
    X_sample = pd.DataFrame(Xf_te[sample_idx], columns=feat_names_full)
    shap_vals = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_sample, max_display=20, show=False)
    plt.title("SHAP Summary — XGBoost (Full Setup)", fontweight="bold")
    plt.tight_layout(); plt.show()
else:
    print("SHAP not installed — skipping. Install with: pip install shap")

# %% [markdown]
# ---
# ## 12. Feature Dictionary
#
# ### Customer Attributes (treated as observation-cutoff snapshots)
#
# | # | Feature | Description | Note |
# |---|---------|-------------|------|
# | 1 | `age` | Customer age | Static |
# | 2 | `household_size` | Household size | Static |
# | 3 | `income_bracket` | Income bracket (ordinal: Low=0 → Very High=3) | Static |
# | 4 | `gender` | Gender (one-hot, drop-first) | Static |
# | 5 | `savings_account` … `insurance_product` | Product holding flags (0/1) | Assumed static |
# | 6 | `active_products` | Count of active products | Assumed static |
# | 7 | `app_logins_frequency` | Monthly login frequency | ⚠ Assumed snapshot |
# | 8 | `feature_usage_diversity` | Number of app features used | ⚠ Assumed snapshot |
# | 9 | `bill_payment_user`, `auto_savings_enabled` | Digital engagement flags | Assumed static |
# | 10 | `credit_utilization_ratio` | Credit utilisation (imputed on train) | ⚠ Assumed snapshot |
# | 11 | `international_transactions`, `failed_transactions` | Transaction risk indicators | ⚠ Assumed snapshot |
# | 12 | `satisfaction_score` | Overall satisfaction (aggregate) | ⚠ Assumed snapshot |
# | 13 | `nps_score` | Net Promoter Score | ⚠ Assumed snapshot |
# | 14 | `support_tickets_count`, `resolved_tickets_ratio` | Support interaction | ⚠ Assumed snapshot |
# | 15 | `feedback_sentiment` | Feedback sentiment (ordinal: Neg=0 → Pos=2) | ⚠ Assumed snapshot |
# | 16 | `customer_tenure` | Tenure in months | ⚠ Assumed snapshot |
# | 17 | `acquisition_channel` | Acquisition channel (one-hot) | Static |
#
# ⚠ = variable is not timestamped; assumed to reflect state at observation
# cutoff. See §2 assumption note and Tx-strict benchmark for mitigation.
#
# ### Observation-Window Transaction Features (Jan–Sep 2023)
#
# | # | Feature | Description |
# |---|---------|-------------|
# | 18 | `tx_count` | Total obs-window transactions |
# | 19 | `tx_total` | Sum of amounts |
# | 20 | `tx_mean`, `tx_median` | Central tendency |
# | 21 | `tx_std`, `tx_cv` | Volatility (std, coefficient of variation) |
# | 22 | `tx_min`, `tx_max` | Amount range |
# | 23 | `tx_recency` | Days from last obs-window tx to Sep 30 |
# | 24 | `tx_span` | Active span (first-to-last tx, days) |
# | 25 | `tx_freq` | Transactions per active day |
# | 26 | `sh_deposit` … `sh_withdrawal` | Transaction-type shares |
# | 27 | `tx_wknd_ratio` | Weekend transaction fraction |
# | 28 | `tx_count_trend` | Linear slope of monthly count |
# | 29 | `tx_amt_trend` | Relative slope of monthly amount |
# | 30 | `tx_max_gap` | Largest gap between consecutive txns |
# | 31 | `tx_late_ratio` | Jul–Sep monthly rate / Jan–Jun monthly rate |

# %% [markdown]
# ---
# ## 13. Methodology Note
#
# ### Problem Framing
#
# No contractual churn label exists in the COFINFAD dataset. The
# pre-computed `churn_probability` column is a continuous score (0.1–0.5),
# not a ground-truth event — using it as feature or target would create
# circularity. We therefore frame the task as **future transactional
# disengagement prediction**: can we identify, from Jan–Sep behaviour,
# which customers will become silent or sharply decline in Oct–Dec?
#
# ### Target Definition
#
# `future_disengaged = 1` when a customer has **zero Q4 transactions** or
# a Q4 monthly transaction rate **below 20 %** of their Jan–Sep rate.
# This is a **behavioural proxy**, not contractual churn. The goal is to
# detect **strong activity decline**, not minor fluctuations.
#
# In production, this proxy would be replaced by actual churn events
# (closures, cancellations) or business-defined inactivity rules.
#
# ### Temporal Design
#
# - **Observation window (Jan–Sep 2023):** all transaction features built here.
# - **Prediction window (Oct–Dec 2023):** target defined here.
# - No information crosses the Sep 30 boundary.
#
# ### Leakage Controls
#
# 1. **Pre-computed aggregates** in `customer_data` span the full year and
#    are **excluded**. All transaction features are rebuilt from raw
#    `transactions_data` restricted to the observation window.
# 2. **`churn_probability`, `customer_lifetime_value`, `clv_segment`** are
#    never used as features. `churn_probability` is only used for a post-hoc
#    correlation check (§10).
# 3. **`customer_segment`** is excluded — it may incorporate future-window
#    behaviour.
# 4. **Preprocessing** (imputation, scaling, encoding) is wrapped in
#    `ColumnTransformer` pipelines fit **only on the training split**,
#    preventing distribution leakage from validation or test data.
# 5. **Customer-level variables** without timestamps are documented as
#    assumed snapshots (§2). The **Tx-strict** benchmark verifies the model
#    retains signal without any customer-level columns.
#
# ### Evaluation Protocol
#
# - **60 / 20 / 20** stratified train / validation / test split.
# - Models are fit on **train** only.
# - **Threshold** is tuned on **validation** (maximising F1).
# - **Final metrics** are reported on the **held-out test** set: ROC AUC,
#   PR AUC, Brier score, F1, precision, recall, P@10 %, R@10 %, lift at
#   top decile.
# - Calibration curves check probability reliability.
#
# ### Limitations
#
# 1. **No contractual churn event** — the target is a future-disengagement
#    proxy.
# 2. **Single temporal split** — one observation/prediction window pair;
#    no walk-forward validation.
# 3. **Customer attributes are assumed static** — in practice, satisfaction
#    or product holdings may change over time. The Tx-strict benchmark
#    mitigates this risk.
# 4. **12-month dataset** — limits generalisability to seasonal patterns.
# 5. **Class imbalance** — ~2 % disengagement rate makes precision on the
#    minority class challenging; PR AUC is the more informative metric.

# %%
print("=" * 60)
print("  Pipeline complete — Future Disengagement Prediction")
print("=" * 60)
print(f"\n  Obs. window:         Jan – Sep 2023")
print(f"  Pred. window:        Oct – Dec 2023")
print(f"  Customers:           {len(y_all):,}")
print(f"  Disengaged (target): {int(y_all.sum()):,} ({y_all.mean():.1%})")
print(f"  Train / Val / Test:  {len(idx_train):,} / {len(idx_val):,} / {len(idx_test):,}")
print(f"  Full features:       {len(d_full[6])}")
print(f"  Tx-strict features:  {len(d_tx[6])}")
print(f"  Cust-only features:  {len(d_cu[6])}")
best_row = eval_df.loc[eval_df["ROC_AUC"].idxmax()]
print(f"\n  Best ROC AUC: {best_row['Setup']} / {best_row['Model']} — {best_row['ROC_AUC']:.4f}")
print(f"  Best threshold (F1 on val): {best_t:.2f} → val F1 = {best_f1_val:.4f}")
print(f"  churn_probability sanity check: ρ = {corr:.3f}")
print(f"\n  Preprocessing:  ColumnTransformer fit on train only ✓")
print(f"  Threshold:      tuned on validation only ✓")
print(f"  Test set:       used for final evaluation only ✓")

# %%
# ── Save artifacts for dashboard ──
import pickle as _pkl

_artifacts = {
    # Evaluation
    "eval_df": eval_df,
    "th_df": th_df,
    "best_algo": best_algo,
    "best_t": best_t,
    "best_f1_val": best_f1_val,
    # Predictions
    "preds_test": preds_test,
    "preds_val": preds_val,
    # Labels
    "y_all": y_all,
    "y_te": y_all[idx_test],
    "y_va": y_all[idx_val],
    # Indices
    "idx_train": idx_train,
    "idx_val": idx_val,
    "idx_test": idx_test,
    # Models & preprocessors
    "trained": trained,
    "pre_full": pre_full,
    # Feature names
    "feat_names_full": d_full[6],
    "feat_names_tx": d_tx[6],
    "feat_names_cu": d_cu[6],
    "FULL_FEAT": FULL_FEAT,
    "TX_FEAT": TX_FEAT,
    "CUST_ALL": CUST_ALL,
    # Data for explorer
    "full_df": full_df,
    "target_df": target_df,
    "cust": cust,
    "sanity_churn": sanity_churn,
    # Sanity
    "corr_sanity": corr,
    # SHAP (if computed)
    "shap_vals": shap_vals if HAS_SHAP else None,
    "shap_sample_idx": sample_idx if HAS_SHAP else None,
}

_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(_out_dir, exist_ok=True)
_art_path = os.path.join(_out_dir, "churn_artifacts.pkl")
with open(_art_path, "wb") as _f:
    _pkl.dump(_artifacts, _f, protocol=4)
print(f"\n  Artifacts saved to {_art_path}")
