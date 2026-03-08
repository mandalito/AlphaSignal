"""
app.py — AlphaSignal: Client Intelligence System
=================================================
Interactive strategic report for Pictet Asset Management.
Structured as an 11-section narrative — from problem definition
through to actionable sales intelligence.

Run with:
    streamlit run app.py

Prerequisites:
    python3 notebooks/churn_modeling.py   (generates outputs/churn_artifacts.pkl)
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score,
    precision_score, recall_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(ROOT, "outputs", "churn_artifacts.pkl")

st.set_page_config(
    page_title="AlphaSignal — Client Intelligence Report",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────
# Design tokens
# ───────────────────────────────────────────────────────────────────────────
C = {
    "primary": "#1976D2",
    "xgb": "#1976D2",
    "rf": "#388E3C",
    "lr": "#FFA000",
    "full": "#1976D2",
    "tx": "#388E3C",
    "cust": "#9C27B0",
    "active": "#BDBDBD",
    "disengaged": "#D32F2F",
    "accent": "#7B1FA2",
}
ALGO_COLORS = {"LR": C["lr"], "RF": C["rf"], "XGB": C["xgb"]}
SETUP_COLORS = {"Full": C["full"], "Tx-strict": C["tx"], "Cust-only": C["cust"]}


# ───────────────────────────────────────────────────────────────────────────
# Artifact loading
# ───────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(
            "Artifacts not found. Run the pipeline first:\n\n"
            "```\npython3 notebooks/churn_modeling.py\n```"
        )
        st.stop()
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def _artifact_mtime():
    try:
        return os.path.getmtime(ARTIFACT_PATH)
    except OSError:
        return 0


if "artifact_mtime" not in st.session_state:
    st.session_state.artifact_mtime = _artifact_mtime()
elif st.session_state.artifact_mtime != _artifact_mtime():
    st.session_state.artifact_mtime = _artifact_mtime()
    load_artifacts.clear()


# ───────────────────────────────────────────────────────────────────────────
# Sidebar — Table of Contents
# ───────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='margin-bottom:0'>📈 AlphaSignal</h2>"
    "<p style='color:grey;font-size:0.85em;margin-top:0'>"
    "Client Intelligence &amp; Distribution Analytics Platform</p>",
    unsafe_allow_html=True,
)

SECTIONS = [
    "§1  Executive Summary",
    "§2  Problem Definition",
    "§3  Dataset Description",
    "§4  Temporal Modeling Strategy",
    "§5  Feature Engineering & Explainability",
    "§6  Machine Learning Models",
    "§7  Probability Calibration",
    "§8  Signal Architecture",
    "§9  Expected Client Value",
    "§10 Opportunity Frontier",
    "§11 Sales Intelligence",
    "§12 Distribution Intelligence",
]

section = st.sidebar.radio("Navigate", SECTIONS)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**COFINFAD** — Colombian Fintech Financial Analytics Dataset  \n"
    "48,723 customers · 3.2 M transactions · 2023  \n\n"
    "*Prepared for Pictet Asset Management*"
)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §1 EXECUTIVE SUMMARY                                                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_executive_summary():
    a = load_artifacts()
    eval_df = a["eval_df"]
    signals_df = a.get("signals_df")
    wf_df = a.get("wf_df")

    st.title("§1 — Executive Summary")
    st.markdown("""
    <div style='background:#f0f4ff;padding:1.2em 1.5em;border-left:4px solid #1976D2;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>AlphaSignal</strong> is a machine-learning-driven client intelligence
    system that transforms raw fintech transaction data into calibrated,
    risk-adjusted commercial opportunity scores — enabling relationship
    managers to prioritise upsell, retention, and monitoring actions with
    quantitative precision.
    </div>
    """, unsafe_allow_html=True)

    # Headline KPIs
    best_full = eval_df[eval_df["Setup"] == "Full"].sort_values(
        "ROC_AUC", ascending=False).iloc[0]
    best_tx = eval_df[eval_df["Setup"] == "Tx-strict"].sort_values(
        "ROC_AUC", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Best Disengagement Predictor",
        f"ROC AUC {best_full['ROC_AUC']:.3f}",
        help=f"{best_full['Setup']} / {best_full['Model']}",
    )
    c2.metric(
        "Transaction-Only Benchmark",
        f"ROC AUC {best_tx['ROC_AUC']:.3f}",
        help="Uses no customer-level attributes",
    )
    if wf_df is not None:
        c3.metric(
            "Walk-Forward Stability",
            f"{wf_df['ROC_AUC'].mean():.3f} ± {wf_df['ROC_AUC'].std():.3f}",
            help=f"{len(wf_df)} expanding-window folds",
        )
    else:
        c3.metric("Customers Scored", f"{len(a['y_all']):,}")
    if signals_df is not None:
        c4.metric(
            "Upsell Candidates Identified",
            f"{(signals_df['recommended_action'] == 'Upsell').sum():,}",
        )
    else:
        c4.metric("Target Rate", f"{a['y_all'].mean():.1%}")

    st.markdown("---")

    # System architecture
    st.subheader("System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────┐
    │  Raw Data            │   48,723 customers × 3.2 M transactions
    │  (COFINFAD 2023)     │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Feature Engineering │   RFM aggregates, trends, type shares,
    │  Jan – Sep 2023      │   weekend ratio, max gaps, …
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐   3 setups × 3 algorithms = 9 models
    │  Model Training      │   + walk-forward temporal validation
    │  + Calibration       │   + isotonic probability calibration
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Signal Layer        │   RedemptionRisk · BuyPropensity · Engagement
    │  (calibrated)        │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐   MasterSignal = BP × (1 − RR) × Engagement
    │  Commercial Scoring  │   ExpectedClientValue = Master × tx_total
    │  + Opportunity       │   OpportunityFrontier = ECV / RR (normalised)
    │    Frontier          │
    └────────┬────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  Actionable Output   │   Upsell / Retention / Monitor
    │  (this report)       │   Risk-adjusted client rankings
    └─────────────────────┘
    ```
    """)

    st.markdown("---")

    # Key findings
    st.subheader("Key Findings")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
        **Predictive Performance**
        - Disengagement prediction achieves **ROC AUC > 0.79** on held-out test data
        - Transaction-derived features carry **>95%** of predictive signal;
          customer demographics alone are near-random
        - Signal is **temporally stable** across 5 walk-forward folds
          (σ < 0.04)
        """)
    with cols[1]:
        st.markdown("""
        **Commercial Intelligence**
        - Three calibrated signals decompose client opportunity into
          **risk**, **propensity**, and **engagement** dimensions
        - The Opportunity Frontier Score provides a **risk-adjusted ranking**
          analogous to Sharpe ratio in portfolio management
        - Every client receives a quantitative **recommended action**
          (Upsell / Retention / Monitor) with an associated expected value
        """)

    st.info(
        "**Navigation →** Use the sidebar to explore each section of this report "
        "in depth. Sections follow the analytical pipeline sequentially — from "
        "raw data through to actionable intelligence."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §2 PROBLEM DEFINITION                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_problem_definition():
    a = load_artifacts()
    st.title("§2 — Problem Definition")

    st.markdown("""
    <div style='background:#fff8e1;padding:1.2em 1.5em;border-left:4px solid #FFA000;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>Business question:</strong> Which clients of a Colombian fintech
    are likely to disengage in the next quarter — and among those who stay,
    who represents the highest commercial opportunity?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Why Machine Learning?

    Traditional rule-based client segmentation (e.g. RFM buckets) cannot
    capture the non-linear, multi-dimensional interactions between
    hundreds of behavioural features that predict future disengagement.
    Supervised learning models discover these complex patterns automatically,
    while rigorous temporal validation ensures robustness.

    ### What We Predict

    | Signal | Definition | Business Use |
    |--------|-----------|--------------|
    | **Redemption Risk** | P(transactional disengagement in Q4 2023) | Retention targeting |
    | **Buy Propensity** | P(meaningful transaction growth in Q4 2023) | Upsell prioritisation |
    | **Engagement Score** | Normalised behavioural intensity | Relationship depth |

    These three signals feed into a **composite scoring framework** that
    mirrors expected-value decomposition in quantitative finance:

    $$\\text{MasterSignal} = \\text{BuyPropensity} \\times (1 - \\text{RedemptionRisk}) \\times \\text{EngagementScore}$$

    ### Why "Disengagement" Rather Than "Churn"?

    The COFINFAD dataset does not contain explicit account closure events.
    Instead, we define **future disengagement** as a behavioural proxy:
    a client who shows transactional silence or a sharp decline (>50%)
    during the prediction window. This is a strictly conservative proxy —
    an important caveat for interpretation.
    """)

    # Target rate visualisation
    y_all = a["y_all"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Clients", f"{len(y_all):,}")
    c2.metric("Disengaged (target = 1)", f"{int(y_all.sum()):,}")
    c3.metric("Base Rate", f"{y_all.mean():.1%}")

    st.warning(
        "**Important caveat:** `future_disengaged` is a behavioural proxy, not "
        "contractual churn. Model outputs should be interpreted as "
        "*propensity to disengage transactionally*, not certainty of account closure."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §3 DATASET DESCRIPTION                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_dataset():
    a = load_artifacts()
    st.title("§3 — Dataset Description")

    cust = a["cust"]
    target_df = a["target_df"]
    full_df = a["full_df"]
    y_all = a["y_all"]

    st.markdown("""
    The **COFINFAD** (Colombian Fintech Financial Analytics Dataset) comprises
    two relational tables covering the full year 2023:

    | Table | Granularity | Size | Key Variables |
    |-------|------------|------|---------------|
    | **Customers** | One row per client | 48,723 × 54 | Demographics, products, satisfaction, NPS |
    | **Transactions** | One row per event | 3.2 M × 4 | customer_id, date, type, amount |

    The data is split into an **observation window** (Jan–Sep 2023) for
    feature construction and a **prediction window** (Oct–Dec 2023) for
    target definition, with a strict temporal barrier at September 30.
    """)

    st.markdown("---")

    # Target distribution
    st.subheader("Target Distribution — future_disengaged")
    c1, c2 = st.columns([1, 2])
    with c1:
        tgt_counts = pd.Series(y_all).value_counts().sort_index()
        st.metric("Active (0)", f"{tgt_counts.get(0, 0):,}")
        st.metric("Disengaged (1)", f"{tgt_counts.get(1, 0):,}")
        st.metric("Imbalance Ratio", f"1 : {tgt_counts.get(0, 1) / max(tgt_counts.get(1, 1), 1):.1f}")
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(
            ["Active", "Disengaged"],
            [tgt_counts.get(0, 0), tgt_counts.get(1, 0)],
            color=[C["active"], C["disengaged"]], alpha=0.85,
        )
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 200,
                    f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Target: future_disengaged", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Customer attribute distributions
    st.subheader("Customer Attributes by Disengagement Status")
    st.caption(
        "Density plots highlight distributional differences between active "
        "and disengaged clients across key demographic and behavioural variables."
    )

    num_features = ["age", "customer_tenure", "active_products",
                    "satisfaction_score", "nps_score",
                    "app_logins_frequency", "credit_utilization_ratio"]
    num_features = [f for f in num_features if f in cust.columns]

    merged = cust[["customer_id"] + num_features].merge(
        target_df[["customer_id", "future_disengaged"]], on="customer_id"
    )

    cols = st.columns(3)
    for i, feat in enumerate(num_features[:6]):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 3))
            active = merged.loc[merged["future_disengaged"] == 0, feat].dropna()
            diseng = merged.loc[merged["future_disengaged"] == 1, feat].dropna()
            ax.hist(active, bins=30, alpha=0.5, color=C["active"],
                    label="Active", density=True)
            ax.hist(diseng, bins=30, alpha=0.6, color=C["disengaged"],
                    label="Disengaged", density=True)
            ax.set_title(feat.replace("_", " ").title(), fontsize=9,
                         fontweight="bold")
            ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # Transaction feature distributions
    st.subheader("Transaction-Derived Features (Observation Window)")
    st.caption(
        "These behavioural indicators are engineered from raw transaction "
        "data during the observation window (Jan–Sep 2023)."
    )

    tx_feats = ["tx_count", "tx_total", "tx_recency", "tx_max_gap",
                "tx_late_ratio", "tx_wknd_ratio"]
    tx_feats = [f for f in tx_feats if f in full_df.columns]

    cols = st.columns(3)
    for i, feat in enumerate(tx_feats):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 3))
            active = full_df.loc[full_df["future_disengaged"] == 0, feat].dropna()
            diseng = full_df.loc[full_df["future_disengaged"] == 1, feat].dropna()
            # Clip at 99th percentile to avoid extreme outliers collapsing
            # all data into a single bin (e.g. tx_total spans 9 orders of
            # magnitude, making the histogram appear empty).
            clip_hi = full_df[feat].quantile(0.99)
            active = active.clip(upper=clip_hi)
            diseng = diseng.clip(upper=clip_hi)
            ax.hist(active, bins=30, alpha=0.5, color=C["active"],
                    label="Active", density=True)
            ax.hist(diseng, bins=30, alpha=0.6, color=C["disengaged"],
                    label="Disengaged", density=True)
            ax.set_title(feat.replace("_", " ").title(), fontsize=9,
                         fontweight="bold")
            ax.legend(fontsize=7)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("""
    ---
    **Insight:** Transaction-derived features show visibly different
    distributions for active vs disengaged clients, while demographic
    attributes are largely overlapping. This foreshadows the modelling
    results in §6 — behavioural data, not demographics, drives prediction.
    """)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §4 TEMPORAL MODELING STRATEGY                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_temporal():
    a = load_artifacts()
    st.title("§4 — Temporal Modeling Strategy")

    st.markdown("""
    A rigorous temporal design prevents **look-ahead bias** — the most
    critical integrity requirement for any predictive model in finance.

    ### Observation / Prediction Split
    """)

    st.markdown("""
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
    """)

    st.markdown("""
    The temporal barrier is **absolute**: no feature is derived from data
    occurring after September 30. The target variable is defined exclusively
    from Q4 2023 transaction patterns.

    ### Data Split

    The dataset is partitioned into three non-overlapping sets:
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train (60%)", f"{len(a['idx_train']):,} clients")
    c2.metric("Validation (20%)", f"{len(a['idx_val']):,} clients")
    c3.metric("Test (20%)", f"{len(a['idx_test']):,} clients")

    st.markdown("""
    - **Train** — model fitting
    - **Validation** — threshold tuning, probability calibration
    - **Test** — final evaluation only (never seen during any optimisation)

    All splits are **stratified** to preserve the target distribution.
    """)

    st.markdown("---")

    # Walk-Forward Validation
    wf_df = a.get("wf_df")
    if wf_df is not None:
        st.subheader("Walk-Forward Temporal Validation")
        st.markdown("""
        To verify that the disengagement signal is **not an artefact of a
        single temporal split**, we run a 5-fold expanding-window walk-forward
        validation. Each fold shifts the observation and prediction windows
        forward by one month, rebuilding all features and targets from scratch.
        """)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean ROC AUC",
                  f"{wf_df['ROC_AUC'].mean():.4f}",
                  help=f"σ = {wf_df['ROC_AUC'].std():.4f}")
        c2.metric("Mean PR AUC",
                  f"{wf_df['PR_AUC'].mean():.4f}",
                  help=f"σ = {wf_df['PR_AUC'].std():.4f}")
        c3.metric("Folds", str(len(wf_df)))

        # Fold results table
        fmt = {c: "{:.4f}" for c in wf_df.columns
               if c not in ("Fold", "Obs_Window", "Pred_Window")}
        if "Target_Rate" in fmt:
            fmt["Target_Rate"] = "{:.3f}"
        st.dataframe(
            wf_df.style.format(fmt).highlight_max(
                subset=["ROC_AUC", "PR_AUC"], color="#c8e6c9"
            ).highlight_min(subset=["Brier"], color="#c8e6c9"),
            width="stretch", hide_index=True,
        )

        # Performance across folds
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        folds_x = wf_df["Fold"].values

        for ax, metric, col, clr in [
            (axes[0], "ROC_AUC", "ROC AUC", C["xgb"]),
            (axes[1], "PR_AUC", "PR AUC", C["rf"]),
        ]:
            vals = wf_df[metric].values
            ax.plot(folds_x, vals, "o-", color=clr, lw=2, ms=8, label=col)
            ax.axhline(vals.mean(), color=clr, ls="--", lw=1, alpha=0.6,
                       label=f"Mean = {vals.mean():.4f}")
            ax.fill_between(folds_x, vals.mean() - vals.std(),
                            vals.mean() + vals.std(), alpha=0.15, color=clr)
            ax.set_xlabel("Fold"); ax.set_ylabel(col)
            ax.set_title(f"Walk-Forward {col}", fontweight="bold")
            ax.set_xticks(folds_x); ax.legend(fontsize=8)

        fig.suptitle("Expanding-Window Walk-Forward Validation",
                     y=1.02, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        roc_std = wf_df["ROC_AUC"].std()
        if roc_std < 0.03:
            st.success(
                f"**Signal is temporally stable.** ROC AUC standard deviation "
                f"across folds is only {roc_std:.4f}, confirming the "
                f"behavioural disengagement pattern persists across different "
                f"time periods."
            )
        else:
            st.warning(
                f"ROC AUC standard deviation across folds is {roc_std:.4f}. "
                f"Some temporal instability detected."
            )

        with st.expander("Fold methodology details"):
            st.markdown("""
            | Fold | Training Window | Prediction Window |
            |------|----------------|-------------------|
            | 1 | Jan – May | Jun – Jul |
            | 2 | Jan – Jun | Jul – Aug |
            | 3 | Jan – Jul | Aug – Sep |
            | 4 | Jan – Aug | Sep – Oct |
            | 5 | Jan – Sep | Oct – Nov |

            Features and targets are reconstructed independently for each
            fold from raw transaction data. No information leaks across
            temporal boundaries.
            """)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §5 FEATURE ENGINEERING & EXPLAINABILITY                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_features():
    a = load_artifacts()
    st.title("§5 — Feature Engineering & Explainability")

    feat_names = a["feat_names_full"]
    trained = a["trained"]

    st.markdown(f"""
    Features are engineered from the **observation window** (Jan–Sep 2023)
    across three setups:

    | Setup | Features | Rationale |
    |-------|---------|-----------|
    | **Full** | Customer attributes + transaction aggregates | Maximise predictive power |
    | **Tx-strict** | Transaction aggregates only | Eliminate any leakage from static snapshots |
    | **Cust-only** | Customer attributes only | Benchmark — isolate demographic signal |

    Total feature dimensionality (Full setup): **{len(feat_names)}** variables
    after preprocessing.
    """)

    st.markdown("---")

    # XGBoost feature importance
    st.subheader("Feature Importance — XGBoost (Full Setup, Gain)")
    st.caption(
        "Gain measures the improvement in the loss function each time a "
        "feature is used for splitting. Higher gain = stronger contribution."
    )

    xgb_full = trained[("Full", "XGB")]
    imp = pd.DataFrame({
        "feature": feat_names,
        "importance": xgb_full.feature_importances_,
    }).sort_values("importance", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    imp_p = imp.sort_values("importance")
    ax.barh(imp_p["feature"], imp_p["importance"], color=C["xgb"], alpha=0.85)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Top 20 Features — XGBoost (Full Setup)", fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # LR coefficients
    st.subheader("Logistic Regression Coefficients")
    st.caption(
        "Standardised coefficients show direction and magnitude of each "
        "feature's effect on disengagement probability. Red = increases risk."
    )

    lr_full = trained[("Full", "LR")]
    coefs = lr_full.coef_[0]
    lr_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    lr_df["abs"] = lr_df["coef"].abs()
    lr_top = lr_df.sort_values("abs", ascending=False).head(20).sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [C["disengaged"] if v > 0 else C["rf"] for v in lr_top["coef"]]
    ax.barh(lr_top["feature"], lr_top["coef"], color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Coefficient (standardised)")
    ax.set_title(
        "Top 20 — Logistic Regression (Full Setup)\n"
        "Red = ↑ disengagement risk   Green = ↓ risk",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # SHAP
    shap_vals = a.get("shap_vals")

    if shap_vals is not None:
        st.subheader("SHAP Analysis — XGBoost")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) provides a game-theoretic
        decomposition of each individual prediction. Unlike global feature
        importance, SHAP reveals **how much each feature pushes a specific
        client's score up or down**.
        """)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        sp = shap_df.sort_values("mean_abs_shap")
        ax.barh(sp["feature"], sp["mean_abs_shap"], color=C["xgb"], alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance", fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # SHAP scatter for top 5
        st.subheader("SHAP Value Distribution — Top 5 Features")
        top5 = shap_df.head(5)["feature"].tolist()

        fig, axes_shap = plt.subplots(1, len(top5),
                                       figsize=(3.5 * len(top5), 3.5))
        if len(top5) == 1:
            axes_shap = [axes_shap]

        for ax_s, feat_name in zip(axes_shap, top5):
            fidx = (list(feat_names).index(feat_name)
                    if feat_name in feat_names else None)
            if fidx is not None:
                ax_s.scatter(range(len(shap_vals)), shap_vals[:, fidx],
                             alpha=0.2, s=6, color=C["xgb"])
                ax_s.axhline(0, color="#9e9e9e", lw=0.5, ls="--")
                ax_s.set_xlabel("Sample index", fontsize=8)
                ax_s.set_ylabel("SHAP value", fontsize=8)
                ax_s.set_title(feat_name.replace("_", " "), fontsize=8,
                               fontweight="bold")
                ax_s.tick_params(labelsize=7)
        fig.suptitle("SHAP Values — Top 5 Features", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("SHAP values not available — re-run the pipeline with SHAP installed.")

    st.markdown("---")

    # Sanity check
    st.subheader("Sanity Check — churn_probability Correlation")
    corr_val = a.get("corr_sanity", 0)
    st.metric("Pearson ρ (model score vs churn_probability)", f"{corr_val:.3f}")
    st.markdown("""
    The dataset's pre-computed `churn_probability` has **near-zero**
    correlation with our model scores. This is expected: our proxy captures
    future transactional disengagement, while `churn_probability` measures
    a different construct. Neither variable is used as a training feature.
    """)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §6 MACHINE LEARNING MODELS                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_models():
    a = load_artifacts()
    st.title("§6 — Machine Learning Models")

    eval_df = a["eval_df"]
    y_te = a["y_te"]
    preds_test = a["preds_test"]

    st.markdown("""
    Three algorithms are compared across three feature setups, yielding
    a **3 × 3 model matrix**. All models are evaluated on the held-out
    **test set** (20% of data, never used for any optimisation).

    | Algorithm | Strengths |
    |-----------|----------|
    | **Logistic Regression (LR)** | Interpretable, linear baseline |
    | **Random Forest (RF)** | Non-linear, robust to outliers |
    | **XGBoost (XGB)** | Gradient boosting — typically best accuracy |
    """)

    st.markdown("---")

    # Full metrics table
    st.subheader("Complete Model Evaluation — Test Set")
    fmt = {c: "{:.4f}" for c in eval_df.columns if c not in ("Setup", "Model")}
    st.dataframe(
        eval_df.style.format(fmt).highlight_max(
            subset=["ROC_AUC", "PR_AUC", "Lift@D1"], color="#c8e6c9"
        ).highlight_min(subset=["Brier"], color="#c8e6c9"),
        width="stretch", hide_index=True,
    )

    st.markdown("---")

    # Setup comparison bar chart
    st.subheader("ROC AUC by Setup × Algorithm")
    fig, ax = plt.subplots(figsize=(10, 5))
    setups = ["Full", "Tx-strict", "Cust-only"]
    algos = ["LR", "RF", "XGB"]
    x = np.arange(len(setups))
    w = 0.25

    for i, algo in enumerate(algos):
        vals = []
        for s in setups:
            row = eval_df[(eval_df["Setup"] == s) & (eval_df["Model"] == algo)]
            vals.append(row["ROC_AUC"].values[0] if len(row) else 0)
        bars = ax.bar(x + i * w, vals, w, label=algo,
                      color=ALGO_COLORS[algo], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w)
    ax.set_xticklabels(setups, fontsize=10)
    ax.axhline(0.5, color="#9e9e9e", ls="--", lw=0.8, label="Random")
    ax.set_ylabel("ROC AUC"); ax.set_ylim(0.3, 0.9)
    ax.set_title("Test Set ROC AUC", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Key insight
    full_best = eval_df[eval_df["Setup"] == "Full"]["ROC_AUC"].max()
    tx_best = eval_df[eval_df["Setup"] == "Tx-strict"]["ROC_AUC"].max()
    cu_best = eval_df[eval_df["Setup"] == "Cust-only"]["ROC_AUC"].max()

    st.success(
        f"**Tx-strict** achieves ROC AUC {tx_best:.4f} — only "
        f"{full_best - tx_best:.4f} below Full ({full_best:.4f}). "
        f"Predictive signal comes primarily from transaction-derived features."
    )
    st.error(
        f"**Cust-only** ROC AUC is {cu_best:.4f} — near random. "
        f"Demographics alone carry negligible predictive power."
    )

    st.markdown("---")

    # Diagnostic curves
    st.subheader("Diagnostic Curves — Full Setup")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    base_rate = y_te.mean()

    for algo, clr in ALGO_COLORS.items():
        key = ("Full", algo)
        if key not in preds_test:
            continue
        yp = preds_test[key]
        fpr, tpr, _ = roc_curve(y_te, yp)
        axes[0].plot(fpr, tpr,
                     label=f"{algo} ({roc_auc_score(y_te, yp):.3f})",
                     lw=1.5, color=clr)
        prec_a, rec_a, _ = precision_recall_curve(y_te, yp)
        axes[1].plot(rec_a, prec_a,
                     label=f"{algo} ({average_precision_score(y_te, yp):.3f})",
                     lw=1.5, color=clr)
        pt, pp = calibration_curve(y_te, yp, n_bins=10, strategy="uniform")
        axes[2].plot(pp, pt, "o-", ms=4, label=algo, lw=1.5, color=clr)

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve", fontweight="bold"); axes[0].legend(fontsize=7)

    axes[1].axhline(base_rate, color="gray", ls="--", lw=0.7,
                    label=f"base {base_rate:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall", fontweight="bold")
    axes[1].legend(fontsize=7)

    axes[2].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    axes[2].set_xlabel("Mean predicted prob")
    axes[2].set_ylabel("Fraction positive")
    axes[2].set_title("Calibration", fontweight="bold")
    axes[2].legend(fontsize=7)

    fig.suptitle("Full Setup — Test Set Evaluation", y=1.02, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Confusion matrices
    st.subheader("Confusion Matrices — Full Setup (threshold 0.5)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, algo in zip(axes, ["LR", "RF", "XGB"]):
        key = ("Full", algo)
        if key not in preds_test:
            continue
        yp = preds_test[key]
        cm = confusion_matrix(y_te, (yp >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["Active", "Disengaged"]).plot(
            ax=ax, cmap="Blues")
        ax.set_title(f"Full / {algo}", fontsize=10, fontweight="bold")
    fig.suptitle("Default threshold 0.5 — Test set", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Threshold analysis
    st.subheader("Threshold Optimisation")
    st.markdown("""
    The decision threshold is tuned on the **validation set** by
    maximising F1-score. The test set is **never** used for threshold selection.
    """)

    th_df = a["th_df"]
    best_t = a["best_t"]
    best_f1_val = a["best_f1_val"]
    best_algo = a["best_algo"]
    yp_te = a["preds_test"][("Full", best_algo)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(th_df["threshold"], th_df["f1"], label="F1", lw=1.8, color=C["xgb"])
    ax.plot(th_df["threshold"], th_df["precision"], label="Precision",
            lw=1, ls="--", color=C["rf"])
    ax.plot(th_df["threshold"], th_df["recall"], label="Recall",
            lw=1, ls="--", color=C["disengaged"])
    ax.axvline(best_t, color="black", lw=0.8, ls=":",
               label=f"Best F1 = {best_f1_val:.4f} @ t = {best_t:.2f}")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Threshold Tuning on Validation — Full / {best_algo}",
                 fontweight="bold")
    ax.legend(); fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Test metrics with tuned threshold
    yhat = (yp_te >= best_t).astype(int)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f"{roc_auc_score(y_te, yp_te):.4f}")
    c2.metric("F1 (tuned)", f"{f1_score(y_te, yhat):.4f}")
    c3.metric("Precision", f"{precision_score(y_te, yhat, zero_division=0):.4f}")
    c4.metric("Recall", f"{recall_score(y_te, yhat):.4f}")

    # Interactive threshold explorer
    with st.expander("Interactive Threshold Explorer"):
        t_select = st.slider("Choose threshold", 0.05, 0.95,
                             float(best_t), 0.01)
        yhat_custom = (yp_te >= t_select).astype(int)
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("F1", f"{f1_score(y_te, yhat_custom):.4f}")
        cc2.metric("Precision",
                   f"{precision_score(y_te, yhat_custom, zero_division=0):.4f}")
        cc3.metric("Recall", f"{recall_score(y_te, yhat_custom):.4f}")
        cc4.metric("Flagged",
                   f"{yhat_custom.sum():,} / {len(yhat_custom):,}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §7 PROBABILITY CALIBRATION                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_calibration():
    a = load_artifacts()
    st.title("§7 — Probability Calibration")

    calib_comp = a.get("calib_comparison")
    rr_calib = a.get("rr_calib_test")
    rr_uncalib = a.get("rr_uncalib_test")
    bp_calib = a.get("bp_calib_test")
    bp_uncalib = a.get("bp_uncalib_test")
    y_te = a["y_te"]
    y_growth_te = a.get("y_growth_te")

    if calib_comp is None:
        st.error(
            "Calibration data not found. Re-run the pipeline:\n\n"
            "```\npython3 notebooks/churn_modeling.py\n```"
        )
        st.stop()

    st.markdown("""
    <div style='background:#e8f5e9;padding:1.2em 1.5em;border-left:4px solid #388E3C;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>Why calibration matters:</strong> The Master Signal formula multiplies
    probabilities. If a model predicts 0.7 but the true event rate is only 0.4,
    the Expected Client Value will be systematically overstated. Isotonic
    calibration aligns predicted probabilities with observed frequencies.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    We apply **isotonic calibration** (a non-parametric monotonic fit) to
    both the Redemption Risk and Buy Propensity models, trained on the
    **validation set** (never test). This ensures that:

    $$P(\\text{event} \\mid \\hat{p} = 0.7) \\approx 0.7$$
    """)

    st.markdown("---")

    # Brier score comparison
    st.subheader("Brier Score Comparison")
    st.caption("Brier score measures the mean squared error of probabilistic predictions (lower = better).")
    st.dataframe(
        calib_comp.style.format({
            "Brier_Before": "{:.6f}", "Brier_After": "{:.6f}",
            "Improvement": "{:.6f}",
        }).highlight_max(subset=["Improvement"], color="#c8e6c9"),
        width="stretch", hide_index=True,
    )

    st.markdown("---")

    # Calibration curves — Redemption Risk
    st.subheader("Redemption Risk — Before vs After Calibration")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for yp, label, ls, clr in [
        (rr_uncalib, "Before calibration", "--", C["disengaged"]),
        (rr_calib, "After calibration", "-", C["xgb"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y_te, yp, n_bins=10,
                                                 strategy="uniform")
        axes[0].plot(mean_pred, frac_pos, f"o{ls}", ms=5, lw=1.5,
                     color=clr, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction of positives")
    axes[0].set_title("Calibration Curve", fontweight="bold")
    axes[0].legend(fontsize=8)

    brier_before = brier_score_loss(y_te, rr_uncalib)
    brier_after = brier_score_loss(y_te, rr_calib)
    axes[1].hist(rr_uncalib, bins=50, alpha=0.5, color=C["disengaged"],
                 label=f"Before (Brier={brier_before:.4f})", density=True)
    axes[1].hist(rr_calib, bins=50, alpha=0.5, color=C["xgb"],
                 label=f"After (Brier={brier_after:.4f})", density=True)
    axes[1].set_title("Prediction Distribution", fontweight="bold")
    axes[1].legend(fontsize=8)

    fig.suptitle("Redemption Risk — Calibration Diagnostics", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Calibration curves — Buy Propensity
    if bp_calib is not None and y_growth_te is not None:
        st.subheader("Buy Propensity — Before vs After Calibration")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for yp, label, ls, clr in [
            (bp_uncalib, "Before calibration", "--", C["disengaged"]),
            (bp_calib, "After calibration", "-", C["rf"]),
        ]:
            frac_pos, mean_pred = calibration_curve(y_growth_te, yp, n_bins=10,
                                                     strategy="uniform")
            axes[0].plot(mean_pred, frac_pos, f"o{ls}", ms=5, lw=1.5,
                         color=clr, label=label)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
        axes[0].set_xlabel("Mean predicted probability")
        axes[0].set_ylabel("Fraction of positives")
        axes[0].set_title("Calibration Curve", fontweight="bold")
        axes[0].legend(fontsize=8)

        bp_brier_before = brier_score_loss(y_growth_te, bp_uncalib)
        bp_brier_after = brier_score_loss(y_growth_te, bp_calib)
        axes[1].hist(bp_uncalib, bins=50, alpha=0.5, color=C["disengaged"],
                     label=f"Before (Brier={bp_brier_before:.4f})", density=True)
        axes[1].hist(bp_calib, bins=50, alpha=0.5, color=C["rf"],
                     label=f"After (Brier={bp_brier_after:.4f})", density=True)
        axes[1].set_title("Prediction Distribution", fontweight="bold")
        axes[1].legend(fontsize=8)

        fig.suptitle("Buy Propensity — Calibration Diagnostics", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    ---
    **Interpretation:** After isotonic calibration, predicted probabilities
    closely track the diagonal (perfect calibration). This means the
    Master Signal and Expected Client Value computations in §8–§9 operate
    on **genuine probability estimates**, not arbitrary scores.
    """)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §8 SIGNAL ARCHITECTURE                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_signals():
    a = load_artifacts()
    st.title("§8 — Signal Architecture")

    signals_df = a.get("signals_df")
    if signals_df is None:
        st.error(
            "Signal data not found. Re-run the pipeline:\n\n"
            "```\npython3 notebooks/churn_modeling.py\n```"
        )
        st.stop()

    st.markdown("""
    <div style='background:#f3e5f5;padding:1.2em 1.5em;border-left:4px solid #7B1FA2;
    border-radius:6px;margin-bottom:1.5em'>
    The signal layer transforms raw model outputs into a <strong>three-dimensional
    client profile</strong> — decomposing commercial opportunity into risk,
    propensity, and engagement. This mirrors factor decomposition in
    quantitative portfolio management.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Signal Definitions

    | Signal | Source | Range | Interpretation |
    |--------|--------|-------|----------------|
    | **Redemption Risk** | Calibrated P(disengagement) | [0, 1] | Higher = more likely to leave |
    | **Buy Propensity** | Calibrated P(tx growth > 50%) | [0, 1] | Higher = more likely to grow |
    | **Engagement Score** | Normalised tx behavioural intensity | [0, 1] | Higher = more active relationship |

    ### Master Signal Formula

    $$\\text{MasterSignal} = \\text{BuyPropensity} \\times (1 - \\text{RedemptionRisk}) \\times \\text{EngagementScore}$$

    This multiplicative formulation ensures that **all three dimensions
    must be favourable** for a client to rank highly — a client with high
    buy propensity but high redemption risk will be penalised, analogous
    to risk-adjusted returns in finance.
    """)

    # Signal summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Buy Propensity",
              f"{signals_df['buy_propensity'].mean():.3f}")
    c2.metric("Mean Redemption Risk",
              f"{signals_df['redemption_risk'].mean():.3f}")
    c3.metric("Mean Engagement",
              f"{signals_df['engagement_score'].mean():.3f}")
    c4.metric("Mean Master Signal",
              f"{signals_df['master_signal'].mean():.3f}")

    st.markdown("---")

    # Individual signal distributions
    st.subheader("Signal Distributions")
    cols = st.columns(3)
    sig_info = [
        ("buy_propensity", "Buy Propensity", "#1976D2"),
        ("engagement_score", "Engagement Score", "#388E3C"),
        ("redemption_risk", "Redemption Risk", "#D32F2F"),
    ]
    for col, (feat, label, color) in zip(cols, sig_info):
        with col:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(signals_df[feat], bins=40, color=color, alpha=0.85,
                    edgecolor="white")
            ax.set_title(label, fontweight="bold", fontsize=10)
            ax.set_ylabel("Count")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # Master Signal distribution
    st.subheader("Master Signal Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(signals_df["master_signal"], bins=50, color=C["primary"],
            alpha=0.85, edgecolor="white")
    ax.set_xlabel("Master Signal (Commercial Opportunity)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Commercial Opportunity Score",
                 fontweight="bold")
    ax.axvline(signals_df["master_signal"].mean(), color="black", ls="--",
               lw=1, label=f"Mean = {signals_df['master_signal'].mean():.3f}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Opportunity Quadrant
    st.subheader("Client Opportunity Map")
    st.markdown("""
    Each point represents a client. The x-axis is **Buy Propensity**,
    the y-axis is **Redemption Risk**, and color encodes **Engagement**.
    The recommended action zones are delineated by dashed lines.
    """)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        signals_df["buy_propensity"],
        signals_df["redemption_risk"],
        c=signals_df["engagement_score"],
        cmap="RdYlGn", alpha=0.4, s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Engagement Score")
    ax.set_xlabel("Buy Propensity")
    ax.set_ylabel("Redemption Risk")
    ax.set_title("Client Opportunity Map", fontweight="bold")
    ax.axhline(0.3, color="#9e9e9e", ls="--", lw=0.7)
    ax.axhline(0.6, color="#9e9e9e", ls="--", lw=0.7)
    ax.axvline(0.6, color="#9e9e9e", ls="--", lw=0.7)
    ax.text(0.8, 0.15, "UPSELL", fontsize=11, fontweight="bold",
            color="#388E3C", ha="center", alpha=0.7)
    ax.text(0.3, 0.8, "RETENTION", fontsize=11, fontweight="bold",
            color="#D32F2F", ha="center", alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Signal correlations
    st.subheader("Signal Correlation Matrix")
    _corr_cols = ["buy_propensity", "redemption_risk",
                  "engagement_score", "master_signal"]
    has_ecv = "expected_client_value" in signals_df.columns
    if has_ecv:
        _corr_cols.append("expected_client_value")
    corr_mat = signals_df[_corr_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Buy Prop.", "Redemp. Risk", "Engagement", "Master"]
    if has_ecv:
        labels.append("Exp. Value")
    n_labels = len(labels)
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(j, i, f"{corr_mat.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson ρ")
    ax.set_title("Signal Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Recommended actions
    st.markdown("---")
    st.subheader("Action Distribution")
    action_counts = signals_df["recommended_action"].value_counts()

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("""
        | Action | Rule |
        |--------|------|
        | **Upsell** | BP > 0.6 AND RR < 0.3 |
        | **Retention** | RR > 0.6 |
        | **Monitor** | All other clients |
        """)
        st.dataframe(action_counts.to_frame("Count"), width="stretch")
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_act = {"Upsell": "#388E3C", "Retention": "#D32F2F",
                      "Monitor": "#FFA000"}
        bars = ax.bar(
            action_counts.index, action_counts.values,
            color=[colors_act.get(a, "#999") for a in action_counts.index],
            alpha=0.85,
        )
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 50,
                    f"{bar.get_height():,}", ha="center", va="bottom",
                    fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Recommended Action Distribution", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §9 EXPECTED CLIENT VALUE                                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_ecv():
    a = load_artifacts()
    st.title("§9 — Expected Client Value")

    signals_df = a.get("signals_df")
    if signals_df is None:
        st.error("Signal data not found. Re-run the pipeline.")
        st.stop()

    has_ecv = "expected_client_value" in signals_df.columns

    st.markdown("""
    <div style='background:#e3f2fd;padding:1.2em 1.5em;border-left:4px solid #1976D2;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>Expected Client Value (ECV)</strong> translates the abstract
    Master Signal into a monetary estimate:
    <br><br>
    <code>ECV = MasterSignal × tx_total</code>
    <br><br>
    where <code>tx_total</code> is the client's total transaction volume during
    the observation window. This monetises the opportunity score, giving
    relationship managers a dollar-denominated ranking.
    </div>
    """, unsafe_allow_html=True)

    if not has_ecv:
        st.warning("Expected Client Value not computed. "
                   "Re-run the pipeline with tx_total available.")
        st.stop()

    # Key metrics
    ecv = signals_df["expected_client_value"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median ECV", f"{ecv.median():,.0f}")
    c2.metric("Mean ECV", f"{ecv.mean():,.0f}")
    c3.metric("Top 1% Threshold", f"{ecv.quantile(0.99):,.0f}")
    c4.metric("Total Scored Clients", f"{len(ecv):,}")

    st.markdown("---")

    # ECV distribution
    st.subheader("ECV Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ecv_clip = ecv.clip(upper=ecv.quantile(0.99))
    ax.hist(ecv_clip, bins=60, color=C["primary"], alpha=0.85,
            edgecolor="white")
    ax.set_xlabel("Expected Client Value (clipped at 99th percentile)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Expected Client Value", fontweight="bold")
    ax.axvline(ecv.median(), color="black", ls="--", lw=1,
               label=f"Median = {ecv.median():,.0f}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ECV vs Redemption Risk scatter
    st.subheader("ECV vs Redemption Risk")
    st.markdown("""
    This scatter reveals the **risk–value tradeoff**:
    - **Upper-left:** High-value, low-risk clients → prime upsell targets
    - **Upper-right:** High-value, high-risk clients → urgent retention
    - **Lower band:** Lower-value clients → monitor
    """)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_ev = ax.scatter(
        signals_df["redemption_risk"],
        signals_df["expected_client_value"].clip(
            upper=signals_df["expected_client_value"].quantile(0.99)),
        c=signals_df["buy_propensity"],
        cmap="YlOrRd", alpha=0.4, s=8,
    )
    plt.colorbar(scatter_ev, ax=ax, label="Buy Propensity")
    ax.set_xlabel("Redemption Risk")
    ax.set_ylabel("Expected Client Value")
    ax.set_title("ECV vs Redemption Risk (coloured by Buy Propensity)",
                 fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §10 OPPORTUNITY FRONTIER                                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_frontier():
    a = load_artifacts()
    st.title("§10 — Opportunity Frontier")

    signals_df = a.get("signals_df")
    if signals_df is None or "opportunity_frontier_score" not in signals_df.columns:
        st.error("Opportunity Frontier data not found. Re-run the pipeline.")
        st.stop()

    has_ecv = "expected_client_value" in signals_df.columns

    st.markdown("""
    <div style='background:#fce4ec;padding:1.2em 1.5em;border-left:4px solid #D32F2F;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>The Opportunity Frontier Score</strong> is a risk-adjusted metric
    inspired by portfolio theory. Just as the Sharpe ratio divides excess
    return by volatility, the Frontier Score divides Expected Client Value
    by Redemption Risk:
    <br><br>
    <code>FrontierScore = normalise(ECV / RedemptionRisk)</code>
    <br><br>
    Clients with the highest risk-adjusted opportunity — high value,
    low risk — rank at the top.
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Frontier Score",
              f"{signals_df['opportunity_frontier_score'].mean():.3f}")
    c2.metric("Median Frontier Score",
              f"{signals_df['opportunity_frontier_score'].median():.3f}")
    if has_ecv:
        c3.metric("Median ECV",
                  f"{signals_df['expected_client_value'].median():,.0f}")
    else:
        c3.metric("Clients", f"{len(signals_df):,}")
    c4.metric("Mean Redemption Risk",
              f"{signals_df['redemption_risk'].mean():.3f}")

    st.markdown("---")

    # Opportunity Frontier scatter
    st.subheader("Opportunity Frontier — Risk vs Propensity")
    st.markdown("""
    **X-axis:** Redemption Risk · **Y-axis:** Buy Propensity ·
    **Colour & size:** Expected Client Value

    Ideal clients occupy the **upper-left** quadrant: high growth
    propensity with low disengagement risk.
    """)

    fig, ax = plt.subplots(figsize=(10, 7))
    _ecv = (signals_df["expected_client_value"] if has_ecv
            else signals_df["master_signal"])
    _ecv_clip = _ecv.clip(upper=_ecv.quantile(0.99))
    _sizes = np.clip(_ecv_clip / max(_ecv_clip.max(), 1e-9) * 60, 2, 60)

    sc = ax.scatter(
        signals_df["redemption_risk"],
        signals_df["buy_propensity"],
        c=_ecv_clip, s=_sizes,
        cmap="plasma", alpha=0.45,
    )
    plt.colorbar(sc, ax=ax, label="Expected Client Value")
    ax.set_xlabel("Redemption Risk")
    ax.set_ylabel("Buy Propensity")
    ax.set_title(
        "Opportunity Frontier\nHigh propensity + low risk = best opportunity",
        fontweight="bold",
    )
    ax.axhline(0.6, color="#9e9e9e", ls="--", lw=0.7)
    ax.axvline(0.3, color="#9e9e9e", ls="--", lw=0.7)
    ax.text(0.15, 0.8, "HIGH OPP\nLOW RISK", fontsize=9, fontweight="bold",
            color="#388E3C", ha="center", alpha=0.7)
    ax.text(0.8, 0.8, "HIGH OPP\nHIGH RISK", fontsize=9, fontweight="bold",
            color="#FFA000", ha="center", alpha=0.7)
    ax.text(0.5, 0.15, "LOW OPPORTUNITY", fontsize=9, fontweight="bold",
            color="#9e9e9e", ha="center", alpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Frontier Score distribution
    st.subheader("Frontier Score Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(signals_df["opportunity_frontier_score"], bins=50,
            color=C["accent"], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Opportunity Frontier Score (normalised)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Opportunity Frontier Score",
                 fontweight="bold")
    ax.axvline(signals_df["opportunity_frontier_score"].mean(),
               color="black", ls="--", lw=1,
               label=f"Mean = {signals_df['opportunity_frontier_score'].mean():.3f}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Ranking comparison
    st.subheader("Ranking Divergence: Frontier vs Master Signal")
    st.markdown("""
    The Opportunity Frontier Score prioritises **risk-adjusted opportunity**
    (high value, low risk), while the Master Signal prioritises **absolute
    opportunity** regardless of risk level. Clients appearing in the
    Frontier top-100 but not the Master top-100 are **hidden gems** —
    moderate absolute value but very low risk.
    """)

    top_master = set(signals_df.nlargest(100, "master_signal")["customer_id"])
    top_front = set(
        signals_df.nlargest(100, "opportunity_frontier_score")["customer_id"]
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Overlap (both top 100)", f"{len(top_master & top_front)}")
    c2.metric("Frontier Only", f"{len(top_front - top_master)}")
    c3.metric("Master Only", f"{len(top_master - top_front)}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §11 SALES INTELLIGENCE                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_sales():
    a = load_artifacts()
    st.title("§11 — Sales Intelligence")

    signals_df = a.get("signals_df")
    top_opportunities = a.get("top_opportunities")

    if signals_df is None:
        st.error("Signal data not found. Re-run the pipeline.")
        st.stop()

    has_ecv = "expected_client_value" in signals_df.columns
    has_frontier = "opportunity_frontier_score" in signals_df.columns

    st.markdown("""
    <div style='background:#e8eaf6;padding:1.2em 1.5em;border-left:4px solid #3F51B5;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>Actionable output.</strong> This section consolidates all
    analytical results into client-level rankings and recommended actions
    that relationship managers can execute immediately.
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Upsell Candidates",
              f"{(signals_df['recommended_action'] == 'Upsell').sum():,}")
    c2.metric("Retention Targets",
              f"{(signals_df['recommended_action'] == 'Retention').sum():,}")
    c3.metric("Monitor",
              f"{(signals_df['recommended_action'] == 'Monitor').sum():,}")
    if has_ecv:
        c4.metric("Total ECV (top 100)",
                  f"{signals_df.nlargest(100, 'expected_client_value')['expected_client_value'].sum():,.0f}")

    st.markdown("---")

    # Top 100 by Expected Client Value
    st.subheader("Top 100 Clients by Expected Client Value")
    st.caption(
        "Ranked by monetised commercial opportunity. Each row includes "
        "the three underlying signals and the recommended commercial action."
    )

    if top_opportunities is not None and len(top_opportunities) > 0:
        _fmt = {
            "buy_propensity": "{:.4f}",
            "redemption_risk": "{:.4f}",
            "engagement_score": "{:.4f}",
            "master_signal": "{:.4f}",
        }
        _grad_col = ["master_signal"]
        if has_ecv and "expected_client_value" in top_opportunities.columns:
            _fmt["expected_client_value"] = "{:,.0f}"
            _grad_col = ["expected_client_value"]
        if has_frontier and "opportunity_frontier_score" in top_opportunities.columns:
            _fmt["opportunity_frontier_score"] = "{:.4f}"

        st.dataframe(
            top_opportunities.style.format(_fmt)
                .background_gradient(subset=_grad_col, cmap="Greens"),
            width="stretch", hide_index=True,
        )
    else:
        # Fallback: build from signals_df
        _sort_col = "expected_client_value" if has_ecv else "master_signal"
        top100 = signals_df.nlargest(100, _sort_col).reset_index(drop=True)
        _fmt = {
            "buy_propensity": "{:.4f}",
            "redemption_risk": "{:.4f}",
            "engagement_score": "{:.4f}",
            "master_signal": "{:.4f}",
        }
        if has_ecv:
            _fmt["expected_client_value"] = "{:,.0f}"
        st.dataframe(
            top100.style.format(_fmt).background_gradient(
                subset=[_sort_col], cmap="Greens"),
            width="stretch", hide_index=True,
        )

    st.markdown("---")

    # Top 100 by Frontier Score (if available)
    if has_frontier:
        st.subheader("Top 100 Clients by Opportunity Frontier Score")
        st.caption(
            "Risk-adjusted ranking — clients with the best "
            "value-to-risk ratio. May surface opportunities missed by "
            "the absolute-value ranking above."
        )

        top_frontier = signals_df.sort_values(
            "opportunity_frontier_score", ascending=False
        ).head(100).reset_index(drop=True)

        _fmt_f = {
            "buy_propensity": "{:.4f}",
            "redemption_risk": "{:.4f}",
            "engagement_score": "{:.4f}",
            "master_signal": "{:.4f}",
            "opportunity_frontier_score": "{:.4f}",
        }
        if has_ecv:
            _fmt_f["expected_client_value"] = "{:,.0f}"

        display_cols = ["customer_id", "opportunity_frontier_score",
                        "buy_propensity", "redemption_risk",
                        "engagement_score", "master_signal",
                        "recommended_action"]
        if has_ecv:
            display_cols.insert(2, "expected_client_value")
        display_cols = [c for c in display_cols if c in top_frontier.columns]

        st.dataframe(
            top_frontier[display_cols].style.format(_fmt_f)
                .background_gradient(
                    subset=["opportunity_frontier_score"], cmap="Purples"),
            width="stretch", hide_index=True,
        )

        st.markdown("---")

    # Action playbook
    st.subheader("Commercial Action Playbook")
    st.markdown("""
    | Action | Criteria | Recommended Response |
    |--------|---------|---------------------|
    | **Upsell** | Buy Propensity > 0.6, Redemption Risk < 0.3 | Propose premium products, increase wallet share |
    | **Retention** | Redemption Risk > 0.6 | Proactive outreach, loyalty incentives, service review |
    | **Monitor** | All other clients | Standard service, periodic check-in |

    ---

    **Next steps for Pictet:**
    1. Integrate client scores into the CRM for automated routing
    2. Run A/B test: Frontier-ranked vs status-quo prioritisation
    3. Refresh model quarterly to capture behavioural drift
    4. Extend signal layer with AUM-weighted Expected Client Value
    """)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  §12 DISTRIBUTION INTELLIGENCE & STRATEGIC INTERPRETATION              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def section_distribution_intelligence():
    a = load_artifacts()
    st.title("§12 — Distribution Intelligence & Strategic Interpretation")

    signals_df = a.get("signals_df")
    full_df = a.get("full_df")

    if signals_df is None:
        st.error("Signal data not found. Re-run the pipeline.")
        st.stop()

    has_ecv = "expected_client_value" in signals_df.columns
    has_frontier = "opportunity_frontier_score" in signals_df.columns

    # ─── Part 1: Commercial Interpretation ─────────────────────────────
    st.markdown("""
    <div style='background:#e8eaf6;padding:1.2em 1.5em;border-left:4px solid #3F51B5;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>From prediction to distribution strategy.</strong> This section
    bridges the gap between statistical modelling and asset management
    distribution analytics — interpreting every signal through a
    commercial lens and sketching the path toward a full
    client × product intelligence platform.
    </div>
    """, unsafe_allow_html=True)

    st.header("Part 1 — Commercial Interpretation of the Signals")

    st.markdown("""
    The AlphaSignal system approximates **expected commercial opportunity**
    using a multiplicative decomposition:

    $$\\text{CommercialOpportunity} \\approx P(\\text{growth}) \\times \\text{Engagement} \\times (1 - P(\\text{disengagement}))$$

    Each component carries a distinct business meaning:

    | Component | Signal | Commercial Interpretation |
    |-----------|--------|---------------------------|
    | $P(\\text{growth})$ | **Buy Propensity** | Probability the client will increase transactional activity — a proxy for wallet-share expansion |
    | Engagement | **Engagement Score** | Intensity and consistency of the existing relationship — behavioural depth of the client |
    | $1 - P(\\text{disengagement})$ | **1 − Redemption Risk** | Probability the client remains active — the retention dimension |

    The **Master Signal** multiplies these three dimensions so that
    **all three must be favourable** for a client to rank highly.
    A client with strong growth propensity but high disengagement risk
    will be penalised — exactly as a high-return but high-volatility
    asset would be penalised in a risk-adjusted portfolio framework.

    This formulation mirrors **expected-value scoring** used in
    institutional client intelligence systems across asset management
    distribution. It identifies clients where:

    - growth probability is **high**
    - engagement is **strong**
    - disengagement risk is **manageable**
    """)

    st.markdown("---")

    # ─── Part 2: Client Drilldown ──────────────────────────────────────
    st.header("Part 2 — Client Drilldown")
    st.markdown("""
    Select a client to inspect their full signal profile and receive
    an automated commercial interpretation — the kind of view a
    relationship manager would consult before a client meeting.
    """)

    # Build a merged view once
    _drill_df = signals_df.copy()

    # Attach key behavioural features from full_df if available
    _beh_cols = ["tx_count", "tx_total", "tx_recency", "tx_max_gap",
                 "tx_late_ratio", "tx_wknd_ratio"]
    if full_df is not None:
        _avail = [c for c in _beh_cols if c in full_df.columns]
        if _avail:
            _beh = full_df[["customer_id"] + _avail].copy()
            _drill_df = _drill_df.merge(_beh, on="customer_id", how="left")

    # Client selector
    client_ids = _drill_df["customer_id"].astype(str).tolist()
    selected_id = st.selectbox(
        "Select a Client ID",
        options=client_ids,
        index=0,
        help="Type to search by client ID",
    )

    row = _drill_df[_drill_df["customer_id"].astype(str) == selected_id]
    if row.empty:
        st.warning("Client not found.")
    else:
        row = row.iloc[0]

        # Signal metrics
        st.subheader(f"Signal Profile — Client {selected_id}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Buy Propensity", f"{row['buy_propensity']:.4f}")
        c2.metric("Redemption Risk", f"{row['redemption_risk']:.4f}")
        c3.metric("Engagement Score", f"{row['engagement_score']:.4f}")
        c4.metric("Master Signal", f"{row['master_signal']:.4f}")

        c5, c6, c7 = st.columns(3)
        if has_ecv:
            c5.metric("Expected Client Value",
                      f"{row['expected_client_value']:,.0f}")
        if has_frontier:
            c6.metric("Opportunity Frontier Score",
                      f"{row['opportunity_frontier_score']:.4f}")
        c7.metric("Recommended Action", row.get("recommended_action", "—"))

        # Behavioural features
        _avail_beh = [c for c in _beh_cols if c in row.index and pd.notna(row[c])]
        if _avail_beh:
            st.subheader("Behavioural Features")
            beh_cols = st.columns(min(len(_avail_beh), 4))
            for i, feat in enumerate(_avail_beh):
                label = feat.replace("_", " ").title()
                val = row[feat]
                fmt_val = f"{val:,.0f}" if val > 100 else f"{val:.3f}"
                beh_cols[i % len(beh_cols)].metric(label, fmt_val)

        # Automated interpretation
        st.subheader("Commercial Interpretation")
        bp = row["buy_propensity"]
        rr = row["redemption_risk"]
        eng = row["engagement_score"]

        if bp > 0.6 and rr < 0.3:
            st.success(
                f"**Strong upsell opportunity.** This client shows high "
                f"growth propensity ({bp:.2f}) with low disengagement risk "
                f"({rr:.2f}). Engagement is {'strong' if eng > 0.5 else 'moderate'} "
                f"({eng:.2f}). **Recommendation:** propose premium products, "
                f"increase wallet share, and schedule a proactive review meeting."
            )
        elif rr > 0.6:
            st.error(
                f"**Retention priority.** This client has elevated "
                f"disengagement risk ({rr:.2f}). Buy propensity is "
                f"{'still positive' if bp > 0.4 else 'low'} ({bp:.2f}) and "
                f"engagement is {'weakening' if eng < 0.3 else 'moderate'} "
                f"({eng:.2f}). **Recommendation:** proactive outreach, "
                f"loyalty incentives, and a service-quality review."
            )
        elif bp > 0.4 and rr < 0.5:
            st.info(
                f"**Emerging opportunity.** Growth propensity is moderate "
                f"({bp:.2f}) with acceptable risk ({rr:.2f}). "
                f"**Recommendation:** nurture the relationship with targeted "
                f"content and periodic check-ins to move the client toward "
                f"the upsell zone."
            )
        else:
            st.warning(
                f"**Monitor.** Signals are mixed — buy propensity {bp:.2f}, "
                f"redemption risk {rr:.2f}, engagement {eng:.2f}. "
                f"**Recommendation:** maintain standard service and revisit "
                f"at the next model refresh."
            )

    st.markdown("---")

    # ─── Part 3: Product Recommendation Signal ─────────────────────────
    st.header("Part 3 — Product Opportunity Signal")

    st.markdown("""
    Distribution analytics in asset management goes beyond *which clients*
    are promising — it also asks **which products** should be proposed
    to each client.

    The **Product Opportunity Signal** extends the system from
    *client intelligence* to **client × product intelligence** by
    estimating the likelihood that a client could adopt a new product
    category based on behavioural similarity with existing adopters.

    ### Conceptual Framework

    This approach parallels **collaborative filtering** in recommender
    systems:

    > *Clients with similar behavioural patterns tend to adopt similar
    > financial products.*

    The signal can be estimated from four dimensions:

    | Dimension | Description |
    |-----------|------------|
    | **Behavioural similarity** | Proximity in transaction-behaviour space (RFM, type mix, trends) |
    | **Historical adoption** | Product categories the client already uses |
    | **Engagement level** | Depth of relationship — engaged clients adopt more readily |
    | **Growth propensity** | High-propensity clients are more receptive to new offerings |

    In practice, a product-level opportunity score would be computed for
    every **client × product** pair, enabling the system to rank
    opportunities not only by client but by **client × product
    combination**.

    This aligns directly with the **Product Distribution Signals**
    objective of the Pictet EPFL research project.
    """)

    # Generate a synthetic product opportunity signal for demonstration
    # Uses engagement × buy_propensity with product-category noise to
    # illustrate the concept without requiring product-level data.
    np.random.seed(42)
    _n = len(signals_df)
    _product_cats = ["Equity Funds", "Fixed Income", "Multi-Asset",
                     "Private Equity", "Structured Products"]

    _base_product_score = (
        signals_df["engagement_score"].values * 0.4
        + signals_df["buy_propensity"].values * 0.4
        + (1 - signals_df["redemption_risk"].values) * 0.2
    )
    _product_noise = np.random.normal(0, 0.08, size=_n)
    _product_opp = np.clip(_base_product_score + _product_noise, 0, 1)

    st.markdown("---")

    # ─── Part 4: Distribution Intelligence Visualisation ───────────────
    st.header("Part 4 — Distribution Intelligence Visualisation")

    st.markdown("""
    The chart below illustrates how a **client × product** distribution
    matrix would look in practice. Each point represents a potential
    opportunity: the x-axis is the client-level commercial opportunity
    score and the y-axis is the simulated product adoption propensity.

    The **upper-right quadrant** identifies the most promising
    distribution opportunities — clients who are both commercially
    attractive and likely to adopt a given product.
    """)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Assign a random product category per client for illustration
    _cat_assign = np.random.choice(_product_cats, size=_n)
    _cat_colors = {
        "Equity Funds": "#1976D2",
        "Fixed Income": "#388E3C",
        "Multi-Asset": "#FFA000",
        "Private Equity": "#7B1FA2",
        "Structured Products": "#D32F2F",
    }

    for cat in _product_cats:
        mask = _cat_assign == cat
        ax.scatter(
            signals_df["master_signal"].values[mask],
            _product_opp[mask],
            s=10, alpha=0.35, label=cat,
            color=_cat_colors[cat],
        )

    ax.set_xlabel("Client Opportunity Score (Master Signal)")
    ax.set_ylabel("Product Adoption Propensity (simulated)")
    ax.set_title(
        "Client × Product Distribution Intelligence\n"
        "Upper-right = highest-priority distribution opportunities",
        fontweight="bold",
    )
    ax.axhline(0.5, color="#9e9e9e", ls="--", lw=0.7)
    ax.axvline(
        signals_df["master_signal"].median(),
        color="#9e9e9e", ls="--", lw=0.7,
    )
    ax.legend(fontsize=8, title="Product Category", title_fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Note: Product adoption propensity is simulated for demonstration "
        "purposes. In a production system this would be derived from "
        "actual product-level holdings and flow data."
    )

    st.markdown("---")

    # ─── Part 4b: Strategic Quadrant — Value vs Risk ───────────────────
    st.header("Strategic Client Segmentation — Value vs Risk")

    st.markdown("""
    The chart below maps every client along two dimensions that drive
    commercial resource allocation in asset management distribution:
    **Expected Client Value** (vertical) and **Redemption Risk**
    (horizontal), coloured by **Buy Propensity**.
    """)

    if has_ecv:
        _ecv_plot = signals_df["expected_client_value"].clip(
            upper=signals_df["expected_client_value"].quantile(0.99))

        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(
            signals_df["redemption_risk"],
            _ecv_plot,
            c=signals_df["buy_propensity"],
            cmap="RdYlGn", alpha=0.40, s=10,
        )
        plt.colorbar(sc, ax=ax, label="Buy Propensity")

        # Quadrant lines
        _rr_mid = 0.5
        _ecv_mid = _ecv_plot.median()
        ax.axvline(_rr_mid, color="#9e9e9e", ls="--", lw=0.8)
        ax.axhline(_ecv_mid, color="#9e9e9e", ls="--", lw=0.8)

        # Quadrant labels
        _y_top = _ecv_plot.quantile(0.92)
        _y_bot = _ecv_plot.quantile(0.08)
        ax.text(0.25, _y_top, "STRATEGIC UPSELL\nHigh Value / Low Risk",
                fontsize=9, fontweight="bold", color="#388E3C",
                ha="center", alpha=0.75)
        ax.text(0.75, _y_top, "RETENTION PRIORITY\nHigh Value / High Risk",
                fontsize=9, fontweight="bold", color="#D32F2F",
                ha="center", alpha=0.75)
        ax.text(0.25, _y_bot, "STABLE / LOW PRIORITY\nLow Value / Low Risk",
                fontsize=9, fontweight="bold", color="#757575",
                ha="center", alpha=0.65)
        ax.text(0.75, _y_bot, "MONITOR\nLow Value / High Risk",
                fontsize=9, fontweight="bold", color="#9E9E9E",
                ha="center", alpha=0.65)

        ax.set_xlabel("Redemption Risk")
        ax.set_ylabel("Expected Client Value (clipped at 99th pctl)")
        ax.set_title(
            "Strategic Client Segmentation\n"
            "Allocating commercial effort across the client base",
            fontweight="bold",
        )
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Each point is a client. Colour intensity reflects Buy Propensity "
            "(green = high growth likelihood). The quadrant framework helps "
            "relationship managers prioritise effort by combining value and risk."
        )
    else:
        st.info("Expected Client Value not available — chart requires ECV.")

    st.markdown("---")

    # ─── Distribution Insight box ──────────────────────────────────────
    st.header("Distribution Insight")

    st.markdown("""
    <div style='background:#e0f2f1;padding:1.2em 1.5em;border-left:4px solid #00897B;
    border-radius:6px;margin-bottom:1.5em'>
    <strong>How AlphaSignal supports distribution strategy.</strong>
    <br><br>
    Asset management distribution is not only about identifying valuable
    clients — it is about <em>allocating relationship management effort
    efficiently</em> across a large, heterogeneous client base.
    <br><br>
    AlphaSignal decomposes commercial opportunity into three behavioural
    dimensions:<br>
    &nbsp;&nbsp;• <strong>Growth potential</strong> — Buy Propensity<br>
    &nbsp;&nbsp;• <strong>Relationship engagement</strong> — Engagement Score<br>
    &nbsp;&nbsp;• <strong>Disengagement risk</strong> — Redemption Risk<br>
    <br>
    Combining these signals with <strong>Expected Client Value</strong>
    (monetised opportunity) and the <strong>Opportunity Frontier Score</strong>
    (risk-adjusted ranking) produces a quantitative, risk-aware
    commercial prioritisation of every client.
    <br><br>
    This architecture mirrors the expected-value and risk-adjustment
    frameworks used by institutional distribution teams to optimise
    coverage models, allocate RM bandwidth, and maximise commercial
    outcomes per unit of relationship effort.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ─── Part 5: Limitations ───────────────────────────────────────────
    st.header("Part 5 — Limitations")

    st.markdown("""
    Transparent acknowledgement of limitations is essential for any
    research prototype presented to institutional stakeholders.

    | Limitation | Detail |
    |-----------|--------|
    | **Behavioural proxy target** | The disengagement target is defined from future transactional activity, not explicit account closures or asset outflows. The model predicts *transactional disengagement*, not contractual churn. |
    | **No product-level data** | The current dataset contains client-level transactions but not product holdings or fund-level flows. The Product Opportunity Signal above is therefore conceptual. |
    | **ECV based on transaction volume** | Expected Client Value is monetised using transaction volume (tx_total) rather than assets under management. In a real system, AUM and fee structures would provide a more accurate proxy. |
    | **Single-year observation** | All features are derived from 2023 data. Multi-year longitudinal data would strengthen trend signals and enable regime-change detection. |
    | **No causal identification** | The models capture correlations, not causal effects. An intervention (e.g. retention campaign) might change client behaviour in ways the model cannot predict without experimental data. |
    """)

    st.markdown("---")

    # ─── Part 6: Research Roadmap ──────────────────────────────────────
    st.header("Part 6 — Research Roadmap")

    st.markdown("""
    The current prototype demonstrates the **architecture** of a client
    intelligence system for asset management distribution. The roadmap
    below outlines how this architecture evolves into a full
    **distribution intelligence platform**.
    """)

    st.markdown("""
    ```
    ┌──────────────────────────────────┐
    │  CURRENT PROTOTYPE               │
    │  Client Intelligence             │
    │  (AlphaSignal v1)                │
    │                                  │
    │  • Disengagement prediction      │
    │  • Buy propensity estimation     │
    │  • Client-level scoring          │
    │  • Risk-adjusted opportunity     │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  PHASE 2                         │
    │  Client × Product Intelligence   │
    │                                  │
    │  • Product-level flow data       │
    │  • Collaborative filtering       │
    │  • Client × product scoring      │
    │  • Distribution opportunity map  │
    └──────────────┬───────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  PHASE 3                         │
    │  Full Distribution Intelligence  │
    │                                  │
    │  • AUM-weighted ECV              │
    │  • Macro / market signal layer   │
    │  • Periodic model refresh        │
    │  • CRM integration & A/B testing │
    │  • Regime-change detection       │
    └──────────────────────────────────┘
    ```
    """)

    st.markdown("""
    | Phase | Extension | Impact |
    |-------|----------|--------|
    | **2a** | Integrate product-level flows (subscriptions, redemptions by fund) | Enables client × product recommendation models |
    | **2b** | Build collaborative-filtering product propensity model | Ranks opportunities by client × product pair |
    | **3a** | Replace tx_total with AUM and fee schedules | ECV reflects true revenue potential |
    | **3b** | Incorporate macro and market signals (rates, flows, sentiment) | Captures regime-dependent behaviour |
    | **3c** | Periodic model refresh with drift monitoring | Maintains predictive accuracy over time |
    | **3d** | CRM integration with automated action routing | Closes the loop from insight to execution |
    """)

    # ─── Final System Architecture ──────────────────────────────────────
    st.header("AlphaSignal — System Architecture")

    st.markdown("""
    The diagram below summarises the complete analytical pipeline — from
    raw data to commercial decision support.
    """)

    st.markdown("""
    ```
    ┌───────────────────────────────────────────────────────────────────┐
    │                        RAW DATA                                  │
    │        48,723 customers  ·  3.2 M transactions  ·  2023          │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │              BEHAVIOURAL FEATURE ENGINEERING                      │
    │     RFM aggregates · trends · type shares · weekend ratio        │
    │     Jan – Sep 2023 (strict temporal barrier)                     │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                     SIGNAL LAYER                                  │
    │                                                                   │
    │   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
    │   │ Redemption Risk  │  │ Buy Propensity   │  │   Engagement    │  │
    │   │ P(disengage)     │  │ P(tx growth)     │  │   Score         │  │
    │   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
    │            │                    │                     │           │
    │            └────────────────────┼─────────────────────┘           │
    │                                │                                  │
    │                                ▼                                  │
    │              MASTER COMMERCIAL SIGNAL                             │
    │         BP × (1 − RR) × Engagement                               │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │              COMMERCIAL SCORING                                   │
    │                                                                   │
    │   Expected Client Value          Opportunity Frontier Score       │
    │   Master × tx_total              ECV / RedemptionRisk             │
    │   (monetised opportunity)        (risk-adjusted ranking)          │
    └──────────────────────────┬────────────────────────────────────────┘
                               │
                               ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │         SALES & DISTRIBUTION INTELLIGENCE                        │
    │                                                                   │
    │   Client rankings · Recommended actions · Client drilldown       │
    │   Strategic segmentation · Product opportunity signal             │
    └───────────────────────────────────────────────────────────────────┘
    ```
    """)

    st.info(
        "**Positioning:** AlphaSignal is a research prototype that "
        "demonstrates the complete architecture of a **client intelligence "
        "and distribution analytics platform** for asset management. "
        "It combines predictive modelling, probability calibration, "
        "signal architecture, expected-value scoring, risk-adjusted "
        "opportunity ranking, and commercial interpretation — and shows "
        "how client intelligence naturally extends to **client × product "
        "distribution intelligence**. This positioning aligns directly "
        "with the research objectives of the Pictet EPFL project."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

PAGES = {
    "§1  Executive Summary": section_executive_summary,
    "§2  Problem Definition": section_problem_definition,
    "§3  Dataset Description": section_dataset,
    "§4  Temporal Modeling Strategy": section_temporal,
    "§5  Feature Engineering & Explainability": section_features,
    "§6  Machine Learning Models": section_models,
    "§7  Probability Calibration": section_calibration,
    "§8  Signal Architecture": section_signals,
    "§9  Expected Client Value": section_ecv,
    "§10 Opportunity Frontier": section_frontier,
    "§11 Sales Intelligence": section_sales,
    "§12 Distribution Intelligence": section_distribution_intelligence,
}

PAGES[section]()
