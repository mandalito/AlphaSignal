"""
app.py
======
Streamlit dashboard for the COFINFAD Future Disengagement prediction pipeline.

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

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_PATH = os.path.join(ROOT, "outputs", "churn_artifacts.pkl")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="COFINFAD — Disengagement Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
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
}

ALGO_COLORS = {"LR": C["lr"], "RF": C["rf"], "XGB": C["xgb"]}
SETUP_COLORS = {"Full": C["full"], "Tx-strict": C["tx"], "Cust-only": C["cust"]}


# ---------------------------------------------------------------------------
# Artifact loading (cached)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/bank-building.png",
    width=64,
)
st.sidebar.title("COFINFAD")
st.sidebar.caption("Future Disengagement Prediction")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Data Explorer",
        "🎯 Model Performance",
        "⚖️ Setup Comparison",
        "🎚️ Threshold Analysis",
        "🧠 Explainability",
        "📡 Master Signal",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**COFINFAD** — Colombian Fintech Financial Analytics Dataset. "
    "Target is a **behavioural proxy** for transactional disengagement, "
    "not contractual churn."
)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1: OVERVIEW                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_overview():
    a = load_artifacts()
    st.title("Future Disengagement Prediction — COFINFAD")

    st.markdown("""
    > **Objective:** Predict which customers of a Colombian fintech will
    > show **transactional disengagement** (silence or sharp decline) in
    > Q4 2023, using only data observable through Sep 2023.
    """)

    # Key metrics
    eval_df = a["eval_df"]
    best_full = eval_df[eval_df["Setup"] == "Full"].sort_values(
        "ROC_AUC", ascending=False).iloc[0]
    best_tx = eval_df[eval_df["Setup"] == "Tx-strict"].sort_values(
        "ROC_AUC", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full — Best ROC AUC",
              f"{best_full['ROC_AUC']:.4f}",
              help=f"{best_full['Setup']} / {best_full['Model']}")
    c2.metric("Tx-strict — Best ROC AUC",
              f"{best_tx['ROC_AUC']:.4f}",
              help=f"{best_tx['Setup']} / {best_tx['Model']}")
    c3.metric("Target Rate",
              f"{a['y_all'].mean():.1%}",
              help=f"{int(a['y_all'].sum()):,} / {len(a['y_all']):,}")
    c4.metric("Best Threshold (val F1)",
              f"{a['best_t']:.2f}",
              help=f"Tuned on validation, F1 = {a['best_f1_val']:.4f}")

    st.markdown("---")

    # Temporal design
    st.subheader("Temporal Design")
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

    # Dataset summary
    st.subheader("Dataset Summary")
    cust = a["cust"]
    y_all = a["y_all"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(cust):,}")
    c2.metric("Disengaged", f"{int(y_all.sum()):,} ({y_all.mean():.1%})")
    c3.metric("Train / Val / Test",
              f"{len(a['idx_train']):,} / {len(a['idx_val']):,} / {len(a['idx_test']):,}")
    c4.metric("Feature Setups", "Full · Tx-strict · Cust-only")

    st.markdown("""
    | Component | Description |
    |---|---|
    | **Data** | COFINFAD — 48,723 customers, 3.2M transactions (2023) |
    | **Observation** | Jan – Sep 2023 — features built from raw transactions + customer attributes |
    | **Prediction** | Oct – Dec 2023 — target = transactional silence or sharp decline |
    | **Split** | 60 / 20 / 20 stratified (train / validation / test) |
    | **Models** | Logistic Regression, Random Forest, XGBoost × 3 feature setups |
    | **Preprocessing** | ColumnTransformer fit on train only (no leakage) |
    | **Threshold** | Tuned on validation set only (never on test) |
    """)

    st.warning(
        "⚠️ The target `future_disengaged` is a **behavioural proxy** — not "
        "contractual churn. Customer-level variables are assumed to be "
        "observation-cutoff snapshots. See the Tx-strict benchmark for "
        "a model that uses only transaction-derived features."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2: DATA EXPLORER                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_data_explorer():
    a = load_artifacts()
    st.title("Data Explorer")

    cust = a["cust"]
    target_df = a["target_df"]
    full_df = a["full_df"]
    y_all = a["y_all"]

    # Target distribution
    st.subheader("Target Distribution")
    c1, c2 = st.columns([1, 2])
    with c1:
        tgt_counts = pd.Series(y_all).value_counts().sort_index()
        st.metric("Active (0)", f"{tgt_counts.get(0, 0):,}")
        st.metric("Disengaged (1)", f"{tgt_counts.get(1, 0):,}")
        st.metric("Rate", f"{y_all.mean():.2%}")
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(["Active", "Disengaged"],
                       [tgt_counts.get(0, 0), tgt_counts.get(1, 0)],
                       color=[C["active"], C["disengaged"]], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Target: future_disengaged", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Customer attribute distributions
    st.subheader("Customer Attributes by Target")

    num_features = ["age", "customer_tenure", "active_products",
                    "satisfaction_score", "nps_score",
                    "app_logins_frequency", "credit_utilization_ratio"]
    num_features = [f for f in num_features if f in cust.columns]

    merged = cust[["customer_id"] + [c for c in num_features if c in cust.columns]].merge(
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

    # Categorical distributions
    st.subheader("Categorical Features")
    cat_feats = ["income_bracket", "gender", "acquisition_channel",
                 "feedback_sentiment"]
    cat_feats = [f for f in cat_feats if f in cust.columns]

    merged_cat = cust[["customer_id"] + cat_feats].merge(
        target_df[["customer_id", "future_disengaged"]], on="customer_id"
    )

    cols = st.columns(2)
    for i, feat in enumerate(cat_feats):
        with cols[i % 2]:
            ct = pd.crosstab(merged_cat[feat], merged_cat["future_disengaged"],
                             normalize="index")
            fig, ax = plt.subplots(figsize=(5, 3))
            ct.plot(kind="bar", stacked=True, ax=ax,
                    color=[C["active"], C["disengaged"]], alpha=0.85)
            ax.set_title(feat.replace("_", " ").title(), fontsize=9,
                         fontweight="bold")
            ax.set_ylabel("Proportion")
            ax.legend(["Active", "Disengaged"], fontsize=7)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # Transaction feature overview
    st.subheader("Key Transaction Features (Observation Window)")
    tx_feats = ["tx_count", "tx_total", "tx_recency", "tx_max_gap",
                "tx_late_ratio", "tx_wknd_ratio"]
    tx_feats = [f for f in tx_feats if f in full_df.columns]

    cols = st.columns(3)
    for i, feat in enumerate(tx_feats):
        with cols[i % 3]:
            fig, ax = plt.subplots(figsize=(5, 3))
            active = full_df.loc[full_df["future_disengaged"] == 0, feat].dropna()
            diseng = full_df.loc[full_df["future_disengaged"] == 1, feat].dropna()
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


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3: MODEL PERFORMANCE                                            ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_model_performance():
    a = load_artifacts()
    st.title("Model Performance — Test Set")

    eval_df = a["eval_df"]
    y_te = a["y_te"]
    preds_test = a["preds_test"]

    # Full metrics table
    st.subheader("All Models × All Setups")
    fmt = {c: "{:.4f}" for c in eval_df.columns if c not in ("Setup", "Model")}
    st.dataframe(
        eval_df.style.format(fmt).highlight_max(
            subset=["ROC_AUC", "PR_AUC", "Lift@D1"], color="#c8e6c9"
        ).highlight_min(
            subset=["Brier"], color="#c8e6c9"
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    # ROC / PR / Calibration for Full setup
    st.subheader("Diagnostic Curves — Full Setup")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    base_rate = y_te.mean()

    for algo, clr in ALGO_COLORS.items():
        key = ("Full", algo)
        if key not in preds_test:
            continue
        yp = preds_test[key]
        fpr, tpr, _ = roc_curve(y_te, yp)
        axes[0].plot(fpr, tpr, label=f"{algo} ({roc_auc_score(y_te, yp):.3f})",
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
    axes[1].set_title("Precision-Recall", fontweight="bold"); axes[1].legend(fontsize=7)

    axes[2].plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    axes[2].set_xlabel("Mean predicted prob"); axes[2].set_ylabel("Fraction positive")
    axes[2].set_title("Calibration", fontweight="bold"); axes[2].legend(fontsize=7)

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


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4: SETUP COMPARISON                                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_setup_comparison():
    a = load_artifacts()
    st.title("Setup Comparison — Full vs Tx-strict vs Cust-only")

    eval_df = a["eval_df"]
    y_te = a["y_te"]
    preds_test = a["preds_test"]

    st.markdown("""
    | Setup | Features | Purpose |
    |-------|----------|---------|
    | **Full** | Customer attributes + obs-window tx features | Best available information |
    | **Tx-strict** | Obs-window tx features only | Ultra-strict — no customer-level variables |
    | **Cust-only** | Customer attributes only | Baseline — demographics + product flags |
    """)

    st.markdown("---")

    # Bar chart: ROC AUC by setup × algo
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
        bars = ax.bar(x + i * w, vals, w, label=algo, color=ALGO_COLORS[algo],
                      alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + w)
    ax.set_xticklabels(setups, fontsize=10)
    ax.axhline(0.5, color="#9e9e9e", ls="--", lw=0.8, label="Random")
    ax.set_ylabel("ROC AUC"); ax.set_ylim(0.3, 0.9)
    ax.set_title("Test Set ROC AUC", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # ROC curves overlaid by setup (best algo per setup)
    st.subheader("ROC Curves — Best Model per Setup")
    fig, ax = plt.subplots(figsize=(8, 5))
    for setup, clr in SETUP_COLORS.items():
        sub = eval_df[eval_df["Setup"] == setup]
        best = sub.loc[sub["ROC_AUC"].idxmax()]
        key = (setup, best["Model"])
        if key in preds_test:
            yp = preds_test[key]
            fpr, tpr, _ = roc_curve(y_te, yp)
            ax.plot(fpr, tpr, lw=1.8, color=clr,
                    label=f"{setup}/{best['Model']} ({best['ROC_AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves — Best per Setup", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Lift at top decile comparison
    st.subheader("Lift at Top Decile")
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, algo in enumerate(algos):
        vals = []
        for s in setups:
            row = eval_df[(eval_df["Setup"] == s) & (eval_df["Model"] == algo)]
            vals.append(row["Lift@D1"].values[0] if len(row) else 0)
        ax.bar(x + i * w, vals, w, label=algo, color=ALGO_COLORS[algo], alpha=0.85)

    ax.set_xticks(x + w)
    ax.set_xticklabels(setups, fontsize=10)
    ax.axhline(1.0, color="#9e9e9e", ls="--", lw=0.8, label="No lift")
    ax.set_ylabel("Lift"); ax.set_title("Lift @ Top Decile", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Insight callouts
    full_best = eval_df[eval_df["Setup"] == "Full"]["ROC_AUC"].max()
    tx_best = eval_df[eval_df["Setup"] == "Tx-strict"]["ROC_AUC"].max()
    cu_best = eval_df[eval_df["Setup"] == "Cust-only"]["ROC_AUC"].max()

    st.success(
        f"**Tx-strict** achieves ROC AUC {tx_best:.4f} — only "
        f"{full_best - tx_best:.4f} below Full ({full_best:.4f}). "
        f"Predictive signal comes primarily from transaction-derived "
        f"features, not customer-level attributes."
    )
    st.error(
        f"**Cust-only** ROC AUC is {cu_best:.4f} (near random). Customer "
        f"attributes alone carry negligible predictive power for future "
        f"disengagement."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 5: THRESHOLD ANALYSIS                                           ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_threshold():
    a = load_artifacts()
    st.title("Threshold Analysis")

    st.markdown(
        "The decision threshold is tuned on the **validation set** by "
        "maximising F1. The test set is **never** used for threshold selection."
    )

    th_df = a["th_df"]
    best_t = a["best_t"]
    best_f1_val = a["best_f1_val"]
    best_algo = a["best_algo"]
    y_te = a["y_te"]
    yp_te = a["preds_test"][("Full", best_algo)]

    # Threshold sweep plot
    st.subheader(f"Threshold Sweep — Full / {best_algo} (Validation Set)")
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

    st.markdown("---")

    # Final test with frozen threshold
    st.subheader(f"Final Test Evaluation — threshold = {best_t:.2f}")
    yhat = (yp_te >= best_t).astype(int)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC AUC", f"{roc_auc_score(y_te, yp_te):.4f}")
    c2.metric("F1", f"{f1_score(y_te, yhat):.4f}")
    c3.metric("Precision", f"{precision_score(y_te, yhat, zero_division=0):.4f}")
    c4.metric("Recall", f"{recall_score(y_te, yhat):.4f}")

    report = classification_report(y_te, yhat,
                                    target_names=["Active", "Disengaged"])
    st.code(report, language="text")

    # Confusion matrix with tuned threshold
    st.subheader("Confusion Matrix — Tuned Threshold")
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_te, yhat)
    ConfusionMatrixDisplay(cm, display_labels=["Active", "Disengaged"]).plot(
        ax=ax, cmap="Blues")
    ax.set_title(f"Full / {best_algo} — threshold {best_t:.2f}", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Interactive threshold explorer
    st.subheader("Interactive Threshold Explorer")
    t_select = st.slider("Choose threshold", 0.05, 0.95, float(best_t), 0.01)
    yhat_custom = (yp_te >= t_select).astype(int)

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("F1", f"{f1_score(y_te, yhat_custom):.4f}")
    cc2.metric("Precision", f"{precision_score(y_te, yhat_custom, zero_division=0):.4f}")
    cc3.metric("Recall", f"{recall_score(y_te, yhat_custom):.4f}")
    cc4.metric("Flagged", f"{yhat_custom.sum():,} / {len(yhat_custom):,}")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 6: EXPLAINABILITY                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_explainability():
    a = load_artifacts()
    st.title("Explainability")

    feat_names = a["feat_names_full"]
    trained = a["trained"]

    # XGBoost feature importance
    st.subheader("Feature Importance — XGBoost (Full Setup, Gain)")
    xgb_full = trained[("Full", "XGB")]
    imp = pd.DataFrame({
        "feature": feat_names,
        "importance": xgb_full.feature_importances_
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
    st.subheader("Logistic Regression Coefficients — Full Setup")
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
    ax.set_title("Top 20 — LR (Full Setup)\nRed = ↑ disengagement risk  "
                 "Green = ↓ risk", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # SHAP
    shap_vals = a.get("shap_vals")

    if shap_vals is not None:
        st.subheader("SHAP Analysis — XGBoost (Full Setup)")

        # SHAP bar chart (mean |SHAP|)
        mean_abs = np.abs(shap_vals).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        sp = shap_df.sort_values("mean_abs_shap")
        ax.barh(sp["feature"], sp["mean_abs_shap"], color=C["xgb"], alpha=0.85)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("SHAP Feature Importance — XGBoost (Full Setup)",
                     fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # SHAP scatter for top 5 features
        st.subheader("SHAP Value Distribution — Top 5 Features")
        top5 = shap_df.head(5)["feature"].tolist()

        fig, axes_shap = plt.subplots(1, len(top5), figsize=(3.5 * len(top5), 3.5))
        if len(top5) == 1:
            axes_shap = [axes_shap]

        for ax_s, feat_name in zip(axes_shap, top5):
            fidx = list(feat_names).index(feat_name) if feat_name in feat_names else None
            if fidx is not None:
                ax_s.scatter(
                    range(len(shap_vals)),
                    shap_vals[:, fidx],
                    alpha=0.2, s=6, color=C["xgb"],
                )
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
        st.info("SHAP values not available. Re-run the pipeline with SHAP installed.")

    st.markdown("---")

    # Sanity check
    st.subheader("Sanity Check — churn_probability Correlation")
    corr_val = a.get("corr_sanity", 0)
    st.metric("Pearson ρ (model score vs churn_probability)", f"{corr_val:.3f}")
    st.markdown(
        "The dataset's pre-computed `churn_probability` has **near-zero** "
        "correlation with our model scores. This is expected: our proxy captures "
        "future transactional disengagement, while `churn_probability` appears "
        "to measure a different construct. Neither is used as a training feature."
    )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 7: MASTER SIGNAL                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def page_master_signal():
    a = load_artifacts()
    st.title("Master Commercial Signal")

    signals_df = a.get("signals_df")
    top_opportunities = a.get("top_opportunities")

    if signals_df is None:
        st.error(
            "Multi-signal data not found. Re-run the pipeline to generate signals:\n\n"
            "```\npython3 notebooks/churn_modeling.py\n```"
        )
        st.stop()

    st.markdown("""
    > Three behavioral signals combined into a **Master Commercial Signal**
    > for client prioritisation.
    >
    > **MasterSignal = 0.5 × BuyPropensity + 0.3 × EngagementScore + 0.2 × (1 − RedemptionRisk)**
    """)

    # Key metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Master Signal",
              f"{signals_df['master_signal'].mean():.3f}")
    c2.metric("Upsell Candidates",
              f"{(signals_df['recommended_action'] == 'Upsell').sum():,}")
    c3.metric("Retention Targets",
              f"{(signals_df['recommended_action'] == 'Retention').sum():,}")
    c4.metric("Monitor",
              f"{(signals_df['recommended_action'] == 'Monitor').sum():,}")

    st.markdown("---")

    # Master Signal distribution
    st.subheader("Master Signal Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(signals_df["master_signal"], bins=50, color=C["primary"],
            alpha=0.85, edgecolor="white")
    ax.set_xlabel("Master Signal"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Master Commercial Signal", fontweight="bold")
    ax.axvline(signals_df["master_signal"].mean(), color="black", ls="--",
               lw=1, label=f"Mean = {signals_df['master_signal'].mean():.3f}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")

    # Individual signal distributions
    st.subheader("Individual Signal Distributions")
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

    # Recommended actions
    st.subheader("Recommended Actions")
    c1, c2 = st.columns([1, 2])
    with c1:
        action_counts = signals_df["recommended_action"].value_counts()
        st.dataframe(action_counts.to_frame("Count"), width="stretch")
        st.markdown("""
        | Action | Rule |
        |--------|------|
        | **Upsell** | BuyPropensity > 0.6 AND RedemptionRisk < 0.3 |
        | **Retention** | RedemptionRisk > 0.6 |
        | **Monitor** | All other clients |
        """)
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
        ax.set_title("Action Distribution", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # Opportunity quadrant
    st.subheader("Opportunity Quadrant")
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

    # Top opportunities
    st.subheader("Top 100 Sales Opportunities")
    st.dataframe(
        top_opportunities.style.format({
            "buy_propensity": "{:.4f}",
            "redemption_risk": "{:.4f}",
            "engagement_score": "{:.4f}",
            "master_signal": "{:.4f}",
        }).background_gradient(subset=["master_signal"], cmap="Greens"),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    # Signal correlations
    st.subheader("Signal Correlations")
    corr_mat = signals_df[["buy_propensity", "redemption_risk",
                           "engagement_score", "master_signal"]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Buy Prop.", "Redemp. Risk", "Engagement", "Master"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr_mat.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson ρ")
    ax.set_title("Signal Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                               ║
# ╚═════════════════════════════════════════════════════════════════════════╝

PAGES = {
    "🏠 Overview": page_overview,
    "📊 Data Explorer": page_data_explorer,
    "🎯 Model Performance": page_model_performance,
    "⚖️ Setup Comparison": page_setup_comparison,
    "🎚️ Threshold Analysis": page_threshold,
    "🧠 Explainability": page_explainability,
    "📡 Master Signal": page_master_signal,
}

PAGES[page]()
