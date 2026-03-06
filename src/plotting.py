"""
plotting.py
===========
Clean, professional visualisations for the Behavioral Master Signal prototype.

Charts produced
---------------
1. Sub-signal distributions (violin + box plot)
2. Model performance summary (ROC AUC bar chart)
3. Precision-by-decile lift chart
4. Feature importance chart (SHAP or built-in)
5. Pipeline architecture diagram (text-based, no external deps)
6. Example client timeline

All figures are saved to outputs/figures/ and returned as Figure objects.
Style is kept intentionally minimal — no flashy palettes or excessive labels.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/script usage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger()

_FIG_DIR = config.FIGURES_DIR
os.makedirs(_FIG_DIR, exist_ok=True)

# Consistent colour palette
_COLORS = {
    "heuristic": "#9e9e9e",
    "logistic":  "#1976D2",
    "gbt":       "#388E3C",
    "buy":       "#1976D2",
    "redeem":    "#D32F2F",
}


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(_FIG_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Sub-signal distributions
# ---------------------------------------------------------------------------

def plot_signal_distributions(df: pd.DataFrame) -> plt.Figure:
    """Violin plots of the four sub-signals, coloured by label."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()

    signals = config.SUB_SIGNAL_COLS
    titles  = [
        "Tenure & Performance Signal",
        "External Flows Signal",
        "News / Attention Signal",
        "Product Launch Signal",
    ]

    for ax, sig, title in zip(axes, signals, titles):
        buy_vals    = df.loc[df["buy_3m"] == 1,    sig].dropna()
        redeem_vals = df.loc[df["redeem_3m"] == 1, sig].dropna()
        no_action   = df.loc[(df["buy_3m"] == 0) & (df["redeem_3m"] == 0), sig].dropna()

        data   = [no_action, buy_vals, redeem_vals]
        labels = ["No Action", "Buy", "Redeem"]
        colors = ["#BDBDBD", _COLORS["buy"], _COLORS["redeem"]]

        parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                               showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_alpha(0.6)
        parts["cmedians"].set_colors(["#333333"])

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Signal value [0–1]", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.suptitle("Sub-signal distributions by outcome label\n"
                 "(synthetic prototype data)", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "01_signal_distributions.png")
    return fig


# ---------------------------------------------------------------------------
# 2. Model performance summary
# ---------------------------------------------------------------------------

def plot_model_performance(results_df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart comparing ROC AUC across models and labels."""
    test_df = results_df[results_df["split"] == "test"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label in zip(axes, ["buy_3m", "redeem_3m"]):
        sub = test_df[test_df["label"] == label].sort_values("model")
        models  = sub["model"].tolist()
        roc_auc = sub["roc_auc"].tolist()
        pr_auc  = sub["pr_auc"].tolist()

        x = np.arange(len(models))
        w = 0.35

        colors = [_COLORS.get(m, "#757575") for m in models]
        bars1 = ax.bar(x - w / 2, roc_auc, w, label="ROC AUC",
                        color=colors, alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + w / 2, pr_auc, w, label="PR AUC",
                        color=colors, alpha=0.50, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in models], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#9e9e9e", linewidth=0.8, linestyle="--",
                   label="Random (0.5)")
        ax.set_title(
            f"{'Buy Propensity' if label == 'buy_3m' else 'Redemption Risk'} — "
            f"Test set",
            fontsize=10, fontweight="bold",
        )
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=8)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7)

    fig.suptitle("Model Performance Comparison (Test Set)\n"
                 "Prototype on synthetic data — not validated on real data",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "02_model_performance.png")
    return fig


# ---------------------------------------------------------------------------
# 3. Precision-by-decile lift chart
# ---------------------------------------------------------------------------

def plot_lift_curves(
    models: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
) -> plt.Figure:
    """Precision-by-decile chart for GBT models (buy and redeem)."""
    from src.utils import time_split
    from src.evaluation import decile_table

    _, _, test_df = time_split(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label in zip(axes, ["buy_3m", "redeem_3m"]):
        bundle   = models[label]
        feat     = bundle["feature_cols"]
        X_test   = test_df[feat].astype(float)
        y_test   = test_df[label].astype(int).values
        base_rate = y_test.mean()

        for model_name, color in [("logistic", _COLORS["logistic"]),
                                    ("gbt",      _COLORS["gbt"])]:
            probs = bundle[model_name].predict_proba(X_test)[:, 1]
            dtbl  = decile_table(y_test, probs)
            ax.plot(dtbl["decile"], dtbl["precision"],
                    marker="o", color=color, linewidth=2,
                    label=model_name.capitalize())

        ax.axhline(base_rate, color="#9e9e9e", linestyle="--",
                   linewidth=1.2, label=f"Base rate ({base_rate:.1%})")
        ax.set_xlabel("Score Decile (1 = top)", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_title(
            f"{'Buy' if label == 'buy_3m' else 'Redemption'} — "
            f"Precision by Decile",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.set_xticks(range(1, 11))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.suptitle("Lift Analysis — Precision by Score Decile\n"
                 "(synthetic prototype data)", fontsize=11)
    fig.tight_layout()
    _save(fig, "03_lift_curves.png")
    return fig


# ---------------------------------------------------------------------------
# 4. Feature importance chart
# ---------------------------------------------------------------------------

def plot_feature_importance(
    explainability_results: Dict[str, Any],
    top_n: int = 15,
) -> plt.Figure:
    """Horizontal bar chart of SHAP or permutation feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label in zip(axes, ["buy_3m", "redeem_3m"]):
        imp_df = explainability_results[label]["shap_feature_importance"].head(top_n)

        col_name = imp_df.columns[-1]  # "mean_abs_shap" or "importance"
        imp_df = imp_df.sort_values(col_name, ascending=True)

        color = _COLORS["buy"] if label == "buy_3m" else _COLORS["redeem"]
        ax.barh(imp_df["feature"], imp_df[col_name], color=color, alpha=0.8)
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_title(
            f"{'Buy Propensity' if label == 'buy_3m' else 'Redemption Risk'}\n"
            f"Top {top_n} feature importances (GBT)",
            fontsize=10, fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Feature Importance — SHAP / Permutation\n"
                 "(synthetic prototype data)", fontsize=11)
    fig.tight_layout()
    _save(fig, "04_feature_importance.png")
    return fig


# ---------------------------------------------------------------------------
# 5. Example client timeline
# ---------------------------------------------------------------------------

def plot_client_timeline(
    df: pd.DataFrame,
    client_id: Optional[str] = None,
) -> plt.Figure:
    """Plot one client's exposure, performance, and sub-signals over time."""
    if client_id is None:
        # Pick a client with a mix of buy and redeem events
        counts = df.groupby("client_id")[["buy_3m", "redeem_3m"]].sum()
        mixed  = counts[(counts["buy_3m"] > 0) & (counts["redeem_3m"] > 0)]
        if not mixed.empty:
            client_id = mixed.index[0]
        else:
            client_id = df["client_id"].iloc[0]

    cdf = df[df["client_id"] == client_id].sort_values("date")

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    # Panel 1: Exposure
    ax = axes[0]
    ax.plot(cdf["date"], cdf["exposure"], color="#455A64", linewidth=1.5,
            label="Exposure (M USD)")
    buy_dates    = cdf.loc[cdf["buy_3m"] == 1, "date"]
    redeem_dates = cdf.loc[cdf["redeem_3m"] == 1, "date"]
    ax.scatter(buy_dates,
               cdf.loc[cdf["buy_3m"] == 1, "exposure"],
               marker="^", color=_COLORS["buy"], s=50, zorder=5,
               label="Buy event")
    ax.scatter(redeem_dates,
               cdf.loc[cdf["redeem_3m"] == 1, "exposure"],
               marker="v", color=_COLORS["redeem"], s=50, zorder=5,
               label="Redeem event")
    ax.set_ylabel("Exposure", fontsize=8)
    ax.legend(fontsize=7)

    # Panel 2: 3M performance
    ax = axes[1]
    ax.plot(cdf["date"], cdf["performance_3m"] * 100,
            color="#00796B", linewidth=1.5)
    ax.axhline(0, color="#9e9e9e", linewidth=0.8, linestyle="--")
    ax.set_ylabel("3M Perf. (%)", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # Panel 3: Sub-signals
    ax = axes[2]
    signal_colors = ["#1976D2", "#388E3C", "#FFA000", "#7B1FA2"]
    for sig, c in zip(config.SUB_SIGNAL_COLS, signal_colors):
        ax.plot(cdf["date"], cdf[sig], linewidth=1.2, label=sig.replace("_", " "), color=c)
    ax.set_ylabel("Signal [0–1]", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlabel("Date", fontsize=9)

    fig.suptitle(f"Client Timeline — {client_id}\n"
                 "(synthetic prototype data)", fontsize=11)
    fig.tight_layout()
    _save(fig, "05_client_timeline.png")
    return fig


# ---------------------------------------------------------------------------
# 6. Score distribution histogram
# ---------------------------------------------------------------------------

def plot_score_distributions(df: pd.DataFrame) -> plt.Figure:
    """Histogram of heuristic master scores, split by label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    pairs = [
        ("buy_3m",    "master_buy_score_heuristic",
         "Buy Propensity — Heuristic Score", _COLORS["buy"]),
        ("redeem_3m", "master_redeem_score_heuristic",
         "Redemption Risk — Heuristic Score", _COLORS["redeem"]),
    ]

    for ax, (label, score_col, title, color) in zip(axes, pairs):
        if score_col not in df.columns:
            ax.set_visible(False)
            continue
        pos = df.loc[df[label] == 1, score_col]
        neg = df.loc[df[label] == 0, score_col]
        ax.hist(neg, bins=40, alpha=0.5, color="#9e9e9e", density=True,
                label="No event")
        ax.hist(pos, bins=40, alpha=0.7, color=color, density=True,
                label="Event")
        ax.set_xlabel("Heuristic score", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("Heuristic Master Score Distributions\n"
                 "(synthetic prototype data)", fontsize=11)
    fig.tight_layout()
    _save(fig, "06_score_distributions.png")
    return fig
