"""
evaluation.py
=============
Evaluation framework for the Behavioral Master Signal prototype.

Metrics computed for each model × label combination:

Discriminative power:
    - ROC AUC
    - PR AUC (Average Precision)

Business-facing (top-decile) metrics:
    - Precision@10%  : fraction of top-10%-scored observations that are true positives
    - Recall@10%     : share of all positive events captured in the top 10%
    - Lift@10%       : how much more concentrated positives are in top decile vs. base rate

Heuristic comparison:
    The same top-decile metrics are computed for the heuristic score baseline,
    giving an explicit "does ML beat heuristic?" answer.

Time awareness:
    Evaluation is always performed on the held-out validation and test sets.
    No metrics are reported on training data to avoid inflated results.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from src import config
from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def top_decile_metrics(y_true: np.ndarray,
                        y_score: np.ndarray) -> Dict[str, float]:
    """Compute precision, recall, and lift for the top 10% of scored rows.

    Parameters
    ----------
    y_true  : binary ground-truth labels (0/1)
    y_score : predicted probabilities or score values (higher = more likely)

    Returns
    -------
    dict with keys: precision_at_10, recall_at_10, lift_at_10
    """
    n = len(y_true)
    if n == 0:
        return {"precision_at_10": np.nan, "recall_at_10": np.nan,
                "lift_at_10": np.nan}

    threshold_idx = int(np.ceil(0.10 * n))
    # Sort descending by score
    order = np.argsort(y_score)[::-1]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[order[:threshold_idx]] = True

    tp_in_top   = y_true[top_mask].sum()
    total_pos   = y_true.sum()
    base_rate   = total_pos / n if n > 0 else np.nan

    precision   = tp_in_top / threshold_idx if threshold_idx > 0 else np.nan
    recall      = tp_in_top / total_pos if total_pos > 0 else np.nan
    lift        = precision / base_rate if base_rate > 0 else np.nan

    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "lift_at_10": lift,
    }


def evaluate_model(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str = "",
    split_name: str = "",
) -> Dict[str, float]:
    """Compute a full suite of evaluation metrics for one model/split combination.

    Parameters
    ----------
    y_true      : binary labels
    y_score     : predicted probability of positive class
    model_name  : for logging only
    split_name  : "valid" or "test" — for logging only

    Returns
    -------
    dict with keys: roc_auc, pr_auc, precision_at_10, recall_at_10, lift_at_10
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Guard against degenerate splits
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in %s/%s — metrics are undefined.",
                       model_name, split_name)
        return {k: np.nan for k in ["roc_auc", "pr_auc",
                                     "precision_at_10", "recall_at_10",
                                     "lift_at_10"]}

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc  = average_precision_score(y_true, y_score)
    decile  = top_decile_metrics(y_true, y_score)

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc, **decile}

    logger.info(
        "[%s | %s | %s] ROC-AUC=%.3f  PR-AUC=%.3f  "
        "Prec@10=%.3f  Rec@10=%.3f  Lift@10=%.2f",
        model_name, split_name,
        "n=%d" % len(y_true),
        roc_auc, pr_auc,
        decile["precision_at_10"], decile["recall_at_10"],
        decile["lift_at_10"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(
    models: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all trained models on validation and test splits.

    Also evaluates the heuristic scores (master_buy/redeem_score_heuristic)
    as a non-ML baseline.

    Parameters
    ----------
    models : output of modeling.train_models()
    df     : full panel DataFrame (used to extract heuristic score columns)

    Returns
    -------
    pd.DataFrame with one row per (label, model, split) combination and
    metric columns.
    """
    from src.utils import time_split

    _, valid_df, test_df = time_split(df)

    records = []

    for label in config.LABEL_COLS:
        bundle = models[label]
        feat   = bundle["feature_cols"]

        for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
            y_true = split_df[label].astype(int).values
            X_split = split_df[feat].astype(float)

            # Heuristic score baseline
            heuristic_col = (
                "master_buy_score_heuristic"
                if label == "buy_3m"
                else "master_redeem_score_heuristic"
            )
            if heuristic_col in split_df.columns:
                h_score = split_df[heuristic_col].values
                m = evaluate_model(y_true, h_score,
                                   model_name="heuristic", split_name=split_name)
                records.append({"label": label, "model": "heuristic",
                                 "split": split_name, **m})

            # Logistic regression
            lr_prob = bundle["logistic"].predict_proba(X_split)[:, 1]
            m = evaluate_model(y_true, lr_prob,
                                model_name="logistic", split_name=split_name)
            records.append({"label": label, "model": "logistic",
                             "split": split_name, **m})

            # GBT
            gbt_prob = bundle["gbt"].predict_proba(X_split)[:, 1]
            m = evaluate_model(y_true, gbt_prob,
                                model_name="gbt", split_name=split_name)
            records.append({"label": label, "model": "gbt",
                             "split": split_name, **m})

    results_df = pd.DataFrame(records)
    return results_df


def decile_table(y_true: np.ndarray,
                  y_score: np.ndarray,
                  n_deciles: int = 10) -> pd.DataFrame:
    """Build a precision/recall lift table broken down by score decile.

    Parameters
    ----------
    y_true    : binary labels
    y_score   : predicted scores (higher = higher propensity)
    n_deciles : number of quantile buckets (default 10)

    Returns
    -------
    DataFrame with columns: decile, n, n_positive, precision, recall, lift
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order   = np.argsort(y_score)[::-1]
    y_true_sorted  = y_true[order]
    total_pos = y_true.sum()
    base_rate = total_pos / len(y_true)

    bucket_size = len(y_true) // n_deciles
    rows = []
    for d in range(n_deciles):
        start = d * bucket_size
        end   = (d + 1) * bucket_size if d < n_deciles - 1 else len(y_true)
        bucket = y_true_sorted[start:end]
        n_pos  = bucket.sum()
        n      = len(bucket)
        prec   = n_pos / n if n > 0 else np.nan
        rec    = n_pos / total_pos if total_pos > 0 else np.nan
        lift   = prec / base_rate if base_rate > 0 else np.nan
        rows.append({
            "decile": d + 1,
            "n": n,
            "n_positive": n_pos,
            "precision": prec,
            "recall": rec,
            "lift": lift,
        })
    return pd.DataFrame(rows)
