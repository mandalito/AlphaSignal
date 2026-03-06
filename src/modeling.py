"""
modeling.py
===========
Trains and persists the ML models for the Behavioral Master Signal prototype.

Two models per label (buy_3m, redeem_3m):
    1. Logistic Regression (L2, scaled features) — interpretable baseline
    2. Gradient Boosted Trees — XGBoost preferred; falls back to
       sklearn HistGradientBoostingClassifier if XGBoost is unavailable.

Training uses a strict time-aware split (no random shuffling across time).
Feature scaling is applied only to logistic regression (tree models are
invariant to monotonic feature transformations).

All models are serialised to disk with joblib for reproducibility.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils import get_logger, time_split

logger = get_logger()


# ---------------------------------------------------------------------------
# Gradient boosted tree: XGBoost with safe fallback
# ---------------------------------------------------------------------------

def _get_gbt_model() -> Any:
    """Return an XGBoost classifier, or HistGradientBoostingClassifier fallback."""
    try:
        from xgboost import XGBClassifier  # type: ignore
        logger.info("Using XGBoost as the gradient boosted tree model.")
        return XGBClassifier(**config.XGBOOST_PARAMS)
    except ImportError:
        logger.warning(
            "XGBoost not found — falling back to "
            "sklearn.HistGradientBoostingClassifier."
        )
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=config.RANDOM_SEED,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_splits(
    df: pd.DataFrame,
    label: str,
    feature_cols: list[str],
) -> Tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """Split the panel into train/valid/test and extract X, y."""
    train_df, valid_df, test_df = time_split(df)

    def _xy(split: pd.DataFrame):
        X = split[feature_cols].astype(float)
        y = split[label].astype(int)
        return X, y

    return (*_xy(train_df), *_xy(valid_df), *_xy(test_df))


def _build_logistic_pipeline() -> Pipeline:
    """Logistic regression preceded by standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**config.LOGISTIC_PARAMS)),
    ])


# ---------------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------------

def train_models(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> Dict[str, Dict[str, Any]]:
    """Train all models for both labels.

    Parameters
    ----------
    df           : full panel DataFrame with features and labels attached
    feature_cols : ordered list of feature column names to use

    Returns
    -------
    Dictionary structured as:
        {
          "buy_3m": {
              "logistic":  trained Pipeline,
              "gbt":       trained GBT model,
              "X_train": ..., "y_train": ...,
              "X_valid": ..., "y_valid": ...,
              "X_test":  ..., "y_test":  ...,
          },
          "redeem_3m": { ... }
        }
    """
    models: Dict[str, Dict[str, Any]] = {}

    for label in config.LABEL_COLS:
        logger.info("Training models for label: %s", label)

        X_tr, y_tr, X_va, y_va, X_te, y_te = _prepare_splits(
            df, label, feature_cols
        )

        logger.info(
            "  Split sizes → train=%d  valid=%d  test=%d | "
            "pos_rate_train=%.1f%%",
            len(y_tr), len(y_va), len(y_te), 100 * y_tr.mean()
        )

        # --- Logistic Regression ---
        lr_pipe = _build_logistic_pipeline()
        lr_pipe.fit(X_tr, y_tr)
        logger.info("  Logistic regression trained.")

        # --- Gradient Boosted Trees ---
        gbt = _get_gbt_model()
        # XGBoost eval_set for early stopping (use validation)
        try:
            gbt.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False,
            )
        except TypeError:
            # HistGradientBoosting does not support eval_set
            gbt.fit(X_tr, y_tr)
        logger.info("  GBT model trained.")

        models[label] = {
            "logistic": lr_pipe,
            "gbt": gbt,
            "X_train": X_tr, "y_train": y_tr,
            "X_valid": X_va, "y_valid": y_va,
            "X_test":  X_te, "y_test":  y_te,
            "feature_cols": feature_cols,
        }

    return models


def save_models(models: Dict[str, Dict[str, Any]],
                directory: str = config.MODELS_DIR) -> None:
    """Serialise trained models to disk."""
    os.makedirs(directory, exist_ok=True)
    for label, bundle in models.items():
        for model_name in ("logistic", "gbt"):
            path = os.path.join(directory, f"{label}_{model_name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(bundle[model_name], f)
    logger.info("Models saved to %s/", directory)


def load_models(
    directory: str = config.MODELS_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Load serialised models from disk."""
    models: Dict[str, Dict[str, Any]] = {}
    for label in config.LABEL_COLS:
        models[label] = {}
        for model_name in ("logistic", "gbt"):
            path = os.path.join(directory, f"{label}_{model_name}.pkl")
            with open(path, "rb") as f:
                models[label][model_name] = pickle.load(f)
    return models
