"""
utils.py
========
Shared utility helpers for the Behavioral Master Signal prototype.

Includes:
- min-max scaling
- time-aware train/validation/test split
- a simple console logger
"""

from __future__ import annotations

import logging
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str = "bms") -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def minmax_scale_series(s: pd.Series, clip: bool = True) -> pd.Series:
    """Scale a Series to [0, 1] using its own min/max.

    Parameters
    ----------
    s    : input Series (numeric)
    clip : whether to clip output to [0, 1] after scaling

    Returns
    -------
    Scaled Series of the same index.
    """
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index, name=s.name)
    scaled = (s - lo) / (hi - lo)
    if clip:
        scaled = scaled.clip(0.0, 1.0)
    return scaled.rename(s.name)


def zscore_series(s: pd.Series) -> pd.Series:
    """Standardise a Series to zero mean and unit variance.

    Available as a building block for feature engineering steps that require
    z-scored inputs before clipping or further transformation (e.g. news burst).
    """
    mu, sigma = s.mean(), s.std(ddof=0)
    if sigma == 0:
        return pd.Series(0.0, index=s.index, name=s.name)
    return ((s - mu) / sigma).rename(s.name)


# ---------------------------------------------------------------------------
# Time-aware data split
# ---------------------------------------------------------------------------

def time_split(
    df: pd.DataFrame,
    date_col: str = "date",
    train_end: str = config.TRAIN_END,
    valid_end: str = config.VALID_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a panel DataFrame into train / validation / test by calendar date.

    The split is strictly forward-looking:
        train  : date <= train_end
        valid  : train_end < date <= valid_end
        test   : date > valid_end

    This prevents any look-ahead bias that would arise from a random split
    across a time-indexed panel.

    Parameters
    ----------
    df        : panel DataFrame with a date column
    date_col  : name of the date column
    train_end : ISO date string marking the end of the training window
    valid_end : ISO date string marking the end of the validation window

    Returns
    -------
    (train_df, valid_df, test_df)
    """
    dates = pd.to_datetime(df[date_col])
    t_end = pd.Timestamp(train_end)
    v_end = pd.Timestamp(valid_end)

    train_mask = dates <= t_end
    valid_mask = (dates > t_end) & (dates <= v_end)
    test_mask = dates > v_end

    return df[train_mask].copy(), df[valid_mask].copy(), df[test_mask].copy()


# ---------------------------------------------------------------------------
# Sigmoid helper (avoids scipy dependency)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
