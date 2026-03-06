"""
scoring.py
==========
Constructs interpretable heuristic master scores from the four sub-signals.

Heuristic design rationale
---------------------------
Before training any ML model, we encode our *prior business hypothesis* about
how the sub-signals should combine.  This serves three purposes:

1. Provides a non-ML baseline to beat (showing whether ML adds value).
2. Documents explicit assumptions about signal directions and weights.
3. Allows immediate commercial use even before model training.

Buy score
---------
    master_buy_score = 0.35 × tenure_perf_signal
                     + 0.30 × external_flows_signal
                     + 0.20 × news_signal
                     + 0.15 × launch_signal

Rationale for weights:
- tenure_perf (35%): The strongest individual predictor of buy intent —
  a client with a profitable, stable relationship is most likely to add.
- external_flows (30%): Category-level flows reflect the macro tide; clients
  typically act with, not against, broad market flows.
- news (20%): Media coverage amplifies decisions but is noisier; secondary role.
- launch (15%): New product pipeline is an opportunity signal, useful but least
  directly correlated with an individual client's near-term buy decision.

Redemption score
----------------
    master_redeem_score = −0.30 × tenure_perf_signal
                        − 0.35 × external_flows_signal
                        + 0.15 × news_signal
                        − 0.20 × launch_signal

Sign directions:
- tenure_perf negative: strong relationship → lower exit propensity.
- external_flows negative: positive flows → clients tend to stay (herding).
- news positive: elevated attention can amplify anxiety → outflow trigger.
- launch negative: a rich product menu reduces the incentive to exit.

IMPORTANT
---------
These weights are a *business hypothesis* and initial prototype assumption.
The ML models in modeling.py will subsequently test whether data-driven
weighting or non-linear interactions improve upon this baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, minmax_scale_series

logger = get_logger()


def compute_heuristic_buy_score(df: pd.DataFrame) -> pd.Series:
    """Compute the heuristic master buy propensity score.

    Score is a weighted linear combination of the four sub-signals,
    rescaled to [0, 1] for comparability.

    Parameters
    ----------
    df : DataFrame with columns tenure_perf_signal, external_flows_signal,
         news_signal, launch_signal (all already in [0,1])

    Returns
    -------
    pd.Series named 'master_buy_score_heuristic', values in [0, 1]
    """
    w = config.HEURISTIC_BUY_WEIGHTS
    raw = (
        w["tenure_perf_signal"]    * df["tenure_perf_signal"]
        + w["external_flows_signal"] * df["external_flows_signal"]
        + w["news_signal"]           * df["news_signal"]
        + w["launch_signal"]         * df["launch_signal"]
    )
    return minmax_scale_series(raw).rename("master_buy_score_heuristic")


def compute_heuristic_redeem_score(df: pd.DataFrame) -> pd.Series:
    """Compute the heuristic master redemption risk score.

    Negative weights on protective factors (tenure, positive flows, new products)
    and a positive weight on news (can amplify anxiety).

    Scores are shifted and rescaled so that higher value = higher redeem risk.

    Parameters
    ----------
    df : DataFrame with columns tenure_perf_signal, external_flows_signal,
         news_signal, launch_signal

    Returns
    -------
    pd.Series named 'master_redeem_score_heuristic', values in [0, 1]
    """
    w = config.HEURISTIC_REDEEM_WEIGHTS
    raw = (
        w["tenure_perf_signal"]    * df["tenure_perf_signal"]
        + w["external_flows_signal"] * df["external_flows_signal"]
        + w["news_signal"]           * df["news_signal"]
        + w["launch_signal"]         * df["launch_signal"]
    )
    # Shift to [0,1]: the raw score may be negative (all weights here sum ≠ 1)
    return minmax_scale_series(raw).rename("master_redeem_score_heuristic")


def attach_heuristic_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Attach both heuristic scores to the panel and return augmented DataFrame."""
    df = df.copy()
    df["master_buy_score_heuristic"]    = compute_heuristic_buy_score(df)
    df["master_redeem_score_heuristic"] = compute_heuristic_redeem_score(df)
    logger.info(
        "Heuristic scores attached | "
        "buy_mean=%.3f  redeem_mean=%.3f",
        df["master_buy_score_heuristic"].mean(),
        df["master_redeem_score_heuristic"].mean(),
    )
    return df
