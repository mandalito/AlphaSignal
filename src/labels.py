"""
labels.py
=========
Defines and attaches binary labels to the synthetic panel.

Label design principles
-----------------------
1. Labels are generated via latent probabilities derived from raw features.
   The causal chain is always:  features → latent_prob → sampled_label
   We NEVER build features from labels (no reverse leakage).

2. Each label represents a 3-month forward window:
   - buy_3m   : 1 if the client meaningfully INCREASES exposure in the next 3m
   - redeem_3m: 1 if the client meaningfully DECREASES exposure (partial/full
                redemption) in the next 3m

3. Labels are probabilistic and somewhat imbalanced, roughly:
   - buy_3m   : ~20–30% positive rate
   - redeem_3m: ~15–25% positive rate
   These rates are in line with real commercial fund activity.

4. Because we only have simulated exposure (no actual transaction log), we
   proxy "buy" / "redeem" through the sign and magnitude of the forward
   exposure change relative to current exposure, combined with latent prob.

Assumptions (documented)
------------------------
- A "buy event" occurs when forward_exposure > current_exposure × (1 + τ_buy)
  AND a Bernoulli draw from the latent buy probability is 1.
  τ_buy = 0.05 (5% increase threshold).
- A "redemption event" occurs when forward_exposure < current_exposure × (1 − τ_redeem)
  AND a Bernoulli draw from the latent redeem probability is 1.
  τ_redeem = 0.05.
- The latent probabilities are functions of the raw features (not the sub-signals,
  to avoid circular dependency).
- A single row can have both labels = 0 (no strong action expected),
  but we suppress the case where both = 1 (a client cannot buy and redeem
  in the same 3-month window in our simplified model; the higher-probability
  action wins).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, sigmoid

logger = get_logger()

# Minimum exposure change thresholds for labelling
THRESHOLD_BUY: float = 0.05    # +5% increase
THRESHOLD_REDEEM: float = 0.05  # −5% decrease


# ---------------------------------------------------------------------------
# Latent probability functions
# ---------------------------------------------------------------------------

def _latent_buy_prob(df: pd.DataFrame) -> np.ndarray:
    """Compute the latent buy propensity for each row.

    Business logic encoded:
    - Positive recent performance → higher buy propensity
    - Positive external flows     → higher buy propensity
    - Low drawdown (near 0)       → higher buy propensity
    - Mature tenure               → slightly higher buy propensity
    - Positive news sentiment     → amplifier
    - Risk-on regime              → higher overall propensity

    Returns a 1-D array of probabilities in (0, 1).
    """
    perf3  = df["performance_3m"].values
    perf12 = df["performance_12m"].values
    flow1  = df["category_flow_1m"].values
    flow3  = df["category_flow_3m"].values
    dd     = df["drawdown"].values          # ≤ 0; closer to 0 = better
    tenure = df["tenure_months"].values
    news_s = df["news_sentiment"].values
    regime = df["market_regime"].values
    news_v = df["news_volume"].values

    # Logit-space linear combination (coefficients are business hypotheses)
    logit = (
        -2.0                              # intercept (calibrates base rate)
        + 6.0  * perf3                    # 3m performance lift
        + 3.0  * perf12                   # 12m performance (secondary)
        + 0.5  * flow1                    # current month flows
        + 0.4  * flow3                    # 3m flow momentum
        + 4.0  * (dd + 0.10).clip(-1, 1)  # drawdown relief (near 0 = good)
        + 0.01 * np.log1p(tenure)        # log-tenure maturity
        + 0.8  * news_s                   # sentiment amplifier
        + 0.3  * (news_v / 10.0)          # volume amplifier
        + 0.7  * regime                   # regime lift
        + np.random.default_rng(config.RANDOM_SEED + 1)
              .normal(0, 0.3, size=len(df))  # irreducible noise
    )
    return sigmoid(logit)


def _latent_redeem_prob(df: pd.DataFrame) -> np.ndarray:
    """Compute the latent redemption propensity for each row.

    Business logic encoded:
    - Significant drawdown          → higher redemption risk
    - Negative external flows       → higher redemption risk
    - Low/negative recent performance → higher redemption risk
    - Short tenure (< 12 months)    → higher churn risk
    - Negative news sentiment        → amplifier
    - Risk-off regime               → higher overall risk

    Returns a 1-D array of probabilities in (0, 1).
    """
    perf3  = df["performance_3m"].values
    flow1  = df["category_flow_1m"].values
    flow3  = df["category_flow_3m"].values
    dd     = df["drawdown"].values
    tenure = df["tenure_months"].values
    news_s = df["news_sentiment"].values
    regime = df["market_regime"].values

    logit = (
        -2.5                               # intercept (calibrates base rate)
        - 5.0 * perf3                      # poor performance → redeem
        - 0.5 * flow1                      # outflows → redeem
        - 0.4 * flow3
        - 8.0 * dd                         # deep drawdown → redeem (dd ≤ 0)
        - 0.008 * np.log1p(tenure)        # long tenure slightly reduces churn
        - 0.6  * news_s                    # negative sentiment → redeem
        - 0.6  * regime                    # risk-off → redeem
        + np.random.default_rng(config.RANDOM_SEED + 2)
              .normal(0, 0.3, size=len(df))  # irreducible noise
    )
    return sigmoid(logit)


# ---------------------------------------------------------------------------
# Forward exposure change helper
# ---------------------------------------------------------------------------

def _forward_exposure_change(
    df: pd.DataFrame,
    horizon: int = config.LABEL_HORIZON_MONTHS,
) -> pd.Series:
    """Compute the relative exposure change over the next `horizon` months.

    For each (client, date) row this is:
        (exposure_t+horizon − exposure_t) / exposure_t

    Rows within `horizon` months of the end of the panel cannot have a valid
    forward change; they are set to NaN and will be excluded from labelling.
    """
    df = df.sort_values(["client_id", "date"])
    # Shift exposure backward by `horizon` positions within each client
    future_exp = (
        df.groupby("client_id")["exposure"]
        .shift(-horizon)
    )
    current_exp = df["exposure"].replace(0, np.nan)
    rel_change = (future_exp - current_exp) / current_exp
    return rel_change


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_labels(
    df: pd.DataFrame,
    horizon: int = config.LABEL_HORIZON_MONTHS,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Add buy_3m and redeem_3m labels to the panel DataFrame.

    Steps
    -----
    1. Compute forward exposure change (proxy for actual buy/redeem events).
    2. Compute latent buy and redeem probabilities from features.
    3. Sample Bernoulli draws conditioned on both the latent prob and the
       direction of the forward exposure change, adding structured noise.
    4. Resolve conflicts: if both buy and redeem would be 1, keep whichever
       has the higher latent probability.
    5. Drop rows where forward change is NaN (end-of-panel boundary).

    Parameters
    ----------
    df      : panel DataFrame from synthetic_data.generate_panel()
    horizon : label horizon in months (default = 3)
    seed    : random seed

    Returns
    -------
    DataFrame with two new boolean integer columns: buy_3m, redeem_3m.
    Rows at the boundary of the panel (where forward data is unavailable)
    are removed.
    """
    rng = np.random.default_rng(seed + 99)
    df = df.sort_values(["client_id", "date"]).copy()

    # Step 1 — forward exposure change
    rel_change = _forward_exposure_change(df, horizon)

    # Step 2 — latent probabilities
    p_buy    = _latent_buy_prob(df)
    p_redeem = _latent_redeem_prob(df)

    # Step 3 — label generation
    # Buy: latent prob is amplified when the client's exposure actually grows
    buy_amplifier = np.where(
        rel_change.values > THRESHOLD_BUY, 1.4, 1.0
    )
    p_buy_adj = np.clip(p_buy * buy_amplifier, 0, 1)

    redeem_amplifier = np.where(
        rel_change.values < -THRESHOLD_REDEEM, 1.4, 1.0
    )
    p_redeem_adj = np.clip(p_redeem * redeem_amplifier, 0, 1)

    # Bernoulli samples
    buy_raw    = (rng.random(size=len(df)) < p_buy_adj).astype(int)
    redeem_raw = (rng.random(size=len(df)) < p_redeem_adj).astype(int)

    # Step 4 — resolve conflicts (cannot buy and redeem simultaneously)
    conflict = (buy_raw == 1) & (redeem_raw == 1)
    buy_wins  = p_buy_adj >= p_redeem_adj
    buy_raw[conflict & ~buy_wins]  = 0
    redeem_raw[conflict & buy_wins] = 0

    df["buy_3m"]    = buy_raw
    df["redeem_3m"] = redeem_raw
    df["_rel_change"] = rel_change.values  # kept for diagnostics, prefixed _

    # Step 5 — drop boundary rows
    df = df.dropna(subset=["_rel_change"]).copy()
    df.drop(columns=["_rel_change"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    buy_rate    = df["buy_3m"].mean()
    redeem_rate = df["redeem_3m"].mean()
    logger.info(
        "Labels attached: %d rows | buy_rate=%.1f%% | redeem_rate=%.1f%%",
        n, 100 * buy_rate, 100 * redeem_rate
    )
    return df
