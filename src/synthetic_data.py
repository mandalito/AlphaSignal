"""
synthetic_data.py
=================
Generates a structurally realistic (but entirely synthetic) client × date panel
for the Behavioral Master Signal prototype.

Design principles
-----------------
- Labels are derived from latent probabilities, **not** from the labels themselves.
  Pattern: features → latent_prob → sampled_label
- Controlled noise ensures the problem is neither trivially easy nor random.
- All randomness is seeded for full reproducibility.
- The data represents a monthly, cross-sectional panel of fund/investment clients.

Columns generated
-----------------
Identifiers:
  client_id, date, segment

Client / performance features:
  tenure_months, exposure, performance_3m, performance_12m,
  drawdown, exposure_stability, market_regime

Category-level external flow features:
  category_flow_1m, category_flow_3m, flow_acceleration, flow_persistence

News / attention features:
  news_volume, news_sentiment, news_burst_zscore

Product-launch features:
  launch_count_recent, launch_acceleration, launch_intensity

Labels (see labels.py for generation logic):
  buy_3m, redeem_3m
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger, sigmoid

logger = get_logger()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _simulate_clients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create static per-client attributes sampled once at inception."""
    segments = rng.choice(
        config.SEGMENTS, size=n, p=config.SEGMENT_WEIGHTS
    )
    # Institutional clients tend to have longer tenure; retail shorter
    base_tenure = {"institutional": 36, "wealth": 24, "retail": 12}
    base_arr = np.array([base_tenure[s] for s in segments], dtype=float)
    tenure_at_start = np.maximum(1, base_arr + rng.normal(0, 8, size=n))

    # Segment-level baseline exposure (log-normal, in M USD)
    base_exposure = {"institutional": 50.0, "wealth": 10.0, "retail": 2.0}
    exp_arr = np.array([base_exposure[s] for s in segments], dtype=float)
    log_exposure = np.log(exp_arr) + rng.normal(0, 0.4, size=n)

    return pd.DataFrame({
        "client_id": [f"C{i:04d}" for i in range(n)],
        "segment": segments,
        "tenure_at_start": tenure_at_start,
        "log_base_exposure": log_exposure,
    })


def _simulate_market_regime(dates: pd.DatetimeIndex,
                             rng: np.random.Generator) -> pd.Series:
    """Simulate a slow-moving market regime: 1 = risk-on, 0 = risk-off.

    Uses a simple Markov chain: high persistence so regimes last several months.
    """
    n = len(dates)
    regime = np.zeros(n, dtype=int)
    regime[0] = rng.integers(0, 2)
    # Transition probabilities: stay in current regime with high probability
    p_stay = 0.85
    for t in range(1, n):
        if rng.random() < p_stay:
            regime[t] = regime[t - 1]
        else:
            regime[t] = 1 - regime[t - 1]
    return pd.Series(regime, index=dates, name="market_regime")


def _add_time_varying_features(
    client_row: pd.Series,
    dates: pd.DatetimeIndex,
    market_regime: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate time-varying features for a single client across all dates."""
    n = len(dates)

    # ---- Tenure --------------------------------------------------------
    tenure = client_row["tenure_at_start"] + np.arange(n)  # grows by 1/month

    # ---- Exposure (log-normal walk) ------------------------------------
    base_exp = np.exp(client_row["log_base_exposure"])
    log_exp = np.log(base_exp) + np.cumsum(rng.normal(0.002, 0.05, size=n))
    exposure = np.exp(log_exp)

    # Exposure stability: rolling std over last 3 months (approx. noise term)
    exp_noise = np.abs(rng.normal(0, 0.03, size=n))
    exposure_stability = 1.0 / (1.0 + exp_noise * 10)  # high → stable

    # ---- Performance ---------------------------------------------------
    # Underlying return: correlated with regime + idio noise
    regime_arr = market_regime.values
    monthly_return = (
        0.005 * regime_arr           # regime-linked drift
        - 0.003 * (1 - regime_arr)   # mild drag in risk-off
        + rng.normal(0, 0.025, size=n)
    )
    cumret = np.cumprod(1 + monthly_return)

    def rolling_ret(window: int) -> np.ndarray:
        out = np.full(n, np.nan)
        for i in range(window - 1, n):
            out[i] = cumret[i] / cumret[i - window + 1] - 1
        return out

    performance_3m = rolling_ret(3)
    performance_12m = rolling_ret(12)

    # Forward-fill NaN at start with 0
    performance_3m = np.where(np.isnan(performance_3m), 0, performance_3m)
    performance_12m = np.where(np.isnan(performance_12m), 0, performance_12m)

    # ---- Drawdown (negative = drawdown) --------------------------------
    rolling_max = np.maximum.accumulate(cumret)
    drawdown = cumret / rolling_max - 1.0  # ≤ 0

    # ---- Category external flows (market-level, same across clients in same
    #       segment, but with client-specific noise) ----------------------
    market_flow_raw = (
        rng.normal(0, 1, size=n)
        + 1.5 * regime_arr          # stronger inflows in risk-on
        - 0.8 * (1 - regime_arr)
    )
    # Small idio component per client
    client_flow_noise = rng.normal(0, 0.3, size=n)
    category_flow_1m = market_flow_raw + client_flow_noise

    # 3-month cumulative flow (rolling sum over 3 months)
    category_flow_3m = np.convolve(category_flow_1m,
                                   np.ones(3) / 3, mode="same")

    # Flow acceleration: diff of 1m flow
    flow_acceleration = np.diff(category_flow_1m, prepend=category_flow_1m[0])
    # Flow persistence: sign of flow × sign of last period flow
    flow_persistence = np.sign(category_flow_1m) * np.sign(
        np.roll(category_flow_1m, 1)
    )
    flow_persistence[0] = 0.0

    # ---- News ----------------------------------------------------------
    news_volume = np.maximum(0, rng.normal(5, 2, size=n))
    news_sentiment = rng.normal(0.1 * regime_arr, 0.3)

    # News burst: z-score of news volume over a rolling window (approx.)
    news_vol_roll_mean = np.convolve(news_volume, np.ones(6) / 6, mode="same")
    news_vol_roll_std = np.maximum(0.1, np.abs(rng.normal(2, 0.5, size=n)))
    news_burst_zscore = (news_volume - news_vol_roll_mean) / news_vol_roll_std

    # ---- Product launches (market-level signal) ------------------------
    # Launches are clustered, e.g. around regime transitions
    launch_base = np.maximum(0, rng.poisson(2, size=n).astype(float))
    # More launches in risk-on
    launch_count_recent = launch_base + regime_arr * rng.poisson(1, size=n)
    launch_acceleration = np.diff(launch_count_recent,
                                  prepend=launch_count_recent[0]).astype(float)
    launch_intensity = launch_count_recent / (
        np.convolve(launch_count_recent, np.ones(6) / 6, mode="same") + 1e-6
    )

    df = pd.DataFrame({
        "client_id": client_row["client_id"],
        "date": dates,
        "segment": client_row["segment"],
        "tenure_months": tenure,
        "exposure": exposure,
        "exposure_stability": exposure_stability,
        "performance_3m": performance_3m,
        "performance_12m": performance_12m,
        "drawdown": drawdown,
        "market_regime": regime_arr,
        "category_flow_1m": category_flow_1m,
        "category_flow_3m": category_flow_3m,
        "flow_acceleration": flow_acceleration,
        "flow_persistence": flow_persistence,
        "news_volume": news_volume,
        "news_sentiment": news_sentiment,
        "news_burst_zscore": news_burst_zscore,
        "launch_count_recent": launch_count_recent,
        "launch_acceleration": launch_acceleration,
        "launch_intensity": launch_intensity,
    })
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_panel(
    n_clients: int = config.N_CLIENTS,
    start_date: str = config.START_DATE,
    end_date: str = config.END_DATE,
    freq: str = config.FREQ,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a synthetic client × date panel dataset.

    Parameters
    ----------
    n_clients  : number of simulated client accounts
    start_date : first observation date (ISO string)
    end_date   : last observation date (ISO string)
    freq       : pandas frequency alias for time index
    seed       : random seed for reproducibility

    Returns
    -------
    pd.DataFrame with shape (n_clients × n_dates, n_features)

    Notes
    -----
    Labels (buy_3m, redeem_3m) are NOT added here — see labels.py.
    This module only generates raw predictors.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_dates = len(dates)

    logger.info("Simulating %d clients × %d months = %d observations …",
                n_clients, n_dates, n_clients * n_dates)

    clients = _simulate_clients(n_clients, rng)
    market_regime = _simulate_market_regime(dates, rng)

    panels: list[pd.DataFrame] = []
    for _, client_row in clients.iterrows():
        # Use a per-client sub-seed so clients are independent but reproducible
        client_seed = seed + int(client_row["client_id"][1:])
        client_rng = np.random.default_rng(client_seed)
        panel = _add_time_varying_features(
            client_row, dates, market_regime, client_rng
        )
        panels.append(panel)

    df = pd.concat(panels, ignore_index=True)
    df.sort_values(["client_id", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Panel generated: %s rows × %s columns", *df.shape)
    return df


def save_panel(df: pd.DataFrame,
               path: str = config.SYNTHETIC_DATA_PATH) -> None:
    """Persist the panel to a Parquet file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Panel saved → %s", path)


def load_panel(path: str = config.SYNTHETIC_DATA_PATH) -> pd.DataFrame:
    """Load panel from a Parquet file."""
    return pd.read_parquet(path)
