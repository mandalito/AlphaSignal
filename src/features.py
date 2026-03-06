"""
features.py
===========
Builds the four interpretable sub-signals that compose the Behavioral Master
Signal framework.

Sub-signal design
-----------------
Each sub-signal is:
1. Constructed from a transparent formula over raw predictors.
2. Scaled to [0, 1] so that signals are comparable and combinable.
3. Documented with a business interpretation.

The four sub-signals:

1. tenure_perf_signal
   Captures how well-established and commercially engaged the client is.
   High score → mature tenure, good recent performance, low drawdown, stable exposure.

2. external_flows_signal
   Captures the momentum of category-level investor flows.
   High score → strong positive flows, accelerating, persistent.

3. news_signal
   Captures the attention environment around the product category.
   High score → elevated, positive-sentiment media attention.

4. launch_signal
   Captures the intensity of new product activity in the ecosystem.
   High score → high recent launch count and accelerating pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, minmax_scale_series, zscore_series

logger = get_logger()


# ---------------------------------------------------------------------------
# Sub-signal 1: Tenure & Performance
# ---------------------------------------------------------------------------

def compute_tenure_perf_signal(df: pd.DataFrame) -> pd.Series:
    """Build the tenure + performance composite sub-signal.

    Formula components (each scaled to [0,1] individually before combining):
    - tenure_score    : log-tenure, normalised  → proxy for relationship maturity
    - perf3_score     : 3-month performance     → short-term momentum
    - perf12_score    : 12-month performance    → long-term return quality
    - drawdown_score  : 1 + drawdown (≤1)       → closeness to high-water mark
    - stability_score : exposure_stability      → client commitment / stickiness

    Weighted combination:
        0.25 * tenure  +  0.25 * perf3  +  0.20 * perf12
        +  0.20 * drawdown  +  0.10 * stability

    Business interpretation:
        High score → a client with a long, profitable, stable relationship —
        the type most likely to increase allocation.
        Low score  → a new or underperforming client under drawdown stress.
    """
    tenure_score    = minmax_scale_series(np.log1p(df["tenure_months"]))
    perf3_score     = minmax_scale_series(df["performance_3m"])
    perf12_score    = minmax_scale_series(df["performance_12m"])
    # drawdown ≤ 0; (1 + drawdown) ∈ [0,1]; already bounded but scale for safety
    drawdown_score  = minmax_scale_series(1.0 + df["drawdown"])
    stability_score = minmax_scale_series(df["exposure_stability"])

    signal = (
        0.25 * tenure_score
        + 0.25 * perf3_score
        + 0.20 * perf12_score
        + 0.20 * drawdown_score
        + 0.10 * stability_score
    )
    return minmax_scale_series(signal).rename("tenure_perf_signal")


# ---------------------------------------------------------------------------
# Sub-signal 2: External Flows
# ---------------------------------------------------------------------------

def compute_external_flows_signal(df: pd.DataFrame) -> pd.Series:
    """Build the external category-flows sub-signal.

    Formula components:
    - flow1_score         : 1-month category flow    → immediate demand pulse
    - flow3_score         : 3-month cumulative flow  → trend strength
    - acceleration_score  : flow acceleration        → momentum change
    - persistence_score   : flow persistence (−1/+1) → directional consistency

    Weighted combination:
        0.35 * flow1  +  0.35 * flow3  +  0.20 * acceleration
        +  0.10 * persistence

    Business interpretation:
        High score → sustained positive inflows to the category — a supportive
        macro backdrop for buy decisions.
        Low score  → outflows or decelerating flows — potential redemption trigger.
    """
    flow1_score        = minmax_scale_series(df["category_flow_1m"])
    flow3_score        = minmax_scale_series(df["category_flow_3m"])
    acceleration_score = minmax_scale_series(df["flow_acceleration"])
    # persistence is already in {-1, 0, 1}; scale to [0,1]
    persistence_score  = minmax_scale_series(df["flow_persistence"])

    signal = (
        0.35 * flow1_score
        + 0.35 * flow3_score
        + 0.20 * acceleration_score
        + 0.10 * persistence_score
    )
    return minmax_scale_series(signal).rename("external_flows_signal")


# ---------------------------------------------------------------------------
# Sub-signal 3: News / Attention
# ---------------------------------------------------------------------------

def compute_news_signal(df: pd.DataFrame) -> pd.Series:
    """Build the news / attention sub-signal.

    Formula components:
    - volume_score   : normalised news volume   → attention level
    - sentiment_score: normalised news sentiment → valence of coverage
    - burst_score    : clipped burst z-score    → abnormal media spike

    Weighted combination:
        0.30 * volume  +  0.50 * sentiment  +  0.20 * burst

    Sentiment dominates because valence (positive/negative) is the primary
    driver of client behaviour, not raw volume.

    Business interpretation:
        High score → positive, elevated media attention — can amplify both
        buy interest and brand confidence.
        Low score  → negative or absent media coverage — may increase anxiety
        and redemption risk.
    """
    volume_score    = minmax_scale_series(df["news_volume"])
    sentiment_score = minmax_scale_series(df["news_sentiment"])
    # Clip burst z-score to [-3, 3] to limit outlier influence
    burst_clipped   = df["news_burst_zscore"].clip(-3, 3)
    burst_score     = minmax_scale_series(burst_clipped)

    signal = (
        0.30 * volume_score
        + 0.50 * sentiment_score
        + 0.20 * burst_score
    )
    return minmax_scale_series(signal).rename("news_signal")


# ---------------------------------------------------------------------------
# Sub-signal 4: Product Launch / Strategy
# ---------------------------------------------------------------------------

def compute_launch_signal(df: pd.DataFrame) -> pd.Series:
    """Build the product launch / strategy sub-signal.

    Formula components:
    - count_score        : recent launch count     → pipeline activity level
    - acceleration_score : launch acceleration     → speed of pipeline growth
    - intensity_score    : launch intensity ratio  → relative pace vs. history

    Weighted combination:
        0.40 * count  +  0.30 * acceleration  +  0.30 * intensity

    Business interpretation:
        High score → active product pipeline with accelerating launches —
        indicates commercial momentum and increases the probability that
        existing clients will find new allocation opportunities.
        Low score  → stagnant pipeline — reduces the commercial stimulus.
    """
    count_score        = minmax_scale_series(df["launch_count_recent"])
    acceleration_score = minmax_scale_series(df["launch_acceleration"])
    intensity_score    = minmax_scale_series(df["launch_intensity"])

    signal = (
        0.40 * count_score
        + 0.30 * acceleration_score
        + 0.30 * intensity_score
    )
    return minmax_scale_series(signal).rename("launch_signal")


# ---------------------------------------------------------------------------
# Public API: build all sub-signals at once
# ---------------------------------------------------------------------------

def build_sub_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all four sub-signals and attach them to the panel.

    Parameters
    ----------
    df : panel DataFrame from synthetic_data + labels pipeline

    Returns
    -------
    DataFrame with four new columns:
        tenure_perf_signal, external_flows_signal, news_signal, launch_signal
    """
    df = df.copy()
    df["tenure_perf_signal"]     = compute_tenure_perf_signal(df)
    df["external_flows_signal"]  = compute_external_flows_signal(df)
    df["news_signal"]            = compute_news_signal(df)
    df["launch_signal"]          = compute_launch_signal(df)

    logger.info(
        "Sub-signals built | means: "
        "tenure_perf=%.3f  flows=%.3f  news=%.3f  launch=%.3f",
        df["tenure_perf_signal"].mean(),
        df["external_flows_signal"].mean(),
        df["news_signal"].mean(),
        df["launch_signal"].mean(),
    )
    return df


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full feature matrix for ML models.

    Combines raw engineered features with the four sub-signals, and adds a
    one-hot encoding of the client segment.

    Returns a DataFrame aligned with `df`, containing only numeric predictors.
    """
    feature_cols = (
        # Raw predictors
        ["tenure_months", "exposure", "exposure_stability",
         "performance_3m", "performance_12m", "drawdown",
         "category_flow_1m", "category_flow_3m",
         "flow_acceleration", "flow_persistence",
         "news_volume", "news_sentiment", "news_burst_zscore",
         "launch_count_recent", "launch_acceleration", "launch_intensity",
         "market_regime"]
        # Sub-signals
        + ["tenure_perf_signal", "external_flows_signal",
           "news_signal", "launch_signal"]
    )

    X = df[feature_cols].copy()

    # One-hot encode segment (drop first to avoid dummy trap)
    seg_dummies = pd.get_dummies(df["segment"], prefix="seg", drop_first=True)
    X = pd.concat([X, seg_dummies], axis=1)

    return X
