"""
config.py
=========
Central configuration for the Behavioral Master Signal prototype.

All seeds, data-generation parameters, column names, and modelling
hyper-parameters live here so that nothing is hard-coded in individual modules.
"""

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Synthetic dataset dimensions
# ---------------------------------------------------------------------------
N_CLIENTS: int = 300          # number of simulated client accounts
START_DATE: str = "2020-01-01"
END_DATE: str = "2023-12-01"  # inclusive; monthly frequency → ~48 periods
FREQ: str = "MS"               # month-start frequency (pandas offset alias)

# ---------------------------------------------------------------------------
# Client segment definitions
# ---------------------------------------------------------------------------
SEGMENTS: list[str] = ["institutional", "wealth", "retail"]
SEGMENT_WEIGHTS: list[float] = [0.30, 0.45, 0.25]  # sampling probabilities

# ---------------------------------------------------------------------------
# Label horizon
# ---------------------------------------------------------------------------
LABEL_HORIZON_MONTHS: int = 3   # predict activity within next 3 months

# ---------------------------------------------------------------------------
# Train / validation / test split cutoffs (ISO date strings)
# NOTE: these are as-of dates for the *feature snapshot*, not the label.
# ---------------------------------------------------------------------------
TRAIN_END: str = "2022-06-01"
VALID_END: str = "2023-03-01"
# everything after VALID_END up to END_DATE − LABEL_HORIZON_MONTHS is test

# ---------------------------------------------------------------------------
# Heuristic master-score weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
HEURISTIC_BUY_WEIGHTS: dict[str, float] = {
    "tenure_perf_signal": 0.35,
    "external_flows_signal": 0.30,
    "news_signal": 0.20,
    "launch_signal": 0.15,
}

HEURISTIC_REDEEM_WEIGHTS: dict[str, float] = {
    "tenure_perf_signal": -0.30,   # negative: strong tenure → lower redeem risk
    "external_flows_signal": -0.35,
    "news_signal": 0.15,           # high news attention can amplify outflows
    "launch_signal": -0.20,        # new product availability lowers exit incentive
}

# ---------------------------------------------------------------------------
# Feature column groups
# ---------------------------------------------------------------------------
RAW_CLIENT_COLS: list[str] = [
    "tenure_months",
    "segment",
    "exposure",
    "performance_3m",
    "performance_12m",
    "drawdown",
    "exposure_stability",
]

RAW_FLOW_COLS: list[str] = [
    "category_flow_1m",
    "category_flow_3m",
    "flow_acceleration",
    "flow_persistence",
]

RAW_NEWS_COLS: list[str] = [
    "news_volume",
    "news_sentiment",
    "news_burst_zscore",
]

RAW_LAUNCH_COLS: list[str] = [
    "launch_count_recent",
    "launch_acceleration",
    "launch_intensity",
]

SUB_SIGNAL_COLS: list[str] = [
    "tenure_perf_signal",
    "external_flows_signal",
    "news_signal",
    "launch_signal",
]

LABEL_COLS: list[str] = ["buy_3m", "redeem_3m"]

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
XGBOOST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

LOGISTIC_PARAMS: dict = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "C": 1.0,
    "solver": "lbfgs",
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
SYNTHETIC_DATA_PATH: str = "data/synthetic/panel.parquet"
OUTPUTS_DIR: str = "outputs"
FIGURES_DIR: str = "outputs/figures"
MODELS_DIR: str = "outputs/models"
