"""
explainability.py
=================
Interpretability outputs for the Behavioral Master Signal prototype.

Three layers of explainability:

1. Feature importance (built-in)
   Available for tree-based models via `.feature_importances_`.
   Summarises which features the model relies on most at the global level.

2. SHAP values (preferred when available)
   SHAP (SHapley Additive exPlanations) provides model-agnostic, theoretically
   grounded feature attributions.  We use TreeExplainer for GBT and
   LinearExplainer for logistic regression.
   Falls back gracefully to permutation importance if SHAP is not installed.

3. Plain-language client narrative
   For selected example observations, generates a human-readable sentence
   summarising the top drivers of the predicted score.

All outputs are deterministic given the same inputs and seed.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# SHAP availability guard
# ---------------------------------------------------------------------------

def _shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_tree_feature_importance(
    model: Any,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract built-in feature importances from a tree model.

    Works with XGBoost, LightGBM, HistGradientBoosting and any model that
    exposes `.feature_importances_`.

    Parameters
    ----------
    model        : trained tree model (not a Pipeline)
    feature_cols : ordered list of feature names
    top_n        : number of top features to return

    Returns
    -------
    DataFrame with columns [feature, importance], sorted descending.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        logger.warning("Model does not expose feature_importances_; skipping.")
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "gbt",
    max_rows: int = 500,
) -> Optional[np.ndarray]:
    """Compute SHAP values for a sample of observations.

    Parameters
    ----------
    model      : trained model (GBT or logistic pipeline)
    X          : feature DataFrame
    model_type : "gbt" or "logistic"
    max_rows   : cap sample size to keep computation fast

    Returns
    -------
    2-D numpy array of shape (n_rows, n_features) or None if SHAP unavailable.
    """
    if not _shap_available():
        logger.warning("SHAP not installed; falling back to permutation importance.")
        return None

    import shap

    X_sample = X.iloc[:max_rows].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if model_type == "gbt":
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_sample)
                # XGBoost binary: shap_values returns array of shape (n, p)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # positive class
            else:
                # Logistic pipeline: extract the scaler and clf
                from sklearn.pipeline import Pipeline
                if isinstance(model, Pipeline):
                    scaler = model.named_steps["scaler"]
                    clf    = model.named_steps["clf"]
                    X_scaled = pd.DataFrame(
                        scaler.transform(X_sample),
                        columns=X_sample.columns,
                    )
                else:
                    clf = model
                    X_scaled = X_sample
                explainer = shap.LinearExplainer(
                    clf,
                    X_scaled,
                    feature_perturbation="interventional",
                )
                shap_vals = explainer.shap_values(X_scaled)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
        except Exception as exc:
            logger.warning("SHAP computation failed (%s); "
                           "falling back to permutation importance.", exc)
            return None

    return np.asarray(shap_vals)


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    seed: int = 42,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute permutation feature importance as SHAP fallback.

    Parameters
    ----------
    model    : trained model with predict_proba
    X        : feature DataFrame
    y        : binary label Series
    n_repeats: number of permutation rounds
    seed     : random seed
    top_n    : number of top features to return

    Returns
    -------
    DataFrame with columns [feature, importance], sorted descending.
    """
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    class _Wrapper:
        """Wrap predict_proba for ROC-AUC scoring."""
        def __init__(self, m):
            self._m = m
        def predict(self, X):
            return self._m.predict_proba(X)[:, 1]
        def score(self, X, y):
            return roc_auc_score(y, self.predict(X))

    result = permutation_importance(
        _Wrapper(model), X, y,
        n_repeats=n_repeats, random_state=seed,
        scoring=lambda est, X, y: roc_auc_score(y, est.predict(X)),
    )
    df = pd.DataFrame({
        "feature": X.columns.tolist(),
        "importance": result.importances_mean,
    })
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# SHAP summary at feature level
# ---------------------------------------------------------------------------

def shap_feature_summary(
    shap_values: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Aggregate SHAP values to mean absolute importance per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-observation narrative
# ---------------------------------------------------------------------------

def generate_client_narrative(
    row: pd.Series,
    shap_row: Optional[np.ndarray],
    feature_cols: list[str],
    label: str = "buy_3m",
    predicted_prob: float = 0.0,
    top_k: int = 3,
) -> str:
    """Generate a plain-English explanation for a single observation.

    Uses SHAP values when available; otherwise falls back to raw sub-signal
    values.

    Parameters
    ----------
    row           : a single row of the feature DataFrame
    shap_row      : SHAP values for this row (or None)
    feature_cols  : feature column names
    label         : "buy_3m" or "redeem_3m"
    predicted_prob: the model's predicted probability
    top_k         : number of top drivers to mention

    Returns
    -------
    Human-readable string.
    """
    action = "buy propensity" if label == "buy_3m" else "redemption risk"
    level  = "elevated" if predicted_prob > 0.5 else "moderate"

    # Identify top drivers
    if shap_row is not None:
        top_idx = np.argsort(np.abs(shap_row))[::-1][:top_k]
        drivers: List[str] = []
        for i in top_idx:
            feat_name = feature_cols[i]
            direction = "high" if shap_row[i] > 0 else "low"
            drivers.append(f"{direction} {feat_name.replace('_', ' ')}")
    else:
        # Fallback: use sub-signal absolute values
        sub_signals = ["tenure_perf_signal", "external_flows_signal",
                       "news_signal", "launch_signal"]
        available   = [s for s in sub_signals if s in feature_cols]
        vals        = row[available].values if hasattr(row, "__getitem__") else []
        if len(vals) > 0:
            top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
            drivers = [
                # Threshold of 0.5 is the natural midpoint of the [0,1] signal
                # scale applied by minmax_scale_series in features.py.
                f"{'high' if vals[i] > 0.5 else 'low'} "
                f"{available[i].replace('_', ' ')}"
                for i in top_idx
            ]
        else:
            drivers = ["insufficient signal data"]

    drivers_str = ", ".join(drivers)
    narrative = (
        f"{action.capitalize()} is {level} "
        f"(predicted probability: {predicted_prob:.1%}). "
        f"Primary drivers: {drivers_str}."
    )
    return narrative


# ---------------------------------------------------------------------------
# Full explainability pipeline
# ---------------------------------------------------------------------------

def run_explainability(
    models: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    n_shap_rows: int = 300,
) -> Dict[str, Any]:
    """Run the full explainability pipeline for both labels.

    Returns a dictionary of explainability artefacts used by plotting.py.
    """
    from src.utils import time_split

    _, _, test_df = time_split(df)

    results: Dict[str, Any] = {}

    for label in ["buy_3m", "redeem_3m"]:
        bundle       = models[label]
        feat         = bundle["feature_cols"]
        gbt_model    = bundle["gbt"]
        lr_model     = bundle["logistic"]
        X_test       = test_df[feat].astype(float)
        y_test       = test_df[label].astype(int)

        # --- Tree feature importance ---
        tree_imp = get_tree_feature_importance(gbt_model, feat)

        # --- SHAP or permutation ---
        shap_vals = compute_shap_values(
            gbt_model, X_test, model_type="gbt", max_rows=n_shap_rows
        )

        if shap_vals is not None:
            feat_imp_df = shap_feature_summary(shap_vals, feat)
            logger.info("[%s] SHAP values computed for %d rows.", label, len(shap_vals))
        else:
            feat_imp_df = compute_permutation_importance(
                gbt_model, X_test, y_test, top_n=20
            )
            logger.info("[%s] Permutation importance computed.", label)

        # --- Example narratives ---
        # Use only the SHAP-sampled subset so indices are consistent
        X_narr     = X_test.iloc[:n_shap_rows]
        sv_narr    = shap_vals  # may be None
        narr_probs = gbt_model.predict_proba(X_narr)[:, 1]

        sorted_idx = np.argsort(narr_probs)
        example_indices = {
            "high_score":   int(sorted_idx[-1]),
            "medium_score": int(sorted_idx[len(sorted_idx) // 2]),
            "low_score":    int(sorted_idx[0]),
        }

        narratives: Dict[str, str] = {}
        for key, idx in example_indices.items():
            row      = X_narr.iloc[idx]
            shap_row = sv_narr[idx] if sv_narr is not None else None
            prob     = float(narr_probs[idx])
            narratives[key] = generate_client_narrative(
                row, shap_row, feat, label, prob
            )

        results[label] = {
            "tree_feature_importance": tree_imp,
            "shap_feature_importance": feat_imp_df,
            "shap_values": shap_vals,
            "X_shap": X_test.iloc[:n_shap_rows],
            "example_narratives": narratives,
        }

        # Log example narratives
        logger.info("[%s] Example narratives:", label)
        for key, narr in narratives.items():
            logger.info("  [%s] %s", key, narr)

    return results
