"""
run_prototype.py
================
End-to-end execution script for the Behavioral Master Signal prototype.

Run this script to:
1. Generate synthetic data
2. Attach labels
3. Build sub-signals and heuristic scores
4. Train ML models (logistic + GBT) on a time-aware split
5. Evaluate all models
6. Compute explainability outputs
7. Generate and save all visualisations

Usage
-----
    python run_prototype.py

All outputs are written to:
    data/synthetic/panel.parquet  — synthetic panel dataset
    outputs/figures/              — PNG charts
    outputs/models/               — serialised models (.pkl)

IMPORTANT DISCLAIMER
--------------------
This prototype runs entirely on synthetic data generated under controlled
assumptions.  It validates the signal design and modelling pipeline but
does NOT claim real-world predictive validity before access to production data.
"""

import os
import sys
import time

# Ensure the repo root is on the Python path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.utils import get_logger
from src.synthetic_data import generate_panel, save_panel
from src.labels import attach_labels
from src.features import build_sub_signals, build_model_features
from src.scoring import attach_heuristic_scores
from src.modeling import train_models, save_models
from src.evaluation import evaluate_all
from src.explainability import run_explainability
from src.plotting import (
    plot_signal_distributions,
    plot_model_performance,
    plot_lift_curves,
    plot_feature_importance,
    plot_client_timeline,
    plot_score_distributions,
)

logger = get_logger()


def main() -> None:
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  Behavioral Master Signal — Prototype Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    logger.info("\n[1/7] Generating synthetic panel …")
    panel = generate_panel()
    save_panel(panel)

    # ------------------------------------------------------------------
    # Step 2: Attach labels
    # ------------------------------------------------------------------
    logger.info("\n[2/7] Attaching labels (buy_3m, redeem_3m) …")
    panel = attach_labels(panel)

    # ------------------------------------------------------------------
    # Step 3: Build sub-signals and heuristic scores
    # ------------------------------------------------------------------
    logger.info("\n[3/7] Building sub-signals and heuristic scores …")
    panel = build_sub_signals(panel)
    panel = attach_heuristic_scores(panel)

    # ------------------------------------------------------------------
    # Step 4: Build ML feature matrix
    # ------------------------------------------------------------------
    logger.info("\n[4/7] Training ML models …")
    X = build_model_features(panel)
    feature_cols = X.columns.tolist()

    # Attach feature matrix back to panel (aligned by index)
    panel_with_features = panel.copy()
    for col in feature_cols:
        if col not in panel_with_features.columns:
            panel_with_features[col] = X[col]

    models = train_models(panel_with_features, feature_cols)
    save_models(models)

    # ------------------------------------------------------------------
    # Step 5: Evaluate
    # ------------------------------------------------------------------
    logger.info("\n[5/7] Evaluating models …")
    results_df = evaluate_all(models, panel_with_features)

    logger.info("\n--- Evaluation Results ---")
    print(results_df.to_string(index=False, float_format="{:.3f}".format))

    # Save results table
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    results_path = os.path.join(config.OUTPUTS_DIR, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved → %s", results_path)

    # ------------------------------------------------------------------
    # Step 6: Explainability
    # ------------------------------------------------------------------
    logger.info("\n[6/7] Computing explainability outputs …")
    explain_results = run_explainability(models, panel_with_features)

    logger.info("\n--- Example Client Narratives ---")
    for label in ["buy_3m", "redeem_3m"]:
        logger.info("Label: %s", label)
        for key, narr in explain_results[label]["example_narratives"].items():
            logger.info("  [%s] %s", key, narr)

    # ------------------------------------------------------------------
    # Step 7: Generate plots
    # ------------------------------------------------------------------
    logger.info("\n[7/7] Generating visualisations …")
    plot_signal_distributions(panel)
    plot_score_distributions(panel)
    plot_model_performance(results_df)
    plot_lift_curves(models, panel_with_features)
    plot_feature_importance(explain_results)
    plot_client_timeline(panel)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline complete in %.1f seconds.", elapsed)
    logger.info("  Outputs:")
    logger.info("    data/synthetic/panel.parquet")
    logger.info("    outputs/evaluation_results.csv")
    logger.info("    outputs/figures/*.png  (6 figures)")
    logger.info("    outputs/models/*.pkl   (4 model files)")
    logger.info("=" * 60)
    logger.info("\nDISCLAIMER: This is a METHODOLOGY PROTOTYPE on synthetic data.")
    logger.info("It does NOT claim real-world predictive validity.")


if __name__ == "__main__":
    main()
