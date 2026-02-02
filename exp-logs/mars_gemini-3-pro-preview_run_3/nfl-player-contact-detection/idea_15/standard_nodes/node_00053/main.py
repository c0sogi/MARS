import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.utils import set_seed, setup_logger, compute_mcc
from library.model_engine import DualStreamModel
from library.feature_builder import FeatureBuilder


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)
    logger = setup_logger("RunFile")

    # Adjust configuration for a fast baseline execution within the time limit
    # Reducing estimators and undersampling ratio to speed up training while preserving logic
    Config.XGB_PARAMS_STREAM_A["n_estimators"] = 2000
    Config.XGB_PARAMS_STREAM_B["n_estimators"] = 2000
    Config.UNDERSAMPLE_RATIO = 5.0

    logger.info("Configuration adjusted for fast baseline execution.")

    # 2. Model Training
    logger.info("Initializing and training Dual-Stream Model...")
    model = DualStreamModel()
    model.train()

    # 3. Threshold Optimization
    logger.info("Optimizing decision thresholds based on validation set...")
    model.optimize_thresholds()

    # 4. Global Validation and Metric Calculation
    logger.info("Performing global validation on hold-out set...")
    fb = FeatureBuilder()

    # --- Stream A Validation (Player-Player) ---
    # Load features (uses cache if available)
    X_val_a, y_val_a, _ = fb.generate_stream_a_features("validation", load_cache=True)
    dval_a = xgb.DMatrix(X_val_a)
    # XGBoost inference (automatically uses GPU if model was trained on GPU)
    probs_a = model.model_a.predict(dval_a)
    preds_a = (probs_a >= model.best_threshold_a).astype(int)

    # --- Stream B Validation (Player-Ground) ---
    X_val_b, y_val_b, _ = fb.generate_stream_b_features("validation", load_cache=True)
    dval_b = xgb.DMatrix(X_val_b)
    probs_b = model.model_b.predict(dval_b)
    preds_b = (probs_b >= model.best_threshold_b).astype(int)

    # --- Combine Results ---
    # Stream A (P2 != G) and Stream B (P2 == G) are disjoint partitions of the dataset.
    # We concatenate them to evaluate the global metric.
    y_true_all = np.concatenate([y_val_a, y_val_b])
    y_pred_all = np.concatenate([preds_a, preds_b])

    final_mcc = compute_mcc(y_true_all, y_pred_all)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mcc}")

    # 5. Failure Analysis
    logger.info("Performing Failure Analysis...")

    def analyze_errors(X, y_true, y_probs, stream_name):
        # Calculate error magnitude
        errors = np.abs(y_true - y_probs)

        # Create a temporary DataFrame for correlation analysis
        # X is already float32 and cleaned by FeatureBuilder
        df_analysis = X.copy()
        df_analysis["error_magnitude"] = errors

        # Calculate correlation of features with error magnitude
        correlations = df_analysis.corr()["error_magnitude"].drop("error_magnitude")

        # Sort by absolute correlation to find most impactful features
        top_corrs = correlations.iloc[correlations.abs().argsort()[::-1]].head(5)

        print(f"\n[{stream_name}] Top Feature Correlations with Error Magnitude:")
        print(top_corrs.to_string())

    analyze_errors(X_val_a, y_val_a, probs_a, "Stream A (Interaction)")
    analyze_errors(X_val_b, y_val_b, probs_b, "Stream B (Context-Impact)")

    # 6. Submission Generation
    # Threshold defined in requirements
    SUBMISSION_THRESHOLD = 0.6938871601521127

    if final_mcc > SUBMISSION_THRESHOLD:
        logger.info(
            f"Validation Metric ({final_mcc}) exceeds threshold ({SUBMISSION_THRESHOLD}). Generating submission..."
        )
        model.generate_submission()
    else:
        logger.warning(
            f"Validation Metric ({final_mcc}) did not exceed threshold ({SUBMISSION_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
