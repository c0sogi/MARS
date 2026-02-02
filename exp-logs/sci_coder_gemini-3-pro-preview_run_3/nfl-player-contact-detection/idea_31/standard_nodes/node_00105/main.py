import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef
import logging

# Import from provided libraries
from library.config import Config
from library.modeling import DualStreamModel
from library.utils import setup_logger


def main():
    # 1. Configuration Overrides
    # Use full dataset and default model complexity (Cite Lesson 00104: Hierarchy of Feature Convergence)
    # We rely on the defaults in Config (DEBUG_SAMPLE_SIZE=None, n_estimators=2000)
    pass

    # Setup Logger
    logger = setup_logger("RunFile")
    logger.info("Starting Fast Baseline Run...")

    # 2. Train Models
    # The DualStreamModel class handles data loading, feature engineering (via pipeline),
    # training, and threshold optimization.
    model = DualStreamModel()
    model.train()

    # 3. Global Validation Assessment
    logger.info("Performing Global Validation...")

    # We need to reconstruct the global validation set to calculate the combined MCC
    # Stream A Validation
    X_val_a, y_val_a, ids_a = model.pipeline.load_data(
        mode="validation", stream="streamA"
    )
    dval_a = xgb.DMatrix(X_val_a)
    probs_a = model.model_a.predict(dval_a)
    preds_a = (probs_a >= model.threshold_a).astype(int)

    # Stream B Validation
    X_val_b, y_val_b, ids_b = model.pipeline.load_data(
        mode="validation", stream="streamB"
    )
    dval_b = xgb.DMatrix(X_val_b)
    probs_b = model.model_b.predict(dval_b)
    preds_b = (probs_b >= model.threshold_b).astype(int)

    # Combine Predictions
    # Create a map of contact_id -> prediction
    pred_map = {}
    for cid, pred in zip(ids_a, preds_a):
        pred_map[cid] = pred
    for cid, pred in zip(ids_b, preds_b):
        pred_map[cid] = pred

    # Load Ground Truth (Validation Metadata)
    df_val_meta = pd.read_csv(Config.VAL_META_PATH)

    # Filter metadata to those present in our validation subset (if DEBUG_SAMPLE_SIZE was applied)
    # Note: load_data might have returned a subset if DEBUG_SAMPLE_SIZE is set.
    # We need to ensure we align predictions with ground truth.
    # The ids returned by load_data correspond to the X_val rows.

    combined_ids = np.concatenate([ids_a, ids_b])
    combined_preds = np.concatenate([preds_a, preds_b])
    combined_truth = np.concatenate([y_val_a, y_val_b])

    # Calculate Global MCC
    # We calculate it based on the union of Stream A and Stream B validation sets processed
    final_mcc = matthews_corrcoef(combined_truth, combined_preds)

    print(f"Final Validation Metric: {final_mcc}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    def analyze_errors(X, y_true, y_prob, stream_name):
        # Calculate residuals (error magnitude)
        residuals = np.abs(y_true - y_prob)

        # Compute correlation between features and residuals
        correlations = {}
        for col in X.columns:
            # Simple correlation
            try:
                corr = np.corrcoef(X[col], residuals)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                pass

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        logger.info(
            f"--- {stream_name} Failure Analysis (Top 5 Correlated Features with Error) ---"
        )
        for feature, corr in sorted_corr[:5]:
            logger.info(f"{feature}: {corr:.4f}")

    analyze_errors(X_val_a, y_val_a, probs_a, "Stream A")
    analyze_errors(X_val_b, y_val_b, probs_b, "Stream B")

    # 5. Submission Generation
    # Only submit if performance is adequate
    if final_mcc > 0.7008:
        logger.info("Validation metric satisfactory. Generating submission...")

        # We need to reset the Config.DEBUG_SAMPLE_SIZE to None or handle test data correctly.
        # However, the pipeline's load_data uses DEBUG_SAMPLE_SIZE mainly for training/val.
        # For test, we process the full set usually.
        # But Config.DEBUG_SAMPLE_SIZE is global. Let's ensure we don't truncate test data if possible,
        # or rely on the fact that for inference we usually want full predictions.
        # In the provided library code:
        # load_data -> process_stream -> _load_metadata
        # In process_stream: "if Config.DEBUG_SAMPLE_SIZE and mode == 'train': df_meta = df_meta.head(...)"
        # So test data is NOT truncated by DEBUG_SAMPLE_SIZE. Safe to proceed.

        model.predict()
    else:
        logger.warning(f"Validation metric {final_mcc} <= 0.7008. Submission skipped.")


if __name__ == "__main__":
    main()
