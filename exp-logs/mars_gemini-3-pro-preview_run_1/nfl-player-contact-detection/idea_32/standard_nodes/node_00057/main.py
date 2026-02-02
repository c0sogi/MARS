import os
import sys
import numpy as np
import pandas as pd
import logging

# Import from the provided library files
from library.config import Config
from library.utils import setup_logging, seed_everything, calc_mcc
from library.training_flow import TrainingPipeline


def main():
    # =========================================================================
    # 1. Configuration for Fast Baseline
    # =========================================================================
    # Modify Config globally to ensure fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50000  # Limit samples for speed

    # Reduce model complexity for baseline speed
    Config.LGBM_PARAMS["n_estimators"] = 100
    Config.XGB_PARAMS["n_estimators"] = 100
    Config.CAT_PARAMS["iterations"] = 100

    # Setup environment
    setup_logging(level=logging.INFO)
    seed_everything(Config.SEED)

    logging.info("Starting runfile.py execution...")
    logging.info(f"Debug Mode: {Config.DEBUG} (Samples: {Config.DEBUG_SAMPLE_SIZE})")

    # Initialize the orchestration pipeline
    pipeline = TrainingPipeline(Config)

    # =========================================================================
    # 2. Data Loading & Feature Engineering
    # =========================================================================
    # Load Train and Validation data (features generated/cached automatically)
    df_train = pipeline.data_pipeline.load_data(
        mode="train", load_cached_data=True, debug=Config.DEBUG
    )
    df_val = pipeline.data_pipeline.load_data(
        mode="val", load_cached_data=True, debug=Config.DEBUG
    )

    # =========================================================================
    # 3. Training Flow (Scouts -> Mining -> Experts)
    # =========================================================================
    # Phase 1: Train Scouts on Balanced Data
    scouts = pipeline.train_scouts(df_train, load_cached_data=True)

    # Phase 2: Mine Hard Negatives
    hard_indices = pipeline.run_mining_phase(df_train, scouts, load_cached_data=True)

    # Phase 3: Train Experts on Anchored Dataset (Positives + Hard Negs + Anchors)
    experts = pipeline.train_experts(
        df_train, df_val, hard_indices, load_cached_data=True
    )

    # =========================================================================
    # 4. Validation & Threshold Optimization
    # =========================================================================
    logging.info("Performing validation and threshold optimization...")

    # Predict on validation set
    y_val = df_val["contact"].values
    y_pred_proba = experts.predict_proba(df_val)

    # Grid search for best threshold
    best_mcc = -1.0
    best_threshold = 0.5
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred_binary = (y_pred_proba > thresh).astype(int)
        score = calc_mcc(y_val, y_pred_binary)
        if score > best_mcc:
            best_mcc = score
            best_threshold = thresh

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {best_mcc}")

    # Cache the best threshold for consistency if needed later
    os.makedirs(os.path.dirname(Config.CACHE_BEST_THRESHOLD), exist_ok=True)
    np.save(Config.CACHE_BEST_THRESHOLD, np.array(best_threshold))

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude: |y_true - y_pred_prob|
    # High error means confident wrong prediction (e.g., predicted 0.9 for class 0, error 0.9)
    errors = np.abs(y_val - y_pred_proba)

    # Identify feature columns (exclude metadata)
    feature_cols = experts._get_feature_cols(df_val)

    correlations = {}
    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df_val[col]):
            # Compute correlation between the feature value and the error magnitude
            # Handle potential NaNs safely
            valid_mask = ~np.isnan(df_val[col]) & ~np.isnan(errors)
            if np.sum(valid_mask) > 1:
                corr = np.corrcoef(df_val[col][valid_mask], errors[valid_mask])[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort by absolute correlation to find strongest associations with error
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features Correlated with Error Magnitude:")
    for name, val in sorted_corr[:10]:
        print(f"{name}: {val:.4f}")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    TARGET_METRIC = 0.6865

    if best_mcc > TARGET_METRIC:
        logging.info(
            f"Validation metric ({best_mcc}) exceeds threshold ({TARGET_METRIC}). Generating submission..."
        )
        pipeline.generate_submission(
            experts, best_threshold, load_cached_data=True, debug=Config.DEBUG
        )
    else:
        logging.warning(
            f"Validation metric ({best_mcc}) is below threshold ({TARGET_METRIC}). Submission skipped."
        )


if __name__ == "__main__":
    main()
