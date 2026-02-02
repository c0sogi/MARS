import os
import sys
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from library.config import LGBM_PARAMS, SEED
from library.data_loader import generate_dataset
from library.model_handler import VolcanoLGBM
from library.utils import calculate_mae, save_submission, setup_logger

# Ensure reproducibility
np.random.seed(SEED)


def main():
    logger = setup_logger()
    logger.info("Starting runfile.py execution...")

    # --- 1. Configuration & GPU Detection ---
    params = LGBM_PARAMS.copy()

    # Check for GPU availability to optimize training/inference
    if torch.cuda.is_available():
        logger.info("GPU detected. Configuring LightGBM to use GPU.")
        params["device"] = "gpu"
        params["gpu_platform_id"] = 0
        params["gpu_device_id"] = 0
    else:
        logger.info("No GPU detected. Using CPU.")
        params["device"] = "cpu"

    # --- 2. Data Loading ---
    # We load the full dataset. Caching in generate_dataset ensures this is fast on subsequent runs.
    logger.info("Loading Training Data...")
    X_train, y_train = generate_dataset("train", load_cached_data=True, debug_size=None)

    logger.info("Loading Validation Data...")
    X_val, y_val = generate_dataset("val", load_cached_data=True, debug_size=None)

    logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # --- 3. Model Training ---
    logger.info("Initializing and Training Model...")
    model = VolcanoLGBM(params=params)
    model.train(X_train, y_train, X_val, y_val)

    # --- 4. Validation & Metric ---
    logger.info("Performing Validation Inference...")
    val_preds = model.predict(X_val)

    final_mae = calculate_mae(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mae}")

    # --- 5. Failure Analysis ---
    logger.info("Performing Failure Analysis...")
    errors = np.abs(y_val - val_preds)

    # Calculate correlation between features and error magnitude
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    print("\n--- Failure Analysis: Top 10 Features Correlated with Error ---")
    # Print top 10 features (excluding the error column itself)
    print(correlations.head(11).drop("error_magnitude", errors="ignore"))

    # --- 6. Submission ---
    THRESHOLD = 2617304.0647319085

    if final_mae < THRESHOLD:
        logger.info(
            f"Validation metric ({final_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        logger.info("Loading Test Data...")
        test_df, _ = generate_dataset("test", load_cached_data=True, debug_size=None)

        # Prepare features for prediction
        # generate_dataset for test returns a DF with segment_id, so we filter it out for prediction
        feature_cols = [
            c for c in test_df.columns if c not in ["segment_id", "time_to_eruption"]
        ]
        X_test = test_df[feature_cols]
        segment_ids = test_df["segment_id"]

        logger.info(f"Predicting on {len(X_test)} test samples...")
        test_preds = model.predict(X_test)

        submission_df = pd.DataFrame(
            {"segment_id": segment_ids, "time_to_eruption": test_preds}
        )

        output_path = "./submission/submission.csv"
        save_submission(submission_df, output_path)
        logger.info("Process completed successfully.")
    else:
        logger.warning(
            f"Validation metric ({final_mae}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
