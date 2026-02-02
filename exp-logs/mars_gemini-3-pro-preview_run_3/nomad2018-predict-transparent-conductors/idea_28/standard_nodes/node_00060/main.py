import pandas as pd
import numpy as np
import logging
import sys
import os

# Import from provided libraries
from library.data_pipeline import generate_features
from library.model import train_model, generate_submission_file, DualTargetRegressor
from library.utils import calculate_rmsle, setup_logger
from library.config import TARGET_COLS

# Setup logger
logger = setup_logger("runfile")


def main():
    logger.info("Starting runfile execution...")

    # 1. Train the model
    # We use load_cached_data=True to utilize any pre-computed features if available.
    # debug=False ensures we use the full dataset for maximum performance.
    # The train_model function handles feature generation/loading, training, and initial evaluation.
    logger.info("Training model...")
    model = train_model(load_cached_data=True, debug=False)

    # 2. Load Validation Data for Assessment and Failure Analysis
    # We load it explicitly here to have access to X_val and y_val variables for detailed analysis
    logger.info("Loading validation data for analysis...")
    X_val, y_val = generate_features(data_type="val", load_cached_data=True)

    # 3. Validation Assessment
    logger.info("Generating validation predictions...")
    val_preds_df = model.predict(X_val)

    # Ensure order of columns matches for metric calculation
    y_val_sorted = y_val[TARGET_COLS]
    val_preds_sorted = val_preds_df[TARGET_COLS]

    # Calculate the final metric using the provided utility
    final_metric = calculate_rmsle(y_val_sorted, val_preds_sorted)

    # Print the required format for the grading system
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    logger.info("Performing failure analysis...")

    # Drop non-numeric columns from X_val for correlation calculation to avoid errors
    X_val_numeric = X_val.select_dtypes(include=[np.number])

    for target in TARGET_COLS:
        y_true = y_val[target]
        y_pred = val_preds_df[target]

        # Calculate absolute error for each sample
        error = np.abs(y_true - y_pred)

        # Calculate correlation between input features and the prediction error
        # This helps identify which features are associated with higher errors (systematic failure modes)
        correlations = X_val_numeric.corrwith(error).abs().sort_values(ascending=False)

        print(f"\n--- Top 5 Features correlated with Error for {target} ---")
        print(correlations.head(5))

        # Print error statistics
        print(f"Mean Absolute Error for {target}: {np.mean(error)}")
        print(f"Max Error for {target}: {np.max(error)}")

    # 5. Submission Generation
    # Check against the threshold defined in the task description
    threshold = 0.054581
    if final_metric < threshold:
        logger.info(
            f"Validation metric {final_metric} is below threshold {threshold}. Generating submission..."
        )
        # generate_submission_file handles loading test data, predicting, and saving to CSV
        generate_submission_file(model, load_cached_data=True)
    else:
        logger.warning(
            f"Validation metric {final_metric} is NOT below threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()
