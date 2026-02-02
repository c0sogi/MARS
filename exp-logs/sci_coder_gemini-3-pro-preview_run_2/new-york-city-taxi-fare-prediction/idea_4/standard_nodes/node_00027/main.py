import os
import sys
import numpy as np
import pandas as pd
import random
import xgboost as xgb

from library.config import Config
from library.utils import calculate_rmse, inverse_log_transform
from library.data_loader import load_and_process_data
from library.model_trainer import XGBoostTrainer


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Note: XGBoost seed is set in Config.XGB_PARAMS


def main():
    # 1. Setup and Configuration
    print("Setting up environment...")
    set_seed(Config.RANDOM_SEED)
    Config.setup()

    # 2. Data Loading
    # Using full dataset (debug=False) with sanitization (implemented in data_loader)
    # Cite solution_lesson_node_00017: Scaling up data works if target is sanitized.
    print("Loading and processing data...")
    train_df, val_df, test_df = load_and_process_data(
        load_cached_data=True, debug=False
    )

    print(f"Data loaded. Train shape: {train_df.shape}, Val shape: {val_df.shape}")

    # 3. Model Training
    print("Initializing XGBoost Trainer...")
    # Limiting n_estimators to 1000 for a fast baseline run, relying on early stopping
    trainer = XGBoostTrainer(n_estimators=1000)

    print("Starting training...")
    trainer.train(train_df, val_df, target_col="fare_amount", key_col="key")

    # 4. Validation Evaluation
    print("Performing validation evaluation...")

    # Generate predictions on the validation set
    val_preds = trainer.predict(val_df)

    # Get Ground Truth
    val_y = val_df["fare_amount"].values

    # Inverse transform if using log target
    if Config.USE_LOG_TARGET:
        val_preds = inverse_log_transform(val_preds)
        val_y = inverse_log_transform(val_y)

    # Calculate RMSE
    rmse = calculate_rmse(val_y, val_preds)
    print(f"Final Validation Metric: {rmse}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error
    errors = np.abs(val_y - val_preds)

    # Get features used by the model
    features = trainer.feature_names

    # Create a temporary dataframe for correlation analysis
    analysis_df = val_df[features].copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlation
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations.sort_values(ascending=False))
    print("========================\n")

    # 6. Submission Generation
    THRESHOLD = 4.278504866347902

    if rmse < THRESHOLD:
        print(
            f"Validation RMSE ({rmse}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Predict on Test Set
        test_preds = trainer.predict(test_df)

        # Inverse transform if using log target
        if Config.USE_LOG_TARGET:
            test_preds = inverse_log_transform(test_preds)

        # Post-Processing: Apply minimum fare floor ($2.50)
        test_preds = np.maximum(test_preds, 2.50)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation RMSE ({rmse}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
