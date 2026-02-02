import os
import sys
import numpy as np
import pandas as pd
import random
import xgboost as xgb

from library.config import Config
from library.utils import inverse_log_transform, calculate_rmse
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
    # We use debug=True with sample_size=12,000,000.
    # The validation set is ~11M rows, so this ensures we load the FULL validation set
    # as required by the task, while limiting the training set to 12M rows for speed.
    print("Loading and processing data...")
    train_df, val_df, test_df = load_and_process_data(
        load_cached_data=True, debug=True, sample_size=12000000
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

    # Generate predictions on the validation set (Log Scale)
    val_preds_log = trainer.predict(val_df)

    # Inverse transform predictions to Dollar Scale
    val_preds = inverse_log_transform(val_preds_log)

    # Get Ground Truth (Inverse transform from log-scale in dataframe)
    val_y_log = val_df["fare_amount"].values
    val_y = inverse_log_transform(val_y_log)

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
        test_preds_log = trainer.predict(test_df)
        test_preds = inverse_log_transform(test_preds_log)

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
