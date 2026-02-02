import sys
import os
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.data_loader import load_train_data, load_val_data, load_test_data
from library.models import LGBMTrainer
from library.utils import seed_everything, compute_mae, save_submission


def main():
    # 1. Setup and Configuration
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Optimize for fast baseline execution as per requirements
    # Ensure sufficient estimators for convergence
    Config.N_ESTIMATORS = 10000
    Config.EARLY_STOPPING_ROUNDS = 100

    # Explicitly update the parameter dictionaries
    Config.LGBM_PARAMS["n_estimators"] = Config.N_ESTIMATORS

    print(f"Configuration set. N_ESTIMATORS: {Config.N_ESTIMATORS}")

    # 2. Data Loading
    # Using load_cached_data=True to leverage existing parquet files if available
    print("\n--- Loading Data ---")
    try:
        X_train, y_train = load_train_data(load_cached_data=True)
        print(f"Training Data Loaded: {X_train.shape}")

        X_val, y_val = load_val_data(load_cached_data=True)
        print(f"Validation Data Loaded: {X_val.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Model Training
    print("\n--- Training Models ---")
    # Initialize the Trainer
    manager = LGBMTrainer()

    # Train the pipeline
    manager.fit_pipeline(X_train, y_train)

    # 4. Validation Assessment
    print("\n--- Validation Assessment ---")
    # Predict on the hold-out validation set
    val_preds = manager.predict(X_val)

    # Compute Metric
    val_mae = compute_mae(y_val, val_preds)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {val_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_val - val_preds)

    # Create a temporary dataframe for correlation analysis
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = errors

    # Calculate correlation between features and error magnitude
    # We drop the error column itself from the correlation calculation against itself
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).abs()

    # Drop the 'error_magnitude' self-correlation if present
    if "error_magnitude" in correlations:
        correlations = correlations.drop("error_magnitude")

    # Sort and display top 10
    top_correlations = correlations.sort_values(ascending=False).head(10)

    print("Top 10 features correlated with prediction error magnitude:")
    print(top_correlations)

    # 6. Submission Generation
    print("\n--- Submission Generation ---")
    TARGET_THRESHOLD = 2739761.2592384242

    if val_mae < TARGET_THRESHOLD:
        print(
            f"Validation MAE ({val_mae}) is better than threshold ({TARGET_THRESHOLD}). Generating submission..."
        )

        # Load Test Data
        # Note: y_test will be None
        X_test, _ = load_test_data(load_cached_data=True)
        print(f"Test Data Loaded: {X_test.shape}")

        # Generate Predictions
        test_preds = manager.predict(X_test)

        # Extract Segment IDs (index of X_test)
        segment_ids = X_test.index

        # Save Submission
        save_submission(segment_ids, test_preds)

    else:
        print(
            f"Validation MAE ({val_mae}) did not meet threshold ({TARGET_THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
