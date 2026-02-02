import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import provided library modules
from library.config import Config
from library.feature_factory import process_data
from library.model_trainer import XGBTrainer


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("Starting pipeline execution...")

    # 2. Data Loading and Processing
    # Utilizes the Factorized Spatiotemporal Feature Engineering pipeline
    # load_cached_data=True ensures we use pre-computed features if available to speed up execution
    print("Loading and processing data...")
    X_train, y_train, X_val, y_val, X_test, test_keys = process_data(
        load_cached_data=True, debug=False
    )

    # 3. Model Training
    # Initialize the trainer which wraps XGBoost with A100 optimization
    print("Training model...")
    trainer = XGBTrainer()

    # Train on the Learner set (subsampled) and monitor with Validation set
    # The trainer internally handles early stopping and model saving
    trainer.train(X_train, y_train, X_val, y_val)

    # 4. Validation & Metric Calculation
    print("Evaluating on Validation set...")
    # Predict on validation set (applies post-processing like min fare floor)
    # Ensure we use the trained model for inference
    val_preds = trainer.predict(X_val)

    # Calculate RMSE
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    # REQUIRED: Print metric in specific format
    print(f"Final Validation Metric: {final_rmse}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    # Calculate absolute error (residuals)
    residuals = np.abs(y_val - val_preds)

    # Create a temporary DataFrame to calculate correlations
    # We use the validation features to see which inputs correlate with high error
    analysis_df = X_val.copy()
    analysis_df["error_magnitude"] = residuals

    # Compute correlation of features with error magnitude
    # corrwith is efficient for calculating correlation of one column against all others
    correlations = analysis_df.corrwith(analysis_df["error_magnitude"]).drop(
        "error_magnitude"
    )

    # Sort by absolute correlation to find strongest relationships
    sorted_corr = correlations.abs().sort_values(ascending=False)

    print("Top features correlated with error magnitude:")
    print(sorted_corr.head(5))

    # 6. Submission Generation
    # Threshold defined in the task
    THRESHOLD = 3.438959912830025

    if final_rmse < THRESHOLD:
        print(
            f"Validation metric {final_rmse} meets threshold {THRESHOLD}. Generating submission..."
        )

        # Predict on Test set
        test_preds = trainer.predict(X_test)

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_keys, "fare_amount": test_preds})

        # Save to disk
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"Validation metric {final_rmse} does not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
