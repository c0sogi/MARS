import os
import sys
import numpy as np
import pandas as pd

# Import from the provided library files
import library.config as config
from library.config import RANDOM_SEED, TARGET_COL, SUBMISSION_PATH, NUMERIC_COLS
from library.utils import seed_everything, calculate_auc, save_submission
import library.data_loader as data_loader
import library.trainer as trainer


def run():
    # 1. Setup
    seed_everything(RANDOM_SEED)
    print("Starting orchestration script...")

    # 2. Load Data
    # Using load_cached_data=True to leverage any existing preprocessed files
    print("Loading data...")
    train_df, val_df, test_df = data_loader.load_data(load_cached_data=True)

    # 3. Train Models
    # We use the granular training functions from trainer.py to get the model objects back
    # for custom validation and failure analysis.

    # Train Random Forest Stream
    rf_model, rf_val_auc = trainer.train_rf_stream(
        train_df, val_df, load_cached_data=True
    )

    # Train MLP Stream
    mlp_model, mlp_val_auc = trainer.train_mlp_stream(
        train_df, val_df, load_cached_data=True
    )

    # 4. Ensemble Validation
    print("\nPerforming Ensemble Validation...")
    # Generate probabilities on validation set
    rf_val_probs = rf_model.predict_proba(val_df, load_cached_data=True)
    mlp_val_probs = mlp_model.predict_proba(val_df, load_cached_data=True)

    # Simple Average Ensemble
    ensemble_val_probs = (rf_val_probs + mlp_val_probs) / 2.0

    # Calculate Final Metric
    y_val = val_df[TARGET_COL].values
    final_auc = calculate_auc(y_val, ensemble_val_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val - ensemble_val_probs)

    # Create a DataFrame for correlation analysis
    # We focus on numeric columns available in the validation set
    analysis_df = val_df[NUMERIC_COLS].copy()
    analysis_df["error_magnitude"] = errors

    # Compute correlations
    correlations = analysis_df.corr()["error_magnitude"].drop("error_magnitude")

    # Sort by absolute correlation
    sorted_corrs = correlations.abs().sort_values(ascending=False)

    print("Top 5 Features correlated with Error Magnitude:")
    print(correlations.loc[sorted_corrs.index[:5]])

    # 6. Submission Generation
    # Threshold check
    THRESHOLD = 0.6959737721862433

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        # Use the trainer's helper to generate ensemble predictions for test set
        submission_df = trainer.generate_predictions(
            rf_model, mlp_model, test_df, load_cached_data=True
        )

        # Save submission
        save_submission(submission_df, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation metric ({final_auc}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    run()
