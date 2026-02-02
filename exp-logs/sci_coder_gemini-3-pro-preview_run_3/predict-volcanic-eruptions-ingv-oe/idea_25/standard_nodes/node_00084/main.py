import sys
import os
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, log_message, calculate_mae, save_submission
from library.dataset import generate_dataset
from library.trainer import EnsembleTrainer


def main():
    # 1. Setup
    # Set random seeds for reproducibility
    seed_everything(Config.SEED)
    log_message("Starting runfile.py execution...")

    # 2. Load Data
    # We load 'train' for training the ensemble (which does internal CV)
    # We load 'val' for the final hold-out evaluation required by the prompt
    log_message("Loading datasets...")

    # Ensure we use cached data if available to save time
    # The generate_dataset function handles feature extraction if cache is missing
    X_train, y_train = generate_dataset("train", load_cached_data=True)
    X_val, y_val = generate_dataset("val", load_cached_data=True)

    log_message(f"Train Data Shape: {X_train.shape}")
    log_message(f"Val Data Shape: {X_val.shape}")

    # 3. Train Model
    trainer = EnsembleTrainer()

    # The trainer performs K-Fold CV on the provided data and saves models
    # It returns OOF predictions for the training set, but we focus on the hold-out val performance
    log_message("Training ensemble...")
    _ = trainer.train_ensemble(X_train, y_train)

    # 4. Validation on Hold-out Set
    log_message("Evaluating on hold-out validation set...")
    # Predict using the ensemble (averaging predictions from all folds)
    val_preds = trainer.predict_ensemble(X_val)

    final_mae = calculate_mae(y_val, val_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    log_message("\n--- Failure Analysis ---")
    errors = np.abs(y_val - val_preds)

    # Create a dataframe for analysis
    # We drop segment_id as it is an identifier, not a feature
    analysis_df = X_val.drop(columns=["segment_id"], errors="ignore")
    analysis_df["error_magnitude"] = errors

    # Calculate correlations between features and the error magnitude
    correlations = (
        analysis_df.corrwith(analysis_df["error_magnitude"])
        .abs()
        .sort_values(ascending=False)
    )

    log_message("Top 10 features correlated with error magnitude:")
    # Filter out the error column itself
    top_corrs = correlations.drop("error_magnitude", errors="ignore").head(10)
    for feat, corr in top_corrs.items():
        print(f"{feat}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 2617304.0647319085

    if final_mae < THRESHOLD:
        log_message(
            f"\nValidation metric {final_mae} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        X_test, _ = generate_dataset("test", load_cached_data=True)

        # Predict
        test_preds = trainer.predict_ensemble(X_test)

        # Save
        # X_test contains segment_id which is needed for the submission file
        save_submission(test_preds, X_test)

    else:
        log_message(
            f"\nValidation metric {final_mae} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
