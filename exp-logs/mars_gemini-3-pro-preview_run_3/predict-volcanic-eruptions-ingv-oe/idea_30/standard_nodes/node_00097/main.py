import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.data_loader import build_feature_dataset
from library.model_engine import run_cross_validation, predict_ensemble
from library.utils import compute_metric, save_submission


def main():
    # 1. Configuration & Setup
    # Set seeds for reproducibility
    np.random.seed(Config.SEED)

    print("Starting execution of runfile.py...")

    # 2. Data Loading
    # Load Training Data
    print("Loading Training Data...")
    X_train_full, y_train_full = build_feature_dataset(
        mode="train", load_cached_data=True
    )

    # Load Hold-Out Validation Data
    print("Loading Validation Data...")
    X_val_holdout, y_val_holdout = build_feature_dataset(
        mode="val", load_cached_data=True
    )

    # Validation check for data loading
    if X_train_full.empty or X_val_holdout.empty:
        print(
            "CRITICAL ERROR: Data loading failed. Training or Validation set is empty."
        )
        sys.exit(1)

    # Prepare Feature Matrices
    # Remove 'segment_id' as it is not a predictive feature
    if "segment_id" in X_train_full.columns:
        X_train = X_train_full.drop(columns=["segment_id"])
    else:
        X_train = X_train_full

    if "segment_id" in X_val_holdout.columns:
        X_val = X_val_holdout.drop(columns=["segment_id"])
    else:
        X_val = X_val_holdout

    # 3. Training
    # Train the ensemble using 5-Fold CV on the Training set
    print(f"Training Ensemble with {Config.N_FOLDS} folds on Training set...")
    models, _ = run_cross_validation(X_train, y_train_full, save_models=True)

    # 4. Validation
    # Evaluate the ensemble on the Hold-Out Validation set
    print("Generating predictions for Hold-Out Validation set...")
    val_preds = predict_ensemble(models, X_val)

    # Compute Final Metric
    final_val_mae = compute_metric(y_val_holdout, val_preds)

    # Print the required metric string
    print(f"Final Validation Metric: {final_val_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_val_holdout - val_preds)

    # Compute correlation between error magnitude and features
    correlations = []
    # Handle potential NaN correlations if a feature is constant
    for col in X_val.columns:
        try:
            corr, _ = pearsonr(X_val[col], errors)
            if np.isnan(corr):
                corr = 0.0
            correlations.append((col, corr))
        except Exception:
            correlations.append((col, 0.0))

    # Sort by absolute correlation strength (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(
        "Top 10 Features correlated with Error Magnitude (Systematic Error Patterns):"
    )
    for feat, corr in correlations[:10]:
        print(f"{feat}: {corr:.4f}")

    # 6. Submission
    THRESHOLD = 2617304.0647319085

    if final_val_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({final_val_mae}) meets the threshold ({THRESHOLD}). Proceeding to submission."
        )

        # Load Test Data
        print("Loading Test Data...")
        X_test_full, _ = build_feature_dataset(mode="test", load_cached_data=True)

        if not X_test_full.empty:
            # Prepare Test Features
            test_ids = X_test_full["segment_id"]
            if "segment_id" in X_test_full.columns:
                X_test = X_test_full.drop(columns=["segment_id"])
            else:
                X_test = X_test_full

            # Generate Predictions
            print("Generating predictions for Test set...")
            test_preds = predict_ensemble(models, X_test)

            # Save Submission
            save_submission(test_ids, test_preds)
        else:
            print("Error: Test data is empty. Cannot generate submission.")
    else:
        print(
            f"\nValidation Metric ({final_val_mae}) is NOT lower than threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
