import os
import sys
import numpy as np
import pandas as pd
import torch
import xgboost as xgb

# Import from the provided library files
from library.data import FeatureLoader
from library.model import DualTargetRegressor
from library.config import get_xgb_params, TARGET_COLS
from library.utils import (
    log_transform,
    inverse_log_transform,
    compute_rmsle,
    save_submission,
)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Device Setup
    # -------------------------------------------------------------------------
    print("Initializing workflow...")

    # Check for GPU availability for XGBoost
    xgb_params = get_xgb_params()
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost to use CUDA.")
        xgb_params["device"] = "cuda"
        # tree_method 'hist' works with device='cuda' in recent XGBoost versions
    else:
        print("No GPU detected. Using CPU for XGBoost.")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # -------------------------------------------------------------------------
    print("\nLoading and preprocessing data...")
    # Initialize FeatureLoader with debug=False to use the full dataset
    loader = FeatureLoader(debug=False)

    # Load training and validation data
    # This handles feature generation, caching, log-transformation of targets,
    # and dropping constant columns.
    X_train, y_train_log, X_val, y_val_log = loader.load_train_val(
        load_cached_data=True
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Validation Data Shape: {X_val.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\nInitializing and training model...")
    model = DualTargetRegressor(params=xgb_params)

    # Fit the model
    # We use early stopping based on the validation set provided within the fit method
    model.fit(
        X_train,
        y_train_log,
        X_val=X_val,
        y_val=y_val_log,
        early_stopping_rounds=100,
        verbose=False,
    )
    print("Training complete.")

    # -------------------------------------------------------------------------
    # 4. Validation and Metric Computation
    # -------------------------------------------------------------------------
    print("\nRunning validation inference...")

    # Predict in log space
    preds_val_log = model.predict(X_val)

    # Inverse transform predictions and ground truth to original scale
    preds_val = inverse_log_transform(preds_val_log)
    y_val_true = inverse_log_transform(y_val_log)

    # Compute RMSLE using the utility function
    # Note: compute_rmsle expects original scale values
    val_metric = compute_rmsle(y_val_true.values, preds_val.values)

    # Print the final metric in the required format
    print(f"Final Validation Metric: {val_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (absolute difference in log space approximates relative error)
    # Using log space errors for correlation analysis as it aligns with the loss function
    errors = np.abs(y_val_log - preds_val_log)

    # Aggregate error across targets (mean error per sample)
    mean_error = errors.mean(axis=1)

    # Correlate error with input features
    # We use the validation features X_val
    correlations = X_val.corrwith(mean_error).abs().sort_values(ascending=False)

    print("Top 5 features correlated with prediction error:")
    print(correlations.head(5))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    # Threshold defined in the task description
    THRESHOLD = 0.056919346405286564

    if val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({val_metric}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Load test data
        X_test, ids_test = loader.load_test(load_cached_data=True)

        # Predict on test set (returns log scale)
        preds_test_log = model.predict(X_test)

        # Inverse transform to original scale
        preds_test = inverse_log_transform(preds_test_log)

        # Extract columns for submission
        formation_energy_pred = preds_test["formation_energy_ev_natom"].values
        bandgap_energy_pred = preds_test["bandgap_energy_ev"].values

        # Save submission
        save_submission(ids_test.values, formation_energy_pred, bandgap_energy_pred)

    else:
        print(
            f"\nValidation metric ({val_metric}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
