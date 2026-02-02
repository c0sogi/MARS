import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import from provided libraries
from library.config import RANDOM_SEED, TARGET_COLS, SUBMISSION_PATH, XGB_PARAMS
from library.data_loader import load_and_process_data, inverse_transform_targets
from library.model import DualTargetRegressor, generate_submission

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def run():
    # -------------------------------------------------------------------------
    # 1. Setup and Device Configuration
    # -------------------------------------------------------------------------
    print("Initializing pipeline...")

    # Configure XGBoost parameters for GPU if available
    xgb_params = XGB_PARAMS.copy()
    if torch.cuda.is_available():
        print("GPU detected. Configuring XGBoost to use CUDA.")
        xgb_params["device"] = "cuda"
        # Ensure tree method is compatible
        xgb_params["tree_method"] = "hist"
    else:
        print("No GPU detected. Using CPU.")
        xgb_params["device"] = "cpu"

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("Loading and processing training data...")
    # Load training data (features cached if available)
    X_train, y_train_log = load_and_process_data(
        dataset_type="train", load_cached_data=False
    )

    print("Loading and processing validation data...")
    # Load validation data
    X_val, y_val_log = load_and_process_data(dataset_type="val", load_cached_data=False)

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("Training model...")
    model = DualTargetRegressor(params=xgb_params)
    model.fit(X_train, y_train_log, X_val, y_val_log)

    # -------------------------------------------------------------------------
    # 4. Validation and Metric Calculation
    # -------------------------------------------------------------------------
    print("Performing validation inference...")
    # Predict on validation set
    # model.predict returns predictions in the original scale (eV)
    preds_original = model.predict(X_val)

    # Convert predictions back to log scale for RMSLE calculation
    # The target y_val_log is already log(1 + y)
    preds_log = np.log1p(preds_original)

    # Calculate Column-wise Root Mean Squared Logarithmic Error (RMSLE)
    # Since we are working in the log domain, RMSE on log values is RMSLE on original values
    rmsle_scores = []
    for col in TARGET_COLS:
        # Calculate RMSE for this target
        rmse = np.sqrt(mean_squared_error(y_val_log[col], preds_log[col]))
        rmsle_scores.append(rmse)
        print(f"RMSLE for {col}: {rmse}")

    # Final metric is the mean of column-wise RMSLEs
    final_metric = np.mean(rmsle_scores)
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 5. Failure Analysis
    # -------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude (absolute error in log space) per sample
    # We average the error across the two targets to get a single error score per sample
    error_df = np.abs(y_val_log - preds_log)
    mean_error_magnitude = error_df.mean(axis=1)

    # Calculate correlation between input features and error magnitude
    # Select only numeric features for correlation
    numeric_features = X_val.select_dtypes(include=[np.number])
    correlations = (
        numeric_features.corrwith(mean_error_magnitude)
        .abs()
        .sort_values(ascending=False)
    )

    print("Top 10 features correlated with model error magnitude:")
    print(correlations.head(10))

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.056919346405286564

    if final_metric < THRESHOLD:
        print(f"\nValidation metric {final_metric} meets threshold {THRESHOLD}.")
        print("Generating submission for test set...")
        generate_submission(model, load_cached_data=True)
    else:
        print(
            f"\nValidation metric {final_metric} does NOT meet threshold {THRESHOLD}."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
