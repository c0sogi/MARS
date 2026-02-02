import sys
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

# Import library modules
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_and_process_data, create_dataloaders
from library.engine import train_gbdt, train_nn, generate_predictions


def main():
    # 1. Setup and Configuration
    set_seed(Config.SEED)

    print("Setting up Fast Baseline Configuration...")
    # Adjust hyperparameters for speed
    FAST_TRAIN_SIZE = 500_000
    FAST_VAL_SIZE = 100_000

    # GBDT Speedups
    Config.GBDT_PARAMS["max_iter"] = 100  # Reduced from 300
    Config.GBDT_PARAMS["max_leaf_nodes"] = 127  # Reduced complexity

    # NN Speedups
    Config.NN_PARAMS["epochs"] = 5  # Reduced from 15
    Config.NN_PARAMS["batch_size"] = 8192  # Increased for faster throughput on A100
    Config.NN_PARAMS["patience"] = 2

    # 2. Load Data
    print("Loading datasets...")
    # load_cached_data=True will use the parquet files in working/ if they exist
    train_df, val_df, test_df = load_and_process_data(load_cached_data=True)

    # 3. Create Subsets for Fast Training
    print(f"Sampling training data to {FAST_TRAIN_SIZE} rows...")
    if len(train_df) > FAST_TRAIN_SIZE:
        train_df_small = train_df.sample(
            n=FAST_TRAIN_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        train_df_small = train_df.copy()

    print(f"Sampling validation data to {FAST_VAL_SIZE} rows for early stopping...")
    if len(val_df) > FAST_VAL_SIZE:
        val_df_small = val_df.sample(
            n=FAST_VAL_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
    else:
        val_df_small = val_df.copy()

    # 4. Train Models
    # Train GBDT
    print("\n=== Training GBDT Stream ===")
    gbdt_model = train_gbdt(train_df_small, val_df_small)

    # Train NN
    print("\n=== Training Neural Network Stream ===")
    # train_nn handles scaling internally. It will fit a scaler on train_df_small.
    # It returns the model and the test_loader (prepared with the same scaler).
    nn_model, test_loader = train_nn(train_df_small, val_df_small, test_df)

    # 5. Final Validation (Full Set)
    print("\n=== Performing Final Validation on Full Hold-out Set ===")

    # A. GBDT Predictions
    print("Predicting with GBDT...")
    gbdt_val_preds = gbdt_model.predict(val_df)

    # B. NN Predictions
    print("Predicting with Neural Network...")
    # We need to create a DataLoader for the full validation set.
    # We must use load_cached_scaler=True to reuse the scaler fitted during train_nn.
    _, val_loader_full, _, _, _, _ = create_dataloaders(
        train_df_small,
        val_df,
        test_df,
        load_cached_scaler=True,
        batch_size=Config.NN_PARAMS["batch_size"],
    )
    nn_val_preds = nn_model.predict(val_loader_full)

    # C. Ensemble
    print("Calculating Ensemble Metrics...")
    final_val_preds = (Config.WEIGHT_GBDT * gbdt_val_preds) + (
        Config.WEIGHT_NN * nn_val_preds
    )

    # D. Metric
    y_val = val_df[Config.TARGET_COL].values
    rmse = np.sqrt(mean_squared_error(y_val, final_val_preds))

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {rmse}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate errors
    errors = y_val - final_val_preds
    abs_errors = np.abs(errors)

    # Create a temporary dataframe for analysis
    analysis_df = val_df.copy()
    analysis_df["abs_error"] = abs_errors

    # Select numerical features for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove target and error cols from features list
    exclude_cols = [Config.TARGET_COL, "abs_error", "prediction", "error"]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Compute correlations
    correlations = (
        analysis_df[feature_cols + ["abs_error"]].corr()["abs_error"].drop("abs_error")
    )
    print("Correlation between Error Magnitude and Features:")
    print(correlations.sort_values(ascending=False).head(10))

    # 7. Submission
    THRESHOLD = 3.3935366001817666

    if rmse < THRESHOLD:
        print(
            f"\nValidation RMSE ({rmse}) is below threshold ({THRESHOLD}). Generating submission..."
        )
        generate_predictions(gbdt_model, nn_model, test_df, test_loader)
    else:
        print(
            f"\nValidation RMSE ({rmse}) is NOT below threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
