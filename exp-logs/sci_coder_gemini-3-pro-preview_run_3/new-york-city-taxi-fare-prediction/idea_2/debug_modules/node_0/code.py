import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import joblib

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data_loader import load_and_process_data, create_dataloaders
from library.engine import train_gbdt, train_nn, generate_predictions


def main():
    print("=== Starting Taxi Fare Prediction Demo Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Debugging
    # ---------------------------------------------------------
    print("[1/5] Configuring environment for rapid execution...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Small sample for demonstration

    # Reduce Feature Engineering complexity
    Config.N_CLUSTERS = 10  # Fewer clusters for faster KMeans

    # Reduce GBDT complexity
    Config.GBDT_PARAMS["max_iter"] = 5
    Config.GBDT_PARAMS["max_leaf_nodes"] = 10
    Config.GBDT_PARAMS["min_samples_leaf"] = 5
    Config.GBDT_PARAMS["validation_fraction"] = None  # We pass explicit val set
    Config.GBDT_PARAMS["verbose"] = 0

    # Reduce NN complexity
    Config.NN_PARAMS["epochs"] = 1
    Config.NN_PARAMS["batch_size"] = 64
    Config.NN_PARAMS["hidden_dims"] = [32, 16]
    Config.NN_PARAMS["embedding_dims"] = {"cluster": 4, "hour": 2, "dow": 2, "year": 2}

    # Clean up working directory to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated. Debug mode enabled.")

    # ---------------------------------------------------------
    # 2. Data Processing & Feature Engineering
    # ---------------------------------------------------------
    print("\n[2/5] Processing Data...")

    # Load and process data (force processing by ignoring cache if any existed)
    # This uses library.feature_engineering internally
    train_df, val_df, test_df = load_and_process_data(load_cached_data=False)

    # Assertions to verify Feature Engineering
    print("Verifying processed data integrity...")
    expected_cols = [
        "pickup_cluster",
        "dropoff_cluster",
        "haversine_dist",
        "pickup_rot_x",
        "year",
        "hour",
        "day_of_week",
    ]

    for df_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        # Check columns
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            raise AssertionError(f"{df_name} DataFrame missing columns: {missing_cols}")

        # Check rows (should match debug sample size or less due to filtering)
        if df_name == "Train" and len(df) > Config.DEBUG_SAMPLE_SIZE:
            raise AssertionError(
                f"{df_name} size {len(df)} exceeds debug limit {Config.DEBUG_SAMPLE_SIZE}"
            )

    # Verify KMeans model was saved
    if not os.path.exists(Config.KMEANS_MODEL_PATH):
        raise AssertionError(f"KMeans model not found at {Config.KMEANS_MODEL_PATH}")

    print(f"Data processed successfully. Train shape: {train_df.shape}")

    # ---------------------------------------------------------
    # 3. GBDT Model Training
    # ---------------------------------------------------------
    print("\n[3/5] Training GBDT Model...")

    gbdt_model = train_gbdt(train_df, val_df)

    # Verify GBDT Artifacts
    if not os.path.exists(Config.GBDT_MODEL_PATH):
        raise AssertionError("GBDT model file was not created.")

    # Verify Prediction capability
    sample_preds = gbdt_model.predict(val_df.head(10))
    if sample_preds.shape != (10,):
        raise AssertionError(
            f"GBDT prediction shape mismatch. Expected (10,), got {sample_preds.shape}"
        )
    if np.isnan(sample_preds).any():
        raise AssertionError("GBDT produced NaN predictions.")

    print("GBDT training and verification complete.")

    # ---------------------------------------------------------
    # 4. Neural Network Training
    # ---------------------------------------------------------
    print("\n[4/5] Training Neural Network...")

    # train_nn handles DataLoader creation and scaling internally
    nn_model, test_loader = train_nn(train_df, val_df, test_df)

    # Verify NN Artifacts
    if not os.path.exists(Config.NN_MODEL_PATH):
        raise AssertionError("NN model file was not created.")
    if not os.path.exists(Config.SCALER_PATH):
        raise AssertionError("Scaler file was not created.")

    # Verify DataLoader
    batch = next(iter(test_loader))
    # Batch should be a tuple (x_cat, x_cont) for test set
    if len(batch) != 2:
        raise AssertionError("Test DataLoader batch structure incorrect.")

    x_cat, x_cont = batch
    # Check categorical input shape: Batch Size x 5 features (p_clus, d_clus, hr, dow, yr)
    if x_cat.shape[1] != 5:
        raise AssertionError(
            f"Categorical input shape incorrect. Expected 5 features, got {x_cat.shape[1]}"
        )

    print("Neural Network training and verification complete.")

    # ---------------------------------------------------------
    # 5. Ensemble & Submission
    # ---------------------------------------------------------
    print("\n[5/5] Generating Submission...")

    submission_df = generate_predictions(gbdt_model, nn_model, test_df, test_loader)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file not found.")

    if submission_df.shape[0] != len(test_df):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(test_df)}, got {len(submission_df)}"
        )

    if list(submission_df.columns) != ["key", "fare_amount"]:
        raise AssertionError(
            f"Submission columns incorrect. Got {submission_df.columns}"
        )

    # Check for valid values
    if submission_df["fare_amount"].isnull().any():
        raise AssertionError("Submission contains NaN values.")

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
