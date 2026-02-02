import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import feature_engineering
from library import model_rf
from library import model_mlp


def run_demo():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    # Set a deterministic seed
    utils.set_seed(42)

    # Override CACHE_DIR to use a temporary demo location.
    # This prevents loading existing full-dataset caches and forces re-computation on our subset.
    demo_cache_dir = os.path.join(config.WORKING_DIR, "demo_cache")
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)

    # Monkeypatch the config object
    config.CACHE_DIR = demo_cache_dir
    # Also update the FeatureEngineer's reference to cache dir by re-instantiating later
    # (The class reads config.CACHE_DIR in __init__)

    print(f"Temporary cache directory set to: {config.CACHE_DIR}")

    # 2. Load Data
    print("\n--- Loading Data ---")
    # We force load from source (CSV) to ensure we get the raw data, then we'll subset it.
    # We pass load_cached_data=False to skip looking for parquet files in the *original* cache dir location
    # if the data_loader uses config.CACHE_DIR (which we just changed).
    train_df, val_df, test_df = data_loader.load_dataset(load_cached_data=False)

    # 3. Subsample Data for Speed
    print("\n--- Subsampling Data for Demonstration ---")
    subset_size = 50
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    print(f"Train subset shape: {train_subset.shape}")
    print(f"Val subset shape: {val_subset.shape}")
    print(f"Test subset shape: {test_subset.shape}")

    # 4. Feature Engineering
    print("\n--- Running Feature Engineering Pipeline ---")
    # Instantiate FeatureEngineer. It will use the modified config.CACHE_DIR.
    fe = feature_engineering.FeatureEngineer()

    # Process data to get features for both streams
    # This handles SBERT, TF-IDF, Metadata scaling, etc.
    (rf_data, mlp_data) = fe.process_data(
        train_subset, val_subset, test_subset, load_cached_data=False  # Force compute
    )

    # Unpack RF Data (Stream A)
    X_train_rf, y_train_rf, X_val_rf, y_val_rf, X_test_rf = rf_data

    # Unpack MLP Data (Stream B)
    X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp, X_test_mlp = mlp_data

    # Assertions to verify feature generation
    assert X_train_rf.shape[0] == subset_size, "RF Train features row count mismatch"
    assert X_train_mlp.shape[0] == subset_size, "MLP Train features row count mismatch"
    assert len(y_train_rf) == subset_size, "Target vector size mismatch"

    print("Feature engineering complete.")

    # 5. Stream A: Random Forest
    print("\n--- Stream A: Random Forest Model ---")
    # Define reduced parameters for speed
    rf_params = {
        "n_estimators": 10,  # Reduced from default 500
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
        "verbose": 0,
    }

    # Train RF
    rf_model, rf_val_auc = model_rf.train_rf_model(
        X_train_rf, y_train_rf, X_val_rf, y_val_rf, params=rf_params
    )
    print(f"RF Validation AUC: {rf_val_auc:.4f}")

    # Predict RF
    rf_test_probs = model_rf.predict_rf_model(rf_model, X_test_rf)

    # Verify predictions
    assert len(rf_test_probs) == subset_size
    assert np.all(
        (rf_test_probs >= 0) & (rf_test_probs <= 1)
    ), "RF probabilities out of bounds"

    # 6. Stream B: MLP
    print("\n--- Stream B: Topology-Aware MLP Model ---")
    # Define reduced parameters for speed
    mlp_params = {
        "epochs": 2,  # Reduced from default 50
        "batch_size": 16,
        "learning_rate": 1e-3,
        "hidden_dim": 64,  # Smaller dimension
        "patience": 1,
    }

    # Train MLP
    mlp_model, mlp_val_auc = model_mlp.train_mlp_model(
        X_train_mlp, y_train_mlp, X_val_mlp, y_val_mlp, params=mlp_params
    )

    # Predict MLP
    mlp_test_probs = model_mlp.predict_mlp_model(mlp_model, X_test_mlp)

    # Verify predictions
    assert len(mlp_test_probs) == subset_size
    assert np.all(
        (mlp_test_probs >= 0) & (mlp_test_probs <= 1)
    ), "MLP probabilities out of bounds"

    # 7. Ensemble and Submission
    print("\n--- Generating Ensemble Submission ---")
    # Weighted average
    w_rf = config.ENSEMBLE_WEIGHT_RF
    w_mlp = config.ENSEMBLE_WEIGHT_MLP

    final_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    # Create DataFrame
    submission_df = pd.DataFrame(
        {
            "request_id": test_subset["request_id"],
            "requester_received_pizza": final_probs,
        }
    )

    print("Submission DataFrame head:")
    print(submission_df.head())

    # 8. Verify Submission Format
    # We need to temporarily mock the sample submission check because we are using a subset.
    # The utils.verify_submission_format checks length against sampleSubmission.csv.
    # Since we only have 50 rows, it would fail.
    # We will manually verify the structure here instead of calling the strict utility
    # or just catch the specific length error if we wanted to be robust.
    # However, to strictly follow "Verify Logic", we check columns and types manually.

    expected_cols = ["request_id", "requester_received_pizza"]
    if list(submission_df.columns) != expected_cols:
        raise ValueError(f"Submission columns mismatch. Expected {expected_cols}")

    if not pd.api.types.is_numeric_dtype(submission_df["requester_received_pizza"]):
        raise ValueError("Prediction column is not numeric")

    print("Manual submission format check passed (Subset mode).")

    # Save submission
    output_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
