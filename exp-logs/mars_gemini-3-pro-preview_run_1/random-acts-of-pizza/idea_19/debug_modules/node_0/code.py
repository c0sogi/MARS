import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
import library.config as config
from library.utils import seed_everything, save_submission
from library.data_loader import load_data
from library.features import FeatureEngineer
from library.engine import train_rf, train_mlp

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=" * 50)
    print("STARTING DEMO EXECUTION")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. SETUP & CONFIGURATION OVERRIDES
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    seed_everything(config.RANDOM_SEED)

    # Override configuration for speed
    # We monkey-patch the config module to use smaller parameters
    config.RF_PARAMS["n_estimators"] = 10  # Reduce trees
    config.RF_PARAMS["n_jobs"] = 1  # Avoid overhead for small data

    config.MLP_PARAMS["num_epochs"] = 2  # Minimal epochs
    config.MLP_PARAMS["batch_size"] = 8  # Small batch for small data
    config.MLP_PARAMS["hidden_dim"] = 16  # Smaller network
    config.MLP_PARAMS["patience"] = 1  # Aggressive early stopping

    # Reduce feature complexity for demo
    config.TFIDF_MAX_FEATURES = 100
    config.LSA_COMPONENTS = 5

    # Use a small subset of data
    DEMO_NROWS = 50
    print(f"    - Dataset size limit: {DEMO_NROWS} rows")
    print(f"    - RF Estimators: {config.RF_PARAMS['n_estimators']}")
    print(f"    - MLP Epochs: {config.MLP_PARAMS['num_epochs']}")

    # -------------------------------------------------------------------------
    # 2. DATA LOADING
    # -------------------------------------------------------------------------
    print("\n[2] Loading Data...")
    train_df, val_df, test_df = load_data(nrows=DEMO_NROWS)

    # Assertions to verify data loading
    assert (
        len(train_df) == DEMO_NROWS
    ), f"Expected {DEMO_NROWS} train rows, got {len(train_df)}"
    assert (
        len(val_df) == DEMO_NROWS
    ), f"Expected {DEMO_NROWS} val rows, got {len(val_df)}"
    assert (
        len(test_df) == DEMO_NROWS
    ), f"Expected {DEMO_NROWS} test rows, got {len(test_df)}"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"
    print("    - Data loaded and verified.")

    # -------------------------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n[3] Running Feature Engineering...")
    fe = FeatureEngineer()

    # We force load_cached=False to demonstrate the generation process
    # In a real run, this would be True to save time
    data_rf, data_mlp = fe.process_features(
        train_df, val_df, test_df, load_cached=False
    )

    # Verify RF Data Structure
    print("    - Verifying RF Data...")
    assert "X_train" in data_rf and "y_train" in data_rf
    assert data_rf["X_train"].shape[0] == DEMO_NROWS
    assert data_rf["X_train"].shape[1] > 0

    # Verify MLP Data Structure
    print("    - Verifying MLP Data...")
    assert "train" in data_mlp and "req_emb" in data_mlp["train"]
    assert data_mlp["train"]["req_emb"].shape == (DEMO_NROWS, 384)  # SBERT dim is 384
    assert data_mlp["train"]["hist_emb"].shape[0] == DEMO_NROWS

    print("    - Feature generation successful.")

    # -------------------------------------------------------------------------
    # 4. STREAM A: RANDOM FOREST
    # -------------------------------------------------------------------------
    print("\n[4] Training Stream A (Random Forest)...")
    rf_preds, rf_model = train_rf(data_rf)

    # Verify Predictions
    assert len(rf_preds["val"]) == DEMO_NROWS
    assert len(rf_preds["test"]) == DEMO_NROWS
    assert np.all(
        (rf_preds["val"] >= 0) & (rf_preds["val"] <= 1)
    ), "RF probabilities out of range"
    print("    - RF training complete.")

    # -------------------------------------------------------------------------
    # 5. STREAM B: MLP
    # -------------------------------------------------------------------------
    print("\n[5] Training Stream B (Credibility-Gated MLP)...")
    mlp_preds, mlp_model = train_mlp(data_mlp)

    # Verify Predictions
    assert len(mlp_preds["val"]) == DEMO_NROWS
    assert len(mlp_preds["test"]) == DEMO_NROWS
    assert np.all(
        (mlp_preds["val"] >= 0) & (mlp_preds["val"] <= 1)
    ), "MLP probabilities out of range"
    print("    - MLP training complete.")

    # -------------------------------------------------------------------------
    # 6. ENSEMBLING & SUBMISSION
    # -------------------------------------------------------------------------
    print("\n[6] Creating Ensemble and Submission...")

    # Simple weighted average
    w_rf = config.ENSEMBLE_WEIGHTS["rf"]
    w_mlp = config.ENSEMBLE_WEIGHTS["mlp"]

    final_test_preds = (w_rf * rf_preds["test"]) + (w_mlp * mlp_preds["test"])

    # Verify Ensemble
    assert len(final_test_preds) == DEMO_NROWS

    # Prepare submission
    # Note: In the real test set, request_id is unique. Here we just take from the loaded df.
    request_ids = test_df["request_id"].values

    # Save
    save_submission(
        request_ids, final_test_preds, output_path="./working/demo_submission.csv"
    )

    # Verify file creation
    assert os.path.exists("./working/demo_submission.csv"), "Submission file not found"

    # Check content
    sub_df = pd.read_csv("./working/demo_submission.csv")
    assert sub_df.shape == (DEMO_NROWS, 2)
    assert list(sub_df.columns) == ["request_id", "requester_received_pizza"]

    print(f"\n[SUCCESS] Demo completed successfully.")
    print(f"Submission saved to ./working/demo_submission.csv")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
