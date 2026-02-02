import sys
import os
import pandas as pd
import numpy as np
import torch

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, compute_auc
from library.data_loader import load_data
from library.features import FeatureProcessor
from library.model_rf import run_rf_pipeline
from library.model_mlp import run_mlp_pipeline


def run_demo():
    print("--- Starting Pipeline Demonstration ---")

    # 1. Configuration Overrides for Speed & Demonstration
    # We modify the Config class attributes directly to run a lightweight version
    print("Configuring for fast execution (Debug Mode)...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for speed
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data

    # Random Forest Speedups
    Config.RF_HYPERPARAMETERS["n_estimators"] = 5

    # MLP Speedups
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16

    # Force reload of data/features to ensure pipeline logic runs
    Config.LOAD_CACHED_DATA = False

    # Set reproducibility seed
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n[Step 1] Loading Data...")
    # load_data handles reading metadata CSVs and basic type conversion
    train_df, val_df, test_df = load_data(debug=Config.DEBUG, load_cached_data=False)

    # Verification
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    assert (
        len(train_df) == Config.DEBUG_SAMPLE_SIZE
    ), "Train size mismatch for debug mode"
    assert len(val_df) == Config.DEBUG_SAMPLE_SIZE, "Val size mismatch for debug mode"
    assert (
        "requester_received_pizza" in train_df.columns
    ), "Target column missing in train"
    assert "request_text_edit_aware" in test_df.columns, "Text column missing in test"

    # 3. Feature Engineering
    print("\n[Step 2] Processing Features...")
    processor = FeatureProcessor()

    # We pass load_cached_data=False to force the processor to run the SBERT/TF-IDF logic
    # instead of loading a potentially existing full-dataset cache.
    processed_data = processor.process_data(
        train_df, val_df, test_df, load_cached_data=False
    )

    # Verify output structure
    expected_keys = [
        "train_rf",
        "train_mlp_sem",
        "train_mlp_rel",
        "train_mlp_comm",
        "train_y",
        "val_rf",
        "val_y",
        "test_rf",
    ]
    for k in expected_keys:
        assert k in processed_data, f"Missing key '{k}' in processed features"

    # Verify feature dimensions
    rf_feats = processed_data["train_rf"]
    mlp_sem = processed_data["train_mlp_sem"]
    print(f"RF Feature Matrix Shape: {rf_feats.shape}")
    print(f"MLP Semantic Matrix Shape: {mlp_sem.shape}")

    assert rf_feats.shape[0] == Config.DEBUG_SAMPLE_SIZE
    assert mlp_sem.shape[0] == Config.DEBUG_SAMPLE_SIZE

    # 4. Random Forest Pipeline
    print("\n[Step 3] Running Random Forest Pipeline...")
    rf_model, rf_val_probs, rf_test_probs = run_rf_pipeline(processed_data)

    # Verify RF Outputs
    assert len(rf_val_probs) == len(val_df)
    assert len(rf_test_probs) == len(test_df)
    assert np.all(
        (rf_val_probs >= 0) & (rf_val_probs <= 1)
    ), "RF probabilities out of bounds"

    rf_auc = compute_auc(processed_data["val_y"], rf_val_probs)
    print(f"Random Forest Validation AUC: {rf_auc:.4f}")

    # 5. MLP Pipeline
    print("\n[Step 4] Running MLP Pipeline...")
    mlp_model, mlp_val_probs, mlp_test_probs = run_mlp_pipeline(processed_data)

    # Verify MLP Outputs
    assert len(mlp_val_probs) == len(val_df)
    assert len(mlp_test_probs) == len(test_df)
    assert np.all(
        (mlp_val_probs >= 0) & (mlp_val_probs <= 1)
    ), "MLP probabilities out of bounds"

    mlp_auc = compute_auc(processed_data["val_y"], mlp_val_probs)
    print(f"MLP Validation AUC: {mlp_auc:.4f}")

    # 6. Ensemble
    print("\n[Step 5] Ensembling...")
    # Simple weighted average
    w_rf = Config.ENSEMBLE_WEIGHT_RF
    w_mlp = Config.ENSEMBLE_WEIGHT_MLP

    ensemble_val_probs = (w_rf * rf_val_probs) + (w_mlp * mlp_val_probs)
    ensemble_test_probs = (w_rf * rf_test_probs) + (w_mlp * mlp_test_probs)

    ensemble_auc = compute_auc(processed_data["val_y"], ensemble_val_probs)
    print(f"Ensemble Validation AUC: {ensemble_auc:.4f}")

    # 7. Generate Submission
    print("\n[Step 6] Generating Demo Submission...")
    submission = pd.DataFrame(
        {
            "request_id": test_df["request_id"],
            "requester_received_pizza": ensemble_test_probs,
        }
    )

    out_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission.to_csv(out_path, index=False)

    # Final Verification
    assert os.path.exists(out_path), "Submission file was not created"
    loaded_sub = pd.read_csv(out_path)
    assert len(loaded_sub) == len(test_df), "Submission length mismatch"

    print(f"Submission saved to {out_path}")
    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    run_demo()
