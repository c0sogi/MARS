import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_all, mcrmse_metric
from library.data import get_dataloaders
from library.model import HighCapacityBiGRU
from library.train import run_training


def main():
    print("=== RNA Degradation Prediction: Library Usage Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Execution
    # ---------------------------------------------------------
    print("1. Configuring environment for rapid demonstration...")

    # Override Config attributes to run a minimal version
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size

    # Setup a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths to point to the demo directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npy")

    # Set seeds for reproducibility
    seed_all(42)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading Demonstration
    # ---------------------------------------------------------
    print("\n2. Verifying Data Loading and Shapes...")

    # Load dataloaders (force reprocessing to ensure cache uses subset)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch a single batch
    inputs, adjacency, targets = next(iter(train_loader))

    print(f"   Input Batch Shape:     {inputs.shape}")
    print(f"   Adjacency Batch Shape: {adjacency.shape}")
    print(f"   Target Batch Shape:    {targets.shape}")

    # Verify Shapes
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)}, got {inputs.shape}"

    # Adjacency: (Batch, Seq_Len=107)
    assert adjacency.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Adjacency shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {adjacency.shape}"

    # Targets: (Batch, Seq_Len=107, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 5)}, got {targets.shape}"

    print("   ✓ Data shapes verified.")

    # ---------------------------------------------------------
    # 3. Metric Logic Verification
    # ---------------------------------------------------------
    print("\n3. Verifying MCRMSE Metric Logic...")

    # Create synthetic data
    # Batch=2, Seq=107, Targets=5
    y_true_dummy = torch.zeros(2, 107, 5)
    y_pred_dummy = torch.zeros(2, 107, 5)

    # Introduce a known error in the first scored position of the first sample
    # Scored targets indices: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # We put an error of 1.0 in index 0.
    y_true_dummy[0, 0, 0] = 1.0
    y_pred_dummy[0, 0, 0] = 0.0

    # Calculate metric using library function
    calculated_score = mcrmse_metric(y_true_dummy, y_pred_dummy)

    # Manual Calculation:
    # We score 3 columns.
    # Col 0 (reactivity): MSE = (1.0^2 + 0...) / (2 * 68) = 1/136. RMSE = sqrt(1/136)
    # Col 1 (deg_Mg_pH10): MSE = 0. RMSE = 0
    # Col 3 (deg_Mg_50C): MSE = 0. RMSE = 0
    # MCRMSE = (sqrt(1/136) + 0 + 0) / 3

    expected_rmse_col0 = np.sqrt(1.0 / (2 * 68))
    expected_score = expected_rmse_col0 / 3.0

    print(f"   Calculated Score: {calculated_score:.6f}")
    print(f"   Expected Score:   {expected_score:.6f}")

    assert np.isclose(
        calculated_score, expected_score, atol=1e-6
    ), "Metric calculation did not match expected value."
    print("   ✓ Metric logic verified.")

    # ---------------------------------------------------------
    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n4. Verifying Model Architecture (Forward Pass)...")

    model = HighCapacityBiGRU()
    model.to(Config.DEVICE)

    # Move batch to device
    inputs_dev = inputs.to(Config.DEVICE)
    adj_dev = adjacency.to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs_dev, adj_dev)

    print(f"   Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 5)}, got {outputs.shape}"
    print("   ✓ Model forward pass verified.")

    # ---------------------------------------------------------
    # 5. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n5. Running Full Training Pipeline (Debug Mode)...")

    # This runs training, validation, and generates submission
    run_training(debug=True)

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission file found: {Config.SUBMISSION_PATH}")
        print(f"   Submission shape: {sub_df.shape}")

        # Expected rows: N_test_samples (20) * Seq_Len (107) = 2140
        expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
        assert (
            len(sub_df) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

        # Check columns
        expected_cols = ["id_seqpos"] + Config.TARGET_COLS
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

        print("   ✓ Pipeline execution and submission generation verified.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
