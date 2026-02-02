import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import ScaledResidualWideStreamBiGRU
from library.train import train_model


def run_demonstration():
    # 1. Setup
    print("--- Setting up environment ---")
    warnings.filterwarnings("ignore")
    seed_everything(Config.SEED)

    # Ensure working directories exist (usually handled by Config.setup, but good to be explicit)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # 2. Metric Verification
    print("\n--- Verifying MCRMSE Metric ---")
    # Create dummy data: 2 samples, 3 columns
    # Sample 1: Error 1.0 on col 0, 0.0 elsewhere
    # Sample 2: Error 0.0 everywhere
    y_true = np.zeros((2, 5, 3))
    y_pred = np.zeros((2, 5, 3))

    # Introduce error
    y_pred[0, :, 0] = 1.0  # Error of 1.0 for all 5 positions in col 0 for sample 0

    # Manual Calculation:
    # Col 0: Sample 0 MSE = 1.0, Sample 1 MSE = 0.0 -> Mean MSE = 0.5 -> RMSE = sqrt(0.5) ≈ 0.7071
    # Col 1: MSE = 0 -> RMSE = 0
    # Col 2: MSE = 0 -> RMSE = 0
    # MCRMSE = (0.7071 + 0 + 0) / 3 ≈ 0.2357

    score = mcrmse_metric(y_true, y_pred)
    expected_score = np.sqrt(0.5) / 3.0

    print(f"Calculated Score: {score:.6f}")
    print(f"Expected Score:   {expected_score:.6f}")

    assert np.isclose(score, expected_score, atol=1e-5), "MCRMSE calculation mismatch!"
    print("Metric verification passed.")

    # 3. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")
    # Load a small subset to verify shapes
    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, debug_subset=20
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Keys
    expected_keys = {"sequence", "loop_type", "pair_dist", "targets"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Sequence: (Batch, 107)
    seq = batch["sequence"]
    assert (
        seq.dim() == 2 and seq.shape[1] == Config.SEQ_LEN
    ), f"Incorrect sequence shape: {seq.shape}"

    # Targets: (Batch, 107, 3)
    targets = batch["targets"]
    assert (
        targets.dim() == 3
        and targets.shape[1] == Config.SEQ_LEN
        and targets.shape[2] == 3
    ), f"Incorrect target shape: {targets.shape}"

    # Pair Dist: (Batch, 107)
    pair_dist = batch["pair_dist"]
    assert pair_dist.shape == seq.shape, "Pair dist shape mismatch"

    print(f"Batch shapes verified. Sequence: {seq.shape}, Targets: {targets.shape}")

    # 4. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    model = ScaledResidualWideStreamBiGRU().to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = batch["loop_type"].to(device)
    dist = pair_dist.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(seq, loop, dist)

    # Check Output Shape: (Batch, 107, 3)
    # 3 corresponds to len(Config.TARGET_COLS)
    expected_shape = (seq.shape[0], Config.SEQ_LEN, len(Config.TARGET_COLS))
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"Model forward pass successful. Output shape: {output.shape}")

    # 5. End-to-End Training Demonstration
    print("\n--- Running End-to-End Training (Demo) ---")
    # Running with a small debug subset and 2 epochs for speed
    # This uses the train_model function from library/train.py

    debug_subset_size = 50
    demo_epochs = 2

    train_model(debug_subset=debug_subset_size, epochs=demo_epochs)

    # 6. Submission Verification
    print("\n--- Verifying Submission File ---")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")

    # Check Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check Row Count
    # Test set has 240 samples (from metadata info), each has 107 positions
    # Note: debug_subset in get_dataloaders only slices the training set, not the test set.
    # So we expect full test set predictions.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    # Check that unscored columns are 0.0 (as per generate_submission logic)
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demonstration()
