import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Add current directory to path to ensure library imports work
sys.path.append(".")

# ---------------------------------------------------------
# 1. Configuration & Setup
# ---------------------------------------------------------
from library.config import Config

# Override Config for a fast demonstration
Config.HIDDEN_DIM = 64  # Reduce model size for speed
Config.NUM_LAYERS = 2  # Reduce number of layers
Config.EPOCHS = 1  # Run only 1 epoch
Config.BATCH_SIZE = 16  # Use small batch size
Config.WORKING_DIR = "./working/demo_run"
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Ensure the working directory exists
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import remaining modules (Config changes will apply to these)
from library.utils import set_seed, mcrmse_metric
from library.dataset import get_dataloaders
from library.model import VectorScaledHighCapacityBiGRU
from library.loss import MaskedMSELoss
from library.train import Trainer


def run_demo():
    print("=== Starting RNA Degradation Library Demonstration ===")

    # Set fixed seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # Demo 1: Data Loading & Processing
    # ---------------------------------------------------------
    print("\n[1/5] Verifying Data Loading...")

    # Initialize dataloaders. load_cached_data=False forces processing from metadata
    # to demonstrate the full pipeline.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))

    # Assert keys exist
    expected_keys = {"seq", "loop", "dist", "targets", "mask"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Assert shapes
    # seq: (Batch, Seq_Len)
    # targets: (Batch, Seq_Len, 3)
    B = batch["seq"].shape[0]
    L = Config.SEQ_LEN
    assert batch["seq"].shape == (B, L), f"Seq shape mismatch: {batch['seq'].shape}"
    assert batch["targets"].shape == (
        B,
        L,
        3,
    ), f"Targets shape mismatch: {batch['targets'].shape}"
    assert batch["mask"].shape == (B, L), f"Mask shape mismatch: {batch['mask'].shape}"

    print("Data loading verified successfully.")

    # ---------------------------------------------------------
    # Demo 2: Model Instantiation & Forward Pass
    # ---------------------------------------------------------
    print("\n[2/5] Verifying Model Architecture...")

    model = VectorScaledHighCapacityBiGRU().to(device)

    # Move batch data to device
    seq = batch["seq"].to(device)
    loop = batch["loop"].to(device)
    dist = batch["dist"].to(device)

    # Perform forward pass
    preds = model(seq, loop, dist)

    # Verify output shape: (Batch, Seq_Len, 3 Targets)
    assert preds.shape == (B, L, 3), f"Prediction shape mismatch: {preds.shape}"

    print(f"Model forward pass successful. Output shape: {preds.shape}")

    # ---------------------------------------------------------
    # Demo 3: Loss Function Logic
    # ---------------------------------------------------------
    print("\n[3/5] Verifying Masked MSE Loss...")

    criterion = MaskedMSELoss()

    # Create deterministic dummy data
    # Batch=1, Len=4, Channels=1
    # Mask = [1, 1, 0, 0] -> Only first two positions are valid
    dummy_pred = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
    dummy_true = torch.tensor([[[1.5], [1.0], [10.0], [10.0]]], dtype=torch.float32)
    dummy_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)

    # Manual Calculation:
    # Pos 0: (1.0 - 1.5)^2 = 0.25
    # Pos 1: (2.0 - 1.0)^2 = 1.00
    # Pos 2 & 3: Masked (0 contribution)
    # Sum = 1.25, Count = 2
    # Mean = 0.625

    loss = criterion(dummy_pred, dummy_true, dummy_mask)
    expected_loss = 0.625

    assert torch.isclose(
        loss, torch.tensor(expected_loss), atol=1e-6
    ), f"Loss calculation incorrect. Expected {expected_loss}, got {loss.item()}"

    print(
        f"Loss function verified. Calculated: {loss.item()}, Expected: {expected_loss}"
    )

    # ---------------------------------------------------------
    # Demo 4: Metric Calculation (MCRMSE)
    # ---------------------------------------------------------
    print("\n[4/5] Verifying MCRMSE Metric...")

    # Create dummy data: Batch=1, Len=5, Channels=3
    # We simulate scoring only the first 2 positions (pred_len=2)
    y_true = np.zeros((1, 5, 3))
    y_pred = np.zeros((1, 5, 3))

    # Col 0: Error of 1.0 at pos 0 and 1 -> MSE=1.0 -> RMSE=1.0
    y_true[0, :2, 0] = 0.0
    y_pred[0, :2, 0] = 1.0

    # Col 1: Error of 2.0 at pos 0 and 1 -> MSE=4.0 -> RMSE=2.0
    y_true[0, :2, 1] = 0.0
    y_pred[0, :2, 1] = 2.0

    # Col 2: No error -> RMSE=0.0

    # MCRMSE = Mean(1.0, 2.0, 0.0) = 1.0

    metric_val = mcrmse_metric(y_true, y_pred, pred_len=2)
    assert np.isclose(
        metric_val, 1.0
    ), f"Metric incorrect. Expected 1.0, got {metric_val}"

    print(f"Metric verified. Value: {metric_val}")

    # ---------------------------------------------------------
    # Demo 5: Full Training & Inference Cycle
    # ---------------------------------------------------------
    print("\n[5/5] Running Full Training Cycle (Trainer)...")

    trainer = Trainer(debug=True)

    # Run fit: Trains for 1 epoch, validates, and generates submission
    # We use load_cached_data=True here to use the cache created in Demo 1
    trainer.fit(load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")

    # Expected rows: 240 test samples * 107 positions
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print("Training cycle and submission generation successful.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
