import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import load_data, RNADataset
from library.model import WideResBiGRU
from library.utils import mcrmse_loss
from library.train import train_model
from library.predict import predict


def run_demo():
    print("Starting RNA Degradation Pipeline Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Modify Config for a fast demo run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.WORKING_DIR = "./working/demo_run"
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )

    # Ensure clean demo directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)
    os.makedirs(os.path.join(Config.WORKING_DIR, "checkpoints"), exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Dataset Verification
    # =========================================================================
    print("\n--- Verifying Dataset Loading ---")
    subset_size = 16
    train_dataset = load_data(mode="train", subset_size=subset_size)

    assert isinstance(
        train_dataset, RNADataset
    ), "load_data should return an RNADataset instance"
    assert len(train_dataset) == subset_size, f"Dataset size should be {subset_size}"

    # Check a single item
    item = train_dataset[0]
    required_keys = {"sequence", "loop_type", "distance", "targets", "id"}
    assert required_keys.issubset(
        item.keys()
    ), f"Dataset item missing keys. Found: {item.keys()}"

    # Check shapes
    seq_len = Config.SEQ_LEN
    assert item["sequence"].shape == (
        seq_len,
    ), f"Sequence shape mismatch: {item['sequence'].shape}"
    assert item["loop_type"].shape == (
        seq_len,
    ), f"Loop type shape mismatch: {item['loop_type'].shape}"
    assert item["distance"].shape == (
        seq_len,
    ), f"Distance shape mismatch: {item['distance'].shape}"
    # Targets should be (107, 3) corresponding to [reactivity, deg_Mg_pH10, deg_Mg_50C]
    assert item["targets"].shape == (
        seq_len,
        3,
    ), f"Targets shape mismatch: {item['targets'].shape}"

    print("Dataset verification passed.")

    # =========================================================================
    # 3. Model Architecture Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    device = "cpu"  # Use CPU for simple logic verification
    model = WideResBiGRU().to(device)
    model.eval()

    # Create a dummy batch
    dataloader = DataLoader(train_dataset, batch_size=2)
    batch = next(iter(dataloader))

    seq = batch["sequence"].to(device)
    loop = batch["loop_type"].to(device)
    dist = batch["distance"].to(device)

    with torch.no_grad():
        output = model(seq, loop, dist)

    # Expected output: (Batch, Seq_Len, Num_Targets)
    expected_shape = (2, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model forward pass verification passed.")

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n--- Verifying MCRMSE Loss ---")
    # Create dummy data
    # Shape: (Batch=1, Seq_Len=107, Targets=3)
    # Config.PRED_LEN is 68. The loss only looks at the first 68 positions.
    y_true = torch.zeros((1, 107, 3))
    y_pred = torch.zeros((1, 107, 3))

    # Case 1: Perfect prediction
    loss_zero = mcrmse_loss(y_true, y_pred)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0)
    ), "Loss should be 0 for perfect predictions"

    # Case 2: Known error
    # Set error of 1.0 at one position in one column within the scored range
    y_pred[0, 0, 0] = 1.0
    # Calculation:
    # Column 0: RMSE = sqrt(mean((0-1)^2 + zeros...)) = sqrt(1/68)
    # Column 1: RMSE = 0
    # Column 2: RMSE = 0
    # MCRMSE = (sqrt(1/68) + 0 + 0) / 3
    expected_val = (np.sqrt(1 / 68)) / 3
    loss_val = mcrmse_loss(y_true, y_pred)

    assert torch.isclose(
        loss_val, torch.tensor(expected_val, dtype=torch.float32), atol=1e-5
    ), f"Loss calculation mismatch. Expected {expected_val}, got {loss_val.item()}"

    print("Loss function verification passed.")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n--- Running Training Demo ---")
    # We use a small subset to ensure it finishes quickly
    # train_model handles loading data, training, validation, and saving the best model
    try:
        train_model(subset_size=32)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training demo completed successfully.")

    # =========================================================================
    # 6. Inference Demonstration
    # =========================================================================
    print("\n--- Running Inference Demo ---")
    # predict handles loading the test set, loading the best model, and generating submission
    try:
        predict(subset_size=10)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    # Verify Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Expected rows: 10 samples * 107 positions = 1070 rows
    expected_rows = 10 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

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
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check that ignored columns (deg_pH10, deg_50C) are 0.0
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    # Check that predicted columns are not all zero (model should have learned something or initialized randomly non-zero)
    # Note: With very little training data/epochs, it might be close to 0, but unlikely exactly 0 everywhere if weights are random.
    assert (
        df_sub["reactivity"] != 0.0
    ).any(), "Reactivity predictions appear to be all zeros."

    print("Inference demo completed successfully.")
    print("\nAll demonstrations passed!")


if __name__ == "__main__":
    run_demo()
