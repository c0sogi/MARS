import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, MaskedL1Loss
from library.model import MCRHNet
from library.train import Runner


def main():
    print("=== Starting MCRH-Net Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Set a specific directory for this demo execution to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset for speed
    Config.EPOCHS = 2  # Minimal epochs for demo
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo script

    # Update dependent paths (since they were initialized at import time)
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Update cache paths used by data_loader and runner
    Config.CACHE_TRAIN_DATA = os.path.join(Config.WORKING_DIR, "train_x.npy")
    Config.CACHE_TRAIN_TARGETS = os.path.join(Config.WORKING_DIR, "train_y.npy")
    Config.CACHE_VAL_DATA = os.path.join(Config.WORKING_DIR, "val_x.npy")
    Config.CACHE_VAL_TARGETS = os.path.join(Config.WORKING_DIR, "val_y.npy")
    Config.CACHE_TEST_DATA = os.path.join(Config.WORKING_DIR, "test_x.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated successfully.")

    # ---------------------------------------------------------
    # 2. Verify Loss Function Logic
    # ---------------------------------------------------------
    print("\n[2] Verifying MaskedL1Loss logic...")
    criterion = MaskedL1Loss()

    # Create dummy data: Batch=2, Seq=5
    # Case 1: Perfect prediction
    pred = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    target = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    u_out = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # All inspiratory
    loss = criterion(pred, target, u_out)
    assert (
        loss.item() == 0.0
    ), f"Expected 0.0 loss for perfect prediction, got {loss.item()}"

    # Case 2: Error in expiratory phase (u_out=1) should be ignored
    pred = torch.tensor([[100.0]])  # Huge error
    target = torch.tensor([[10.0]])
    u_out = torch.tensor([[1.0]])  # Expiratory
    loss = criterion(pred, target, u_out)
    # The loss function returns 0.0 if no valid elements, or ignores masked elements
    assert (
        loss.item() == 0.0
    ), f"Expected 0.0 loss for masked expiratory phase, got {loss.item()}"

    # Case 3: Error in inspiratory phase (u_out=0) should be counted
    pred = torch.tensor([[12.0]])
    target = torch.tensor([[10.0]])
    u_out = torch.tensor([[0.0]])
    loss = criterion(pred, target, u_out)
    assert abs(loss.item() - 2.0) < 1e-6, f"Expected 2.0 loss, got {loss.item()}"

    print("MaskedL1Loss logic verified.")

    # ---------------------------------------------------------
    # 3. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[3] Verifying MCRHNet architecture...")
    input_features = 15  # Approximate feature count
    model = MCRHNet(input_dim=input_features)

    # Create dummy input: (Batch, Seq_Len, Features)
    dummy_input = torch.randn(4, 80, input_features)

    # Forward pass
    output = model(dummy_input)

    # Check output shape: (Batch, Seq_Len, 1)
    expected_shape = (4, 80, 1)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    print("MCRHNet forward pass successful.")

    # ---------------------------------------------------------
    # 4. Run Training Pipeline
    # ---------------------------------------------------------
    print("\n[4] Initializing Runner and starting training loop...")
    # The Runner handles data loading (calling prepare_data), model init, and training
    runner = Runner()

    # Verify data loading happened correctly by checking internal state
    assert runner.train_loader is not None
    assert runner.input_shape > 0
    print(f"Data loaded. Input features: {runner.input_shape}")

    # Run training
    runner.train()

    # Check if model checkpoint was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint verified at {Config.MODEL_PATH}")
    else:
        # If validation loss didn't improve (unlikely in 2 epochs starting from scratch),
        # we check if the script completed without error.
        print(
            "Training finished (no checkpoint saved due to short run or no improvement)."
        )

    # ---------------------------------------------------------
    # 5. Generate Submission
    # ---------------------------------------------------------
    print("\n[5] Generating submission...")
    runner.generate_submission()

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission generated with shape: {sub_df.shape}")

    # Verify columns
    expected_cols = ["id", "pressure"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(sub_df.columns)}"

    # Verify no NaN values
    assert not sub_df.isnull().values.any(), "Submission contains NaN values"

    # Verify IDs are integers (common requirement)
    assert pd.api.types.is_numeric_dtype(sub_df["id"]), "ID column is not numeric"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
