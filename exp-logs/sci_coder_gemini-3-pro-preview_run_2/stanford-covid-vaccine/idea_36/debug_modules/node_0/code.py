import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_loader
from library.model import CF_DCN
from library.train import train_one_epoch, validate, generate_submission


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("1. Setting up configuration for demo execution...")

    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to use the demo directory
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Update model and submission paths
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Reduce compute load for demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG_SIZE = 20  # Use only 20 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = str(device)

    # Set seed
    seed_everything(Config.SEED)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Device: {device}")

    # =========================================================================
    # 2. Data Loading Demonstration
    # =========================================================================
    print("\n2. Testing Data Loading...")

    # Load Train Loader in Debug mode
    train_loader = get_loader(mode="train", debug=True, load_cached_data=False)

    # Fetch a single batch
    inputs, partner_indices, targets = next(iter(train_loader))

    # Assertions for Data Shapes
    # Inputs: (Batch, Seq_Len, Channels) -> (4, 107, 18)
    expected_channels = 4 + 3 + 7 + 4  # 18
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        expected_channels,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, expected_channels)}, got {inputs.shape}"

    # Partner Indices: (Batch, Seq_Len) -> (4, 107)
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Partner indices shape mismatch. Got {partner_indices.shape}"

    # Targets: (Batch, Seq_Len, 5) -> (4, 107, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), f"Target shape mismatch. Got {targets.shape}"

    print("   Data loaded successfully. Shapes verified.")

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n3. Testing Model Initialization and Forward Pass...")

    model = CF_DCN().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Run full inference forward pass
    preds = model(inputs, partner_indices)

    # Assert Output Shape
    assert (
        preds.shape == targets.shape
    ), f"Prediction shape {preds.shape} does not match target shape {targets.shape}"

    print("   Forward pass successful.")

    # =========================================================================
    # 4. Loss Calculation
    # =========================================================================
    print("\n4. Testing Loss Calculation...")

    loss = mcrmse_loss(preds, targets, Config.SCORED_TARGET_INDICES)

    assert loss.dim() == 0, "Loss should be a scalar tensor."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"   MCRMSE Loss calculated: {loss.item():.4f}")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n5. Running Training Loop (1 Epoch)...")

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device)

    print(f"   Epoch 1 Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # Validate
    val_loader = get_loader(mode="val", debug=True, load_cached_data=False)
    val_loss = validate(model, val_loader, device)

    print(f"   Validation Loss: {val_loss:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN."

    # Save the model (required for submission generation step)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print("   Model saved.")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n6. Generating Submission...")

    # Ensure test data cache is created for the debug subset
    # Note: generate_submission calls get_loader internally.
    # Since we set Config.DEBUG_SIZE, we need to ensure the test loader respects that
    # or we just let it run on the full test set (it's small, 240 samples).
    # However, generate_submission in library/train.py hardcodes load_cached_data=True.
    # We will manually trigger the processing first to ensure cache exists.
    _ = get_loader(mode="test", debug=False, load_cached_data=False)

    # Generate submission
    generate_submission(model, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
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
    ), f"Submission columns mismatch. Got {df_sub.columns}"

    # Check row count
    # Test set has 240 samples. Seq length is 107. Total rows = 240 * 107 = 25680.
    # Note: If we ran debug on test loader inside generate_submission, count would differ.
    # But generate_submission uses standard loader call, so it processes all 240 test samples.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    print("   Submission verification passed.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
