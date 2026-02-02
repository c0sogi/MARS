import sys
import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import TransformerResFunnel
from library.train import run_training


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("Step 1: Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True  # Use a small subset (10,000 samples)
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 1024  # Reasonable batch size for the subset
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Ensure working directory exists (handled by library, but good to double check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\nStep 2: Verifying Data Loaders...")

    # Get dataloaders with debug=True
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    x_cont = batch["cont"]
    x_cat = batch["cat"]
    y = batch["target"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Continuous features shape: {x_cont.shape}")
    print(f"Categorical features shape: {x_cat.shape}")
    print(f"Target shape: {y.shape}")

    # Assertions for Data Integrity
    # Continuous features: Batch x 30 (f_00 to f_30 excluding f_27)
    assert x_cont.shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Expected continuous shape ({Config.BATCH_SIZE}, 30), got {x_cont.shape}"

    # Categorical features: Batch x 10 (f_27 is 10 chars long)
    assert x_cat.shape == (
        Config.BATCH_SIZE,
        10,
    ), f"Expected categorical shape ({Config.BATCH_SIZE}, 10), got {x_cat.shape}"

    # Target: Batch size
    assert y.shape == (
        Config.BATCH_SIZE,
    ), f"Expected target shape ({Config.BATCH_SIZE},), got {y.shape}"

    print("Data Loader verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\nStep 3: Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = TransformerResFunnel().to(device)

    # Move batch to device
    x_cont_dev = x_cont.to(device)
    x_cat_dev = x_cat.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(x_cont_dev, x_cat_dev)

    print(f"Logits shape: {logits.shape}")

    # Assertions for Model Output
    # Output should be (Batch, 1) - raw logits
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    print("Model architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\nStep 4: Executing Training Loop (1 Epoch, Debug Subset)...")

    # We use the provided run_training function which encapsulates the loop
    # passing debug=True and epochs=1 explicitly
    run_training(debug=Config.DEBUG, epochs=Config.EPOCHS)

    # Check if model file was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"

    print("Training execution completed successfully.")

    # --------------------------------------------------------------------------
    # 5. Submission Verification
    # --------------------------------------------------------------------------
    print("\nStep 5: Verifying Submission File...")

    submission_path = Config.SUBMISSION_SAVE_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Assertions for Submission
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission missing required columns 'id' or 'target'"

    # In debug mode, the test set is also sliced.
    # Config.MAX_DEBUG_SAMPLES is 10000.
    # The submission should reflect this size.
    expected_len = Config.MAX_DEBUG_SAMPLES
    assert (
        len(df_sub) == expected_len
    ), f"Expected {expected_len} predictions in debug mode, got {len(df_sub)}"

    # Check probability range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Predictions contain values outside [0, 1] probability range"

    print("Submission verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
