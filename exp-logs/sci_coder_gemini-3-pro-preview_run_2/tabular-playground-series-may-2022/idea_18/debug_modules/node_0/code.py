import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import MSResFunnel
from library.engine import train_model


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # --------------------------------------------------------------------------
    print("Configuring execution parameters...")

    # Enable debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000  # Small sample size for speed

    # Reduce training duration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 32

    # Set output paths for this demo run
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("\nLoading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force processing to demonstrate pipeline
        debug=Config.DEBUG,
    )

    # Verify Data Structure
    print("Verifying data loader output...")
    batch = next(iter(train_loader))

    # Check keys
    assert "continuous" in batch, "Batch missing 'continuous' key"
    assert "categorical" in batch, "Batch missing 'categorical' key"
    assert "target" in batch, "Batch missing 'target' key"

    # Check shapes
    # Continuous: (Batch, 30)
    assert batch["continuous"].shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONT_FEATURES,
    ), f"Incorrect continuous shape: {batch['continuous'].shape}"

    # Categorical: (Batch, 10)
    assert batch["categorical"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect categorical shape: {batch['categorical'].shape}"

    # Target: (Batch)
    assert batch["target"].shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {batch['target'].shape}"

    print("Data loader verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("\nInitializing model...")
    device = Config.DEVICE
    model = MSResFunnel().to(device)

    # Verify Forward Pass
    print("Verifying model forward pass...")
    cont_input = batch["continuous"].to(device)
    cat_input = batch["categorical"].to(device)

    with torch.no_grad():
        output = model(cont_input, cat_input)

    # Check output shape: (Batch, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect model output shape: {output.shape}"

    # Check value range (Sigmoid output should be [0, 1])
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output out of probability range [0, 1]"

    print("Model verification passed.")

    # --------------------------------------------------------------------------
    # 4. Training Engine Execution
    # --------------------------------------------------------------------------
    print("\nStarting training engine...")

    # Run the full training and inference pipeline
    # This handles loops, validation, saving checkpoints, and generating submission
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_ids=test_ids,
        epochs=Config.EPOCHS,
        patience=1,  # Strict early stopping for demo
        device=device,
    )

    # --------------------------------------------------------------------------
    # 5. Output Verification
    # --------------------------------------------------------------------------
    print("\nVerifying outputs...")

    # Check for submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Expected rows = DEBUG_SAMPLES (since test set is also truncated in debug mode)
    expected_rows = Config.DEBUG_SAMPLES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for model checkpoint
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {best_model_path}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
