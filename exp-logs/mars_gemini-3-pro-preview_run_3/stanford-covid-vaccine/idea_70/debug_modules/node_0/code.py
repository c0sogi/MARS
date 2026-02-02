import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library import utils, data, model, engine


def main():
    # =========================================================================
    # 1. Configuration & Setup for Demo
    # =========================================================================
    print("Setting up demonstration configuration...")

    # Override Config parameters to ensure speed and isolation
    # We use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Point caches to the demo directory so we don't overwrite or use full datasets
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data_demo.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data_demo.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data_demo.npz")

    # Output paths
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Enable Debug mode to use a tiny subset of data (e.g., 50 samples)
    Config.debug = True
    Config.debug_subset_size = 50

    # Reduce Model Capacity for fast initialization and forward pass
    Config.hidden_dim = 32  # Original: 384
    Config.gate_hidden_dim = 32  # Original: 384
    Config.stem_channels = 16  # Original: 256
    Config.num_layers = 1  # Original: 4

    # Training Hyperparameters for fast execution
    Config.epochs = 2
    Config.batch_size = 8
    Config.num_workers = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    utils.set_seed(Config.seed)

    print(f"Demo Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.debug} (Subset size: {Config.debug_subset_size})")

    # =========================================================================
    # 2. Data Processing & Loading Verification
    # =========================================================================
    print("\n--- Verifying Data Loading ---")

    # Force load_cached_data=False to ensure we process the debug subset from metadata
    # and save it to our new demo cache paths.
    train_loader, val_loader, test_loader = data.get_loaders(
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        load_cached_data=False,
    )

    # Fetch a single batch to verify shapes
    x, pair_indices, pair_mask, y = next(iter(train_loader))

    print(f"Feature Batch Shape: {x.shape}")
    print(f"Target Batch Shape: {y.shape}")

    # Assertions
    # x: (Batch, Seq_Len=107, Input_Dim=14)
    assert x.shape == (
        Config.batch_size,
        107,
        14,
    ), f"Incorrect feature shape: {x.shape}"
    # y: (Batch, Seq_Scored=68, Num_Targets=5)
    assert y.shape == (Config.batch_size, 68, 5), f"Incorrect target shape: {y.shape}"
    # pair_indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.batch_size,
        107,
    ), "Incorrect pair_indices shape"

    print("Data loading verified successfully.")

    # =========================================================================
    # 3. Model Initialization & Forward Pass Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")

    device = torch.device(Config.device)
    net = model.RNAModel(config=Config).to(device)

    # Move batch to device
    x = x.to(device)
    pair_indices = pair_indices.to(device)
    pair_mask = pair_mask.to(device)
    y = y.to(device)

    # Forward Pass
    preds = net(x, pair_indices, pair_mask)
    print(f"Prediction Shape: {preds.shape}")

    # Assertions
    # Output should be (Batch, Seq_Len=107, Num_Targets=5)
    assert preds.shape == (
        Config.batch_size,
        107,
        5,
    ), f"Incorrect output shape: {preds.shape}"

    print("Model forward pass verified successfully.")

    # =========================================================================
    # 4. Loss Function Verification
    # =========================================================================
    print("\n--- Verifying Loss Function ---")

    criterion = utils.MCRMSELoss()
    loss = criterion(preds, y)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Loss function verified successfully.")

    # =========================================================================
    # 5. Full Training Pipeline Execution
    # =========================================================================
    print("\n--- Executing Training Pipeline (Mini-Run) ---")

    # engine.run_training() uses the Config class we modified.
    # It will re-load data. Since we set load_cached_data=True inside run_training (hardcoded),
    # it will pick up the cache files we generated in Step 2.
    engine.run_training()

    print("Training pipeline finished.")

    # =========================================================================
    # 6. Submission Verification
    # =========================================================================
    print("\n--- Verifying Submission File ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print("First 5 rows:")
    print(sub_df.head())

    # Validation Logic
    # We used a debug subset of 50 samples for the test set.
    # Each sample has 107 positions.
    # Total rows should be 50 * 107 = 5350.
    expected_rows = Config.debug_subset_size * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.target_cols
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaNs"

    print("Submission verified successfully.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
