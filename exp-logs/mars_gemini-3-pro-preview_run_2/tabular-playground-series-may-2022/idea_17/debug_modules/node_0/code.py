import sys
import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import HybridTransformerResFunnel
from library.train import train, inference


def run_demonstration():
    print("=== Starting Demonstration of Manufacturing Control Pipeline ===")

    # --------------------------------------------------------------------------
    # 1. Setup Environment
    # --------------------------------------------------------------------------
    print("\n[Step 1] Setting up environment...")
    # Ensure reproducibility
    seed_everything(Config.RANDOM_STATE)

    # Detect device
    device = get_device()
    print(f"Device selected: {device}")

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading and Processing...")

    # We use a small debug subset (2048 samples) to ensure this step runs quickly.
    # This will trigger the internal process_data() function, which handles:
    # - Loading metadata and raw CSVs
    # - Feature Engineering (Sequence encoding, Normalization, Binning)
    # - Caching the processed data to disk
    subset_size = 2048

    print(f"Loading data with debug_subset={subset_size}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug_subset=subset_size
    )

    # Fetch a single batch to verify structure and shapes
    print("Fetching a batch from train_loader...")
    batch = next(iter(train_loader))

    # Verify Dictionary Keys
    expected_keys = {"x_seq", "x_raw", "x_binned", "target"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing required keys. Found: {batch.keys()}"

    # Verify Tensor Shapes
    # Note: Config.BATCH_SIZE is 1024. With subset 2048, we expect full batches.
    current_batch_size = batch["x_seq"].size(0)
    print(f"Batch size: {current_batch_size}")

    # x_seq: Character sequence indices (Batch, Sequence_Length)
    assert batch["x_seq"].shape == (
        current_batch_size,
        Config.SEQUENCE_LENGTH,
    ), f"x_seq shape mismatch. Expected {(current_batch_size, Config.SEQUENCE_LENGTH)}, got {batch['x_seq'].shape}"

    # x_raw: Normalized continuous features (Batch, Num_Continuous)
    assert batch["x_raw"].shape == (
        current_batch_size,
        Config.NUM_CONTINUOUS_FEATURES,
    ), f"x_raw shape mismatch. Expected {(current_batch_size, Config.NUM_CONTINUOUS_FEATURES)}, got {batch['x_raw'].shape}"

    # x_binned: Quantized continuous features (Batch, Num_Continuous)
    assert batch["x_binned"].shape == (
        current_batch_size,
        Config.NUM_CONTINUOUS_FEATURES,
    ), f"x_binned shape mismatch. Expected {(current_batch_size, Config.NUM_CONTINUOUS_FEATURES)}, got {batch['x_binned'].shape}"

    # target: Binary labels (Batch,)
    assert batch["target"].shape == (
        current_batch_size,
    ), f"target shape mismatch. Expected {(current_batch_size,)}, got {batch['target'].shape}"

    print("Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    # Instantiate the model
    model = HybridTransformerResFunnel().to(device)
    print("Model instantiated.")

    # Move batch data to the correct device
    x_seq = batch["x_seq"].to(device)
    x_raw = batch["x_raw"].to(device)
    x_binned = batch["x_binned"].to(device)

    # Perform Forward Pass
    print("Performing forward pass...")
    logits = model(x_seq, x_raw, x_binned)

    # Verify Output Shape: Should be (Batch, 1) for binary classification logits
    assert logits.shape == (
        current_batch_size,
        1,
    ), f"Model output shape mismatch. Expected {(current_batch_size, 1)}, got {logits.shape}"

    # Verify Validity: Check for NaNs which indicate numerical instability
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("Model forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[Step 4] Verifying Training Loop...")

    # We call the library's train function.
    # We override epochs to 1 and use the debug_subset to ensure it finishes in seconds.
    # This validates the optimizer, loss calculation, backprop, and checkpointing logic.
    print("Starting short training run (1 epoch, 2048 samples)...")

    train(debug_subset=subset_size, epochs=1, patience=1)

    # Verify that the model checkpoint was created
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint was not created at {Config.MODEL_SAVE_PATH}"

    print(f"Training loop completed. Checkpoint verified at {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference Verification
    # --------------------------------------------------------------------------
    print("\n[Step 5] Verifying Inference Pipeline...")

    # The inference function loads the best model (saved in Step 4)
    # and generates predictions for the FULL test set.
    # Inference on 100k samples is fast enough to run fully (~10-30s on GPU).
    print("Generating submission for full test set...")
    inference()

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check Columns
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], f"Submission columns incorrect. Found: {df_sub.columns}"

    # Check Row Count (Should be 100,000 for the test set)
    assert (
        len(df_sub) == 100000
    ), f"Submission row count incorrect. Expected 100000, got {len(df_sub)}"

    # Check Probability Range
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Submission target values are outside the probability range [0, 1]."

    print("Inference verified successfully.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
