import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.dataset import get_dataloader
from library.model import SPMHABiGRU
from library.train import run

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("==== Initializing Demonstration ====")

    # --------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides for Speed
    # --------------------------------------------------------------------------
    # We modify the Config class attributes directly to affect the library modules.
    # This ensures we run a fast, miniature version of the pipeline.

    Config.DEBUG = True  # Use subset of data (slicing datasets)
    Config.DEBUG_SIZE = 50  # Use only 50 samples for train/val/test
    Config.EPOCHS = 2  # Minimal epochs to verify training loop
    Config.BATCH_SIZE = 8  # Small batch size

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"

    # Since path attributes in Config are strings computed at import time,
    # we must explicitly update them if we change WORKING_DIR.
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")
    Config.CACHE_TRAIN = os.path.join(Config.WORKING_DIR, "train_cache_debug.npz")
    Config.CACHE_VAL = os.path.join(Config.WORKING_DIR, "val_cache_debug.npz")
    Config.CACHE_TEST = os.path.join(Config.WORKING_DIR, "test_cache_debug.npz")

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")
    print(f"Working directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Demonstrate Data Loading
    # --------------------------------------------------------------------------
    print("\n--- Demonstrating Data Loading ---")
    # We use 'train' split. The get_dataloader function handles caching/processing.
    # It will respect Config.DEBUG and Config.DEBUG_SIZE.
    train_loader = get_dataloader("train", batch_size=Config.BATCH_SIZE, shuffle=True)

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    targets = batch["targets"].to(device)
    mask = batch["mask"].to(device)

    print(f"Batch keys: {list(batch.keys())}")

    # Verify shapes
    # Inputs: (Batch, Seq_Len=107, Input_Dim=14)
    expected_input_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"

    # Pair Indices: (Batch, Seq_Len=107)
    expected_pair_shape = (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert (
        pair_indices.shape == expected_pair_shape
    ), f"Pair indices shape mismatch. Expected {expected_pair_shape}, got {pair_indices.shape}"

    # Targets: (Batch, Seq_Len=107, Num_Targets=5)
    expected_target_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("Data Loading verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Demonstrate Model Instantiation & Forward Pass
    # --------------------------------------------------------------------------
    print("\n--- Demonstrating Model Architecture ---")
    model = SPMHABiGRU().to(device)

    # Forward pass
    # The model expects inputs and pair_indices
    outputs = model(inputs, pair_indices)

    # Verify output shape: (Batch, Seq_Len, Num_Targets)
    assert (
        outputs.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {outputs.shape}"

    print("Model forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Demonstrate Loss Calculation
    # --------------------------------------------------------------------------
    print("\n--- Demonstrating Loss Calculation ---")
    criterion = MCRMSELoss()

    # MCRMSELoss expects (Batch, Seq, Channels)
    # We simulate the masking logic used in the training loop (filtering for valid positions)
    # to ensure the loss calculation is robust.
    active_mask = mask > 0
    masked_outputs = outputs[active_mask].unsqueeze(0)
    masked_targets = targets[active_mask].unsqueeze(0)

    loss = criterion(masked_outputs, masked_targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"

    print("Loss function verified successfully.")

    # --------------------------------------------------------------------------
    # 5. Demonstrate Full Training Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Executing Full Training Pipeline (Miniature Run) ---")
    # This calls train_model() and predict_and_submit() from library.train.
    # It uses the modified Config (DEBUG=True, EPOCHS=2), so it should finish quickly.

    run()

    # --------------------------------------------------------------------------
    # 6. Verify Outputs
    # --------------------------------------------------------------------------
    print("\n--- Verifying Pipeline Outputs ---")

    # Check if model checkpoint exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    print("Model checkpoint found.")

    # Check if submission file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: Num_Test_Samples * Seq_Len
    # In DEBUG mode, the test set is sliced to Config.DEBUG_SIZE (50).
    # Expected rows = 50 * 107 = 5350
    expected_rows = Config.DEBUG_SIZE * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print("Pipeline execution verified successfully.")
    print("==== Demo Completed ====")


if __name__ == "__main__":
    main()
