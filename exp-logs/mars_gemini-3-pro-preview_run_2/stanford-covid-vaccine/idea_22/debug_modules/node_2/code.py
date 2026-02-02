import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.data import get_loader
from library.model import DecoupledDenseNet
from library.loss import MaskedMCRMSELoss
from library.train import run_training


def main():
    print("=== Starting Demo Script ===")

    # 1. Setup Configuration for Demo
    # We override the Config attributes to use a demo directory and fast settings.
    # This affects the global state of Config, which is used by all library modules.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Configuring demo environment in {DEMO_DIR}...")

    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE = os.path.join(DEMO_DIR, "train_decoupled_dense_v1.npz")
    Config.VAL_CACHE = os.path.join(DEMO_DIR, "val_decoupled_dense_v1.npz")
    Config.TEST_CACHE = os.path.join(DEMO_DIR, "test_decoupled_dense_v1.npz")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure the directory exists via the class method
    Config.setup()

    # 2. Data Loading Demonstration
    print("\n--- Demonstrating Data Loading ---")
    # We load the train loader. This will trigger processing and caching of the data.
    train_loader = get_loader(split="train", shuffle=True, load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    partner_indices = batch["partner_indices"]
    targets = batch["targets"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Partner indices shape: {partner_indices.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions to verify data integrity
    # Expected: (Batch, Seq_Len=107, Input_Dim=18)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Expected inputs shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)}, got {inputs.shape}"

    # Expected: (Batch, Seq_Len=107)
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Expected partner_indices shape {(Config.BATCH_SIZE, Config.SEQ_LEN)}, got {partner_indices.shape}"

    # Expected: (Batch, Seq_Len=107, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Expected targets shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {targets.shape}"

    print("Data loading verification passed.")

    # 3. Model Instantiation and Forward Pass
    print("\n--- Demonstrating Model Forward Pass ---")
    device = torch.device("cpu")  # Use CPU for simple verification
    model = DecoupledDenseNet().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)

    # Forward pass
    outputs = model(inputs, partner_indices)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    # Expected output: (Batch, Seq_Len=107, Num_Targets=5)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {outputs.shape}"

    print("Model forward pass verification passed.")

    # 4. Loss Calculation
    print("\n--- Demonstrating Loss Calculation ---")
    criterion = MaskedMCRMSELoss()

    # Move targets to device
    targets = targets.to(device)

    # Calculate loss
    loss = criterion(outputs, targets)

    print(f"Loss value: {loss.item()}")

    # Assertions
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss should not be NaN"

    print("Loss calculation verification passed.")

    # 5. Full Training Pipeline Execution
    print("\n--- Executing Full Training Pipeline (Reduced Epochs) ---")
    # This runs the training loop, validation, and generates submission.csv
    # We rely on the Config overrides set at the beginning.
    run_training()

    # 6. Output Verification
    print("\n--- Verifying Submission Output ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {df_sub.columns.tolist()}")

    # Verify row count
    # Test set has 240 samples. Each has 107 positions. Total rows = 240 * 107 = 25680.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Verify columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
