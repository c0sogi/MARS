import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_loader
from library.model import InteractionEnrichedDenseNet
from library.loss import MaskedMCRMSELoss
from library.train import train_model


def run_demo():
    print("=== Starting Demonstration of RNA Degradation Prediction Library ===")

    # 1. Setup and Configuration Overrides for Speed
    # We modify the Config class attributes directly to create a "Demo Mode"
    print("\n[1] Configuring environment for rapid demonstration...")

    # Use a separate working directory for the demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce training parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATIENCE = 2

    # Use a unique cache version to avoid conflicts with full training runs
    Config.CACHE_VERSION = "demo_v1"

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Device: {device}")
    print(f"    Epochs: {Config.EPOCHS}")

    # 2. Verify Data Pipeline
    print("\n[2] Verifying Data Pipeline...")

    # Load a tiny subset of training data
    limit_size = 10
    train_loader = get_loader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging/demo
        load_cached_data=False,  # Force regeneration for demo
        limit_size=limit_size,
    )

    # Fetch one batch
    inputs, partner_indices, targets = next(iter(train_loader))

    # Check shapes
    # Inputs: (Batch, SeqLen, InputDim=18)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.INPUT_DIM)}, got {inputs.shape}"

    # Partner Indices: (Batch, SeqLen)
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Partner indices shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH)}, got {partner_indices.shape}"

    # Targets: (Batch, PredLen=68, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        5,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.PRED_LEN, 5)}, got {targets.shape}"

    print("    Data Loader shapes verified successfully.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    model = InteractionEnrichedDenseNet().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)

    # Forward pass
    outputs = model(inputs, partner_indices)

    # Check output shape: Model outputs predictions for the full sequence length (107)
    expected_out_shape = (Config.BATCH_SIZE, Config.SEQ_LENGTH, 5)
    assert (
        outputs.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {outputs.shape}"

    print("    Model forward pass successful. Output shape verified.")

    # 4. Verify Loss Function Logic
    print("\n[4] Verifying Masked MCRMSE Loss Logic...")
    criterion = MaskedMCRMSELoss()

    # Create dummy predictions and targets to manually verify calculation
    # Scored indices are [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)

    # Case: Perfect prediction
    # Preds: (1, 107, 5), Targets: (1, 68, 5)
    # Note: Loss function handles slicing of preds to match target length (68)
    dummy_preds = torch.zeros((1, 107, 5), dtype=torch.float32)
    dummy_targets = torch.zeros((1, 68, 5), dtype=torch.float32)
    loss_zero = criterion(dummy_preds, dummy_targets)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0)
    ), "Loss should be 0 for perfect predictions."

    # Case: Known error
    # Set targets to 1.0 for scored columns, preds remain 0.0
    # Squared Error = (0-1)^2 = 1
    # MSE = 1, RMSE = 1
    # MCRMSE = Mean(RMSE_col1, RMSE_col2, RMSE_col3) = 1
    dummy_targets[:, :, [0, 1, 3]] = 1.0

    # Set non-scored columns to random values to ensure they are ignored
    dummy_targets[:, :, [2, 4]] = 99.0

    loss_one = criterion(dummy_preds, dummy_targets)
    assert torch.isclose(
        loss_one, torch.tensor(1.0)
    ), f"Loss should be 1.0, got {loss_one.item()}"

    print("    Loss function logic (slicing and masking) verified.")

    # 5. Verify Full Training Pipeline
    print("\n[5] Executing Full Training Pipeline (Mini-Run)...")

    # We run the actual training function with a small debug limit
    # This tests train loop, validation loop, checkpointing, and submission generation
    debug_dataset_size = 20
    train_model(debug_limit=debug_dataset_size)

    # 6. Verify Outputs
    print("\n[6] Verifying Output Files...")

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")

    assert os.path.exists(best_model_path), "best_model.pth was not created."
    assert os.path.exists(submission_path), "submission.csv was not created."

    # Check submission content
    sub_df = pd.read_csv(submission_path)
    print(f"    Submission file loaded. Shape: {sub_df.shape}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # However, we used debug_limit in train_model, but generate_submission loads the FULL test set
    # unless we also patch get_loader inside train.py or Config.TEST_CSV.
    # The provided train_model function calls get_loader(mode='test') without limit.
    # Since the test set is small (240 samples), this is acceptable for the demo.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols, "Submission columns mismatch."

    print("    Output files verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
