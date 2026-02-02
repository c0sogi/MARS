import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import RHIDFN
from library.train import run_training


def main():
    print(">>> [1/6] Setting up environment...")
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")

    print("\n>>> [2/6] Demonstrating Data Loading (Debug Mode)...")
    # Load a small subset of data (Train: 64, Val: 32, Test: 32 samples)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch a single batch to verify data structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    partner_indices = batch["partner_indices"]
    targets = batch["targets"]

    print(f"    Batch Size: {inputs.shape[0]}")
    print(f"    Input Tensor Shape: {inputs.shape}")  # Expected: (16, 107, 18)
    print(f"    Partner Indices Shape: {partner_indices.shape}")  # Expected: (16, 107)
    print(f"    Targets Tensor Shape: {targets.shape}")  # Expected: (16, 107, 5)

    # Validate Data Shapes
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        18,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, 18)}, got {inputs.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        5,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH, 5)}, got {targets.shape}"
    assert partner_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Partner indices shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LENGTH)}, got {partner_indices.shape}"

    print("\n>>> [3/6] Demonstrating Model Instantiation & Forward Pass...")
    model = RHIDFN().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # Perform Forward Pass
    # The model returns two outputs: y1 (Pass 1, no feedback) and y2 (Pass 2, with feedback)
    y1, y2 = model(inputs, partner_indices)

    print(f"    Pass 1 Output Shape: {y1.shape}")
    print(f"    Pass 2 Output Shape: {y2.shape}")

    # Validate Model Output Shapes
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LENGTH, 5)
    assert (
        y1.shape == expected_shape
    ), f"Model Pass 1 output mismatch. Expected {expected_shape}, got {y1.shape}"
    assert (
        y2.shape == expected_shape
    ), f"Model Pass 2 output mismatch. Expected {expected_shape}, got {y2.shape}"

    print("\n>>> [4/6] Demonstrating Loss Calculation...")
    criterion = MCRMSELoss()

    # Calculate loss on the refined prediction (y2)
    loss = criterion(y2, targets)

    print(f"    Calculated MCRMSE Loss: {loss.item():.6f}")

    # Validate Loss
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    print("\n>>> [5/6] Executing Full Training Pipeline (1 Epoch, Debug Mode)...")
    # run_training handles the loop, validation, saving best model, and generating submission
    run_training(debug=True, epochs=1)

    print("\n>>> [6/6] Verifying Submission File...")
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission Shape: {sub_df.shape}")
    print(f"    Submission Columns: {sub_df.columns.tolist()}")

    # Validate Submission Dimensions
    # Debug mode uses 32 test samples.
    # Each sample has 107 sequence positions.
    # Total rows should be 32 * 107 = 3424.
    num_test_samples = 32
    expected_rows = num_test_samples * Config.SEQ_LENGTH

    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Validate Submission Columns
    required_cols = ["id_seqpos"] + Config.TARGET_COLS
    for col in required_cols:
        assert col in sub_df.columns, f"Missing required column: {col}"

    print("\n>>> All demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
