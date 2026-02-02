import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import GSRDN
from library.loss import MCRMSELoss
from library.train import run_training, generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    print("\n[1] Setting up configuration and environment...")
    set_seed(42)

    # Initialize Config in debug mode for speed
    config = Config(debug=True)

    # Verify directories
    assert os.path.exists(config.input_dir), "Input directory not found."
    assert os.path.exists(config.metadata_dir), "Metadata directory not found."
    assert os.path.exists(config.working_dir), "Working directory not created."

    print(f"    Device: {config.device}")
    print(f"    Batch Size: {config.batch_size}")
    print(f"    Input Channels: {config.input_channels}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch from train loader
    features, pidx, targets = next(iter(train_loader))

    print(f"    Features shape: {features.shape}")
    print(f"    Partner Indices shape: {pidx.shape}")
    print(f"    Targets shape: {targets.shape}")

    # Assertions for shapes
    # Features: (Batch, 107, 18)
    assert features.shape == (
        config.batch_size,
        config.seq_len,
        config.input_channels,
    ), f"Mismatch in features shape. Expected {(config.batch_size, config.seq_len, config.input_channels)}, got {features.shape}"

    # Partner Indices: (Batch, 107)
    assert pidx.shape == (
        config.batch_size,
        config.seq_len,
    ), f"Mismatch in partner indices shape. Expected {(config.batch_size, config.seq_len)}, got {pidx.shape}"

    # Targets: (Batch, 68, 5) - Note: targets are sliced to seq_scored (68) in preprocessing
    assert targets.shape == (
        config.batch_size,
        config.pred_len,
        5,
    ), f"Mismatch in targets shape. Expected {(config.batch_size, config.pred_len, 5)}, got {targets.shape}"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    model = GSRDN().to(config.device)
    features = features.to(config.device)
    pidx = pidx.to(config.device)

    # Forward pass (Pass 1 - No feedback)
    preds = model(features, pidx)

    print(f"    Predictions shape: {preds.shape}")

    # Assertions for output shape: (Batch, 107, 5)
    # Model outputs predictions for full sequence length
    assert preds.shape == (
        config.batch_size,
        config.seq_len,
        5,
    ), f"Mismatch in predictions shape. Expected {(config.batch_size, config.seq_len, 5)}, got {preds.shape}"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")
    criterion = MCRMSELoss().to(config.device)
    targets = targets.to(config.device)

    loss = criterion(preds, targets)
    print(f"    Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop (Debug Mode)...")
    # Run for 1 epoch to be fast
    best_score = run_training(debug=True, epochs=1)

    print(f"    Training finished. Best Score: {best_score}")

    # Verify model checkpoint exists
    assert os.path.exists(
        config.model_save_path
    ), f"Model checkpoint not found at {config.model_save_path}"

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(
        config.submission_path
    ), f"Submission file not found at {config.submission_path}"

    # Check submission content
    sub_df = pd.read_csv(config.submission_path)
    print(f"    Submission shape: {sub_df.shape}")
    print(f"    Submission columns: {sub_df.columns.tolist()}")

    # Test set has 240 samples, sequence length 107. Total rows = 240 * 107 = 25680
    expected_rows = 240 * config.seq_len
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    expected_cols = ["id_seqpos"] + config.target_cols
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print("\n=== Demonstration Complete: All checks passed ===")


if __name__ == "__main__":
    main()
