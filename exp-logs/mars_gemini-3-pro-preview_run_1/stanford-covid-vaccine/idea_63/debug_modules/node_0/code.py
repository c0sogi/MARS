import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse_loss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import run_training


def main():
    print("=== Starting Demonstration of RNA Degradation Prediction Pipeline ===\n")

    # 1. Setup and Configuration Overrides
    # We use a specific directory for this demo to avoid overwriting existing work
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config settings for speed and isolation
    Config.CACHE_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"Configuration set. Working directory: {DEMO_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Data Loading
    print("\n--- Verifying Data Loading ---")
    # Load a tiny subset of data
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        max_samples=20,
        load_cached_data=False,  # Force processing from parquet
    )

    # Fetch one batch
    seq, loop, dist, tgt = next(iter(train_loader))

    # Assertions
    print(
        f"Batch shapes -> Seq: {seq.shape}, Loop: {loop.shape}, Dist: {dist.shape}, Tgt: {tgt.shape}"
    )

    # Sequence length is 107
    assert seq.shape == (Config.BATCH_SIZE, 107), "Incorrect sequence shape"
    assert loop.shape == (Config.BATCH_SIZE, 107), "Incorrect loop shape"
    assert dist.shape == (Config.BATCH_SIZE, 107), "Incorrect distance shape"
    # Targets should be (Batch, 107, 3) - padded from 68
    assert tgt.shape == (Config.BATCH_SIZE, 107, 3), "Incorrect target shape"

    print("Data loading verification passed.")

    # 3. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = RNAModel().to(Config.DEVICE)

    # Move batch to device
    seq = seq.to(Config.DEVICE)
    loop = loop.to(Config.DEVICE)
    dist = dist.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(seq, loop, dist)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 107, 3), "Model output shape mismatch"
    print("Model architecture verification passed.")

    # 4. Verify Metric Logic (MCRMSE)
    print("\n--- Verifying Metric Logic ---")
    # Create dummy data:
    # Target is all 1.0 for the first 68 positions
    # Prediction is all 0.0
    # RMSE for one column = sqrt(mean((1-0)^2)) = 1.0
    # MCRMSE = mean([1.0, 1.0, 1.0]) = 1.0

    dummy_pred = torch.zeros(2, 107, 3)
    dummy_tgt = torch.zeros(2, 107, 3)
    dummy_tgt[:, :68, :] = 1.0  # Only first 68 are scored

    loss = mcrmse_loss(dummy_pred, dummy_tgt)
    print(f"Calculated MCRMSE (expected ~1.0): {loss:.6f}")

    assert (
        abs(loss - 1.0) < 1e-5
    ), f"Metric calculation failed. Expected 1.0, got {loss}"
    print("Metric logic verification passed.")

    # 5. Run Training Pipeline
    print("\n--- Running Training Pipeline (Subsample) ---")
    # We use a small subset (50 samples) and 2 epochs to ensure it finishes quickly
    # This function handles training, validation, checkpointing, and submission generation
    best_model_path = run_training(epochs=2, max_samples=50)

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training complete. Model saved to {best_model_path}")

    # 6. Verify Submission
    print("\n--- Verifying Submission File ---")
    submission_path = os.path.join(Config.SUBMISSION_DIR, Config.SUBMISSION_FILE)
    assert os.path.exists(submission_path), "Submission file not found."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print(f"Submission columns: {list(sub_df.columns)}")

    # Expected rows: 50 samples * 107 positions = 5350
    expected_rows = 50 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Check required columns
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column: {col}"

    # Check if unscored columns are 0.0 (deg_pH10, deg_50C) as per format_submission logic
    assert (sub_df["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (sub_df["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
