import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNADilatedNet
from library.train import (
    masked_mse_loss,
    train_one_epoch,
    validate,
    generate_submission,
)


def run_demo():
    print("Initializing Demo for RNA Degradation Prediction...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    # Override Config for a fast demonstration run
    Config.DEBUG = True  # Use a small subset of data
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    # Update paths based on new working dir
    Config.CACHE_TRAIN = os.path.join(Config.WORKING_DIR, "cache", "train_data.npz")
    Config.CACHE_VAL = os.path.join(Config.WORKING_DIR, "cache", "val_data.npz")
    Config.CACHE_TEST = os.path.join(Config.WORKING_DIR, "cache", "test_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist (Config.setup creates the base dirs, we need cache dir)
    Config.setup()
    os.makedirs(os.path.dirname(Config.CACHE_TRAIN), exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")
    print(f"Device: {device}")

    # =========================================================================
    # 2. Metric Verification
    # =========================================================================
    print("\nVerifying Metric Calculation...")
    # Create dummy ground truth and predictions
    # Shape: (Samples, SeqLen, Targets) -> (1, 2, 2) for simplicity
    # Target 1: [1.0, 3.0], Pred 1: [1.1, 3.2] -> Diffs: 0.1, -0.2 -> MSE: 0.025 -> RMSE: ~0.15811
    # Target 2: [2.0, 4.0], Pred 2: [1.9, 3.8] -> Diffs: 0.1, 0.2  -> MSE: 0.025 -> RMSE: ~0.15811
    y_true_dummy = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    y_pred_dummy = np.array([[[1.1, 1.9], [3.2, 3.8]]])

    metric_val = mcrmse_metric(y_true_dummy, y_pred_dummy)
    expected_val = np.sqrt(0.025)

    assert np.isclose(
        metric_val, expected_val, atol=1e-5
    ), f"Metric verification failed. Got {metric_val}, expected {expected_val}"
    print(f"Metric verified: {metric_val:.5f}")

    # =========================================================================
    # 3. Data Loading
    # =========================================================================
    print("\nLoading Data (Debug Mode)...")
    # Force processing by ensuring cache doesn't exist or just relying on get_dataloaders logic
    # Since we changed paths, it will process from parquet
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Data Shapes
    batch = next(iter(train_loader))
    seq, struct, loop = batch["seq"], batch["struct"], batch["loop"]
    targets, mask = batch["targets"], batch["mask"]
    ids = batch["id"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Seq shape: {seq.shape}")

    # Assertions
    assert seq.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect sequence shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Incorrect target shape"
    assert mask.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect mask shape"
    assert len(ids) == Config.BATCH_SIZE, "Incorrect ID list length"

    # Verify vocab indices are within bounds
    assert seq.max() < Config.VOCAB_SIZE_SEQ, "Sequence index out of bounds"
    assert struct.max() < Config.VOCAB_SIZE_STRUCT, "Structure index out of bounds"
    assert loop.max() < Config.VOCAB_SIZE_LOOP, "Loop index out of bounds"
    print("Data loading verified.")

    # =========================================================================
    # 4. Model Initialization & Forward Pass
    # =========================================================================
    print("\nInitializing Model...")
    model = RNADilatedNet(Config).to(device)

    # Move batch to device
    seq = seq.to(device)
    struct = struct.to(device)
    loop = loop.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    print("Running Forward Pass...")
    preds = model(seq, struct, loop)

    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {preds.shape}"
    print("Forward pass successful.")

    # =========================================================================
    # 5. Loss Calculation
    # =========================================================================
    print("\nVerifying Loss Function...")
    loss = masked_mse_loss(preds, targets, mask)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"Initial Loss: {loss.item():.6f}")

    # =========================================================================
    # 6. Training Loop Demonstration
    # =========================================================================
    print("\nStarting Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        # Simple check to ensure values are changing/valid
        assert train_loss > 0, "Train loss should be positive"
        assert val_score > 0, "Validation score should be positive"

    # Save model
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"
    print("Training loop complete and model saved.")

    # =========================================================================
    # 7. Inference & Submission
    # =========================================================================
    print("\nGenerating Submission...")

    # Load model (good practice to verify loading works)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    # Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Expected rows: Number of test samples * Seq Length
    # In Debug mode, we subset test data to 100 samples (or less if original is smaller)
    # The original test set is 240. Debug logic in data.py sets it to 100.
    # So expected rows = 100 * 107 = 10700
    expected_rows = 100 * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print("Submission verified.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
