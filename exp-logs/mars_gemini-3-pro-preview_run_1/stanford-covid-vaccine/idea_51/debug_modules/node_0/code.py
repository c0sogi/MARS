import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import seed_everything, mcrmse, get_device
from library.data import get_dataloaders, RNADataset
from library.model import RNAModel
from library.engine import run_training


def demonstrate_config():
    print("\n=== Demonstrating Configuration ===")
    # Initialize Config in debug mode
    config = Config(debug=True)

    # Verify debug overrides
    assert config.EPOCHS == 2, f"Debug EPOCHS should be 2, got {config.EPOCHS}"
    assert (
        config.BATCH_SIZE == 16
    ), f"Debug BATCH_SIZE should be 16, got {config.BATCH_SIZE}"
    assert config.SEQ_LEN == 107, "Sequence length mismatch"
    print("Configuration loaded and validated successfully.")


def demonstrate_data_loading():
    print("\n=== Demonstrating Data Loading ===")
    # Get dataloaders in debug mode (loads subset, uses cache if avail)
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=True
    )

    # Fetch a single batch from training loader
    batch = next(iter(train_loader))

    # Verify batch keys
    expected_keys = {"sequence", "loop_type", "pair_dist", "targets", "id"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify shapes
    # Config.BATCH_SIZE is 16 in debug mode
    batch_size = batch["sequence"].shape[0]
    seq_len = batch["sequence"].shape[1]

    assert batch_size == 16, f"Expected batch size 16, got {batch_size}"
    assert seq_len == 107, f"Expected sequence length 107, got {seq_len}"

    # Check data types
    assert batch["sequence"].dtype == torch.long, "Sequence should be LongTensor"
    assert batch["pair_dist"].dtype == torch.float32, "Pair dist should be FloatTensor"
    assert batch["targets"].dtype == torch.float32, "Targets should be FloatTensor"

    print(f"Data batch validated. Shape: {batch['sequence'].shape}")
    return batch


def demonstrate_model(batch):
    print("\n=== Demonstrating Model Architecture ===")
    config = Config(debug=True)
    device = get_device()

    # Instantiate model
    model = RNAModel(config).to(device)

    # Prepare inputs
    inputs = {
        "sequence": batch["sequence"].to(device),
        "loop_type": batch["loop_type"].to(device),
        "pair_dist": batch["pair_dist"].to(device),
    }

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Verify output shape: (Batch, Seq_Len, 3)
    expected_shape = (16, 107, 3)
    assert (
        outputs.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {outputs.shape}"

    # Verify values are finite (no NaNs from positional encoding or GRU)
    assert torch.isfinite(outputs).all(), "Model output contains NaNs or Infs"

    print(f"Model forward pass successful. Output shape: {outputs.shape}")
    return outputs, batch["targets"].to(device)


def demonstrate_metric(outputs, targets):
    print("\n=== Demonstrating MCRMSE Metric ===")
    # The metric calculates RMSE over the first 68 positions (Config.PRED_LEN)
    # We will manually calculate it for a small slice to verify logic

    # Slice to scored length
    pred_scored = outputs[:, :68, :]
    true_scored = targets[:, :68, :]

    # Manual Calculation
    # 1. MSE per column (averaged over batch and sequence)
    mse = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 1))
    # 2. RMSE per column
    rmse = torch.sqrt(mse)
    # 3. Mean of RMSEs
    manual_score = torch.mean(rmse)

    # Library Calculation
    library_score = mcrmse(targets, outputs, num_scored=68)

    # Compare
    diff = torch.abs(manual_score - library_score).item()
    assert (
        diff < 1e-5
    ), f"Metric mismatch: Manual {manual_score:.6f} vs Library {library_score:.6f}"

    print(f"Metric validation passed. Score: {library_score:.6f}")


def demonstrate_full_training_loop():
    print("\n=== Demonstrating Full Training Loop (Debug Mode) ===")
    # This runs the engine.run_training function which orchestrates
    # data loading, training, validation, and submission generation.

    # Ensure clean state for submission file
    config = Config(debug=True)
    if os.path.exists(config.SUBMISSION_PATH):
        os.remove(config.SUBMISSION_PATH)

    # Run training
    try:
        run_training(debug=True)
    except Exception as e:
        raise RuntimeError(f"Training loop failed: {e}")

    # Verify artifacts
    assert os.path.exists(
        config.BEST_MODEL_PATH
    ), "Best model file not found after training."
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), "Submission file not found after training."

    # Verify submission content format
    df = pd.read_csv(config.SUBMISSION_PATH)
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert all(
        col in df.columns for col in required_cols
    ), "Submission missing required columns"

    # Check row count: In debug mode, we use a subset.
    # get_dataloaders(debug=True) slices test set to batch_size * 2 = 32 samples.
    # Each sample has 107 positions. Total rows = 32 * 107 = 3424.
    expected_rows = 32 * 107
    assert (
        len(df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df)}"

    print(
        f"Full training loop completed successfully. Submission generated at {config.SUBMISSION_PATH}"
    )


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Config Demo
    demonstrate_config()

    # 3. Data Demo
    batch = demonstrate_data_loading()

    # 4. Model Demo
    outputs, targets = demonstrate_model(batch)

    # 5. Metric Demo
    demonstrate_metric(outputs, targets)

    # 6. Full Engine Demo
    demonstrate_full_training_loop()

    print("\nAll demonstrations passed successfully!")
