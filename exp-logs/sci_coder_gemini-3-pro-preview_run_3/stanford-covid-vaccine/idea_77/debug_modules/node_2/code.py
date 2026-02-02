import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders, get_test_dataloader
from library.model import RNAModel
from library.train import run_training, generate_submission


def main():
    # =========================================================================
    # 1. Setup and Configuration Override
    # =========================================================================
    print(">>> Setting up configuration for demonstration...")

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Define a specific working directory for this demo to avoid cache conflicts
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config defaults for speed and isolation
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.NUM_LAYERS = 2  # Fewer layers

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # =========================================================================
    # 2. Data Loading and Verification
    # =========================================================================
    print(">>> Verifying Data Loading pipeline...")

    # Load a tiny subset of data (16 samples) to verify shapes
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing to test raw data loading
        max_train_samples=16,
        max_val_samples=8,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    pair_mask = batch["pair_mask"]
    targets = batch["targets"]

    # Assertions for Data Shapes
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Expected inputs shape {(Config.BATCH_SIZE, 107, 14)}, got {inputs.shape}"

    # Pair Indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Expected pair_indices shape {(Config.BATCH_SIZE, 107)}, got {pair_indices.shape}"

    # Targets: (Batch, Pred_Len=68, Targets=5)
    # Note: Targets are sliced to 68 in process_data if using the provided logic,
    # or they might be full length depending on implementation.
    # Checking library/config.py: targets are initialized with shape (N, 68, 5).
    assert targets.shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"Expected targets shape {(Config.BATCH_SIZE, 68, 5)}, got {targets.shape}"

    print("    Data shapes verified successfully.")

    # =========================================================================
    # 3. Model Initialization and Forward Pass
    # =========================================================================
    print(">>> Verifying Model Forward Pass...")

    device = Config.DEVICE
    model = RNAModel(Config).to(device)

    # Move batch to device
    inputs = inputs.to(device)
    pair_indices = pair_indices.to(device)
    pair_mask = pair_mask.to(device)

    # Forward pass
    outputs = model(inputs, pair_indices, pair_mask)

    # Assert Output Shape: (Batch, Seq_Len=107, Targets=5)
    # The model outputs predictions for the full sequence length (107)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Expected output shape {(Config.BATCH_SIZE, 107, 5)}, got {outputs.shape}"

    print("    Model forward pass successful.")

    # =========================================================================
    # 4. Metric Verification (MCRMSE)
    # =========================================================================
    print(">>> Verifying MCRMSE Metric Logic...")

    # Create dummy predictions and targets
    # Preds: All ones
    # Targets: All zeros
    # RMSE should be 1.0

    # Shape: (Batch=2, Seq=107, Cols=5)
    dummy_preds = torch.ones(2, 107, 5)
    # Shape: (Batch=2, Seq=68, Cols=5) - Targets are usually shorter
    dummy_targets = torch.zeros(2, 68, 5)

    # Initialize metric slicing to 68
    metric_fn = MCRMSE(pred_len=68, scored_indices=None)

    loss = metric_fn(dummy_preds, dummy_targets)

    # Expected:
    # 1. Slice preds to 68 -> ones(2, 68, 5)
    # 2. Diff is 1.0, Squared is 1.0
    # 3. Mean over batch/seq is 1.0
    # 4. Sqrt is 1.0
    # 5. Mean over columns is 1.0
    assert (
        abs(loss.item() - 1.0) < 1e-6
    ), f"MCRMSE calculation failed. Expected 1.0, got {loss.item()}"

    # Test with scored_indices (e.g., first column only)
    metric_fn_subset = MCRMSE(pred_len=68, scored_indices=[0])
    loss_subset = metric_fn_subset(dummy_preds, dummy_targets)
    assert abs(loss_subset.item() - 1.0) < 1e-6, "MCRMSE subset calculation failed."

    print("    Metric logic verified.")

    # =========================================================================
    # 5. Training Loop Execution
    # =========================================================================
    print(">>> Executing Training Loop (Demo)...")

    # Run training for 1 epoch with limited samples
    best_model_path = run_training(
        max_epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        max_train_samples=32,
        max_val_samples=16,
    )

    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"    Training complete. Model saved to {best_model_path}")

    # =========================================================================
    # 6. Inference and Submission
    # =========================================================================
    print(">>> Generating Submission...")

    generate_submission(best_model_path, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check row count
    # We have 240 test sequences * 107 length = 25680 rows
    # The provided test.json has 240 lines.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    print("    Submission generated and verified successfully.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
