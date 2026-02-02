import os
import sys
import warnings
import pandas as pd
import torch
import numpy as np

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library import utils, dataset, model, engine


def run_demo():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Setup and Configuration Patching
    # We modify the Config class directly to optimize for a fast demonstration run.
    print("\n[1] Configuring environment and patching parameters...")

    utils.seed_everything(42)

    # Patch Config for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.HIDDEN_DIM = 64  # Reduced from 768 for demo speed
    Config.CNN_FILTERS = 32  # Reduced from 256 for demo speed
    Config.NUM_WORKERS = 2
    Config.IDEA_NAME = "demo_execution"  # Isolate this run

    # Update paths based on new IDEA_NAME
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_NAME)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update dependent paths
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    print(f"    Device: {Config.DEVICE}")
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}")

    # 2. Dataset Loading and Verification
    print("\n[2] Verifying Dataset...")

    # Load training dataset (this will trigger processing from metadata if not cached)
    # We force load_cached_data=False to demonstrate processing logic,
    # though in practice the hash check handles it.
    ds_train = dataset.get_dataset("train", load_cached_data=False)

    print(f"    Train Dataset Size: {len(ds_train)}")

    # Get a single sample
    sample = ds_train[0]
    x, adj, y, sample_id = sample["x"], sample["adj"], sample["y"], sample["id"]

    print(f"    Sample ID: {sample_id}")
    print(f"    Feature Shape (x): {x.shape} (Expected: 107, 14)")
    print(f"    Adjacency Shape (adj): {adj.shape} (Expected: 107)")
    print(f"    Target Shape (y): {y.shape} (Expected: 68, 5)")

    # Assertions to validate logic
    assert x.shape == (107, 14), f"Incorrect feature shape: {x.shape}"
    assert adj.shape == (107,), f"Incorrect adjacency shape: {adj.shape}"
    assert y.shape == (68, 5), f"Incorrect target shape: {y.shape}"
    assert isinstance(x, torch.Tensor), "Features must be a torch.Tensor"
    assert isinstance(y, torch.Tensor), "Targets must be a torch.Tensor"

    # 3. Model Initialization and Forward Pass
    print("\n[3] Verifying Model Architecture...")

    net = model.RNAModel().to(Config.DEVICE)

    # Create a dummy batch to verify forward pass
    batch_size = 4
    dummy_x = torch.randn(batch_size, 107, 14).to(
        Config.DEVICE
    )  # Float input simulates one-hot
    dummy_adj = torch.randint(0, 107, (batch_size, 107)).to(Config.DEVICE)

    print("    Performing dummy forward pass...")
    with torch.no_grad():
        output = net(dummy_x, dummy_adj)

    print(f"    Output Shape: {output.shape} (Expected: {batch_size}, 107, 5)")

    # Assertions
    assert output.shape == (batch_size, 107, 5), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function...")

    # Create dummy targets for the scored sequence length (68)
    dummy_targets = torch.randn(batch_size, 68, 5).to(Config.DEVICE)

    # Slice model output to match scored length
    output_scored = output[:, :68, :]

    loss = engine.mcrmse_loss(output_scored, dummy_targets)
    print(f"    Calculated MCRMSE Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss cannot be negative"

    # 5. Full Training Execution
    print("\n[5] Executing Training Loop...")
    # This calls the engine's train_fn which handles data loading, loops, validation, and saving
    engine.train_fn()

    # Verify model artifact creation
    if os.path.exists(Config.MODEL_PATH):
        print(f"    Success: Model saved to {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    # 6. Inference and Submission
    print("\n[6] Generating Submission...")
    engine.generate_submission()

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Success: Submission saved to {Config.SUBMISSION_PATH}")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # 7. Validate Submission Format
    print("\n[7] Validating Submission Format...")
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    num_test_samples = 240
    seq_len = 107
    expected_rows = num_test_samples * seq_len

    print(f"    Submission Rows: {len(sub_df)} (Expected: {expected_rows})")
    print(f"    Columns: {list(sub_df.columns)}")

    assert (
        len(sub_df) == expected_rows
    ), f"Row count mismatch. Got {len(sub_df)}, expected {expected_rows}"
    assert "id_seqpos" in sub_df.columns, "Missing 'id_seqpos' column"
    assert "reactivity" in sub_df.columns, "Missing target columns"

    # Check for NaN in predictions
    if sub_df.isnull().values.any():
        raise AssertionError("Submission contains NaN values.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
