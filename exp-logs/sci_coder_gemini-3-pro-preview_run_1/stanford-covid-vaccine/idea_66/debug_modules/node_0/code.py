import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, mcrmse_loss, mcrmse_metric
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import generate_submission


def run_demonstration():
    print("=== RNA Degradation Prediction Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # Modify Config for a fast demonstration run
    print("\n[1] Configuring environment for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 40  # Use a small subset of data
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 1

    # Use temporary directories in ./working to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache_exec"
    Config.SUBMISSION_DIR = "./working/demo_submission_exec"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any existing demo artifacts
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    # Initialize directories and seed
    Config.initialize()
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SAMPLES}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Pipeline...")
    # load_cached_data=False forces the processing logic to run from metadata
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    seq = batch["seq"]
    loop = batch["loop"]
    struct = batch["struct"]
    target = batch["target"]

    print(f"    Batch Keys: {list(batch.keys())}")
    print(
        f"    Seq Shape: {seq.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}])"
    )
    print(
        f"    Target Shape: {target.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, 3])"
    )

    # Assertions to ensure data integrity
    assert seq.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect sequence shape"
    assert target.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), "Incorrect target shape"
    assert (
        struct.dtype == torch.float32
    ), "Structure should be float32 (signed distance)"
    print("    Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")
    model = RNAModel().to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = loop.to(device)
    struct = struct.to(device)
    target = target.to(device)

    # Forward Pass
    preds = model(seq, loop, struct)
    print(f"    Predictions Shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), "Output shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaN predictions"
    print("    Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss and Metric Calculation
    # -------------------------------------------------------------------------
    print("\n[4] Testing Loss and Metric Functions...")

    # Calculate Loss (Differentiable)
    loss = mcrmse_loss(preds, target, num_scored=Config.PRED_LEN)
    print(f"    MCRMSE Loss: {loss.item():.6f}")

    # Calculate Metric (Evaluation)
    metric = mcrmse_metric(preds, target, num_scored=Config.PRED_LEN)
    print(f"    MCRMSE Metric: {metric:.6f}")

    # Assertions
    assert loss.item() >= 0, "Loss cannot be negative"
    assert metric >= 0, "Metric cannot be negative"
    print("    Loss/Metric calculation successful.")

    # -------------------------------------------------------------------------
    # 5. Optimization Step (Training Simulation)
    # -------------------------------------------------------------------------
    print("\n[5] Testing Optimization Step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist
    grads_found = any(p.grad is not None for p in model.parameters())
    assert grads_found, "No gradients computed after backward pass"

    optimizer.step()
    print("    Backward pass and optimizer step successful.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation (Inference)
    # -------------------------------------------------------------------------
    print("\n[6] Testing Submission Generation...")

    # Use the test loader (which has no targets)
    # In debug mode, this will process Config.DEBUG_SAMPLES items
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify the output file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission File Loaded. Shape: {df_sub.shape}")

    # Expected rows: Number of samples * Seq Length
    expected_rows = Config.DEBUG_SAMPLES * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Verify content validity (no NaNs)
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("    Submission generation verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
