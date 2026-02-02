import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, compute_auc
from library.data import get_dataloaders
from library.model import get_model
from library.train import run_training


# ------------------------------------------------------------------------------
# 1. Configuration Setup for Demo
# ------------------------------------------------------------------------------
class DemoConfig(Config):
    """
    Configuration overrides for a fast demonstration run.
    """

    # Use a separate directory to avoid interfering with main runs
    WORKING_DIR = "./working/demo_execution"

    # Use a unique cache file for the demo to ensure fresh processing
    CACHE_FILE = "processed_data_demo.npz"

    # Output path for the submission
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Reduce training duration
    EPOCHS = 2
    BATCH_SIZE = 128  # Smaller batch size for the small subset

    # Model parameters (kept small for speed if needed, but defaults are fine)
    # We keep defaults to test the actual architecture provided

    # Scheduler
    SCHEDULER_STEP_SIZE = 1

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print("=== Starting Demonstration Script ===")

    # Set seeds for reproducibility
    seed_everything(42)

    # Initialize Config
    config = DemoConfig()

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[Step 1] Verifying Data Pipeline...")

    # We limit the samples to 2000 for quick loading and processing
    max_samples = 2000

    train_dl, val_dl, test_dl, data_dict = get_dataloaders(
        config,
        load_cached_data=False,  # Force processing from scratch for the demo
        max_train_samples=max_samples,
        max_val_samples=max_samples,
    )

    # Fetch one batch
    batch = next(iter(train_dl))
    x_seq, x_cont, y = batch

    # Assertions
    print(
        f"  Batch shapes - Seq: {x_seq.shape}, Cont: {x_cont.shape}, Target: {y.shape}"
    )

    # Expected: (Batch, Seq_Len=10), (Batch, 30), (Batch,)
    assert x_seq.shape == (
        config.BATCH_SIZE,
        config.SEQ_LEN,
    ), "Incorrect Sequence Input Shape"
    assert x_cont.shape == (config.BATCH_SIZE, 30), "Incorrect Continuous Input Shape"
    assert y.shape == (config.BATCH_SIZE,), "Incorrect Target Shape"
    assert x_seq.dtype == torch.long, "Sequence data should be LongTensor"
    assert x_cont.dtype == torch.float32, "Continuous data should be FloatTensor"

    print("  Data Pipeline Verification Passed.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    vocab_size = data_dict["vocab_size"]
    print(f"  Vocabulary Size: {vocab_size}")

    model = get_model(config, vocab_size).to(config.DEVICE)

    # Move batch to device
    x_seq_dev = x_seq.to(config.DEVICE)
    x_cont_dev = x_cont.to(config.DEVICE)

    # Forward Pass
    with torch.no_grad():
        logits = model(x_seq_dev, x_cont_dev)

    print(f"  Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN values"

    print("  Model Architecture Verification Passed.")

    # --------------------------------------------------------------------------
    # 4. Metric Verification
    # --------------------------------------------------------------------------
    print("\n[Step 3] Verifying Metric Computation...")

    # Synthetic data
    y_true_np = np.array([0, 0, 1, 1])
    y_pred_np = np.array([0.1, 0.4, 0.35, 0.8])

    # 0,0,1,1 vs 0.1,0.4,0.35,0.8
    # Pairs:
    # (0, 0.1) vs (1, 0.35) -> Correct
    # (0, 0.1) vs (1, 0.8)  -> Correct
    # (0, 0.4) vs (1, 0.35) -> Incorrect
    # (0, 0.4) vs (1, 0.8)  -> Correct
    # AUC = 3/4 = 0.75

    auc_score = compute_auc(y_true_np, y_pred_np)
    print(f"  Computed AUC: {auc_score}")
    assert (
        auc_score == 0.75
    ), f"AUC Computation Incorrect. Expected 0.75, got {auc_score}"

    # Test with Tensors
    y_true_t = torch.tensor([0, 0, 1, 1], device=config.DEVICE)
    y_pred_t = torch.tensor([0.1, 0.4, 0.35, 0.8], device=config.DEVICE)
    auc_score_t = compute_auc(y_true_t, y_pred_t)
    assert np.isclose(
        auc_score, auc_score_t
    ), "AUC computation differs between Numpy and Tensor inputs"

    print("  Metric Verification Passed.")

    # --------------------------------------------------------------------------
    # 5. Full Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[Step 4] Executing Training Loop...")

    # We use run_training from library.train
    # We pass the config and the sample limits
    # Note: run_training handles its own data loading, but since we verified the data logic
    # and caching is enabled (or we can force reload), it will work fine.
    # To save time, we let it load the cache we just created in Step 2 if possible,
    # but `get_dataloaders` in Step 2 was called with load_cached_data=False.
    # It saved the cache to `processed_data_demo.npz`.
    # So `run_training` with `load_cached_data=True` will pick it up.

    run_training(
        config=config,
        load_cached_data=True,
        max_train_samples=max_samples,
        max_val_samples=max_samples,
    )

    print("  Training Loop Completed.")

    # --------------------------------------------------------------------------
    # 6. Output Validation
    # --------------------------------------------------------------------------
    print("\n[Step 5] Verifying Submission Output...")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    print(f"  Submission Shape: {df_sub.shape}")
    print(f"  Columns: {df_sub.columns.tolist()}")

    # Assertions
    # The submission should contain all test IDs (100,000), even if we trained on a subset.
    # The test set loading in `load_and_process_data` loads the full test set.
    expected_test_size = 100000  # From metadata description
    assert (
        len(df_sub) == expected_test_size
    ), f"Submission row count mismatch. Expected {expected_test_size}, got {len(df_sub)}"

    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"
    assert (
        df_sub["id"].dtype == "int64" or df_sub["id"].dtype == "int32"
    ), "ID column type incorrect"
    assert (
        df_sub["target"].dtype == "float64" or df_sub["target"].dtype == "float32"
    ), "Target column type incorrect"

    # Check value range
    min_val = df_sub["target"].min()
    max_val = df_sub["target"].max()
    assert 0.0 <= min_val <= 1.0, "Probabilities < 0 found"
    assert 0.0 <= max_val <= 1.0, "Probabilities > 1 found"

    print("  Submission Verification Passed.")
    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
