import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataloaders
from library.model import RNAModel, masked_mse_loss
from library.train import run_training


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting RNA Degradation Prediction Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("[1/7] Configuring environment for rapid demonstration...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 40  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size for stability with small data
    Config.EPOCHS = 2  # Minimal epochs to prove the loop works
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect directories to ./working to ensure write permissions and isolation
    Config.CACHE_DIR = "./working/demo_cache/"
    Config.SUBMISSION_DIR = "./working/demo_submission/"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up any previous demo runs
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("      Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying Metric Calculation (MCRMSE)...")

    # Create dummy data: 2 samples, 3 targets
    # Target 1: Error 1.0 -> MSE 1.0
    # Target 2: Error 2.0 -> MSE 4.0
    # Target 3: Error 0.0 -> MSE 0.0
    y_true = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y_pred = np.array([[2.0, 4.0, 3.0], [0.0, 0.0, 3.0]])

    # Col 1 diffs: (1-2)^2=1, (1-0)^2=1 -> Mean MSE=1.0 -> RMSE=1.0
    # Col 2 diffs: (2-4)^2=4, (2-0)^2=4 -> Mean MSE=4.0 -> RMSE=2.0
    # Col 3 diffs: (3-3)^2=0, (3-3)^2=0 -> Mean MSE=0.0 -> RMSE=0.0
    # MCRMSE = (1.0 + 2.0 + 0.0) / 3 = 1.0

    metric = calculate_mcrmse(y_true, y_pred)
    print(f"      Calculated MCRMSE: {metric}")

    assert (
        abs(metric - 1.0) < 1e-6
    ), f"Metric verification failed. Expected 1.0, got {metric}"
    print("      Metric logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[3/7] Verifying Data Loading and Processing...")

    # Load dataloaders (this will trigger processing and caching)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    seq, loop, dist, targets, mask = next(iter(train_loader))

    print(f"      Batch shapes:")
    print(
        f"      - Seq: {seq.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}])"
    )
    print(
        f"      - Loop: {loop.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}])"
    )
    print(
        f"      - Dist: {dist.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.EMBED_DIM_DIST}])"
    )
    print(
        f"      - Targets: {targets.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.NUM_TARGETS}])"
    )
    print(
        f"      - Mask: {mask.shape} (Expected: [{Config.BATCH_SIZE}, {Config.SEQ_LEN}])"
    )

    # Assertions
    assert seq.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert dist.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.EMBED_DIM_DIST)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)

    # Verify Mask (First 68 should be 1, rest 0)
    assert mask[0, : Config.PRED_LEN].sum() == Config.PRED_LEN
    assert mask[0, Config.PRED_LEN :].sum() == 0
    print("      Data pipeline verified.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4/7] Verifying Model Architecture...")

    model = RNAModel().to(Config.DEVICE)

    # Move batch to device
    seq = seq.to(Config.DEVICE)
    loop = loop.to(Config.DEVICE)
    dist = dist.to(Config.DEVICE)

    # Forward pass
    outputs = model(seq, loop, dist)

    print(f"      Output shape: {outputs.shape}")
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    print("      Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[5/7] Verifying Loss Function...")

    targets = targets.to(Config.DEVICE)
    mask = mask.to(Config.DEVICE)

    loss = masked_mse_loss(outputs, targets, mask)

    print(f"      Loss value: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print("      Loss function verified.")

    # -------------------------------------------------------------------------
    # 6. Integration Run (Full Training Cycle)
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Full Training Cycle (Integration Test)...")

    # Run the training loop provided in library/train.py
    # This handles training, validation, early stopping, and submission generation
    run_training(epochs=Config.EPOCHS, debug=Config.DEBUG, patience=1)

    print("      Integration run completed.")

    # -------------------------------------------------------------------------
    # 7. Submission Validation
    # -------------------------------------------------------------------------
    print("\n[7/7] Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"      Submission shape: {df_sub.shape}")
    print(f"      Columns: {list(df_sub.columns)}")

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements"

    # Verify row count
    # In debug mode, we load Config.DEBUG_SAMPLES for train/val/test.
    # However, get_dataloaders logic for 'test' might load independent test file.
    # The provided get_data function applies DEBUG limit to whatever file it loads.
    # So test set size is min(240, DEBUG_SAMPLES).
    expected_rows = min(240, Config.DEBUG_SAMPLES) * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Verify values (unscored columns should be 0)
    assert df_sub["deg_pH10"].sum() == 0, "deg_pH10 should be 0.0"
    assert df_sub["deg_50C"].sum() == 0, "deg_50C should be 0.0"

    print("      Submission file verified successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
