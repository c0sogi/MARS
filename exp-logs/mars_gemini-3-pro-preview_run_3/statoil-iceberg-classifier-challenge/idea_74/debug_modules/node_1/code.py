import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import library modules
from library.config import Config
from library.utils import set_seed, AverageMeter
from library.data_loader import get_fold_loaders, get_test_loader
from library.model import HCICNN
from library.train_eval import train_fold, generate_submission


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config parameters to run a minimal version of the task
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.NUM_EPOCHS = 1  # Run only 1 epoch per fold
    Config.NUM_FOLDS = 2  # Run only 2 folds (indices 0 and 1)
    Config.BATCH_SIZE = 16  # Small batch size
    Config.PATIENCE = 1  # Minimal patience

    # Re-setup directories based on new paths
    Config.setup_directories()

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Folds: {Config.NUM_FOLDS}")

    # -------------------------------------------------------------------------
    # 2. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test set_seed
    set_seed(42)
    rand_val_1 = np.random.rand()
    set_seed(42)
    rand_val_2 = np.random.rand()
    assert (
        rand_val_1 == rand_val_2
    ), "set_seed failed to produce reproducible numpy results"
    print(" - Random seed verification passed.")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=1)
    meter.update(20, n=1)
    assert meter.avg == 15.0, f"AverageMeter computed {meter.avg}, expected 15.0"
    assert meter.count == 2, f"AverageMeter count {meter.count}, expected 2"
    print(" - AverageMeter logic verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loader Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loaders...")

    # Get Fold 0 loaders
    # Note: This will process JSONs into NPY files in the new cache dir
    train_loader, val_loader = get_fold_loaders(fold_idx=0, load_cached_data=True)

    # Verify Train Batch
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    print(f" - Train Batch Image Shape: {images.shape}")
    print(f" - Train Batch Angle Shape: {angles.shape}")
    print(f" - Train Batch Label Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"

    # Verify Test Loader
    test_loader = get_test_loader(load_cached_data=True)
    test_batch = next(iter(test_loader))
    assert "label" not in test_batch, "Test batch should not contain labels"
    assert "id" in test_batch, "Test batch must contain IDs"
    print(" - Data Loader shapes and keys verified.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HCICNN().to(device)

    # Create dummy input
    dummy_img = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_ang = torch.randn(Config.BATCH_SIZE).to(
        device
    )  # Loader gives (B,), model handles view

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    print(f" - Model Output Shape: {output.shape}")

    # Assertions
    # Model output is (B, 1) logits
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output (B, 1), got {output.shape}"
    print(" - Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Fast Mode)...")

    # Train Fold 0 and Fold 1 to satisfy the NUM_FOLDS=2 requirement for submission
    fold_scores = []
    for fold_idx in range(Config.NUM_FOLDS):
        print(f"\n--- Training Fold {fold_idx} ---")
        best_loss = train_fold(fold_idx, load_cached_data=True)
        fold_scores.append(best_loss)

        # Verify checkpoint exists
        ckpt_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {fold_idx} was not saved."
        print(f" - Checkpoint saved: {ckpt_path}")
        print(f" - Best Val Loss: {best_loss:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    generate_submission(load_cached_data=True)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_FILE):
        df_sub = pd.read_csv(Config.SUBMISSION_FILE)
        print(f" - Submission file created at: {Config.SUBMISSION_FILE}")
        print(f" - Rows: {len(df_sub)}")
        print(f" - Columns: {list(df_sub.columns)}")

        # Check format
        assert list(df_sub.columns) == [
            "id",
            "is_iceberg",
        ], "Incorrect submission columns"
        assert len(df_sub) > 0, "Submission file is empty"

        # Check probability range
        probs = df_sub["is_iceberg"].values
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"
        print(" - Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
