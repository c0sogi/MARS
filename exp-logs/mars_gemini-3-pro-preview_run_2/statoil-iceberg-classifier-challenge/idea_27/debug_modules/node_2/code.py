import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.utils import seed_everything, load_data
from library.model import SWDINet, DualPooling, CBAMBlock
from library.data_loader import get_fold_loaders, IcebergDataset
from library.train_eval import run_training


def run_demo():
    # 1. Setup
    print("=== 1. Setup & Configuration ===")
    DEMO_DIR = "./working/demo_execution"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    SUBMISSION_DIR = DEMO_DIR

    # Clean up previous runs if they exist
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    seed_everything(42)
    print("Random seeds set.")

    # 2. Data Loading Verification
    print("\n=== 2. Verifying Data Loading Logic ===")
    # We use a custom cache dir to force processing from scratch (or verify saving)
    data = load_data(cache_dir=CACHE_DIR, load_cached_data=True)

    # assertions to verify data integrity
    X_train = data["X_train"]
    y_train = data["y_train"]
    inc_train = data["inc_angle_train"]
    X_test = data["X_test"]

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # Verify Shapes: (N, 75, 75, 3)
    if X_train.ndim != 4 or X_train.shape[1:] != (75, 75, 3):
        raise AssertionError(
            f"Expected X_train shape (N, 75, 75, 3), got {X_train.shape}"
        )

    if y_train.ndim != 1 or len(y_train) != len(X_train):
        raise AssertionError("Mismatch between X_train and y_train length")

    # Verify Channel Construction (Band 3 should be avg of Band 1 and 2)
    # Check a random sample
    idx = 0
    b1 = X_train[idx, :, :, 0]
    b2 = X_train[idx, :, :, 1]
    b3 = X_train[idx, :, :, 2]
    expected_b3 = (b1 + b2) / 2.0

    if not np.allclose(b3, expected_b3, atol=1e-5):
        raise AssertionError("3rd Channel is not the average of Band 1 and Band 2")
    print("Data structure and channel composition verified.")

    # 3. Model Component Verification
    print("\n=== 3. Verifying Model Architecture ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test DualPooling
    # Input: (B, C, H, W) -> Output: (B, 2*C, H/2, W/2)
    pool = DualPooling(kernel_size=2, stride=2).to(device)
    dummy_input = torch.randn(2, 64, 10, 10).to(device)
    pool_out = pool(dummy_input)

    expected_shape = (2, 128, 5, 5)
    if pool_out.shape != expected_shape:
        raise AssertionError(
            f"DualPooling output shape mismatch. Expected {expected_shape}, got {pool_out.shape}"
        )
    print("DualPooling verified.")

    # Test CBAMBlock
    # Input: (B, C, H, W) -> Output: (B, C, H, W)
    cbam = CBAMBlock(channels=64).to(device)
    cbam_out = cbam(dummy_input)

    if cbam_out.shape != dummy_input.shape:
        raise AssertionError(
            f"CBAMBlock output shape mismatch. Expected {dummy_input.shape}, got {cbam_out.shape}"
        )
    print("CBAMBlock verified.")

    # Test Full SWDINet
    model = SWDINet().to(device)
    # Input images: (B, 3, 75, 75), Input angles: (B, 1)
    dummy_imgs = torch.randn(4, 3, 75, 75).to(device)
    dummy_angles = torch.randn(4, 1).to(device)

    model_out = model(dummy_imgs, dummy_angles)

    # Output should be (B, 1) logits
    if model_out.shape != (4, 1):
        raise AssertionError(
            f"SWDINet output shape mismatch. Expected (4, 1), got {model_out.shape}"
        )
    print("SWDINet forward pass verified.")

    # 4. Training Pipeline Execution
    print("\n=== 4. Executing Training Pipeline (Fast Mode) ===")
    # We run with 1 epoch and debug=False to ensure submission file is generated.
    # The library's run_training function handles CV loops.

    try:
        run_training(
            epochs=1,
            batch_size=32,
            patience=1,
            seed=42,
            output_dir=SUBMISSION_DIR,
            debug=False,  # Set to False to generate submission, but keep epochs low for speed
        )
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed: {e}")

    # 5. Submission Verification
    print("\n=== 5. Verifying Submission Output ===")
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id", "is_iceberg"]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check ID count matches test set
    if len(df_sub) != len(X_test):
        raise AssertionError(
            f"Submission row count {len(df_sub)} does not match test set size {len(X_test)}"
        )

    # Check probability range
    if df_sub["is_iceberg"].min() < 0 or df_sub["is_iceberg"].max() > 1:
        raise AssertionError("Submission probabilities out of range [0, 1]")

    print("Submission file verified successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
