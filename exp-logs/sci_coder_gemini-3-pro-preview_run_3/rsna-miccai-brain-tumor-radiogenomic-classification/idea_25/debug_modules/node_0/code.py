import os
import shutil
import pandas as pd
import torch
import numpy as np
import sys

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.model import S3HDNetwork
from library.trainer import run_training_pipeline


def main():
    print("Starting S3HD Network Demo Script...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Define a separate directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    demo_meta_dir = os.path.join(demo_dir, "metadata")

    # Clean up previous run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_meta_dir, exist_ok=True)

    print(f"Demo directory created at: {demo_dir}")

    # Override Config attributes to use the demo environment
    # We reduce epochs and batch size for speed
    Config.WORK_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2  # Small batch size for the few samples we will use

    # ==========================================
    # 2. Data Preparation (Subsetting)
    # ==========================================
    print("Creating subset metadata for rapid demonstration...")

    # Load original metadata
    # We assume these files exist as per the problem description
    try:
        orig_train = pd.read_parquet("./metadata/train.parquet")
        orig_val = pd.read_parquet("./metadata/val.parquet")
        orig_test = pd.read_parquet("./metadata/test.parquet")
    except FileNotFoundError as e:
        print(f"Critical Error: Metadata files not found. {e}")
        sys.exit(1)

    # Create tiny subsets
    # We select a few samples to ensure the code runs through the data loading logic
    # without processing the entire dataset (which would take > 1 hour)
    demo_train_df = orig_train.head(4).copy()
    demo_val_df = orig_val.head(2).copy()
    demo_test_df = orig_test.head(2).copy()

    # Save demo metadata
    demo_train_path = os.path.join(demo_meta_dir, "train.parquet")
    demo_val_path = os.path.join(demo_meta_dir, "val.parquet")
    demo_test_path = os.path.join(demo_meta_dir, "test.parquet")

    demo_train_df.to_parquet(demo_train_path)
    demo_val_df.to_parquet(demo_val_path)
    demo_test_df.to_parquet(demo_test_path)

    # Update Config to point to demo metadata
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    print("Subset metadata saved. Config updated.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\nVerifying Model Architecture...")
    device = get_device()
    model = S3HDNetwork().to(device)

    # Expected Input: (Batch, Channels, Height, Width)
    # Channels = 128 (32 slices * 4 modalities)
    # Size = 224x224
    dummy_input = torch.randn(2, 128, 224, 224).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Model forward pass successful. Output shape: {output.shape}")

        # Assertions
        assert output.shape == (
            2,
            1,
        ), f"Expected output shape (2, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Model output contains NaNs"

    except Exception as e:
        print(f"Model verification failed: {e}")
        raise e

    # ==========================================
    # 4. Pipeline Execution
    # ==========================================
    print("\nExecuting Training Pipeline...")
    print("Note: load_cached_data=False forces DICOM processing from scratch.")

    # This runs: Data Loading -> Training (1 Epoch) -> Validation -> Inference
    run_training_pipeline(load_cached_data=False)

    # ==========================================
    # 5. Output Verification
    # ==========================================
    print("\nVerifying Pipeline Artifacts...")

    # 5.1 Check Model Checkpoint
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")
    print(" - Best model checkpoint found.")

    # 5.2 Check Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(" - Submission file loaded.")
    print(submission_df)

    # 5.3 Validate Submission Content
    expected_cols = ["BraTS21ID", "MGMT_value"]
    if list(submission_df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"
        )

    if len(submission_df) != 2:
        raise ValueError(
            f"Expected 2 predictions (matching demo test set), got {len(submission_df)}"
        )

    if submission_df.isnull().values.any():
        raise ValueError("Submission contains NaN values.")

    # Check value range (probabilities should be 0-1)
    probs = submission_df["MGMT_value"].values
    if not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError("Predictions are out of probability range [0, 1].")

    print("\n" + "=" * 40)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
