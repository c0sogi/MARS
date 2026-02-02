import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config, seed_everything
from library.data import get_datasets, BraTSDataset
from library.model import VAMSHDNet
from library.train import run_training
from library.utils import get_device


def main():
    # ==========================================
    # 1. Configuration for Fast Demonstration
    # ==========================================
    print("Configuring parameters for fast demonstration...")

    # Override Config to run in Debug mode with minimal resources
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small sample size for speed
    Config.EPOCHS = 2  # Minimal epochs to verify loop
    Config.BATCH_SIZE = 4  # Small batch size

    # Use a separate cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_execution"
    if not os.path.exists(Config.CACHE_DIR):
        os.makedirs(Config.CACHE_DIR)

    # Redirect submission output to working directory
    Config.SUBMISSION_DIR = "./working"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Set seed for reproducibility
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Demonstrate Data Loading & Processing
    # ==========================================
    print("\n=== Demonstrating Data Loading ===")

    # We call get_datasets with load_cached_data=False to force the processing logic
    # to run on the small debug subset. This also populates the cache for the training step.
    print("Processing datasets (Debug Mode)...")
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False)

    print(f"Train Dataset Size: {len(train_ds)}")
    print(f"Val Dataset Size:   {len(val_ds)}")
    print(f"Test Dataset Size:  {len(test_ds)}")

    # Assertions to verify data loading logic
    assert (
        len(train_ds) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"
    assert (
        len(val_ds) <= Config.DEBUG_SAMPLE_SIZE
    ), "Val dataset size exceeds debug limit"

    # Inspect a single sample
    sample_img, sample_label = train_ds[0]
    print(f"Sample Input Shape: {sample_img.shape}")
    print(f"Sample Label:       {sample_label}")

    # Verify Input Dimensions: (32 slices * 4 modalities, 224, 224)
    expected_channels = Config.NUM_SLICES_PER_MODALITY * Config.NUM_MODALITIES
    expected_shape = (expected_channels, Config.IMG_SIZE, Config.IMG_SIZE)

    assert (
        sample_img.shape == expected_shape
    ), f"Shape mismatch! Expected {expected_shape}, got {sample_img.shape}"
    assert isinstance(sample_label, torch.Tensor), "Label should be a torch.Tensor"

    # ==========================================
    # 3. Demonstrate Model Architecture
    # ==========================================
    print("\n=== Demonstrating Model Architecture ===")

    model = VAMSHDNet().to(device)
    model.eval()

    # Create a dummy batch to verify forward pass
    # Batch size = 2
    dummy_input = torch.randn(2, *expected_shape).to(device)

    print("Executing forward pass on dummy input...")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions for model output
    # Output should be (Batch_Size, 1) logits
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs!"

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Demonstrate Full Training Pipeline
    # ==========================================
    print("\n=== Demonstrating Full Training Pipeline ===")

    # run_training() encapsulates the entire loop:
    # 1. Loads data (will use the cache we just generated)
    # 2. Initializes model/optimizer
    # 3. Trains for Config.EPOCHS
    # 4. Generates predictions on Test set
    # 5. Saves submission file
    run_training()

    # ==========================================
    # 5. Verify Submission Output
    # ==========================================
    print("\n=== Verifying Submission ===")

    if os.path.exists(Config.SUBMISSION_PATH):
        submission_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        print(submission_df.head())

        # Verify submission content
        assert len(submission_df) == len(
            test_ds
        ), f"Submission rows ({len(submission_df)}) do not match test set size ({len(test_ds)})"

        required_cols = ["BraTS21ID", "MGMT_value"]
        for col in required_cols:
            assert col in submission_df.columns, f"Missing column: {col}"

        # Verify values are probabilities (or logits if raw, but usually probs)
        # The library code applies sigmoid before saving, so they should be [0, 1]
        preds = submission_df["MGMT_value"]
        assert (
            preds.min() >= 0.0 and preds.max() <= 1.0
        ), "Predictions are not valid probabilities [0, 1]"

        print("Submission verification passed successfully.")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
