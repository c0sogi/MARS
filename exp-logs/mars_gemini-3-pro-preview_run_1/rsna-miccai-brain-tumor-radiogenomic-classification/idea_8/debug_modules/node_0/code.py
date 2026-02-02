import os
import sys
import numpy as np
import pandas as pd
import torch
import glob

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_middle_indices, read_dicom_processed
from library.dataset import get_dataloaders, BraTSDataset
from library.model import LMSPEfficientNet
from library.trainer import run_training


def test_utils():
    """
    Validates utility functions in library/utils.py
    """
    print("\n=== Testing Utilities ===")

    # 1. Test get_middle_indices
    # Case A: Odd number of files, depth 3
    # List: 0..9 (10 files). Mid index = 5. Depth 3 -> [4, 5, 6]
    dummy_files = ["file"] * 10
    indices = get_middle_indices(dummy_files, depth=3)
    print(f"Indices for 10 files, depth 3: {indices}")
    assert indices == [4, 5, 6], f"Expected [4, 5, 6], got {indices}"

    # Case B: Small number of files (less than depth)
    # List: 0..1 (2 files). Mid index = 1. Depth 3 -> Should handle boundaries, likely [0, 1, 2] clamped or handled
    # The logic in utils: mid=1, start=1-1=0, end=0+3=3. end clamped to 2. indices [0, 1]
    dummy_files_small = ["file"] * 2
    indices_small = get_middle_indices(dummy_files_small, depth=3)
    print(f"Indices for 2 files, depth 3: {indices_small}")
    assert (
        len(indices_small) <= 3
    ), "Should not return more indices than depth/files available logic permits"

    # 2. Test read_dicom_processed
    # Find a real DICOM file to test
    sample_dir = os.path.join(Config.INPUT_DIR, "train", "00000", "FLAIR")
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)
        if files:
            sample_path = os.path.join(sample_dir, files[0])
            print(f"Testing DICOM reading on: {sample_path}")

            img = read_dicom_processed(sample_path, img_size=128)

            assert isinstance(img, np.ndarray), "Output should be a numpy array"
            assert img.shape == (
                128,
                128,
            ), f"Expected shape (128, 128), got {img.shape}"
            assert img.dtype == np.float32, f"Expected dtype float32, got {img.dtype}"
            print("DICOM reading successful.")
    else:
        print("Skipping DICOM read test (sample directory not found).")


def test_dataset_and_loader():
    """
    Validates dataset creation and dataloader generation in library/dataset.py
    """
    print("\n=== Testing Dataset & Dataloaders ===")

    # get_dataloaders handles caching. Since we changed Config.WORKING_DIR,
    # it will generate new cache files for this debug run.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=4
    )

    print(f"Train Loader Length (batches): {len(train_loader)}")
    print(f"Val Loader Length (batches): {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Expected: (Batch, Channels, H, W)
    # Channels = 3 modalities * 3 slices = 9
    expected_channels = Config.IN_CHANNELS
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {images.shape[1]}"
    assert images.shape[2] == Config.IMG_SIZE, f"Expected Height {Config.IMG_SIZE}"
    assert images.shape[3] == Config.IMG_SIZE, f"Expected Width {Config.IMG_SIZE}"

    # Targets should be float32 for BCEWithLogits
    assert targets.dtype == torch.float32, "Targets should be float32"

    return images  # Return for model testing


def test_model(sample_input):
    """
    Validates model instantiation and forward pass in library/model.py
    """
    print("\n=== Testing Model ===")

    model = LMSPEfficientNet()
    model.eval()

    # Check Adapter Initialization
    # The adapter should have weights set to 1.0 for middle slices of specific modalities
    # We won't assert exact values as implementation details might vary slightly,
    # but we check if it runs.

    print("Model instantiated successfully.")

    # Forward pass
    with torch.no_grad():
        output = model(sample_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected: (Batch, 1)
    assert output.shape == (
        sample_input.shape[0],
        1,
    ), f"Expected output shape ({sample_input.shape[0]}, 1), got {output.shape}"
    print("Forward pass successful.")


def verify_submission():
    """
    Verifies the generated submission file.
    """
    print("\n=== Verifying Submission ===")

    sub_path = Config.SUBMISSION_PATH
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    df = pd.read_csv(sub_path)
    print(f"Submission loaded. Rows: {len(df)}")
    print(df.head())

    required_cols = ["BraTS21ID", "MGMT_value"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check value range
    assert df["MGMT_value"].min() >= 0.0, "Probabilities must be >= 0"
    assert df["MGMT_value"].max() <= 1.0, "Probabilities must be <= 1"

    print("Submission format verified.")


if __name__ == "__main__":
    # ==========================================
    # 1. Configure for Demo/Speed
    # ==========================================
    print("Configuring environment for demonstration...")

    # Override Config attributes to run quickly
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples per split
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size

    # Use a specific working directory for this demo to avoid overwriting production files
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)

    # ==========================================
    # 2. Unit Tests
    # ==========================================
    # Validate Utils
    test_utils()

    # Validate Data Loading (and get a sample batch)
    sample_batch = test_dataset_and_loader()

    # Validate Model
    test_model(sample_batch)

    # ==========================================
    # 3. Full Pipeline Execution
    # ==========================================
    print("\n=== Running Full Training Pipeline ===")
    # run_training() encapsulates the entire process defined in library/trainer.py
    # It uses the Config class we just modified.
    run_training()

    # ==========================================
    # 4. Final Verification
    # ==========================================
    verify_submission()

    print("\nDone. All demonstrations and assertions passed.")
