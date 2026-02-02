import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import read_dicom, get_sorted_file_list
from library.data import get_dataloaders
from library.model import MGMTNet
from library.engine import run


def main():
    print("=== Starting Demonstration Script ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Patching
    # ------------------------------------------------------------------------
    # We modify the Config class directly to optimize for speed and demonstration purposes.
    print("Step 1: Patching Configuration for Speed...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Only use 10 subjects for train/val
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.IMG_SIZE = 64  # Small image size for faster processing
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in demo
    Config.PRETRAINED = (
        False  # Disable downloading weights (faster, no internet needed)
    )
    Config.VOI_RANGE = (0.45, 0.55)  # Narrow volume of interest
    Config.NUM_INFERENCE_SLICES = 3  # Fewer slices for inference ensemble
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Configuration patched successfully.\n")

    # ------------------------------------------------------------------------
    # 2. Utility Verification
    # ------------------------------------------------------------------------
    print("Step 2: Verifying Utilities...")

    # Load metadata to find a valid file path
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, sample_row["flair_path"])

    # Test file listing
    files = get_sorted_file_list(flair_dir)
    if files:
        sample_file = files[0]
        print(f"Found {len(files)} files in {flair_dir}")
        print(f"Testing read_dicom on: {os.path.basename(sample_file)}")

        # Test DICOM reading
        img = read_dicom(sample_file, size=Config.IMG_SIZE)

        # Assertions
        assert isinstance(img, np.ndarray), "read_dicom should return a numpy array"
        assert img.shape == (
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        ), f"Image shape mismatch. Expected ({Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"
        assert img.dtype == np.uint8, "Image dtype should be uint8"
        print("read_dicom verification passed.")
    else:
        print(
            "Warning: No files found for utility verification. Skipping specific file check."
        )
    print("")

    # ------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # ------------------------------------------------------------------------
    print("Step 3: Verifying DataLoaders...")

    # Initialize DataLoaders
    # Note: This uses the cached file lists logic in library.utils
    train_loader, val_loader, test_loader = get_dataloaders()

    # Verify Train Loader (2D Slices)
    try:
        train_images, train_labels, train_ids = next(iter(train_loader))
        print(f"Train Batch Shape: {train_images.shape}")

        # Assertions
        # Shape: (Batch, Channels, Height, Width)
        assert train_images.dim() == 4, "Train images should be 4D (B, C, H, W)"
        assert (
            train_images.shape[1] == Config.NUM_CHANNELS
        ), f"Expected {Config.NUM_CHANNELS} channels"
        assert train_images.shape[2] == Config.IMG_SIZE, "Height mismatch"
        assert train_images.shape[3] == Config.IMG_SIZE, "Width mismatch"
        assert (
            train_labels.shape[0] == train_images.shape[0]
        ), "Batch size mismatch between images and labels"
        print("Train DataLoader shape verification passed.")
    except StopIteration:
        print("Train DataLoader is empty (likely due to DEBUG subsetting).")

    # Verify Test Loader (2.5D Ensemble Stacks)
    try:
        test_images, _, test_ids = next(iter(test_loader))
        print(f"Test Batch Shape: {test_images.shape}")

        # Assertions
        # Shape: (Batch, Num_Slices, Channels, Height, Width)
        assert test_images.dim() == 5, "Test images should be 5D (B, N, C, H, W)"
        assert (
            test_images.shape[1] == Config.NUM_INFERENCE_SLICES
        ), f"Expected {Config.NUM_INFERENCE_SLICES} inference slices"
        assert test_images.shape[2] == Config.NUM_CHANNELS, "Channel count mismatch"
        print("Test DataLoader shape verification passed.")
    except StopIteration:
        print("Test DataLoader is empty.")
    print("")

    # ------------------------------------------------------------------------
    # 4. Model Logic Verification
    # ------------------------------------------------------------------------
    print("Step 4: Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = MGMTNet(pretrained=False)  # Force false again to be sure
    model.to(device)
    model.eval()

    # Use the fetched train batch for a forward pass check
    if "train_images" in locals():
        with torch.no_grad():
            inputs = train_images.to(device)
            outputs = model(inputs)

        print(f"Model Output Shape: {outputs.shape}")

        # Assertions
        assert outputs.shape == (
            train_images.shape[0],
            Config.NUM_CLASSES,
        ), "Output shape mismatch"
        print("Model forward pass verification passed.")
    print("")

    # ------------------------------------------------------------------------
    # 5. Full Engine Execution
    # ------------------------------------------------------------------------
    print("Step 5: Running Engine (Train/Val/Inference)...")

    # We call the run function which encapsulates the training loop and inference
    # Since we patched Config, this will run quickly.
    run(train_loader, val_loader, test_loader)
    print("Engine execution complete.\n")

    # ------------------------------------------------------------------------
    # 6. Submission Verification
    # ------------------------------------------------------------------------
    print("Step 6: Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        print(df_sub.head())

        # Assertions
        assert "BraTS21ID" in df_sub.columns, "Submission missing BraTS21ID column"
        assert "MGMT_value" in df_sub.columns, "Submission missing MGMT_value column"
        assert len(df_sub) > 0, "Submission file is empty"

        # Check probability range
        probs = df_sub["MGMT_value"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Predictions out of probability range [0, 1]"

        print("Submission format verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
