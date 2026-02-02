import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
import random
import shutil

# Import from the provided library
from library.config import Config
from library.dicom_utils import read_dicom_file, normalize_min_max
from library.data_loader import (
    get_dataloaders,
    generate_roi_cache,
    MGMTDataset,
    get_sorted_files,
)
from library.model import MILEfficientNet
from library.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_subset_metadata(source_csv, dest_csv, n=5):
    """Creates a subset of the metadata file for quick demonstration."""
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source metadata not found: {source_csv}")

    df = pd.read_csv(source_csv)
    # Take top n samples
    subset = df.head(n).copy()
    subset.to_csv(dest_csv, index=False)
    print(f"Created subset metadata at {dest_csv} with {len(subset)} samples.")
    return subset


def run_demo():
    print("=== Starting Glioblastoma Classification Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Use a separate working directory for the demo
    DEMO_WORKING_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Modify Config global attributes to speed up processing
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Reduce image size and complexity
    Config.IMG_SIZE = 128  # Smaller images for faster forward pass
    Config.NUM_INSTANCES = 2  # Fewer instances per bag
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Point to subset metadata locations (we will create these shortly)
    Config.TRAIN_METADATA = os.path.join(DEMO_WORKING_DIR, "train_subset.csv")
    Config.VAL_METADATA = os.path.join(DEMO_WORKING_DIR, "val_subset.csv")
    Config.TEST_METADATA = os.path.join(DEMO_WORKING_DIR, "test_subset.csv")

    set_seed(Config.SEED)
    print("   Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Prepare Data Subsets
    # -------------------------------------------------------------------------
    print("\n2. Preparing data subsets...")

    # Create subsets from the original metadata provided in ./metadata
    orig_train = "./metadata/train.csv"
    orig_val = "./metadata/val.csv"
    orig_test = "./metadata/test.csv"

    # We use very few samples: 4 train, 2 val, 2 test
    df_train_sub = create_subset_metadata(orig_train, Config.TRAIN_METADATA, n=4)
    df_val_sub = create_subset_metadata(orig_val, Config.VAL_METADATA, n=2)
    df_test_sub = create_subset_metadata(orig_test, Config.TEST_METADATA, n=2)

    # -------------------------------------------------------------------------
    # 3. Verify DICOM Utils
    # -------------------------------------------------------------------------
    print("\n3. Verifying DICOM utilities...")

    # Pick a sample file from the training subset
    sample_row = df_train_sub.iloc[0]
    flair_path_rel = sample_row["path_FLAIR"]
    flair_full_dir = os.path.join(Config.INPUT_DIR, flair_path_rel)

    # Get files
    files = get_sorted_files(flair_full_dir)
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {flair_full_dir}")

    sample_dcm_path = os.path.join(flair_full_dir, files[len(files) // 2])

    # Test Read
    img = read_dicom_file(sample_dcm_path)
    print(f"   Read DICOM shape: {img.shape}, dtype: {img.dtype}")

    # Assertions
    assert isinstance(img, np.ndarray), "read_dicom_file should return numpy array"
    assert img.ndim == 2, "DICOM image should be 2D"

    # Test Normalize
    norm_img = normalize_min_max(img)
    print(f"   Normalized range: [{norm_img.min():.4f}, {norm_img.max():.4f}]")
    assert norm_img.max() <= 1.0 + 1e-6, "Max value should be <= 1"
    assert norm_img.min() >= 0.0 - 1e-6, "Min value should be >= 0"

    # -------------------------------------------------------------------------
    # 4. Verify Data Pipeline (Cache & Loader)
    # -------------------------------------------------------------------------
    print("\n4. Verifying Data Pipeline...")

    # Generate ROI Cache
    print("   Generating ROI cache...")
    roi_cache = generate_roi_cache(
        [df_train_sub, df_val_sub, df_test_sub],
        load_cached_data=False,  # Force generation
    )
    assert not roi_cache.empty, "ROI cache should not be empty"
    assert "roi_indices" in roi_cache.columns, "ROI cache missing 'roi_indices'"

    # Instantiate Dataset
    print("   Instantiating MGMTDataset...")
    # We skip transforms for this check to verify raw tensor output
    ds = MGMTDataset(df_train_sub, roi_cache, transform=None, mode="train")

    # Fetch one item
    inputs, target = ds[0]
    print(f"   Dataset item shape: {inputs.shape}, Target: {target}")

    # Verify Shapes
    # Shape: (Num_Instances, Channels, H, W)
    # Channels = 4 modalities * 3 slices = 12
    expected_channels = len(Config.MODALITIES) * Config.SLICES_PER_MODALITY

    assert (
        inputs.shape[0] == Config.NUM_INSTANCES
    ), f"Expected {Config.NUM_INSTANCES} instances, got {inputs.shape[0]}"
    assert (
        inputs.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == Config.IMG_SIZE
    ), f"Expected height {Config.IMG_SIZE}, got {inputs.shape[2]}"
    assert (
        inputs.shape[3] == Config.IMG_SIZE
    ), f"Expected width {Config.IMG_SIZE}, got {inputs.shape[3]}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # -------------------------------------------------------------------------
    # 5. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n5. Verifying Model Architecture...")

    model = MILEfficientNet()
    model.eval()  # Set to eval mode

    # Create dummy input batch: (Batch, Num_Instances, Channels, H, W)
    dummy_input = torch.randn(
        Config.BATCH_SIZE,
        Config.NUM_INSTANCES,
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )

    print(f"   Forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output.shape}"

    # -------------------------------------------------------------------------
    # 6. Run Trainer (Training & Validation)
    # -------------------------------------------------------------------------
    print("\n6. Running Trainer (Train/Val Loop)...")

    # Initialize Trainer
    # Note: Trainer initializes its own model and optimizer internally.
    # It calls get_dataloaders() which uses the Config paths we overrode.
    trainer = Trainer()

    # Run the training loop
    # Since we set NUM_EPOCHS=1 and small subsets, this should be fast.
    trainer.run()

    # Verify model checkpoint creation
    assert os.path.exists(trainer.model_path), "Best model checkpoint was not saved."
    print("   Training cycle completed and model saved.")

    # -------------------------------------------------------------------------
    # 7. Verify Submission Generation
    # -------------------------------------------------------------------------
    print("\n7. Verifying Submission Output...")

    # Check if submission file exists
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("   Submission head:")
    print(df_sub.head())

    # Assertions
    assert len(df_sub) == len(
        df_test_sub
    ), f"Submission rows {len(df_sub)} mismatch test subset size {len(df_test_sub)}"
    assert "BraTS21ID" in df_sub.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Submission missing MGMT_value column"
    assert df_sub["MGMT_value"].dtype == float, "MGMT_value should be float"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
