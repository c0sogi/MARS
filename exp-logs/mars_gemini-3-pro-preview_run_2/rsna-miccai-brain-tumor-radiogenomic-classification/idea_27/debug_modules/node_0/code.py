import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import read_dicom_robust, resize_image, normalize_min_max
from library.data import get_dataloaders, MRIDataset, get_roi_anchors
from library.model import AsymmetricEfficientNet
from library.train import run_training


def create_subset_metadata(n_samples=8):
    """
    Creates subset CSVs for train, val, and test to speed up the demo.
    """
    os.makedirs(Config.METADATA_DIR, exist_ok=True)

    # Define paths for subset metadata
    subset_dir = os.path.join(Config.WORKING_DIR, "subset_metadata")
    os.makedirs(subset_dir, exist_ok=True)

    subset_train_path = os.path.join(subset_dir, "train.csv")
    subset_val_path = os.path.join(subset_dir, "val.csv")
    subset_test_path = os.path.join(subset_dir, "test.csv")

    # Load original metadata
    # Note: We assume the original metadata files exist as per the problem description
    orig_train = pd.read_csv(os.path.join(Config.METADATA_DIR, "train.csv"))
    orig_val = pd.read_csv(os.path.join(Config.METADATA_DIR, "val.csv"))
    orig_test = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

    # Sample subsets
    sub_train = orig_train.head(n_samples).copy()
    sub_val = orig_val.head(n_samples).copy()
    sub_test = orig_test.head(n_samples).copy()

    # Save subsets
    sub_train.to_csv(subset_train_path, index=False)
    sub_val.to_csv(subset_val_path, index=False)
    sub_test.to_csv(subset_test_path, index=False)

    print(f"Created subset metadata with {n_samples} samples each.")
    return subset_train_path, subset_val_path, subset_test_path


def verify_utils():
    """
    Verifies utility functions: reading DICOM, resizing, normalizing.
    """
    print("\n--- Verifying Utils ---")

    # Pick a sample file from the dataset
    # We know input/train/00000/FLAIR/ exists from the description
    # We need to find a valid file. Let's look at the subset dataframe we just made.
    df = pd.read_csv(Config.TRAIN_CSV)
    row = df.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])
    files = os.listdir(flair_dir)
    # Filter for dcm
    dcm_files = [f for f in files if f.endswith(".dcm")]
    if not dcm_files:
        print("No DICOM files found in sample directory. Skipping util verification.")
        return

    sample_path = os.path.join(flair_dir, dcm_files[0])

    # 1. Test read_dicom_robust
    img = read_dicom_robust(sample_path)
    print(f"Read DICOM shape: {img.shape}, dtype: {img.dtype}")

    assert len(img.shape) == 2, "DICOM image should be 2D"

    # 2. Test resize_image
    target_size = 128
    img_resized = resize_image(img, size=target_size)
    print(f"Resized shape: {img_resized.shape}")
    assert img_resized.shape == (target_size, target_size), "Resize failed"

    # 3. Test normalize_min_max
    img_norm = normalize_min_max(img_resized)
    print(f"Normalized range: [{img_norm.min():.4f}, {img_norm.max():.4f}]")
    assert (
        img_norm.min() >= 0.0 and img_norm.max() <= 1.0
    ), "Normalization out of bounds"
    assert img_norm.dtype == np.float32, "Normalization should return float32"

    print("Utils verification passed.")


def verify_data_pipeline():
    """
    Verifies Dataset and DataLoader logic.
    """
    print("\n--- Verifying Data Pipeline ---")

    # Use the subset CSVs configured in Config
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    # Expected: [Batch_Size, 12, 224, 224]
    assert images.shape[1] == 12, f"Expected 12 channels, got {images.shape[1]}"
    assert images.shape[2] == Config.IMG_SIZE, "Wrong height"
    assert images.shape[3] == Config.IMG_SIZE, "Wrong width"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and labels"

    print("Data pipeline verification passed.")
    return train_loader, val_loader, test_loader


def verify_model(device):
    """
    Verifies Model initialization and forward pass.
    """
    print("\n--- Verifying Model ---")

    model = AsymmetricEfficientNet().to(device)

    # Create dummy input: [Batch=2, Channels=12, H=224, W=224]
    dummy_input = torch.randn(2, 12, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    # Forward pass
    output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Expected: [Batch, 1] (Logits)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model verification passed.")
    return model


def run_demo():
    print("Initializing Demo...")
    seed_everything(42)

    # ---------------------------------------------------------
    # 1. Override Configuration for Demo
    # ---------------------------------------------------------
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.ROI_CACHE_PATH = os.path.join(Config.WORKING_DIR, "roi_cache_demo.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Speed optimizations
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Create and link subset metadata
    t_path, v_path, te_path = create_subset_metadata(n_samples=8)
    Config.TRAIN_CSV = t_path
    Config.VAL_CSV = v_path
    Config.TEST_CSV = te_path

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Verify Components
    # ---------------------------------------------------------
    verify_utils()
    train_loader, val_loader, test_loader = verify_data_pipeline()
    model = verify_model(device)

    # ---------------------------------------------------------
    # 3. Run Training Loop
    # ---------------------------------------------------------
    print("\n--- Starting Training Demo ---")
    # We pass the loaders we already created to avoid re-loading
    trained_model = run_training(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=Config.EPOCHS,
        load_cached_data=False,  # We force re-calc for the subset to ensure logic runs
    )

    # ---------------------------------------------------------
    # 4. Verify Submission
    # ---------------------------------------------------------
    print("\n--- Verifying Submission ---")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Assertions
    assert "BraTS21ID" in df_sub.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Missing MGMT_value column"
    assert len(df_sub) == 8, f"Expected 8 predictions (subset size), got {len(df_sub)}"
    assert df_sub["MGMT_value"].min() >= 0.0, "Probabilities must be >= 0"
    assert df_sub["MGMT_value"].max() <= 1.0, "Probabilities must be <= 1"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
