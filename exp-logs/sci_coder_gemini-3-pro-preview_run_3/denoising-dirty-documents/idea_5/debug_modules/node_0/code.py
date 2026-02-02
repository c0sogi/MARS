import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import library modules
from library.config import Config
from library.utils import seed_everything, load_image, calculate_rmse
from library.dataset import prepare_datasets
from library.model import RDN
from library.train import run_training
from library.inference import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("--- Starting Library Verification Demo ---")

    # 1. Setup & Configuration Overrides
    # We modify the Config class attributes directly to create a "Demo Mode"
    # that runs quickly and uses a separate working directory.
    print("\n[1/6] Configuring environment for rapid execution...")

    demo_working_dir = "./working/demo_execution"
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Paths
    Config.WORKING_DIR = demo_working_dir
    Config.TRAIN_PATCHES_CACHE = os.path.join(demo_working_dir, "train_patches.npy")
    Config.VAL_PATCHES_CACHE = os.path.join(demo_working_dir, "val_patches.npy")
    Config.MODEL_PATH = os.path.join(demo_working_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "demo_submission.csv")

    # Override Data Hyperparameters for Speed
    # Large stride = fewer patches = faster processing
    Config.STRIDE = 200
    Config.PATCH_SIZE = 32

    # Override Model Hyperparameters for Speed (Tiny Model)
    Config.RDN_G0 = 16
    Config.RDN_NUM_BLOCKS = 1
    Config.RDN_NUM_LAYERS = 2

    # Override Training Hyperparameters
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Set Seed
    seed_everything(Config.SEED)
    print("Configuration updated successfully.")

    # 2. Verify Utilities
    print("\n[2/6] Verifying Utility Functions...")

    # Test calculate_rmse
    y_true = np.zeros((10, 10))
    y_pred = np.ones((10, 10))
    rmse = calculate_rmse(y_true, y_pred)
    assert abs(rmse - 1.0) < 1e-6, f"RMSE calculation failed. Expected 1.0, got {rmse}"
    print("  - calculate_rmse: OK")

    # Test load_image
    # Get a valid image path from metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    sample_rel_path = df_train.iloc[0]["input_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    img = load_image(sample_full_path)
    assert isinstance(img, np.ndarray), "load_image should return a numpy array"
    assert (
        img.ndim == 2
    ), f"load_image should return grayscale (2D), got ndim={img.ndim}"
    assert (
        0.0 <= img.min() and img.max() <= 1.0
    ), "Image values must be normalized to [0, 1]"
    print(f"  - load_image: OK (Shape: {img.shape})")

    # 3. Verify Dataset Pipeline
    print("\n[3/6] Verifying Dataset Pipeline...")

    # Force regeneration of data with new sparse stride
    if os.path.exists(Config.TRAIN_PATCHES_CACHE):
        os.remove(Config.TRAIN_PATCHES_CACHE)
    if os.path.exists(Config.VAL_PATCHES_CACHE):
        os.remove(Config.VAL_PATCHES_CACHE)

    train_ds, val_ds = prepare_datasets(load_cached_data=False)

    assert len(train_ds) > 0, "Training dataset is empty"
    assert len(val_ds) > 0, "Validation dataset is empty"

    # Check item structure
    sample_item = train_ds[0]
    # Expecting tuple (patch, target)
    assert isinstance(sample_item, tuple), "Dataset item should be a tuple"
    patch, target = sample_item

    # Check Tensor conversion
    assert isinstance(patch, torch.Tensor), "Patch should be a torch.Tensor"
    assert isinstance(target, torch.Tensor), "Target should be a torch.Tensor"

    # Check Shapes: (1, H, W)
    expected_shape = (1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert (
        patch.shape == expected_shape
    ), f"Patch shape mismatch. Expected {expected_shape}, got {patch.shape}"
    assert (
        target.shape == expected_shape
    ), f"Target shape mismatch. Expected {expected_shape}, got {target.shape}"

    print(f"  - Dataset creation: OK (Train: {len(train_ds)}, Val: {len(val_ds)})")

    # 4. Verify Model Architecture
    print("\n[4/6] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = RDN().to(device)

    # Create dummy input batch
    dummy_input = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(device)

    # Forward pass
    try:
        output = model(dummy_input)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")

    assert (
        output.shape == dummy_input.shape
    ), f"Output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"

    print("  - Model forward pass: OK")

    # 5. Verify Training Loop
    print("\n[5/6] Verifying Training Loop (1 Epoch)...")

    # Run training
    # This uses the modified Config (1 epoch, small batch)
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,  # Use the cache we just generated
        num_workers=Config.NUM_WORKERS,
    )

    assert os.path.exists(
        Config.MODEL_PATH
    ), "Model checkpoint was not saved after training."
    print("  - Training execution: OK")

    # 6. Verify Inference & Submission
    print("\n[6/6] Verifying Inference and Submission Generation...")

    # To save time, we'll temporarily mock the test metadata to only include 2 images
    # Otherwise inference on all test images might take too long even with a small model
    original_test_metadata_path = Config.TEST_METADATA

    # Read original, take head, save to temp location
    df_test_full = pd.read_csv(original_test_metadata_path)
    df_test_subset = df_test_full.head(2)

    temp_test_metadata = os.path.join(demo_working_dir, "temp_test.csv")
    df_test_subset.to_csv(temp_test_metadata, index=False)

    # Point Config to temp metadata
    Config.TEST_METADATA = temp_test_metadata

    try:
        generate_submission(
            model_path=Config.MODEL_PATH, submission_output=Config.SUBMISSION_PATH
        )
    finally:
        # Restore config path just in case
        Config.TEST_METADATA = original_test_metadata_path

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission missing required columns 'id' and 'value'"

    # Check values
    assert (
        df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
    ), "Submission values out of range [0, 1]"

    # Check row count logic
    # We processed 2 images. Each image is roughly 540x420 (varies).
    # Just checking it's not empty and has substantial rows.
    assert len(df_sub) > 1000, "Submission file seems too small."

    print("  - Inference execution: OK")
    print("  - Submission format: OK")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
