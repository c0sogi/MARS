import os
import sys
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_class_weights
from library.data import get_dataloaders
from library.model import CassavaClassifier
from library.engine import train_one_epoch, run


def main():
    print("=== Starting Cassava Leaf Disease Classification Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Overrides for Speed and Testing
    # ---------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")
    # Enable debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Use only 32 images for this test

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Isolate output for this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Setup directories based on new config
    Config.setup_directories()
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Test Utils
    # ---------------------------------------------------------
    print("\n[2] Testing Library: Utils...")
    seed_everything(42)

    print("    Computing class weights...")
    # Force re-computation (load_cached_data=False) to test logic
    weights = compute_class_weights(
        Config.TRAIN_METADATA,
        load_cached_data=False,
        debug=True,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Verify weights
    assert isinstance(weights, torch.Tensor), "Weights must be a torch.Tensor"
    assert (
        weights.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} weights, got {weights.shape[0]}"
    assert (
        weights.device.type == Config.DEVICE
    ), f"Weights should be on device {Config.DEVICE}"
    print("    -> Utils verification successful.")

    # ---------------------------------------------------------
    # 3. Test Data
    # ---------------------------------------------------------
    print("\n[3] Testing Library: Data...")
    print("    Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Loader Content
    assert len(train_loader) > 0, "Train loader should not be empty"

    # Fetch one batch
    images, labels = next(iter(train_loader))
    print(f"    Fetched batch shape - Images: {images.shape}, Labels: {labels.shape}")

    # Verify Shapes
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"
    print("    -> Data verification successful.")

    # ---------------------------------------------------------
    # 4. Test Model
    # ---------------------------------------------------------
    print("\n[4] Testing Library: Model...")
    print("    Instantiating CassavaClassifier...")
    # Use pretrained=False to speed up initialization for this unit test
    model = CassavaClassifier(pretrained=False)
    model.to(Config.DEVICE)

    # Verify Forward Pass
    print("    Running forward pass...")
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"

    # Verify Freeze/Unfreeze Logic
    print("    Testing backbone freezing...")
    model.freeze_backbone()
    # Check the first parameter of the backbone
    param = next(model.backbone.parameters())
    assert (
        param.requires_grad is False
    ), "Backbone parameters should be frozen (requires_grad=False)"

    model.unfreeze_backbone()
    param = next(model.backbone.parameters())
    assert (
        param.requires_grad is True
    ), "Backbone parameters should be unfrozen (requires_grad=True)"
    print("    -> Model verification successful.")

    # ---------------------------------------------------------
    # 5. Test Engine (Full Execution)
    # ---------------------------------------------------------
    print("\n[5] Testing Library: Engine (Full Run)...")
    print("    Executing training and inference pipeline...")

    # Ensure clean state for submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Run the engine
    # Note: This will re-instantiate the model (with pretrained=True) and run the loop
    run(train_loader, val_loader, test_loader, epochs=1, learning_rate=1e-4)

    # ---------------------------------------------------------
    # 6. Validate Submission
    # ---------------------------------------------------------
    print("\n[6] Validating Submission...")
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Check columns
    assert (
        "image_id" in df_sub.columns and "label" in df_sub.columns
    ), "Submission missing required columns"

    # Check row count (should match test subset size)
    # In Debug mode, test set is also sliced to DEBUG_SUBSET_SIZE
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    print("    -> Submission verification successful.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
