import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_device
from library.roi_selection import get_roi_indices
from library.data_loader import get_dataloader
from library.model_arch import AsymmetricEfficientNet
from library.engine import train_model, predict_consensus


def run_demo():
    print("=== Starting MGMT Classification Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Modify Config for a fast demonstration run
    print("[Setup] Configuring parameters for demo run...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Small batch size for speed/memory
    Config.NUM_WORKERS = 2

    # Use a specific working directory for this demo to isolate outputs
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths that depend on WORKING_DIR or are specific to this run
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"[Setup] Device: {device}")
    print(f"[Setup] Working Directory: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. ROI Selection Verification
    # --------------------------------------------------------------------------
    print("\n[Verification] Testing ROI Selection Logic...")

    # Load validation metadata to test ROI generation
    val_meta_path = Config.VAL_METADATA_PATH
    assert os.path.exists(val_meta_path), "Validation metadata file missing."
    df_val = pd.read_csv(val_meta_path)

    # Generate ROI indices (force recompute to test logic, ignore cache for demo safety)
    # We use a unique split name to avoid messing with existing caches
    roi_df = get_roi_indices(df_val, split_name="demo_val", load_cached_data=False)

    # Assertions
    assert isinstance(roi_df, pd.DataFrame), "ROI function should return a DataFrame"
    assert "roi_anchor1_idx" in roi_df.columns, "Missing anchor 1 index column"
    assert "roi_anchor2_idx" in roi_df.columns, "Missing anchor 2 index column"
    assert len(roi_df) == len(df_val), "ROI DataFrame length mismatch"
    print(
        f"[Verification] ROI Selection passed. Generated indices for {len(roi_df)} subjects."
    )

    # --------------------------------------------------------------------------
    # 3. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n[Verification] Testing Data Loader...")

    # Create Training DataLoader
    # We use 'train' split which returns flattened views (2x samples)
    train_loader = get_dataloader(
        split_name="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        load_cached_data=True,  # Use caching to speed up if run multiple times
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Assertions
    # Expected shape: (Batch, 12, 224, 224)
    expected_channels = Config.IN_CHANNELS
    expected_size = Config.IMG_SIZE

    print(f"[Verification] Batch Shape: {images.shape}")
    print(f"[Verification] Labels Shape: {labels.shape}")

    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert (
        images.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {images.shape[1]}"
    assert (
        images.shape[2] == expected_size
    ), f"Expected height {expected_size}, got {images.shape[2]}"
    assert (
        images.shape[3] == expected_size
    ), f"Expected width {expected_size}, got {images.shape[3]}"
    assert labels.dim() == 1, "Labels should be 1D tensor"

    print("[Verification] Data Loader passed.")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[Verification] Testing Model Architecture...")

    model = AsymmetricEfficientNet()
    model.to(device)

    # Forward pass with the fetched batch
    images = images.to(device)
    logits = model(images)

    # Assertions
    assert logits.shape == (
        images.size(0),
        1,
    ), f"Output shape mismatch. Expected ({images.size(0)}, 1), got {logits.shape}"
    print(f"[Verification] Model Forward Pass passed. Output shape: {logits.shape}")

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[Execution] Starting Training Loop (1 Epoch)...")

    # Get Validation Loader (returns grouped views)
    val_loader = get_dataloader(
        split_name="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # Setup Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run Training
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        epochs=Config.EPOCHS,
    )

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"[Execution] Training finished. Checkpoint saved at {best_model_path}")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission
    # --------------------------------------------------------------------------
    print("\n[Execution] Generating Predictions on Test Set...")

    # Get Test Loader
    test_loader = get_dataloader(
        split_name="test",
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        load_cached_data=True,
    )

    # Run Inference
    df_submission = predict_consensus(trained_model, test_loader, device)

    # Verify Submission
    print(f"[Execution] Submission generated with shape: {df_submission.shape}")
    print(df_submission.head())

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found on disk."
    assert list(df_submission.columns) == [
        "BraTS21ID",
        "MGMT_value",
    ], "Incorrect submission columns."
    assert (
        len(df_submission) == 59
    ), f"Expected 59 test predictions, got {len(df_submission)}"

    # Check value range
    preds = df_submission["MGMT_value"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure no warnings clutter the output
    import warnings

    warnings.filterwarnings("ignore")

    try:
        run_demo()
    except AssertionError as e:
        print(f"\n!!! Verification Failed: {e} !!!")
        sys.exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
