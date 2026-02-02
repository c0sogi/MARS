import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from library import config, utils, data_processing, dataset, model, train_eval


def run_demo():
    print("============================================================")
    print(" MGMT Methylation Prediction - Pipeline Demo")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override config defaults to run quickly
    config.DEBUG = True
    config.DEBUG_DATASET_SIZE = 20  # Small subset for speed
    config.BATCH_SIZE = 4
    config.EPOCHS = 1
    config.NUM_FOLDS = 2  # Minimal folds to verify CV loop
    config.NUM_WORKERS = 2

    # Use a specific cache directory for this demo to avoid conflicts
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_execution")
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Set submission path to demo file
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Ensure reproducibility
    utils.seed_everything(42)
    device = utils.get_device()
    print(f"    Device: {device}")
    print(f"    Cache Directory: {config.CACHE_DIR}")

    # ------------------------------------------------------------------
    # 2. Verify Data Processing
    # ------------------------------------------------------------------
    print("\n[2] Verifying Data Processing Logic...")

    # Load training metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    print(f"    Loaded metadata with {len(df_train)} rows.")

    # Process a small subset manually to verify output shapes
    # This simulates what happens inside the DataLoader
    print("    Processing subset of data...")
    ids, images, labels = data_processing.process_dataset(
        df_train, dataset_name="demo_check", load_cached_data=False
    )

    # Assertions
    print(
        f"    Processed Shapes -> IDs: {ids.shape}, Images: {images.shape}, Labels: {labels.shape}"
    )

    assert len(ids) == config.DEBUG_DATASET_SIZE or len(ids) == len(
        df_train
    ), "Processed dataset size does not match debug configuration."

    # Check Image Dimensions: (N, Channels, Height, Width)
    # Config defines INPUT_CHANNELS = 9 (3 slices * 3 modalities)
    assert images.shape[1] == 9, f"Expected 9 channels, got {images.shape[1]}"
    assert images.shape[2] == config.IMG_SIZE, "Height mismatch"
    assert images.shape[3] == config.IMG_SIZE, "Width mismatch"

    # Check Normalization (0-1 range)
    img_min, img_max = images.min(), images.max()
    print(f"    Pixel Value Range: [{img_min:.4f}, {img_max:.4f}]")
    assert (
        img_min >= 0.0 and img_max <= 1.0
    ), "Images are not properly normalized to [0, 1]"

    # ------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # ------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoader...")

    # Create DataLoader using the library function
    # Note: We use a dummy phase name to trigger fresh processing/caching if needed
    loader = dataset.get_dataloader(
        df_train,
        phase="demo_loader",
        batch_size=config.BATCH_SIZE,
        num_workers=0,  # 0 for main thread debugging
        load_cached_data=False,
    )

    # Fetch a single batch
    batch = next(iter(loader))
    b_ids = batch["BraTS21ID"]
    b_imgs = batch["image"]
    b_targets = batch["target"]

    print(f"    Batch Keys: {list(batch.keys())}")
    print(f"    Batch Image Tensor Shape: {b_imgs.shape}")

    # Verify Tensor properties
    assert b_imgs.shape == (config.BATCH_SIZE, 9, config.IMG_SIZE, config.IMG_SIZE)
    assert b_targets.shape[0] == config.BATCH_SIZE
    assert b_imgs.dtype == torch.float32
    assert b_targets.dtype == torch.float32

    # ------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model (using pretrained=False for speed/offline safety in demo)
    net = model.EfficientNet9Channel(pretrained=False, num_classes=1)
    net.to(device)

    # Check 1: First layer adaptation
    first_conv = net.backbone.conv_stem
    print(f"    First Conv Layer Input Channels: {first_conv.in_channels}")
    assert (
        first_conv.in_channels == 9
    ), "Model first layer was not adapted to 9 channels."

    # Check 2: Forward pass
    print("    Running forward pass on batch...")
    with torch.no_grad():
        logits = net(b_imgs.to(device))

    print(f"    Output Logits Shape: {logits.shape}")
    assert logits.shape == (config.BATCH_SIZE, 1), "Model output shape is incorrect."

    # ------------------------------------------------------------------
    # 5. Execute Training Loop (Mini-CV)
    # ------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Mini Cross-Validation)...")

    # This function handles the CV loop, training, and model saving
    # It respects the config overrides (DEBUG=True, EPOCHS=1)
    train_eval.run_training()

    # Verify artifacts
    expected_model_path = os.path.join(config.CACHE_DIR, "best_model_fold0.pth")
    if os.path.exists(expected_model_path):
        print(f"    Success: Model file found at {expected_model_path}")
    else:
        raise FileNotFoundError(
            f"Training failed to produce model file at {expected_model_path}"
        )

    # ------------------------------------------------------------------
    # 6. Generate Submission
    # ------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # This function loads the saved models and predicts on test set
    train_eval.generate_submission()

    if os.path.exists(config.SUBMISSION_PATH):
        print(f"    Success: Submission file generated at {config.SUBMISSION_PATH}")

        # Validate content
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print("    Submission Head:")
        print(sub_df.head())

        assert "BraTS21ID" in sub_df.columns
        assert "MGMT_value" in sub_df.columns
        assert len(sub_df) > 0
        assert sub_df["MGMT_value"].dtype == float
    else:
        raise FileNotFoundError("Submission generation failed.")

    print("\n============================================================")
    print(" Demo Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
