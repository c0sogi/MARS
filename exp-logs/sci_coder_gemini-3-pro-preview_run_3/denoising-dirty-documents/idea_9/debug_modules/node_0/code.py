import os
import torch
import numpy as np
import pandas as pd
import shutil
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import ZIResDnCNN
from library.train import train_model
from library.inference import run_inference


def run_demo():
    print("Starting Demo Execution...")

    # ==========================================
    # 1. Configuration Overrides for Demo
    # ==========================================
    # We modify the Config class attributes directly to run a lightweight version
    # of the pipeline suitable for demonstration and quick validation.

    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config paths to point to the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_PATCHES_CACHE = os.path.join(DEMO_DIR, "train_patches.npy")
    Config.TRAIN_TARGETS_CACHE = os.path.join(DEMO_DIR, "train_targets.npy")
    Config.VAL_PATCHES_CACHE = os.path.join(DEMO_DIR, "val_patches.npy")
    Config.VAL_TARGETS_CACHE = os.path.join(DEMO_DIR, "val_targets.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Update Hyperparameters for speed
    Config.DEBUG = True  # Enable debug mode to limit data loading
    Config.DEBUG_SUBSET_SIZE = 500  # Only process enough patches for a few batches
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.USE_TTA = False  # Disable TTA for speed during demo inference

    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading & Validation
    # ==========================================
    print("\n--- Validating Data Loading ---")

    # Force reload of data (load_cached_data=False) to test extraction logic
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Fetch a single batch to verify shapes
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check data extraction logic.")

    print(f"Batch shapes - Input: {inputs.shape}, Target: {targets.shape}")

    # Assertions
    # Shape should be (Batch_Size, Channels, Height, Width)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Incorrect input shape: {inputs.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}"
    assert inputs.dtype == torch.float32, "Inputs should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"

    print("Data loading verification passed.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- Validating Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = ZIResDnCNN(
        num_blocks=2,  # Reduce blocks for demo speed
        num_channels=16,  # Reduce channels for demo speed
        kernel_size=3,
        padding=1,
        use_zero_gamma=True,
    ).to(device)

    # Move batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Output shape: {outputs.shape}")

    # Assertions
    assert (
        outputs.shape == inputs.shape
    ), f"Model output shape mismatch. Expected {inputs.shape}, got {outputs.shape}"

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # We call train_model. Note: train_model internally instantiates a new ZIResDnCNN
    # using Config parameters. Since we updated Config, it will use those settings
    # (though we can't easily reduce num_blocks inside train_model without changing code,
    # so it will use the default 20 blocks defined in Config class definition if not overridden.
    # To make it fast, we will temporarily monkey-patch the Config class defaults for model arch).

    original_blocks = Config.NUM_BLOCKS
    original_channels = Config.NUM_CHANNELS
    Config.NUM_BLOCKS = 2
    Config.NUM_CHANNELS = 16

    try:
        train_model(
            num_epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            load_cached_data=True,  # We just cached it in step 2
        )
    finally:
        # Restore config just in case
        Config.NUM_BLOCKS = original_blocks
        Config.NUM_CHANNELS = original_channels

    # Verify model checkpoint exists
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print("Training execution passed.")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    print("\n--- Executing Inference ---")

    # Run inference using the trained demo model
    # We need to ensure we use the same architecture parameters for loading
    # Since run_inference instantiates the model based on Config, we must ensure
    # Config matches the trained model (which used 2 blocks, 16 channels).
    Config.NUM_BLOCKS = 2
    Config.NUM_CHANNELS = 16

    run_inference(
        checkpoint_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        metadata_path=Config.TEST_METADATA_PATH,
        use_tta=Config.USE_TTA,
        device=Config.DEVICE,
    )

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Validate Submission Format
    print("Validating submission format...")
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert list(df_sub.columns) == ["id", "value"], f"Invalid columns: {df_sub.columns}"

    # Check first row ID format (image_row_col)
    first_id = df_sub.iloc[0]["id"]
    parts = first_id.split("_")
    assert len(parts) >= 3, f"Invalid ID format: {first_id}"

    # Check value range
    min_val = df_sub["value"].min()
    max_val = df_sub["value"].max()
    assert (
        min_val >= 0 and max_val <= 1
    ), f"Values out of range [0, 1]. Min: {min_val}, Max: {max_val}"

    print(f"Submission generated with {len(df_sub)} rows.")
    print("Inference verification passed.")

    print("\nAll demo steps completed successfully!")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    run_demo()
