import os
import sys
import shutil
import warnings
import pandas as pd
import torch
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_decode
from library.dataset import get_loaders, get_test_loader
from library.model import AttentionUNet25D
from library.loss import TverskyLoss
from library.train import run_training
from library.inference import predict_and_submit


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Create a separate directory for demo outputs
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters to run quickly
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pth")
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Training constraints
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small number of samples for training loop
    Config.ENCODER_NAME = "efficientnet_b0"

    # Setup directories based on new config
    Config.setup()

    # Create a mini test metadata file to speed up inference
    # We take a small subset of the actual test metadata
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_df = full_test_df.head(20).copy()  # Process 20 slices
    mini_test_path = os.path.join(DEMO_DIR, "mini_test_metadata.csv")
    mini_test_df.to_csv(mini_test_path, index=False)

    # Point Config to this mini test file
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading and Validation
    # -------------------------------------------------------------------------
    print("\n[2] Validating Data Loading...")

    # Get loaders (force no cache to demonstrate processing)
    train_loader, val_loader = get_loaders(load_cached_data=False)

    # Fetch one batch
    images, masks = next(iter(train_loader))

    # Verify shapes
    # Images: (Batch, Channels=3, Height=256, Width=256) -> 2.5D stack
    # Masks:  (Batch, Classes=3, Height=256, Width=256)
    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Mask Batch Shape: {masks.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        256,
        256,
    ), "Incorrect image batch shape"
    assert masks.shape == (Config.BATCH_SIZE, 3, 256, 256), "Incorrect mask batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"

    # Verify value ranges
    print(f"    Image Max Value: {images.max().item():.2f} (Normalized)")
    print(f"    Mask Unique Values: {torch.unique(masks).tolist()}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture and Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Validating Model and Loss...")

    device = torch.device(Config.DEVICE)
    model = AttentionUNet25D().to(device)
    loss_fn = TverskyLoss()

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)

    # Forward pass
    outputs = model(images)

    # Verify output shape
    assert outputs.shape == (
        Config.BATCH_SIZE,
        3,
        256,
        256,
    ), "Model output shape mismatch"

    # Calculate loss
    loss = loss_fn(outputs, masks)
    print(f"    Initial Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Debug Mode)...")

    # run_training handles the loop, validation, and model saving
    # debug=True limits the number of batches per epoch
    run_training(debug=True, load_cached_data=True)

    # Verify model was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Model successfully saved to {Config.MODEL_SAVE_PATH}")
        file_size = os.path.getsize(Config.MODEL_SAVE_PATH) / (1024 * 1024)
        print(f"    Model size: {file_size:.2f} MB")
    else:
        raise FileNotFoundError("Model file was not created by run_training.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference and Submission Generation...")

    # predict_and_submit loads the saved model and generates the CSV
    # We use load_cached_data=False to force it to read our new mini_test_metadata
    predict_and_submit(load_cached_data=False)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file created at {Config.SUBMISSION_PATH}")
        print(f"    Rows: {len(df_sub)}")
        print(f"    Columns: {list(df_sub.columns)}")

        # Basic assertions on submission content
        assert "id" in df_sub.columns
        assert "class" in df_sub.columns
        assert "predicted" in df_sub.columns

        # Check if we have rows for the classes
        unique_classes = df_sub["class"].unique()
        print(f"    Classes found: {unique_classes}")
        assert "large_bowel" in unique_classes
        assert "small_bowel" in unique_classes
        assert "stomach" in unique_classes

        # Check an example prediction
        example_rle = df_sub.iloc[0]["predicted"]
        print(f"    Example RLE (first row): '{example_rle}'")

    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
