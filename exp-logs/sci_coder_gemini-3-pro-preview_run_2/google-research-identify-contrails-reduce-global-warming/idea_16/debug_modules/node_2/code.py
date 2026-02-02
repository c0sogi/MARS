import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coef
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.loss import HybridLoss
from library.train import train_model
from library.inference import inference


def verify_utilities():
    print("\n[1] Verifying Utilities (RLE and Dice)...")

    # --- Test RLE Encoding ---
    # Create a simple 4x4 mask
    # Pixels are numbered top-to-bottom, then left-to-right
    # Mask:
    # 1 0 0 0
    # 1 0 0 0
    # 0 0 0 0
    # 0 0 0 0
    # Flattened (Fortran/Column-major): 1, 1, 0, 0, 0...
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 0] = 1

    encoded = rle_encode(mask)
    expected = "1 2"  # Start at 1, length 2
    assert (
        encoded == expected
    ), f"RLE Encoding failed. Expected '{expected}', got '{encoded}'"

    # Test Empty RLE
    empty_mask = np.zeros((4, 4), dtype=np.uint8)
    assert rle_encode(empty_mask) == "-", "RLE Encoding for empty mask failed."

    print("   -> RLE Encoding verified.")

    # --- Test Dice Coefficient ---
    # Perfect overlap
    pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    dice = dice_coef(pred, target, smooth=0)
    assert torch.isclose(dice, torch.tensor(1.0)), f"Dice (Perfect) failed. Got {dice}"

    # No overlap
    pred = torch.tensor([0.0, 0.0, 1.0, 1.0])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    dice = dice_coef(pred, target, smooth=1e-6)
    # 2*0 / (2+2) = 0
    assert torch.isclose(
        dice, torch.tensor(0.0), atol=1e-5
    ), f"Dice (No overlap) failed. Got {dice}"

    print("   -> Dice Coefficient verified.")


def verify_dataset_and_loader():
    print("\n[2] Verifying Dataset and Data Loading...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Subset for speed
    subset_df = train_df.head(10).copy()

    # Initialize Dataset
    ds = ContrailDataset(subset_df, stage="train", load_cached_data=True)

    # Check length
    assert len(ds) == 10, "Dataset length mismatch."

    # Fetch one sample
    img, mask, rid = ds[0]

    # Verify shapes
    # Image: (6, 256, 256) -> 6 channels (3 Ash + 3 Temporal Diff)
    assert img.shape == (
        6,
        256,
        256,
    ), f"Image shape incorrect. Expected (6, 256, 256), got {img.shape}"
    assert mask.shape == (
        1,
        256,
        256,
    ), f"Mask shape incorrect. Expected (1, 256, 256), got {mask.shape}"
    assert isinstance(rid, str), "Record ID should be a string."

    # Verify values are normalized (approx [0, 1] or standardized)
    # Since we use ToTensorV2 and custom normalization in dataset.py, check basic range
    # Note: dataset.py clips to [0, 1]
    assert (
        img.min() >= 0.0 and img.max() <= 1.0
    ), "Image data out of expected range [0, 1]."

    print(f"   -> Dataset verified. Sample ID: {rid}, Img Shape: {img.shape}")


def verify_model_architecture(device):
    print("\n[3] Verifying Model Architecture...")

    model = ConvNeXtUNet().to(device)

    # Create dummy batch: (Batch, Channels, Height, Width)
    batch_size = 2
    dummy_input = torch.randn(
        batch_size, Config.IN_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE
    ).to(device)

    # Forward pass
    output = model(dummy_input)

    # Verify output shape: (Batch, Num_Classes, Height, Width)
    expected_shape = (batch_size, Config.NUM_CLASSES, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("   -> Model forward pass verified.")
    return model


def verify_loss_function(device):
    print("\n[4] Verifying Loss Function...")

    loss_fn = HybridLoss().to(device)

    # Dummy logits (raw model output) and targets (binary mask)
    logits = torch.randn(2, 1, 256, 256, device=device, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss should be non-negative."

    # Verify backward pass capability
    loss.backward()

    print(f"   -> Loss function verified. Value: {loss.item():.4f}")


def run_demo_training():
    print("\n[5] Running Demo Training Loop...")

    # Run the library's training function
    # It uses the Config we patched in main()
    try:
        train_model()
        print("   -> Training loop completed successfully.")
    except Exception as e:
        print(f"   -> Training failed with error: {e}")
        raise e


def run_demo_inference():
    print("\n[6] Running Demo Inference...")

    # Create a subset test metadata file to speed up inference
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    subset_test_df = test_df.head(10).copy()

    # Save to a temporary location
    temp_test_meta_path = os.path.join(Config.WORKING_DIR, "temp_test_metadata.csv")
    subset_test_df.to_csv(temp_test_meta_path, index=False)

    # Patch Config to point to this temp file
    original_test_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = temp_test_meta_path

    try:
        inference()

        # Verify submission file
        if os.path.exists(Config.SUBMISSION_PATH):
            sub_df = pd.read_csv(Config.SUBMISSION_PATH)
            assert (
                len(sub_df) == 10
            ), f"Submission length mismatch. Expected 10, got {len(sub_df)}"
            assert (
                "record_id" in sub_df.columns and "encoded_pixels" in sub_df.columns
            ), "Submission columns missing."
            print(
                f"   -> Inference completed. Submission saved to {Config.SUBMISSION_PATH}"
            )
        else:
            raise FileNotFoundError("Submission file was not created.")

    finally:
        # Restore Config
        Config.TEST_METADATA_PATH = original_test_path


if __name__ == "__main__":
    # 1. Setup Environment
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Patch Configuration for Demo/Speed
    print("Configuring environment for rapid demonstration...")
    Config.PROJECT_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.MAX_TRAIN_SAMPLES = 20  # Only use 20 samples for training
    Config.MAX_VAL_SAMPLES = 10  # Only use 10 samples for validation
    Config.NUM_WORKERS = 2

    # Print config to confirm
    # Config.print_config() # Optional, keeping output clean

    # 3. Execution Steps
    verify_utilities()
    verify_dataset_and_loader()
    verify_model_architecture(device)
    verify_loss_function(device)

    # Run Training Integration
    run_demo_training()

    # Run Inference Integration
    run_demo_inference()

    print("\nAll demonstration steps completed successfully.")
