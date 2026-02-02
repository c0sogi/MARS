import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Configuration Overrides
# We modify the Config class attributes before importing other modules that might use them.
from library.config import Config

# Set DEBUG to True to use only 50 samples
Config.DEBUG = True
# Use a separate working directory for this demonstration to avoid cache conflicts
Config.WORK_DIR = "./working/demo_execution_script"
Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
# Reduce training parameters for speed
Config.EPOCHS = 1
Config.BATCH_SIZE = 8
Config.ENCODER_NAME = "resnet18"  # Use a lighter encoder for the demo

# Ensure directories exist
Config.setup()

# 2. Imports from Library
from library.utils import (
    seed_everything,
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
)
from library.dataset import SaltDataset
from library.model import SaltSegModel
from library.losses import BCEDiceLoss, DeepSupervisionLoss
from library.engine import train_one_epoch, validate, generate_submission


def run_demo():
    print(f"Starting Demo Run in {Config.WORK_DIR}...")
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 3. Utility Verification
    # -------------------------------------------------------------------------
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding/Decoding
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[20:30, 20:30] = 1
    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, (101, 101))
    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Encode/Decode failed consistency check."
    print("RLE Encode/Decode: OK")

    # Test Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101, 3), dtype=np.uint8)
    padded_img = pad_image(dummy_img, target_size=128)
    assert padded_img.shape == (
        128,
        128,
        3,
    ), f"Padding failed. Shape: {padded_img.shape}"
    unpadded_img = unpad_image(padded_img, original_size=101)
    assert np.array_equal(
        dummy_img, unpadded_img
    ), "Pad/Unpad failed consistency check."
    print("Image Padding/Unpadding: OK")

    # -------------------------------------------------------------------------
    # 4. Dataset Initialization
    # -------------------------------------------------------------------------
    print("\n--- Initializing Datasets ---")
    # load_cached_data=False forces the dataset to process the raw files
    # and save new cache files in our specific Config.WORK_DIR
    train_dataset = SaltDataset(mode="train", fold_index=0, load_cached_data=False)
    val_dataset = SaltDataset(mode="val", fold_index=0, load_cached_data=False)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Assert we are in debug mode (size should be small, split from 50 total samples)
    # StratifiedKFold with 5 folds on 50 samples -> approx 40 train, 10 val
    assert len(train_dataset) < 100, "Dataset size too large for DEBUG mode."

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple debug script stability
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Shapes
    imgs, masks, ids = next(iter(train_loader))
    print(f"Batch Image Shape: {imgs.shape}")  # Should be (B, 3, 128, 128)
    print(f"Batch Mask Shape: {masks.shape}")  # Should be (B, 1, 128, 128)

    assert imgs.shape == (Config.BATCH_SIZE, 3, 128, 128)
    assert masks.shape == (Config.BATCH_SIZE, 1, 128, 128)

    # -------------------------------------------------------------------------
    # 5. Model Setup
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    # Using resnet18 and pretrained=False for speed
    model = SaltSegModel(encoder_name="resnet18", pretrained=False)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, 128, 128).to(device)
    with torch.no_grad():
        outputs = model(dummy_input)

    # Model uses Deep Supervision, so it returns a list of tensors
    assert isinstance(outputs, list), "Model output should be a list."
    print(f"Number of Deep Supervision outputs: {len(outputs)}")
    print(f"Final Output Shape: {outputs[-1].shape}")
    assert outputs[-1].shape == (2, 1, 128, 128)

    # -------------------------------------------------------------------------
    # 6. Training Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Loop ---")
    criterion = DeepSupervisionLoss(BCEDiceLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Train for 1 epoch
    avg_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, epoch=1
    )
    assert not np.isnan(avg_loss), "Training Loss is NaN"

    # -------------------------------------------------------------------------
    # 7. Validation Loop
    # -------------------------------------------------------------------------
    print("\n--- Starting Validation Loop ---")
    val_loss, map_score = validate(model, val_loader, criterion, device)
    print(f"Validation Result -> Loss: {val_loss:.4f}, mAP: {map_score:.4f}")
    assert not np.isnan(val_loss), "Validation Loss is NaN"

    # -------------------------------------------------------------------------
    # 8. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    test_dataset = SaltDataset(mode="test", load_cached_data=False)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    generate_submission(model, test_loader, device, output_path=Config.SUBMISSION_PATH)

    # Verify file creation
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created successfully at {Config.SUBMISSION_PATH}")
        print(f"Rows in submission: {len(df_sub)}")
        print("First 3 rows:")
        print(df_sub.head(3))

        # Check format
        assert "id" in df_sub.columns and "rle_mask" in df_sub.columns
        assert len(df_sub) == len(test_dataset)
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
