import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import rasterio
import json
import cv2

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode
from library.data import get_dataloaders
from library.arch import ResNet34UNetPlusPlus
from library.engine import train_one_epoch, validate, DiceBCELoss
from library.predict import predict_slide


def main():
    print("--- Starting HuBMAP Library Demonstration ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override the default configuration to run a fast demo (mini-epoch, small batch)
    print("\n[1] Configuring environment...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")

    # Reduce computational load
    Config.EPOCHS = 1
    Config.TRAIN_NUM_SAMPLES = 64  # Only process 64 tiles per "epoch"
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce workers for simple demo

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    print("\n[2] Preparing DataLoaders...")

    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # SUBSET DATA: Use only 1 image for training and validation to speed up mask generation
    # This avoids processing gigabytes of TIFFs for a simple logic verification
    train_subset = train_df.iloc[:1].copy()
    val_subset = val_df.iloc[:1].copy()

    print(f"    Training on subset: {len(train_subset)} images")
    print(f"    Validating on subset: {len(val_subset)} images")

    # Initialize DataLoaders
    # This triggers mask preprocessing (RLE -> .npy) and coordinate caching
    dataloaders = get_dataloaders(
        train_df=train_subset, val_df=val_subset, test_df=None
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Verification: Check batch shape
    images, masks = next(iter(train_loader))
    print(f"    Batch Image Shape: {images.shape}")  # Should be (B, 3, H, W)
    print(f"    Batch Mask Shape: {masks.shape}")  # Should be (B, 1, H, W)

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect image batch shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect mask batch shape"
    assert masks.max() <= 1.0 and masks.min() >= 0.0, "Mask values out of range [0, 1]"

    # 3. Model Initialization
    print("\n[3] Initializing Model...")
    model = ResNet34UNetPlusPlus(in_channels=3, classes=1)
    model.to(device)

    # Verification: Forward pass with dummy tensor
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"    Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Model output shape mismatch"

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Loop (1 Epoch)...")
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = DiceBCELoss()

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    val_dice = validate(model, val_loader, device)
    print(f"    Validation Dice: {val_dice:.4f}")
    assert 0.0 <= val_dice <= 1.0, "Dice score out of valid range [0, 1]"

    # Save checkpoint for inference step
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print("    Model checkpoint saved.")

    # 5. Inference Demonstration (Predict Slide)
    print("\n[5] Demonstrating Inference on Synthetic Data...")

    # Create a dummy TIFF image in working directory to test predict_slide logic
    # We do this to avoid processing a massive 2GB file from input which takes too long for a demo.
    dummy_tiff_path = os.path.join(Config.WORKING_DIR, "dummy_test.tiff")
    dummy_anat_path = os.path.join(Config.WORKING_DIR, "dummy_anat.json")

    H, W = 2048, 2048
    dummy_img_data = np.random.randint(0, 255, (3, H, W), dtype=np.uint8)

    # Save as TIFF
    with rasterio.open(
        dummy_tiff_path,
        "w",
        driver="GTiff",
        height=H,
        width=W,
        count=3,
        dtype=dummy_img_data.dtype,
    ) as dst:
        dst.write(dummy_img_data)

    # Create dummy anatomical JSON (Cortex polygon)
    # Define a box in the middle as "Cortex"
    cortex_poly = [[[500, 500], [1500, 500], [1500, 1500], [500, 1500], [500, 500]]]
    anat_data = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": cortex_poly},
            "properties": {"classification": {"name": "Cortex", "colorRGB": -1}},
        }
    ]
    with open(dummy_anat_path, "w") as f:
        json.dump(anat_data, f)

    print(f"    Created dummy image: {dummy_tiff_path} ({H}x{W})")

    # Run Inference
    # This tests tiling, TTA, gaussian weighting, and anatomical filtering
    pred_mask = predict_slide(model, dummy_tiff_path, dummy_anat_path, device)

    print(f"    Prediction Shape: {pred_mask.shape}")
    assert pred_mask.shape == (H, W), "Prediction mask shape mismatch"
    assert pred_mask.dtype == np.uint8, "Prediction mask should be uint8"

    # Verify Anatomical Filter: Pixels outside the cortex box (e.g., 100, 100) should be 0
    # The model is random/untrained so it might predict 1s, but the filter forces 0s outside ROI.
    # Note: predict_slide applies filter at end.

    # We check a pixel definitely outside the polygon (0,0)
    pixel_val_outside = pred_mask[0, 0]
    assert (
        pixel_val_outside == 0
    ), "Anatomical filter failed: Pixel outside Cortex is not 0"

    # 6. Submission Encoding
    print("\n[6] Encoding Submission...")
    rle_str = rle_encode(pred_mask)

    # Verify RLE decoding matches
    decoded_mask = rle_decode(rle_str, (H, W))
    match = np.array_equal(pred_mask, decoded_mask)
    print(f"    RLE Encode/Decode Match: {match}")
    assert match, "RLE encoding/decoding cycle failed"

    # Create submission DataFrame format
    sub_df = pd.DataFrame([{"id": "dummy_test", "predicted": rle_str}])
    print("    Submission DataFrame created successfully.")
    print(sub_df.head())

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
