import os
import shutil
import numpy as np
import torch
import pandas as pd

# Import library components
from library.config import Config
from library.utils import rle_encode, rle_decode, keep_largest_component_3d
from library.data import get_dataloaders
from library.model import EfficientNetFPN
from library.loss import BCEDiceLoss
from library.train import train_model, set_seed


def run_demo():
    print("=== Starting Demonstration Script ===")

    # 1. Modify Configuration for Speed and Debugging
    print("\n[1] Configuring environment for rapid execution...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Use a tiny subset of data
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.IMG_SIZE = 128  # Reduce image size for faster processing
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists and is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, IMG_SIZE=128")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Round-trip
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[20:40, 20:40] = 1
    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, (100, 100))

    if not np.array_equal(dummy_mask, decoded):
        raise AssertionError("RLE Encode -> Decode round-trip failed.")
    print("RLE encoding/decoding verified.")

    # Test 3D Connected Component Analysis
    # Create a volume with two disconnected components
    vol = np.zeros((5, 20, 20), dtype=np.uint8)
    # Component 1 (Small): 1 pixel
    vol[0, 5, 5] = 1
    # Component 2 (Large): 4x4 block
    vol[2, 10:14, 10:14] = 1

    cleaned_vol = keep_largest_component_3d(vol)

    if cleaned_vol[0, 5, 5] != 0:
        raise AssertionError("3D Component Analysis failed to remove small component.")
    if np.sum(cleaned_vol[2, 10:14, 10:14]) != 16:
        raise AssertionError("3D Component Analysis damaged the largest component.")
    print("3D Connected Component analysis verified.")

    # 3. Verify Data Loading
    print("\n[3] Verifying Data Loading Pipeline...")
    # load_cached_data=False forces the processor to respect the new Config settings (like IMG_SIZE)
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    print(f"Train Loader: {len(train_loader)} batches")
    print(f"Val Loader: {len(val_loader)} batches")

    # Fetch a single batch
    images, masks = next(iter(train_loader))

    # Validate Shapes
    # Expected: (Batch, Channels=3, Height, Width)
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Image batch shape mismatch. Expected {expected_shape}, got {images.shape}"
        )
    if masks.shape != expected_shape:
        raise AssertionError(
            f"Mask batch shape mismatch. Expected {expected_shape}, got {masks.shape}"
        )

    # Validate Data Types and Ranges
    if images.dtype != torch.float32:
        raise AssertionError(f"Images should be float32, got {images.dtype}")
    if masks.dtype != torch.float32:
        raise AssertionError(f"Masks should be float32, got {masks.dtype}")

    if images.max() > 1.0 or images.min() < 0.0:
        raise AssertionError("Images are not properly normalized to [0, 1].")

    print(f"Data batch verified. Shape: {images.shape}")

    # 4. Verify Model and Loss
    print("\n[4] Verifying Model and Loss...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (pretrained=False to avoid download overhead during demo)
    model = EfficientNetFPN(
        encoder_name="efficientnet_b0", pretrained=False, num_classes=3
    )
    model = model.to(device)

    images = images.to(device)
    masks = masks.to(device)

    # Forward Pass
    logits = model(images)
    if logits.shape != masks.shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {masks.shape}, got {logits.shape}"
        )
    print("Model forward pass successful.")

    # Loss Calculation
    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, masks)

    if torch.isnan(loss):
        raise AssertionError("Computed loss is NaN.")
    if loss.item() <= 0:
        print(
            f"Warning: Loss is {loss.item()}, expected positive value (unless perfect match)."
        )

    print(f"Loss calculation successful. Loss: {loss.item():.4f}")

    # 5. Verify Training Loop
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Run the training loop
    # We use load_cached_data=True here because we just generated the cache in step 3
    train_model(epochs=Config.EPOCHS, load_cached_data=True)

    # Verify Output
    if not os.path.exists(Config.MODEL_PATH):
        # Note: train_model only saves if val_dice improves over -1.0.
        # Since dice >= 0, it should always save after epoch 1.
        raise FileNotFoundError("Model checkpoint was not created after training.")

    print(f"Training complete. Model saved to {Config.MODEL_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
