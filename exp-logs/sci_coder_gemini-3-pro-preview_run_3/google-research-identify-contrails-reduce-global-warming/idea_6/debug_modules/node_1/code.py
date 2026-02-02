import os
import torch
import numpy as np
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, dice_score
from library.data import get_loaders
from library.model import ContrailUnet
from library.loss import DiceBCELoss
from library.train import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Contrail Segmentation Library Demo ===\n")

    # --- 1. Setup & Configuration ---
    print("1. Setting up configuration and environment...")

    # Modify Config for this demo to run quickly and separately
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Reduce computational load for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2

    # Re-run setup to create the new directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(42)
    print(f"   Working Directory: {Config.WORKING_DIR}")
    print(f"   Device: {Config.DEVICE}")
    print("   Configuration updated successfully.\n")

    # --- 2. Verify Utilities (RLE Encoding) ---
    print("2. Verifying Utility Functions (RLE)...")
    # Create a simple 3x3 mask
    # Pixel indices (column-major):
    # 1 (0,0), 2 (1,0), 3 (2,0)
    # 4 (0,1), 5 (1,1), 6 (2,1)
    # 7 (0,2), 8 (1,2), 9 (2,2)
    dummy_mask = np.array(
        [
            [1, 0, 0],  # (0,0) is 1 -> index 1
            [1, 1, 0],  # (1,0) is 1 -> index 2, (1,1) is 1 -> index 5
            [0, 0, 0],
        ]
    )
    # Expected RLE: "1 2 5 1" (Run starting at 1 length 2, Run starting at 5 length 1)
    encoded = rle_encode(dummy_mask)
    print(f"   Input Mask Shape: {dummy_mask.shape}")
    print(f"   Encoded RLE: {encoded}")

    assert (
        encoded == "1 2 5 1"
    ), f"RLE Encoding failed. Expected '1 2 5 1', got '{encoded}'"
    print("   RLE Encoding verified.\n")

    # --- 3. Data Loading ---
    print("3. Verifying Data Loading...")
    # Use debug=True to load a tiny subset
    loaders = get_loaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=True
    )
    train_loader = loaders["train"]

    # Fetch one batch
    images, masks = next(iter(train_loader))

    print(f"   Batch Image Shape: {images.shape}")  # Expected: (B, 6, 256, 256)
    print(f"   Batch Mask Shape: {masks.shape}")  # Expected: (B, 1, 256, 256)

    assert images.shape == (
        Config.BATCH_SIZE,
        Config.N_CHANNELS,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Image Shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect Mask Shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"
    print("   Data Loading verified.\n")

    # --- 4. Model Instantiation & Inference ---
    print("4. Verifying Model Architecture...")
    model = ContrailUnet()
    model.to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    masks = masks.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(images)

    print(f"   Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Output shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("   Model forward pass verified.\n")

    # --- 5. Loss & Metrics ---
    print("5. Verifying Loss and Metrics...")
    criterion = DiceBCELoss()

    # Calculate Loss
    loss = criterion(logits, masks)
    print(f"   Calculated Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss should be non-negative"

    # Calculate Dice Score
    # Apply sigmoid to logits for metric calculation
    preds_prob = torch.sigmoid(logits)
    score = dice_score(preds_prob, masks, threshold=0.5)
    print(f"   Calculated Dice Score: {score:.6f}")

    assert 0.0 <= score <= 1.0, "Dice score must be between 0 and 1"
    print("   Loss and Metrics verified.\n")

    # --- 6. Full Training Loop (Trainer) ---
    print("6. Verifying Trainer (Running 1 Epoch)...")

    # Initialize Trainer with debug=True
    trainer = Trainer(debug=True)

    # Run training
    # This will run for Config.EPOCHS (set to 1 above)
    trainer.fit()
    print("   Training loop completed.\n")

    # --- 7. Checkpoint Verification ---
    print("7. Verifying Artifact Generation...")
    saved_checkpoints = os.listdir(Config.CHECKPOINT_DIR)
    print(f"   Found checkpoints: {saved_checkpoints}")

    assert len(saved_checkpoints) > 0, "No checkpoints were saved."

    # Verify we can load the checkpoint
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, saved_checkpoints[0])
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state_dict" in checkpoint, "Checkpoint missing model state dict"
    assert "dice" in checkpoint, "Checkpoint missing dice score"
    print("   Checkpoint verification successful.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
