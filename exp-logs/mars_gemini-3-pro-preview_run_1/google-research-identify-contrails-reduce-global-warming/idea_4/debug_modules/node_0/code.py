import os
import torch
import numpy as np
import pandas as pd
from library.config import get_config, Config
from library.utils import set_seed, rle_encode, dice_coefficient
from library.dataset import get_dataloader, ContrailDataset
from library.model import HRNetSegmenter
from library.loss import ContrailLoss
from library.train import train_model


def demonstrate_utils():
    print("\n=== 1. Demonstrating Utilities ===")

    # Test RLE Encoding
    # Create a 4x4 mask:
    # 0 1 0 0
    # 0 1 0 0
    # 0 0 0 0
    # 0 0 0 0
    # Flattened (column-major): 0,0,0,0, 1,1,0,0, 0,0,0,0, 0,0,0,0
    # Indices (1-based): 5, 6 are 1s.
    # Run: start at 5, length 2.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 1] = 1
    mask[1, 1] = 1

    encoded = rle_encode(mask)
    print(f"RLE Encode Input (4x4, ones at (0,1) and (1,1)): {encoded}")
    expected_rle = "5 2"
    assert (
        encoded == expected_rle
    ), f"RLE mismatch. Expected '{expected_rle}', got '{encoded}'"
    print("RLE Encoding verified.")

    # Test Dice Coefficient
    # Perfect match
    y_true = torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.float32)
    y_pred = torch.tensor(
        [[[[0.1, 0.9], [0.2, 0.8]]]], dtype=torch.float32
    )  # Logits/Probs simulation
    # Threshold 0.5 -> pred becomes [[0, 1], [0, 1]]

    dice = dice_coefficient(y_pred, y_true, threshold=0.5)
    print(f"Dice Coefficient (Perfect Match): {dice:.4f}")
    assert np.isclose(dice, 1.0), f"Dice should be 1.0, got {dice}"

    # Mismatch
    y_pred_bad = torch.zeros_like(y_true)
    dice_bad = dice_coefficient(y_pred_bad, y_true, threshold=0.5)
    print(f"Dice Coefficient (No Overlap): {dice_bad:.4f}")
    assert np.isclose(dice_bad, 0.0, atol=1e-5), f"Dice should be ~0.0, got {dice_bad}"
    print("Dice Coefficient verified.")


def demonstrate_data_loading(device):
    print("\n=== 2. Demonstrating Data Loading ===")

    # Initialize Config in Debug mode
    conf = get_config(debug=True)

    # Create DataLoader
    # We use a small batch size for demonstration
    dataloader = get_dataloader(mode="train", batch_size=4, num_workers=0, debug=True)

    print(f"DataLoader initialized. Number of batches: {len(dataloader)}")

    # Fetch one batch
    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]
    record_ids = batch["record_id"]

    print(f"Batch Shapes -> Images: {images.shape}, Masks: {masks.shape}")
    print(f"Sample Record ID: {record_ids[0]}")

    # Verify Shapes
    # Expected: (B, 3, 256, 256) and (B, 1, 256, 256)
    assert images.dim() == 4 and images.shape[1] == 3, "Image tensor shape incorrect"
    assert masks.dim() == 4 and masks.shape[1] == 1, "Mask tensor shape incorrect"

    # Verify Normalization (Ash composite should be roughly 0-1)
    # It might slightly exceed if min/max bounds in config don't cover outliers,
    # but `normalize_range` clips data, so it must be exactly [0, 1].
    min_val, max_val = images.min().item(), images.max().item()
    print(f"Image Value Range: [{min_val:.4f}, {max_val:.4f}]")
    assert (
        min_val >= 0.0 and max_val <= 1.0
    ), "Image data not properly normalized to [0, 1]"

    return images.to(device), masks.to(device)


def demonstrate_model_and_loss(images, masks, device):
    print("\n=== 3. Demonstrating Model & Loss ===")

    # Initialize Model
    model = HRNetSegmenter(pretrained=False)  # False for speed
    model.to(device)
    model.eval()  # Eval mode for deterministic check

    # Forward Pass
    print("Running forward pass...")
    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")
    assert (
        logits.shape == masks.shape
    ), f"Output shape mismatch. Expected {masks.shape}, got {logits.shape}"

    # Loss Calculation
    # We need gradients for loss verification usually, but here just checking computation
    loss_fn = ContrailLoss()

    # Switch to train mode to ensure no issues with grad (though we won't step)
    model.train()
    logits_grad = model(images)
    loss = loss_fn(logits_grad, masks)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Verify backward pass capability
    loss.backward()
    print("Backward pass successful.")


def demonstrate_full_training_loop():
    print("\n=== 4. Demonstrating Full Training Loop (Simulation) ===")

    # Override Config for a super-fast run
    # The train_model function re-loads Config, so we modify the class attributes directly.
    Config.EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 10  # Very small dataset
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print("Starting training simulation (1 Epoch, tiny dataset)...")
    try:
        train_model(debug=True)
        print("Training simulation completed successfully.")
    except Exception as e:
        print(f"Training simulation failed: {e}")
        raise e

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at: {checkpoint_path}")
    else:
        # It's possible no improvement happened in 1 epoch if init was lucky,
        # but usually valid dice starts at 0 so any result improves it.
        print("No checkpoint found (validation might not have improved).")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Utils
    demonstrate_utils()

    # 2. Data Loading
    # We keep the batch to pass to the model
    images, masks = demonstrate_data_loading(device)

    # 3. Model & Loss
    demonstrate_model_and_loss(images, masks, device)

    # 4. Full Pipeline
    demonstrate_full_training_loop()

    print("\nAll demonstrations passed.")
