import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.utils import set_seed, rle_encode
from library.dataset import load_and_cache_data, SaltDataset, get_transforms
from library.model import ResFiLM_LinkNet34
from library.losses import CombinedLoss
from library.trainer import run_training, generate_submission

# Constants for the demonstration
DEMO_EPOCHS = 1
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def verify_rle_encoding():
    """Verifies the RLE encoding utility."""
    print("Verifying RLE encoding...")
    # Create a simple 3x3 mask
    # 0 1 0
    # 1 1 1
    # 0 0 0
    # Flattened (column-major): 0,1,0, 1,1,0, 0,1,0
    mask = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    # Expected RLE:
    # Column 1: 0, 1, 0 -> Start index 2, length 1
    # Column 2: 1, 1, 0 -> Start index 4, length 2
    # Column 3: 0, 1, 0 -> Start index 8, length 1
    # Note: 1-based indexing.
    # Col 1 indices: 1, 2, 3. Value at 2 is 1. -> '2 1'
    # Col 2 indices: 4, 5, 6. Values at 4, 5 are 1. -> '4 2'
    # Col 3 indices: 7, 8, 9. Value at 8 is 1. -> '8 1'
    # Result: '2 1 4 2 8 1'

    encoded = rle_encode(mask)
    expected = "2 1 4 2 8 1"
    assert (
        encoded == expected
    ), f"RLE encoding failed. Expected '{expected}', got '{encoded}'"
    print("RLE encoding verified.")


def verify_dataset_and_loading():
    """Verifies data loading and dataset construction."""
    print("\nVerifying Dataset and Data Loading...")

    # 1. Load and Cache Data
    data = load_and_cache_data(load_cached=True)

    required_keys = [
        "train_images",
        "train_masks",
        "train_depths",
        "val_images",
        "val_masks",
        "val_depths",
        "test_images",
        "test_depths",
        "test_ids",
    ]
    for k in required_keys:
        assert k in data, f"Missing key {k} in loaded data."
        assert len(data[k]) > 0, f"Data for {k} is empty."

    print(f"Data loaded successfully. Train size: {len(data['train_images'])}")

    # 2. Instantiate Dataset
    # Calculate depth stats
    all_depths = np.concatenate([data["train_depths"], data["val_depths"]])
    d_mean = all_depths.mean()
    d_std = all_depths.std()

    ds = SaltDataset(
        data["train_images"],
        data["train_masks"],
        data["train_depths"],
        transform=get_transforms("train"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=True,
    )

    # 3. Check Item
    img, mask, depth = ds[0]

    # Check shapes
    # Image should be (1, 128, 128) after transform (ToTensorV2 adds channel dim if grayscale? No, usually (C, H, W))
    # Albumentations ToTensorV2 converts HWC to CHW.
    # If input is (101, 101), PadIfNeeded makes it (128, 128).
    # If read as grayscale (H, W), ToTensorV2 might add dim if configured, or just return (H, W).
    # Let's check the actual output.

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")
    print(f"Sample Depth Shape: {depth.shape}")

    # Assertions
    # Expecting (1, 128, 128) or (128, 128) depending on transform details,
    # but model expects (B, 1, H, W).
    if img.ndim == 2:
        img = img.unsqueeze(0)

    assert img.shape[-2:] == (128, 128), f"Image height/width mismatch. Got {img.shape}"
    assert mask.shape[-2:] == (
        128,
        128,
    ), f"Mask height/width mismatch. Got {mask.shape}"
    assert depth.numel() == 1, "Depth should be a single scalar."

    print("Dataset verification passed.")
    return (
        img.unsqueeze(0),
        mask.unsqueeze(0),
        depth.unsqueeze(0),
    )  # Return batch for model test


def verify_model_and_loss(sample_batch):
    """Verifies model architecture and loss calculation."""
    print("\nVerifying Model and Loss...")

    img_batch, mask_batch, depth_batch = sample_batch
    img_batch = img_batch.to(DEVICE)
    mask_batch = mask_batch.to(DEVICE)
    depth_batch = depth_batch.to(DEVICE)

    # 1. Instantiate Model
    model = ResFiLM_LinkNet34().to(DEVICE)

    # 2. Forward Pass
    logits = model(img_batch, depth_batch)

    # Check output shape: (B, 1, 128, 128)
    assert logits.shape == (
        1,
        1,
        128,
        128,
    ), f"Model output shape mismatch. Expected (1, 1, 128, 128), got {logits.shape}"

    # 3. Loss Calculation
    criterion = CombinedLoss()
    loss = criterion(logits, mask_batch)

    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss should be non-negative."

    # 4. Backward Check
    loss.backward()
    # Check if gradients exist
    assert model.final_conv.weight.grad is not None, "Gradients not computed."

    print(f"Model and Loss verification passed. Initial Loss: {loss.item():.4f}")


def run_pipeline():
    """Runs the training and submission pipeline."""
    print("\nRunning Training Pipeline...")

    # Run training for 1 epoch to demonstrate functionality
    # The trainer module handles data loading internally, but we've verified it works.
    best_thresh, d_mean, d_std = run_training(
        epochs=DEMO_EPOCHS, batch_size=BATCH_SIZE, lr=1e-3
    )

    print(f"Training complete. Best Threshold: {best_thresh:.2f}")

    # Check if best model was saved
    model_path = "./working/idea_12/best_model.pth"
    assert os.path.exists(model_path), "best_model.pth was not saved."

    print("\nGenerating Submission...")
    generate_submission(best_thresh, d_mean, d_std)

    # Check submission file
    sub_path = "submission/submission.csv"
    assert os.path.exists(sub_path), "submission.csv was not generated."

    df = pd.read_csv(sub_path)
    assert len(df) == 1000, f"Submission should have 1000 rows, got {len(df)}"
    print("Pipeline execution successful.")


if __name__ == "__main__":
    # Ensure clean state
    set_seed(42)

    # 1. Verify Utils
    verify_rle_encoding()

    # 2. Verify Data Loading & Dataset
    sample_batch = verify_dataset_and_loading()

    # 3. Verify Model & Loss
    verify_model_and_loss(sample_batch)

    # 4. Run Full Pipeline (Train + Inference)
    run_pipeline()

    print("\nAll tasks completed successfully.")
