import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_encode, dice_score_batch
from library.dataset import ContrailDataset, get_transforms
from library.model import ProgressiveConvNeXtUNet
from library.loss import HybridLoss
from library.train import train_one_epoch, validate


def run_demo():
    print("--- Starting Contrail Identification Demo ---")

    # 1. Setup and Configuration
    # We override some Config values for the purpose of this quick demo
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)

    # 2. Data Preparation
    # Load metadata
    print("\n[1/6] Loading Metadata and Creating Subset...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
        )

    full_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Take a tiny subset for speed (e.g., 8 samples)
    subset_df = full_df.head(8).copy()
    print(f"Subset size: {len(subset_df)}")

    # Instantiate Dataset
    # We use 'train' split which includes augmentations
    dataset = ContrailDataset(
        subset_df, split="train", transform=get_transforms("train")
    )

    # Verify Dataset Item
    sample = dataset[0]
    img = sample["image"]
    mask = sample["mask"]
    record_id = sample["record_id"]

    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Assertions for Data
    # Image: (C=6, H=256, W=256)
    assert img.shape == (
        6,
        256,
        256,
    ), f"Expected image shape (6, 256, 256), got {img.shape}"
    # Mask: (C=1, H=256, W=256)
    assert mask.shape == (
        1,
        256,
        256,
    ), f"Expected mask shape (1, 256, 256), got {mask.shape}"
    # Check normalization (approximate range, mostly 0-1 but standardized inputs might vary slightly depending on normalization logic)
    # The dataset class normalizes to [0, 1] using min/max.
    assert (
        img.min() >= 0.0 and img.max() <= 1.0 + 1e-5
    ), "Image data should be normalized to [0, 1]"

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("Dataset and DataLoader verified.")

    # 3. Model Initialization
    print("\n[2/6] Initializing Model...")
    model = ProgressiveConvNeXtUNet()
    model.to(device)

    # Count parameters (sanity check)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {params:,}")
    print("Model initialized.")

    # 4. Forward Pass & Loss Calculation
    print("\n[3/6] Testing Forward Pass and Loss...")
    loss_fn = HybridLoss()

    # Get a batch
    batch = next(iter(dataloader))
    b_images = batch["image"].to(device)
    b_masks = batch["mask"].to(device)

    # Forward
    logits = model(b_images)

    # Assertions for Model Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
        256,
        256,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1, 256, 256), got {logits.shape}"

    # Calculate Loss
    loss = loss_fn(logits, b_masks)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("Forward pass and loss calculation verified.")

    # 5. Training Loop Simulation
    print("\n[4/6] Simulating Training Step...")
    optimizer = AdamW(model.parameters(), lr=1e-4)

    model.train()
    optimizer.zero_grad()

    # Re-compute logits/loss to ensure graph is connected for backward
    logits = model(b_images)
    loss = loss_fn(logits, b_masks)
    loss.backward()
    optimizer.step()

    print("Optimizer step completed successfully.")

    # 6. Metric Verification (Dice)
    print("\n[5/6] Verifying Dice Metric...")
    # Create synthetic predictions and targets
    # Case 1: Perfect match
    pred_perfect = torch.ones((1, 1, 10, 10))
    target_perfect = torch.ones((1, 1, 10, 10))
    score_perfect = dice_score_batch(pred_perfect, target_perfect)
    print(f"Dice Score (Perfect Match): {score_perfect:.4f}")
    assert abs(score_perfect - 1.0) < 1e-5, "Dice score for perfect match should be 1.0"

    # Case 2: No overlap
    pred_none = torch.zeros((1, 1, 10, 10))
    target_full = torch.ones((1, 1, 10, 10))
    score_none = dice_score_batch(pred_none, target_full)
    print(f"Dice Score (No Overlap): {score_none:.4f}")
    assert abs(score_none - 0.0) < 1e-5, "Dice score for no overlap should be 0.0"

    # Real batch evaluation
    with torch.no_grad():
        model.eval()
        logits = model(b_images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        batch_dice = dice_score_batch(preds, b_masks)
        print(f"Batch Dice Score (Random Init Model): {batch_dice:.4f}")

    # 7. RLE Encoding Verification
    print("\n[6/6] Verifying RLE Encoding...")
    # Create a simple 4x4 mask
    # 0 0 0 0
    # 1 1 0 0
    # 0 0 1 0
    # 0 0 0 0
    # Flattened (Column-major order for RLE in this comp):
    # Col 0: 0, 1, 0, 0
    # Col 1: 0, 1, 0, 0
    # Col 2: 0, 0, 1, 0
    # Col 3: 0, 0, 0, 0
    # Sequence: 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
    # Indices (1-based):
    # 1: 0
    # 2: 1 (Start run 1)
    # 3: 0 (End run 1, len 1) -> "2 1"
    # ...
    # 6: 1 (Start run 2)
    # 7: 0 (End run 2, len 1) -> "6 1"
    # ...
    # 11: 1 (Start run 3)
    # 12: 0 (End run 3, len 1) -> "11 1"

    # Let's construct this mask
    dummy_mask = np.zeros((4, 4), dtype=int)
    dummy_mask[1, 0] = 1  # Pixel (2,1) -> Index 2 in col-major
    dummy_mask[1, 1] = 1  # Pixel (2,2) -> Index 6 in col-major (4 + 2)
    dummy_mask[2, 2] = 1  # Pixel (3,3) -> Index 11 in col-major (8 + 3)

    encoded = rle_encode(dummy_mask)
    print(f"RLE Output: {encoded}")

    # Expected: '2 1 6 1 11 1'
    expected_rle = "2 1 6 1 11 1"
    assert (
        encoded == expected_rle
    ), f"RLE failed. Expected '{expected_rle}', got '{encoded}'"

    # Test Empty
    empty_mask = np.zeros((10, 10), dtype=int)
    encoded_empty = rle_encode(empty_mask)
    assert (
        encoded_empty == "-"
    ), f"Empty RLE failed. Expected '-', got '{encoded_empty}'"
    print("RLE Encoding verified.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
