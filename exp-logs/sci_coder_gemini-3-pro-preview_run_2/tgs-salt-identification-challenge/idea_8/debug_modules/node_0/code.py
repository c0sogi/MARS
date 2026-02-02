import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

# Import provided library components
from library.config import (
    SEED,
    DEVICE,
    WORKING_DIR,
    METADATA_DIR,
    SUBMISSION_PATH,
    IMG_SIZE,
)
from library.utils import set_seed, rle_encode, rle_decode, metric_map
from library.dataset import SaltDataset, get_depth_stats, get_transforms
from library.model import DepthRegularizedWideLinkNet
from library.losses import MixedLoss
from library.engine import train_one_epoch, validate, predict_test

# -----------------------------------------------------------------------------
# Helper Functions for Demo
# -----------------------------------------------------------------------------


def create_mini_metadata(n_samples=32):
    """
    Creates small subsets of the metadata files to speed up the demonstration.
    """
    print(f"Creating mini metadata files with {n_samples} samples each...")

    # Paths
    train_path = os.path.join(METADATA_DIR, "train.csv")
    val_path = os.path.join(METADATA_DIR, "val.csv")
    test_path = os.path.join(METADATA_DIR, "test.csv")

    # Read original metadata
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Sample
    df_train_mini = df_train.head(n_samples).copy()
    df_val_mini = df_val.head(n_samples).copy()
    df_test_mini = df_test.head(n_samples).copy()

    # Save to working directory
    mini_train_path = os.path.join(WORKING_DIR, "train_mini.csv")
    mini_val_path = os.path.join(WORKING_DIR, "val_mini.csv")
    mini_test_path = os.path.join(WORKING_DIR, "test_mini.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)
    df_test_mini.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Create Mini Datasets for Speed
    mini_train_path, mini_val_path, mini_test_path = create_mini_metadata(n_samples=32)

    # 3. Verify Utilities (RLE and Metric)
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding/Decoding
    # Create a simple 101x101 mask with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(dummy_mask, decoded), "RLE decode did not match original mask"
    print("RLE Encode/Decode logic verified.")

    # Test Metric Map
    # Perfect match should give score 1.0
    score_perfect = metric_map(dummy_mask[None, ...], dummy_mask[None, ...])
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match metric should be 1.0, got {score_perfect}"

    # No overlap should give score 0.0 (unless empty-empty case)
    empty_mask = np.zeros((101, 101), dtype=np.uint8)
    score_mismatch = metric_map(dummy_mask[None, ...], empty_mask[None, ...])
    assert score_mismatch == 0.0, f"Mismatch metric should be 0.0, got {score_mismatch}"
    print("Metric calculation verified.")

    # 4. Dataset and DataLoader
    print("\n--- Verifying Dataset & DataLoader ---")

    # Get depth stats from the mini train set
    depth_mean, depth_std = get_depth_stats(mini_train_path)
    print(f"Depth Stats -> Mean: {depth_mean:.2f}, Std: {depth_std:.2f}")

    # Instantiate Dataset
    train_dataset = SaltDataset(
        metadata_path=mini_train_path,
        mode="train",
        depth_stats=(depth_mean, depth_std),
        transform=get_transforms("train"),
    )

    # Instantiate DataLoader
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging
        drop_last=True,
    )

    # Fetch one batch
    images, masks, depths, ids = next(iter(train_loader))

    # Check Shapes
    # Image: (B, 3, 128, 128) - RGB channels, padded size
    assert images.shape == (
        batch_size,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    # Mask: (B, 1, 128, 128)
    assert masks.shape == (
        batch_size,
        1,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Incorrect mask shape: {masks.shape}"
    # Depth: (B, 1)
    assert depths.shape == (batch_size, 1), f"Incorrect depth shape: {depths.shape}"

    print(
        f"Batch shapes verified: Images {images.shape}, Masks {masks.shape}, Depths {depths.shape}"
    )

    # 5. Model Initialization
    print("\n--- Verifying Model ---")

    model = DepthRegularizedWideLinkNet(n_classes=1)
    model.to(DEVICE)

    # Forward pass with the batch fetched earlier
    images = images.to(DEVICE)
    depths = depths.to(DEVICE)

    logits = model(images, depths)

    # Output should be (B, 1, 128, 128)
    assert logits.shape == (
        batch_size,
        1,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Model output shape mismatch: {logits.shape}"
    print("Model forward pass successful.")

    # 6. Loss Function
    print("\n--- Verifying Loss Function ---")

    loss_fn = MixedLoss(bce_weight=0.5, lovasz_weight=0.5)
    masks = masks.to(DEVICE)

    loss = loss_fn(logits, masks)

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f"Calculated Loss: {loss.item():.4f}")

    # 7. Training and Validation Loop
    print("\n--- Verifying Training & Validation Loop ---")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch
    print("Training for 1 epoch...")
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device=DEVICE)
    print(f"Epoch Train Loss: {train_loss:.4f}")
    assert train_loss > 0, "Train loss should be positive"

    # Validate
    # Setup Val Loader
    val_dataset = SaltDataset(
        metadata_path=mini_val_path,
        mode="val",
        depth_stats=(depth_mean, depth_std),
        transform=get_transforms("val"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print("Running validation...")
    val_loss, val_score, best_threshold = validate(
        model, val_loader, loss_fn, device=DEVICE
    )

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val mAP Score: {val_score:.4f}")
    print(f"Best Threshold: {best_threshold:.2f}")

    assert 0 <= val_score <= 1.0, "Validation score out of range [0, 1]"
    assert 0 < best_threshold < 1.0, "Threshold out of range (0, 1)"

    # 8. Inference / Prediction
    print("\n--- Verifying Inference (Test Prediction) ---")

    # Setup Test Loader
    test_dataset = SaltDataset(
        metadata_path=mini_test_path,
        mode="test",
        depth_stats=(
            depth_mean,
            depth_std,
        ),  # Passed but effectively ignored for z_norm=0
        transform=get_transforms("test"),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Run Prediction
    predict_test(model, test_loader, best_threshold, device=DEVICE)

    # Verify Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert (
        "id" in df_sub.columns and "rle_mask" in df_sub.columns
    ), "Submission columns missing"
    # Check row count matches mini test set
    assert (
        len(df_sub) == 32
    ), f"Submission row count mismatch. Expected 32, got {len(df_sub)}"

    print("Inference and submission generation successful.")
    print("\nAll verification steps passed successfully.")
