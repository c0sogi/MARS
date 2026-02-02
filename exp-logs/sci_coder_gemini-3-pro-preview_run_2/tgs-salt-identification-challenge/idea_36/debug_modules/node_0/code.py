import os
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import library modules
from library.utils import set_seed
from library.dataset import (
    preload_data,
    get_depth_stats,
    get_transforms,
    SaltDataset,
    IMG_SIZE_TARGET,
)
from library.model import SaltNet
from library.losses import MixedLoss
from library.engine import (
    train_one_epoch,
    validate,
    optimize_threshold,
    generate_submission,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Constants
METADATA_TRAIN = "./metadata/train.csv"
METADATA_TEST = "./metadata/test.csv"
WORKING_DIR = "./working"
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_WORKERS = 0  # Set to 0 for stability in simple scripts
SEED = 42


def main():
    print(f"Starting demonstration on device: {DEVICE}")
    set_seed(SEED)

    # ==========================================
    # 1. Data Loading & Dataset Verification
    # ==========================================
    print("\n--- 1. Data Loading & Dataset Verification ---")

    # Load metadata
    if not os.path.exists(METADATA_TRAIN):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_TRAIN}")

    full_train_df = pd.read_csv(METADATA_TRAIN)

    # Use a small subset for speed
    subset_size = 16
    train_subset_df = full_train_df.head(subset_size).copy()

    # Save subset metadata to working dir for preload_data to consume
    subset_meta_path = os.path.join(WORKING_DIR, "train_subset.csv")
    train_subset_df.to_csv(subset_meta_path, index=False)

    print(f"Created subset metadata with {len(train_subset_df)} samples.")

    # Preload data (caches to disk)
    # Note: preload_data handles caching logic.
    df_train, data_dict_train = preload_data(subset_meta_path, phase="train")

    # Calculate depth stats
    depth_stats = get_depth_stats(df_train)
    print(f"Depth Stats: Mean={depth_stats['mean']:.4f}, Std={depth_stats['std']:.4f}")

    # Create Dataset and DataLoader
    train_transform = get_transforms(phase="train")
    train_dataset = SaltDataset(
        data_dict_train, transform=train_transform, depth_stats=depth_stats
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    # Verify Batch
    images, masks, depths, ids = next(iter(train_loader))

    print(
        f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Assertions
    assert images.shape == (
        BATCH_SIZE,
        1,
        IMG_SIZE_TARGET,
        IMG_SIZE_TARGET,
    ), "Incorrect image shape"
    assert masks.shape == (
        BATCH_SIZE,
        1,
        IMG_SIZE_TARGET,
        IMG_SIZE_TARGET,
    ), "Incorrect mask shape"
    assert depths.shape == (BATCH_SIZE, 1), "Incorrect depth shape"
    assert len(ids) == BATCH_SIZE, "Incorrect IDs length"
    print("Dataset verification passed.")

    # ==========================================
    # 2. Model Initialization & Verification
    # ==========================================
    print("\n--- 2. Model Initialization & Verification ---")

    model = SaltNet()
    model.to(DEVICE)

    # Dummy forward pass
    images = images.to(DEVICE)
    depths = depths.to(DEVICE)

    with torch.no_grad():
        logits = model(images, depths)

    print(f"Output logits shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        BATCH_SIZE,
        1,
        IMG_SIZE_TARGET,
        IMG_SIZE_TARGET,
    ), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("Model verification passed.")

    # ==========================================
    # 3. Training Loop Demonstration
    # ==========================================
    print("\n--- 3. Training Loop Demonstration ---")

    criterion = MixedLoss(alpha=1.0, beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train for 1 epoch
    loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Training Epoch Loss: {loss:.4f}")

    assert np.isfinite(loss), "Training loss is not finite"
    print("Training loop verification passed.")

    # ==========================================
    # 4. Validation & Metrics Demonstration
    # ==========================================
    print("\n--- 4. Validation & Metrics Demonstration ---")

    # Use the same subset as validation for demonstration purposes
    val_transform = get_transforms(phase="val")
    val_dataset = SaltDataset(
        data_dict_train, transform=val_transform, depth_stats=depth_stats
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    val_loss, val_map = validate(model, val_loader, criterion, DEVICE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")

    assert np.isfinite(val_loss), "Validation loss is not finite"
    assert 0.0 <= val_map <= 1.0, "mAP score out of range [0, 1]"
    print("Validation verification passed.")

    # ==========================================
    # 5. Inference & Submission Demonstration
    # ==========================================
    print("\n--- 5. Inference & Submission Demonstration ---")

    # Optimize Threshold
    best_thresh = optimize_threshold(model, val_loader, DEVICE)
    print(f"Optimal Threshold: {best_thresh}")

    # Load Test Data Subset
    full_test_df = pd.read_csv(METADATA_TEST)
    test_subset_df = full_test_df.head(10).copy()
    test_subset_path = os.path.join(WORKING_DIR, "test_subset.csv")
    test_subset_df.to_csv(test_subset_path, index=False)

    _, data_dict_test = preload_data(test_subset_path, phase="test")

    test_dataset = SaltDataset(
        data_dict_test,
        transform=val_transform,  # Use val transform (no augs) for test
        depth_stats=depth_stats,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Generate Submission
    generate_submission(
        model, test_loader, DEVICE, output_path=SUBMISSION_PATH, threshold=best_thresh
    )

    # Verify Submission File
    assert os.path.exists(SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {sub_df.shape}")
    print(f"Columns: {list(sub_df.columns)}")

    assert (
        "id" in sub_df.columns and "rle_mask" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) == 10, "Submission row count mismatch"

    # Check RLE format (basic check)
    # If mask is empty, RLE is empty string or NaN. If not empty, it's string of numbers.
    # We just check no errors occurred during generation.
    print("Submission verification passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
