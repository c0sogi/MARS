import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import cv2

# Import from provided library files
from library.utils import rle_encode, rle_decode, calculate_iou_batch, calculate_map
from library.dataset import SaltDataset, get_depth_stats
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import (
    set_seed,
    train_model,
    predict_with_tta,
    generate_submission,
    center_crop,
)

# Configuration
WORKING_DIR = "./working/demo_execution"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CKPT_DIR = os.path.join(WORKING_DIR, "checkpoints")
INPUT_ROOT = "./input"
METADATA_TRAIN = "./metadata/train.csv"
METADATA_VAL = "./metadata/val.csv"


def setup_directories():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)
    os.makedirs(CACHE_DIR)
    os.makedirs(CKPT_DIR)


def create_subset_metadata(source_path, dest_path, n=16):
    """Creates a small subset of metadata for rapid demonstration."""
    df = pd.read_csv(source_path)
    subset = df.head(n).copy()
    subset.to_csv(dest_path, index=False)
    return dest_path


def verify_utils():
    print("Verifying Utils...")
    # 1. Test RLE Encoding/Decoding
    # Create a 101x101 mask with a 10x10 square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    if not np.array_equal(mask, decoded):
        raise AssertionError(
            "RLE Encode -> Decode failed to reconstruct original mask."
        )

    # 2. Test IoU Calculation
    # Perfect match
    iou_perfect = calculate_iou_batch(mask, mask)
    if not np.isclose(iou_perfect, 1.0):
        raise AssertionError(f"IoU for perfect match should be 1.0, got {iou_perfect}")

    # No overlap
    mask_inv = np.zeros((101, 101), dtype=np.uint8)
    mask_inv[30:40, 30:40] = 1
    iou_zero = calculate_iou_batch(mask, mask_inv)
    if not np.isclose(iou_zero, 0.0):
        raise AssertionError(f"IoU for no overlap should be 0.0, got {iou_zero}")

    print("Utils verification passed.")


def verify_dataset_and_loader(train_csv, val_csv):
    print("Verifying Dataset and DataLoaders...")

    # Calculate stats from the subset
    depth_mean, depth_std = get_depth_stats(train_csv)

    # Instantiate Datasets
    # Note: We use a unique cache dir to avoid conflicts with other runs
    train_ds = SaltDataset(
        mode="train",
        metadata_file=train_csv,
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir=INPUT_ROOT,
        cache_dir=CACHE_DIR,
        load_cached_data=False,  # Force reload for demo
    )

    val_ds = SaltDataset(
        mode="val",
        metadata_file=val_csv,
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir=INPUT_ROOT,
        cache_dir=CACHE_DIR,
        load_cached_data=False,
    )

    # Check item structure
    # Item: (image, mask, depth, id)
    img, mask, depth, img_id = train_ds[0]

    # Check Shapes (Albumentations pads to 128x128 in get_transforms)
    if img.shape != (1, 128, 128):
        raise AssertionError(f"Expected image shape (1, 128, 128), got {img.shape}")
    if mask.shape != (128, 128):
        raise AssertionError(f"Expected mask shape (128, 128), got {mask.shape}")
    if not isinstance(depth.item(), float):
        raise AssertionError("Depth should be a float tensor.")

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    print("Dataset verification passed.")
    return train_loader, val_loader


def verify_model_and_training(train_loader, val_loader):
    print("Verifying Model and Training Loop...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate Model
    model = ResNet34WideLinkNet().to(device)

    # Verify Forward Pass with dummy data
    dummy_img = torch.randn(2, 1, 128, 128).to(device)
    dummy_depth = torch.randn(2, 1).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_depth)

    # Model output should match input spatial dimensions (128x128)
    if output.shape != (2, 1, 128, 128):
        raise AssertionError(
            f"Expected model output (2, 1, 128, 128), got {output.shape}"
        )

    # Setup Training Components
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Run Training for 1 Epoch
    # This tests engine.train_one_epoch, engine.evaluate, and engine.train_model
    best_thresh = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=1,
        patience=1,
        output_dir=CKPT_DIR,
    )

    if not os.path.exists(os.path.join(CKPT_DIR, "best_model.pth")):
        raise AssertionError("best_model.pth was not saved.")

    print(f"Training verification passed. Best threshold: {best_thresh}")
    return model, best_thresh


def verify_inference(model, val_loader, threshold):
    print("Verifying Inference and Submission...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predict using TTA
    predictions, ids = predict_with_tta(model, val_loader, device)

    # Check prediction shape (N, 101, 101) - predict_with_tta crops back to original size
    expected_n = len(val_loader.dataset)
    if predictions.shape != (expected_n, 101, 101):
        raise AssertionError(
            f"Expected predictions shape ({expected_n}, 101, 101), got {predictions.shape}"
        )

    # Generate Submission
    sub_path = os.path.join(WORKING_DIR, "submission_demo.csv")
    generate_submission(predictions, ids, threshold, sub_path)

    if not os.path.exists(sub_path):
        raise AssertionError("Submission file was not created.")

    # Verify submission content
    df = pd.read_csv(sub_path)
    if len(df) != expected_n:
        raise AssertionError(f"Submission has {len(df)} rows, expected {expected_n}")
    if list(df.columns) != ["id", "rle_mask"]:
        raise AssertionError("Submission columns incorrect.")

    print("Inference verification passed.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_directories()

    # 2. Verify Utils
    verify_utils()

    # 3. Prepare Subset Data (to ensure speed)
    train_subset_path = os.path.join(WORKING_DIR, "train_subset.csv")
    val_subset_path = os.path.join(WORKING_DIR, "val_subset.csv")

    create_subset_metadata(METADATA_TRAIN, train_subset_path, n=16)
    create_subset_metadata(METADATA_VAL, val_subset_path, n=8)

    # 4. Verify Dataset
    train_loader, val_loader = verify_dataset_and_loader(
        train_subset_path, val_subset_path
    )

    # 5. Verify Model & Training
    model, best_thresh = verify_model_and_training(train_loader, val_loader)

    # 6. Verify Inference
    verify_inference(model, val_loader, best_thresh)

    print("\nAll demonstrations completed successfully.")
