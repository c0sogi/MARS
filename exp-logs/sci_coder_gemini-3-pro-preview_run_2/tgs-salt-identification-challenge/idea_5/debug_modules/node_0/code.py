import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, calc_iou_metric
from library.dataset import load_data, SaltDataset, get_transforms, get_depth_stats
from library.model import DepthAwareLinkNet
from library.losses import BCEDiceLoss
from library.train_eval import train_one_epoch, validate, predict_with_tta


def monkey_patch_config():
    """
    Override Config values to ensure the demo runs quickly.
    """
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.DEBUG = True
    # We will use a very small subset, so patience doesn't matter much,
    # but setting it ensures no long waits if we were looping.
    Config.EARLY_STOPPING_PATIENCE = 1


def verify_utils():
    """
    Verifies RLE encoding and decoding logic.
    """
    print("Verifying Utils (RLE Encode/Decode)...")
    # Create a synthetic 101x101 mask with a known square
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    # Encode
    rle = rle_encode(mask)

    # Decode
    decoded_mask = rle_decode(rle, shape=(101, 101))

    # Verify
    assert np.array_equal(mask, decoded_mask), "RLE Decode did not match original mask"
    print("Utils verification passed.")


def main():
    # 1. Setup
    monkey_patch_config()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Verify Utils
    verify_utils()

    # 3. Data Loading
    print("\nLoading Data...")
    # load_data uses Config.TRAIN_CSV internally
    # We load the training data
    data_dict = load_data(Config.TRAIN_CSV, "train", load_cached_data=True)

    # Create a tiny subset for speed (e.g., 16 samples)
    subset_size = 16
    print(f"Subsetting data to {subset_size} samples for demonstration.")

    mini_data = {
        "images": data_dict["images"][:subset_size],
        "masks": data_dict["masks"][:subset_size],
        "depths": data_dict["depths"][:subset_size],
        "ids": data_dict["ids"][:subset_size],
    }

    # Calculate depth stats from the subset (or use global if available, here using subset for self-containment)
    depth_mean = mini_data["depths"].mean()
    depth_std = mini_data["depths"].std()
    depth_stats = (depth_mean, depth_std)

    # 4. Dataset & DataLoader
    print("Creating Dataset and DataLoader...")
    train_transform = get_transforms(phase="train")
    val_transform = get_transforms(
        phase="val"
    )  # No augmentation, just padding/tensor conversion

    train_dataset = SaltDataset(
        mini_data, transform=train_transform, depth_stats=depth_stats
    )
    # We'll use the same subset for validation demonstration
    val_dataset = SaltDataset(
        mini_data, transform=val_transform, depth_stats=depth_stats
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for this small demo
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify a batch
    images, masks, depths, ids = next(iter(train_loader))
    print(
        f"Batch shapes - Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Expected shape is (B, 1, 128, 128) because Config.IMG_SIZE = 128
    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Unexpected image shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Unexpected mask shape"

    # 5. Model Initialization
    print("\nInitializing Model...")
    model = DepthAwareLinkNet()
    model.to(device)

    # Dummy forward pass check
    with torch.no_grad():
        dummy_out = model(images.to(device), depths.to(device))

    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)}, got {dummy_out.shape}"
    print("Model initialized and forward pass successful.")

    # 6. Loss Function
    print("Initializing Loss...")
    criterion = BCEDiceLoss()
    # Check loss calculation
    loss_val = criterion(dummy_out, masks.to(device))
    assert (
        torch.is_tensor(loss_val) and loss_val.ndim == 0
    ), "Loss should be a scalar tensor"
    print(f"Initial Loss check: {loss_val.item():.4f}")

    # 7. Training Loop Demonstration
    print("\nStarting Training Loop (1 Epoch)...")
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch 1 Training Loss: {avg_train_loss:.4f}")

    # 8. Validation Demonstration
    print("\nStarting Validation...")
    val_loss, best_map, best_thresh = validate(model, val_loader, criterion, device)
    print(
        f"Validation Stats - Loss: {val_loss:.4f}, mAP: {best_map:.4f}, Best Threshold: {best_thresh}"
    )

    # 9. Inference & Post-Processing Demonstration
    print("\nDemonstrating Inference and Post-Processing...")
    # We use the val_loader for inference demo (simulating test set)
    # predict_with_tta returns numpy arrays
    preds, pred_ids = predict_with_tta(model, val_loader, device)

    print(f"Prediction shape (raw): {preds.shape}")  # Should be (N, 1, 128, 128)

    # Post-processing: Crop back to 101x101
    # The padding strategy in dataset.py is A.PadIfNeeded with min_height=128, min_width=128.
    # By default, Albumentations centers the image.
    # 128 - 101 = 27. 27 // 2 = 13.
    # So we crop [13 : 13+101, 13 : 13+101]

    pad_total = Config.IMG_SIZE - Config.ORIG_IMG_SIZE
    pad_start = pad_total // 2
    pad_end = pad_start + Config.ORIG_IMG_SIZE

    print(
        f"Cropping predictions from {Config.IMG_SIZE}x{Config.IMG_SIZE} to {Config.ORIG_IMG_SIZE}x{Config.ORIG_IMG_SIZE}..."
    )
    print(f"Crop range: {pad_start}:{pad_end}")

    # Take the first prediction as an example
    pred_raw = preds[0, 0, :, :]  # (128, 128)
    pred_cropped = pred_raw[pad_start:pad_end, pad_start:pad_end]

    assert pred_cropped.shape == (
        101,
        101,
    ), f"Cropped shape mismatch: {pred_cropped.shape}"

    # Binarize using the best threshold found during validation
    pred_binary = (pred_cropped > best_thresh).astype(np.uint8)

    # Encode
    rle_str = rle_encode(pred_binary)
    print(f"Sample RLE for ID {pred_ids[0]}: '{rle_str[:20]}...'")

    print("\nDemonstration Complete.")


if __name__ == "__main__":
    main()
