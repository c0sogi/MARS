import os
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, create_submission
from library.dataset import get_data_arrays, SaltDataset, get_transforms
from library.model import ResNet34WideLinkNet
from library.losses import CombinedLoss
from library.engine import train_one_epoch, train_student_epoch, validate, predict


def center_crop(images, orig_h=101, orig_w=101):
    """
    Crops the center of the images/masks to return to original dimensions.
    Assumes input shape (N, H, W) or (H, W).
    """
    if images.ndim == 3:
        h, w = images.shape[1], images.shape[2]
    elif images.ndim == 2:
        h, w = images.shape[0], images.shape[1]
    else:
        raise ValueError("Unsupported shape for cropping")

    start_h = (h - orig_h) // 2
    start_w = (w - orig_w) // 2

    if images.ndim == 3:
        return images[:, start_h : start_h + orig_h, start_w : start_w + orig_w]
    else:
        return images[start_h : start_h + orig_h, start_w : start_w + orig_w]


if __name__ == "__main__":
    # 1. Setup and Configuration
    print("--- Setting up environment ---")
    Config.setup()
    set_seed(Config.SEED)

    # Override Config for speed in this demo
    Config.BATCH_SIZE = 8
    Config.EPOCHS_STAGE1 = 1
    DEMO_SUBSET_SIZE = 32

    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading (Subset for Speed)
    print("\n--- Loading Data ---")
    # Load Train Data
    train_imgs, train_masks, train_depths, train_ids = get_data_arrays(
        Config.TRAIN_METADATA_PATH, prefix="train", load_cached_data=True
    )

    # Load Val Data
    val_imgs, val_masks, val_depths, val_ids = get_data_arrays(
        Config.VAL_METADATA_PATH, prefix="val", load_cached_data=True
    )

    # Slice to create a tiny subset
    print(f"Original Train size: {len(train_imgs)}")
    train_imgs = train_imgs[:DEMO_SUBSET_SIZE]
    train_masks = train_masks[:DEMO_SUBSET_SIZE]
    train_depths = train_depths[:DEMO_SUBSET_SIZE]
    train_ids = train_ids[:DEMO_SUBSET_SIZE]

    val_imgs = val_imgs[:DEMO_SUBSET_SIZE]
    val_masks = val_masks[:DEMO_SUBSET_SIZE]
    val_depths = val_depths[:DEMO_SUBSET_SIZE]
    val_ids = val_ids[:DEMO_SUBSET_SIZE]
    print(f"Subset Train size: {len(train_imgs)}")

    # Calculate depth stats for normalization
    depth_mean = np.mean(train_depths)
    depth_std = np.std(train_depths)
    depth_stats = (depth_mean, depth_std)

    # 3. Dataset and DataLoader
    print("\n--- Creating Datasets ---")
    # Training Dataset
    train_dataset = SaltDataset(
        images=train_imgs,
        masks=train_masks,
        depths=train_depths,
        ids=train_ids,
        transforms=get_transforms("train"),
        depth_stats=depth_stats,
        depth_dropout_prob=Config.BERNOULLI_DEPTH_PROB,
    )

    # Validation Dataset
    val_dataset = SaltDataset(
        images=val_imgs,
        masks=val_masks,
        depths=val_depths,
        ids=val_ids,
        transforms=get_transforms("valid"),
        depth_stats=depth_stats,
        depth_dropout_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging/demo
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Data Shapes
    sample_img, sample_mask, sample_depth, _ = train_dataset[0]
    print(f"Sample Image Tensor Shape: {sample_img.shape}")  # Should be (1, 128, 128)
    print(f"Sample Mask Tensor Shape: {sample_mask.shape}")  # Should be (1, 128, 128)
    assert sample_img.shape == (
        1,
        128,
        128,
    ), "Image shape mismatch (expected 128x128 padded)"
    assert sample_mask.shape == (1, 128, 128), "Mask shape mismatch"

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = ResNet34WideLinkNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    loss_fn = CombinedLoss()

    # 5. Training Loop (Supervised)
    print("\n--- Starting Training (Stage 1) ---")
    epoch_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
    print(f"Epoch 1 Loss: {epoch_loss:.4f}")
    assert not np.isnan(epoch_loss), "Training loss is NaN"

    # 6. Validation
    print("\n--- Validating ---")
    val_loss, val_map = validate(model, val_loader, device, loss_fn)
    print(f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}")

    # 7. Student Training (Pseudo-labeling Demo)
    print("\n--- Starting Student Training Demo (Stage 3) ---")
    # For demo, we use the validation set as 'unlabeled' data
    student_dataset = SaltDataset(
        images=val_imgs,
        masks=None,  # Unlabeled
        depths=val_depths,
        ids=val_ids,
        transforms=get_transforms("student"),
        depth_stats=depth_stats,
        depth_dropout_prob=1.0,  # Drop depth for robustness
    )
    student_loader = DataLoader(
        student_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # We need a teacher model to generate pseudo-labels, but here we train the model
    # against itself (or a dummy target) just to demonstrate the loop function.
    # In a real scenario, 'student_loader' would yield soft masks from a teacher.
    # The provided `train_student_epoch` expects the loader to yield masks.
    # Since our `student_dataset` has masks=None, it yields zero masks.
    # To properly demonstrate `train_student_epoch`, we need a loader that yields *some* masks.
    # Let's reuse train_dataset as "unlabeled" but with student transforms for the demo.

    pseudo_dataset = SaltDataset(
        images=train_imgs,
        masks=train_masks,  # Using GT as "pseudo-labels" for mechanical demo
        depths=train_depths,
        ids=train_ids,
        transforms=get_transforms("student"),
        depth_stats=depth_stats,
    )
    pseudo_loader = DataLoader(
        pseudo_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    student_loss = train_student_epoch(
        model,
        labeled_loader=train_loader,
        unlabeled_loader=pseudo_loader,
        optimizer=optimizer,
        device=device,
        loss_fn_labeled=loss_fn,
    )
    print(f"Student Epoch Loss: {student_loss:.4f}")

    # 8. Inference and Submission
    print("\n--- Generating Submission ---")
    # We use validation set as test set for this demo
    test_ids, predictions = predict(model, val_loader, device)

    print(f"Raw Prediction Shape: {predictions.shape}")  # (N, 128, 128)

    # Post-processing: Crop back to 101x101
    # The model works on 128x128 (padded), but submission requires 101x101.
    # Albumentations PadIfNeeded defaults to center padding.
    predictions_cropped = center_crop(predictions, 101, 101)
    print(f"Cropped Prediction Shape: {predictions_cropped.shape}")

    assert predictions_cropped.shape[1:] == (101, 101), "Cropping failed"

    # Generate CSV
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    create_submission(test_ids, predictions_cropped, output_path=submission_path)
    print(f"Submission saved to: {submission_path}")

    # 9. Verify RLE Utilities
    print("\n--- Verifying RLE Utilities ---")
    # Create a simple mask: 101x101 with a 2x2 square of 1s at (10,10)
    # Note: 1-based indexing for RLE, pixels numbered top-to-bottom, then left-to-right.
    # (10,10) is row 10, col 10 (0-indexed).
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:12, 10:12] = 1  # 2x2 square

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(dummy_mask, decoded), "RLE Encode -> Decode failed"
    print("RLE Encode/Decode verification passed.")

    # Verify IoU logic
    iou = library.utils.calculate_iou(dummy_mask, decoded)
    assert iou == 1.0, "IoU calculation failed for identical masks"

    print("\nAll demonstrations completed successfully.")
