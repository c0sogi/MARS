import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, calculate_iou_batch
from library.dataset import SaltDataset, PseudoLabelDataset
from library.model import ResNet34WideLinkNetMTL
from library.losses import CombinedMTLLoss
from library.engine import train_teacher_epoch, train_student_epoch, validate, predict


def main():
    print(">>> Starting Salt Segmentation Demo Script")

    # 1. Setup Configuration and Reproducibility
    print("\n[1] Setting up Configuration...")
    # Enable debug mode to reduce epochs and batch sizes in Config defaults
    Config.setup(debug=True)

    # Further override for extreme speed in this demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE3 = 1

    # Set fixed seed
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # 2. Verify Utility Functions (RLE & IoU)
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    # Create a simple 10x10 mask with a 2x2 square of 1s at (1,1)
    # Note: RLE functions expect column-major order
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[1:3, 1:3] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(10, 10))

    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE Decode does not match original mask!"
    print("    RLE Encode/Decode check passed.")

    # Test IoU Calculation
    # Perfect match
    iou_perfect = calculate_iou_batch(
        dummy_mask[None, ...], dummy_mask[None, ...], threshold=0.5
    )
    assert np.isclose(iou_perfect, 1.0), f"Expected IoU 1.0, got {iou_perfect}"

    # No overlap
    dummy_mask_2 = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask_2[5:7, 5:7] = 1
    iou_zero = calculate_iou_batch(
        dummy_mask[None, ...], dummy_mask_2[None, ...], threshold=0.5
    )
    assert np.isclose(iou_zero, 0.0), f"Expected IoU 0.0, got {iou_zero}"
    print("    IoU Calculation check passed.")

    # 3. Data Loading & Verification
    print("\n[3] Loading Datasets...")

    # Initialize Datasets
    # We use the provided metadata files.
    train_ds = SaltDataset(mode="train")
    val_ds = SaltDataset(mode="val")

    print(f"    Full Train Size: {len(train_ds)}")
    print(f"    Full Val Size: {len(val_ds)}")

    # Create Subsets for Speed
    subset_indices = list(range(16))  # Only use 16 images
    train_subset = Subset(train_ds, subset_indices)
    val_subset = Subset(val_ds, subset_indices)

    train_loader = DataLoader(
        train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Shapes
    batch = next(iter(train_loader))
    imgs, masks, depths = batch["image"], batch["mask"], batch["depth"]

    # Expected: Image (B, 1, 128, 128), Mask (B, 1, 128, 128), Depth (B, 1)
    # Note: Config.IMG_SIZE is 128 (padded from 101)
    print(f"    Image Shape: {imgs.shape}")
    print(f"    Mask Shape: {masks.shape}")
    print(f"    Depth Shape: {depths.shape}")

    assert imgs.shape == (Config.BATCH_SIZE, 1, 128, 128), "Incorrect Image shape"
    assert masks.shape == (Config.BATCH_SIZE, 1, 128, 128), "Incorrect Mask shape"
    assert depths.shape == (Config.BATCH_SIZE, 1), "Incorrect Depth shape"
    print("    Data shapes verified.")

    # 4. Model Instantiation & Forward Pass
    print("\n[4] Initializing Model...")
    model = ResNet34WideLinkNetMTL().to(device)

    # Forward pass check
    imgs = imgs.to(device)
    depths = depths.to(device)

    with torch.no_grad():
        outputs = model(imgs, depths)

    assert (
        "mask" in outputs and "depth" in outputs
    ), "Model output dictionary missing keys"
    assert outputs["mask"].shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), "Model output mask shape mismatch"
    assert outputs["depth"].shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output depth shape mismatch"
    print("    Model forward pass successful.")

    # 5. Loss Function Verification
    print("\n[5] Verifying Loss Function...")
    criterion = CombinedMTLLoss().to(device)

    targets = {"mask": masks.to(device), "depth": depths.to(device)}

    # Calculate loss (requires grad usually, but here just checking computation)
    loss, metrics = criterion(outputs, targets)

    print(f"    Total Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert "loss_bce" in metrics, "Metric loss_bce missing"
    assert "loss_lovasz" in metrics, "Metric loss_lovasz missing"
    print("    Loss calculation successful.")

    # 6. Teacher Training Loop
    print("\n[6] Running Teacher Training Epoch...")
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    teacher_metrics = train_teacher_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=1,
    )
    print(f"    Teacher Epoch Metrics: {teacher_metrics}")

    # 7. Validation
    print("\n[7] Running Validation...")
    val_metrics = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        force_zero_depth=True,
    )
    print(f"    Validation Metrics: {val_metrics}")

    # 8. Inference / Pseudo-Label Generation
    print("\n[8] Running Inference (Pseudo-Labeling)...")
    # Load Test Data (Subset)
    test_ds = PseudoLabelDataset(soft_labels=None)
    test_subset = Subset(test_ds, subset_indices)
    test_loader = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    predictions = predict(model, test_loader, device)

    assert len(predictions) == len(
        test_subset
    ), "Number of predictions does not match subset size"

    # Check prediction shape (should be HxW numpy array, unpadded/original size logic handled by user,
    # but here the model outputs 128x128. The predict function returns raw probability maps).
    # Wait, the predict function in engine.py returns the raw output from model which is 128x128.
    # The submission usually requires resizing back to 101x101, but for pseudo-labeling we keep model dim.
    sample_id = list(predictions.keys())[0]
    assert predictions[sample_id].shape == (
        128,
        128,
    ), f"Prediction shape mismatch: {predictions[sample_id].shape}"
    print("    Inference successful.")

    # 9. Student Training Loop (Noisy Student)
    print("\n[9] Running Student Training Epoch...")

    # Create PseudoLabelDataset with generated soft labels
    # We use the predictions we just made
    student_unlabeled_ds = PseudoLabelDataset(soft_labels=predictions)
    student_unlabeled_subset = Subset(student_unlabeled_ds, subset_indices)

    # Loaders
    labeled_loader = DataLoader(
        train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    unlabeled_loader = DataLoader(
        student_unlabeled_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Re-init model/optimizer for student (optional, but cleaner for demo)
    student_model = ResNet34WideLinkNetMTL().to(device)
    student_optimizer = optim.AdamW(student_model.parameters(), lr=Config.LEARNING_RATE)

    student_metrics = train_student_epoch(
        model=student_model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        optimizer=student_optimizer,
        criterion=criterion,
        device=device,
        epoch=1,
    )
    print(f"    Student Epoch Metrics: {student_metrics}")

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    main()
