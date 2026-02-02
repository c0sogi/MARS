import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import from the provided library files
from library.config import CFG
from library.utils import seed_everything
from library.dataset import CassavaDataset
from library.augmentations import get_transforms
from library.model import CassavaModel
from library.engine import train_one_epoch, valid_one_epoch
from library.mixup import mixup_data, cutmix_data


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print("--- Setting up environment and configuration ---")
    seed_everything(CFG.seed)

    # Override CFG for a quick demo run
    CFG.model_name = "tf_efficientnet_b0_ns"  # Use a smaller model for speed
    CFG.img_size = 224  # Smaller image size for speed
    CFG.train_batch_size = 8
    CFG.valid_batch_size = 8
    CFG.epochs = 1
    CFG.print_freq = 1
    CFG.debug = True  # Although we manually slice, this indicates intent

    device = CFG.device
    print(f"Device: {device}")

    # 2. Data Preparation
    print("\n--- Preparing Data ---")
    # Load metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)

    # Use a tiny subset for demonstration
    train_subset = train_df.head(32).copy().reset_index(drop=True)
    val_subset = val_df.head(16).copy().reset_index(drop=True)

    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    # Instantiate Datasets
    train_dataset = CassavaDataset(
        train_subset, transform=get_transforms(data="train"), output_label=True
    )
    valid_dataset = CassavaDataset(
        val_subset, transform=get_transforms(data="valid"), output_label=True
    )

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Verification: Check Data Loading
    print("Verifying DataLoader output...")
    dummy_imgs, dummy_labels = next(iter(train_loader))

    # Check shapes
    assert dummy_imgs.ndim == 4, f"Expected 4D image tensor, got {dummy_imgs.ndim}"
    assert dummy_imgs.shape[1] == 3, f"Expected 3 channels, got {dummy_imgs.shape[1]}"
    assert (
        dummy_imgs.shape[2] == CFG.img_size
    ), f"Expected height {CFG.img_size}, got {dummy_imgs.shape[2]}"
    assert (
        dummy_imgs.shape[3] == CFG.img_size
    ), f"Expected width {CFG.img_size}, got {dummy_imgs.shape[3]}"
    assert dummy_labels.shape[0] == CFG.train_batch_size, "Label batch size mismatch"
    print("DataLoader verification passed.")

    # 3. Augmentation Logic Verification (Mixup/Cutmix)
    print("\n--- Verifying MixUp and CutMix logic ---")
    # Create dummy inputs on CPU for logic check
    dummy_x = torch.randn(4, 3, 224, 224)
    dummy_y = torch.tensor([0, 1, 2, 3])

    # Test Mixup
    mixed_x, y_a, y_b, lam = mixup_data(dummy_x, dummy_y, alpha=1.0)
    assert mixed_x.shape == dummy_x.shape, "Mixup output shape mismatch"
    assert y_a.shape == dummy_y.shape, "Mixup label shape mismatch"
    assert 0 <= lam <= 1, "Mixup lambda out of range"

    # Test Cutmix
    cut_x, cy_a, cy_b, clam = cutmix_data(dummy_x, dummy_y, alpha=1.0)
    assert cut_x.shape == dummy_x.shape, "Cutmix output shape mismatch"
    assert cy_a.shape == dummy_y.shape, "Cutmix label shape mismatch"
    assert 0 <= clam <= 1, "Cutmix lambda out of range"
    print("Augmentation logic verification passed.")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    # Using the smaller model defined in config override
    model = CassavaModel(model_name=CFG.model_name, pretrained=True)
    model.to(device)

    # Verification: Forward pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        dummy_imgs = dummy_imgs.to(device)
        logits = model(dummy_imgs)

    assert logits.shape == (
        CFG.train_batch_size,
        CFG.num_classes,
    ), f"Expected output shape {(CFG.train_batch_size, CFG.num_classes)}, got {logits.shape}"
    print("Model forward pass verification passed.")

    # 5. Training Loop Demonstration
    print("\n--- Starting Training Loop Demo ---")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Train for 1 epoch
    avg_train_loss = train_one_epoch(
        epoch=0,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
    )
    print(f"Epoch 0 Training completed. Avg Loss: {avg_train_loss:.4f}")

    # Verify loss is valid
    assert not np.isnan(avg_train_loss), "Training loss is NaN"
    assert avg_train_loss > 0, "Training loss should be positive"

    # 6. Validation Loop Demonstration
    print("\n--- Starting Validation Loop Demo ---")
    avg_val_loss, val_acc = valid_one_epoch(
        epoch=0,
        model=model,
        val_loader=valid_loader,
        device=device,
        criterion=criterion,
    )
    print(
        f"Epoch 0 Validation completed. Avg Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}"
    )

    # Verify metrics
    assert not np.isnan(avg_val_loss), "Validation loss is NaN"
    assert 0.0 <= val_acc <= 1.0, "Accuracy out of range [0, 1]"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
