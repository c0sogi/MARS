import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_pos_weights
from library.dataset import get_dataloaders, load_images_with_cache
from library.models import BirdModel
from library.engine import train_one_epoch, validate, inference, mixup_data


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("[1/7] Setting up configuration...")
    # Initialize the configuration (sets seeds, creates directories)
    Config.setup()

    # Override parameters for a fast demonstration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True

    # Verify Working Directory Creation
    assert os.path.exists(Config.WORKING_DIR), "Working directory was not created."
    print(f"Configuration loaded. Working directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    print("\n[2/7] Loading metadata and preparing DataLoaders...")

    # Load a small subset of the metadata for speed
    train_df = pd.read_csv(Config.TRAIN_CSV).head(16)
    val_df = pd.read_csv(Config.VAL_CSV).head(8)
    test_df = pd.read_csv(Config.TEST_CSV).head(8)

    print(
        f"Subset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )

    # Calculate positive class weights for loss balancing
    device = Config.DEVICE
    pos_weights = get_pos_weights(train_df, device)
    assert pos_weights.shape == (
        Config.NUM_CLASSES,
    ), "Positive weights shape mismatch."

    # Create DataLoaders
    # We use 'resnet18' which maps to resolution (224, 448) in Config
    model_name = "resnet18"
    dataloaders = get_dataloaders(
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        load_cached_data=False,  # Force loading from disk to verify raw image reading
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]

    # Verify Batch Shapes
    images, targets = next(iter(train_loader))
    expected_h, expected_w = Config.get_resolution(model_name)

    # Expected shape: (Batch, Channels, Height, Width)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        expected_h,
        expected_w,
    ), f"Image batch shape mismatch. Got {images.shape}, expected {(Config.BATCH_SIZE, 3, expected_h, expected_w)}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Target batch shape mismatch. Got {targets.shape}"

    print("DataLoaders initialized and batch shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3/7] Initializing Model...")
    # Using pretrained=False to avoid downloading weights during this demo
    model = BirdModel(model_name=model_name, pretrained=False)
    model.to(device)

    # Verify Forward Pass with Dummy Input
    dummy_input = torch.randn(2, 3, expected_h, expected_w).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Got {output.shape}, expected (2, {Config.NUM_CLASSES})"
    print("Model initialized and forward pass verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4/7] Running Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Verify Mixup Functionality independently
    mixed_x, mixed_y = mixup_data(images.to(device), targets.to(device), alpha=0.4)
    assert mixed_x.shape == images.shape, "Mixup image shape mismatch."
    assert mixed_y.shape == targets.shape, "Mixup target shape mismatch."

    # Run one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, epoch=1
    )
    print(f"Training completed. Avg Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # -------------------------------------------------------------------------
    # 5. Validation Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5/7] Running Validation Loop...")
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation completed. Loss: {val_loss:.4f}, Macro AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN."

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Inference on Test Set...")
    predictions = inference(model, test_loader, device)

    assert predictions.shape == (
        len(test_df),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Got {predictions.shape}"

    # Check values are probabilities (0 to 1)
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Predictions contain values outside [0, 1]."
    print("Inference completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Semi-Supervised Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[7/7] Demonstrating Semi-Supervised Data Loading...")
    # Simulate pseudo-labels for the test set
    pseudo_labels = np.random.rand(len(test_df), Config.NUM_CLASSES).astype(np.float32)

    # Create dataloaders combining Train (Hard Labels) + Test (Soft Pseudo Labels)
    ss_dataloaders = get_dataloaders(
        model_name=model_name,
        train_df=train_df,
        test_df=test_df,
        pseudo_labels=pseudo_labels,
        load_cached_data=True,  # Use cache this time
    )

    ss_train_loader = ss_dataloaders["train"]
    total_samples = len(train_df) + len(test_df)

    assert (
        len(ss_train_loader.dataset) == total_samples
    ), f"Semi-supervised dataset size mismatch. Got {len(ss_train_loader.dataset)}, expected {total_samples}"

    # Check if soft labels are being returned
    ss_images, ss_targets = next(iter(ss_train_loader))
    assert ss_targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "SS Target shape mismatch."
    print("Semi-supervised dataloader created successfully.")

    print("\n=== Demonstration Complete: All components verified ===")


if __name__ == "__main__":
    main()
