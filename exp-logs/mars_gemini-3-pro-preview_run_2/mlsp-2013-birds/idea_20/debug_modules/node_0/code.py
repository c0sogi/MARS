import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import albumentations as A

# Ensure the current directory is in the path to import library modules
sys.path.append("./")

from library.config import Config
from library.utils import set_seed, compute_roc_auc
from library.dataset import get_data, get_dataloaders, get_test_dataloader
from library.augmentations import get_transforms
from library.network import BirdModel
from library.losses import SiameseConsistencyLoss
from library.engine import train_one_epoch, validate, predict_with_tta


def main():
    print("Starting Demonstration Script...")

    # --------------------------------------------------------------------------
    # 1. Setup and Configuration
    # --------------------------------------------------------------------------
    print("\n[1] Setting up configuration and environment...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for Speed/Demo purposes
    Config.DEBUG = True  # Use a small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.WORKING_DIR = "./working/demo_execution"  # Separate working dir

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading Data...")

    # Load data (this will use the metadata files and load images)
    # We set load_cached_data=False to demonstrate the processing logic
    df_dev, images_dev, labels_dev, df_test, images_test = get_data(
        load_cached_data=False
    )

    # Assertions to verify data loading
    print(f"Dev Images Shape: {images_dev.shape}")
    print(f"Dev Labels Shape: {labels_dev.shape}")

    assert (
        len(df_dev) == len(images_dev) == len(labels_dev)
    ), "Mismatch in dev set lengths"
    assert images_dev.dtype == np.uint8, "Images should be uint8"
    assert (
        labels_dev.shape[1] == Config.NUM_SPECIES
    ), f"Labels should have {Config.NUM_SPECIES} columns"

    # --------------------------------------------------------------------------
    # 3. DataLoaders and Augmentations
    # --------------------------------------------------------------------------
    print("\n[3] Creating DataLoaders and Verifying Augmentations...")

    # Get dataloaders for Fold 0
    train_loader, val_loader = get_dataloaders(
        fold_idx=0,
        df_dev=df_dev,
        images_dev=images_dev,
        labels_dev=labels_dev,
        batch_size=Config.BATCH_SIZE,
    )

    # Fetch a single batch to verify shapes and transforms
    images_batch, labels_batch = next(iter(train_loader))

    print(f"Batch Images Shape: {images_batch.shape}")  # Should be (B, C, H, W)
    print(f"Batch Labels Shape: {labels_batch.shape}")  # Should be (B, Num_Classes)

    # Verify Tensor shapes
    assert images_batch.dim() == 4, "Images batch should be 4D (B, C, H, W)"
    assert images_batch.shape[1] == 3, "Images should have 3 channels"
    assert images_batch.shape[2] == Config.IMG_SIZE[0], "Height mismatch"
    assert images_batch.shape[3] == Config.IMG_SIZE[1], "Width mismatch"
    assert labels_batch.shape[1] == Config.NUM_SPECIES, "Label dimension mismatch"

    # Verify Augmentation Pipeline instantiation
    transforms_train = get_transforms(mode="train")
    assert isinstance(
        transforms_train, A.Compose
    ), "Transforms should be an Albumentations Compose object"

    # --------------------------------------------------------------------------
    # 4. Model Initialization
    # --------------------------------------------------------------------------
    print("\n[4] Initializing Model...")

    # Initialize model (using resnet18 for speed, pretrained=False to avoid download in restricted envs)
    model = BirdModel(
        model_name="resnet18", pretrained=False, num_classes=Config.NUM_SPECIES
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images_batch.to(Config.DEVICE)
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_SPECIES,
    ), "Output logits shape mismatch"

    # --------------------------------------------------------------------------
    # 5. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Custom Loss Function...")

    # Calculate positive weights for class imbalance handling
    pos_weights = Config.get_pos_weights(df_dev).to(Config.DEVICE)

    # Initialize SiameseConsistencyLoss
    criterion = SiameseConsistencyLoss(pos_weights=pos_weights, consistency_lambda=1.0)

    # Create dummy inputs for loss calculation
    dummy_logits = torch.randn(Config.BATCH_SIZE, Config.NUM_SPECIES).to(Config.DEVICE)
    dummy_logits_roll = torch.randn(Config.BATCH_SIZE, Config.NUM_SPECIES).to(
        Config.DEVICE
    )
    dummy_targets = (
        torch.randint(0, 2, (Config.BATCH_SIZE, Config.NUM_SPECIES))
        .float()
        .to(Config.DEVICE)
    )

    # Compute loss
    loss_val = criterion(dummy_logits, dummy_logits_roll, dummy_targets)

    print(f"Calculated Loss: {loss_val.item()}")
    assert not torch.isnan(loss_val), "Loss should not be NaN"
    assert loss_val.item() > 0, "Loss should be positive"

    # --------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training for one epoch
    train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        pos_weights=pos_weights,
    )

    print(f"Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # --------------------------------------------------------------------------
    # 7. Validation Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[7] Running Validation Loop...")

    val_loss, val_auc = validate(
        model=model, loader=val_loader, device=Config.DEVICE, pos_weights=pos_weights
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Basic sanity checks
    assert val_loss >= 0, "Validation loss cannot be negative"
    assert 0.0 <= val_auc <= 1.0, "AUC must be between 0 and 1"

    # --------------------------------------------------------------------------
    # 8. Inference / Prediction Demonstration
    # --------------------------------------------------------------------------
    print("\n[8] Running Inference with TTA...")

    test_loader = get_test_dataloader(
        df_test, images_test, batch_size=Config.BATCH_SIZE
    )

    preds = predict_with_tta(model=model, loader=test_loader, device=Config.DEVICE)

    print(f"Predictions Shape: {preds.shape}")

    if len(preds) > 0:
        assert preds.shape[0] == len(
            df_test
        ), "Number of predictions should match test set size"
        assert (
            preds.shape[1] == Config.NUM_SPECIES
        ), "Prediction columns should match num species"
        assert (preds >= 0).all() and (
            preds <= 1
        ).all(), "Predictions should be probabilities [0, 1]"
    else:
        print(
            "Warning: No predictions generated (Test set might be empty in debug mode if filtered heavily)"
        )

    print("\n" + "=" * 50)
    print("Demonstration Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
