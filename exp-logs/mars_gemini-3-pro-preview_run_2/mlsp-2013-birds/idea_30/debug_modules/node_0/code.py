import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, calculate_pos_weights, compute_auc
from library.dataset import get_processed_data, BirdDataset
from library.models import get_model
from library.losses import WeightedBCE, DistillationLoss
from library.engine import (
    train_one_epoch,
    valid_one_epoch,
    distill_one_epoch,
    tta_inference,
)


def main():
    print("Starting Library Demonstration...")

    # 1. Setup and Configuration
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # Override Config for speed in this demo
    BATCH_SIZE = 8  # Small batch size for demo
    Config.BATCH_SIZE = BATCH_SIZE

    # 2. Data Loading and Processing
    print("\n--- Data Loading ---")
    # Load training data (uses caching mechanism)
    # We use 'train' split. get_processed_data handles loading metadata and processing images.
    try:
        images, labels, rec_ids = get_processed_data("train", load_cached_data=True)
        print(f"Loaded {len(images)} training samples.")
        print(f"Image shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Create a small subset for rapid demonstration (e.g., 32 samples)
    subset_size = 32
    if len(images) > subset_size:
        images_sub = images[:subset_size]
        labels_sub = labels[:subset_size]
        rec_ids_sub = rec_ids[:subset_size]
    else:
        images_sub = images
        labels_sub = labels
        rec_ids_sub = rec_ids

    # Initialize Dataset
    train_dataset = BirdDataset(
        images=images_sub, labels=labels_sub, rec_ids=rec_ids_sub, mode="train"
    )

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple debugging/demo
        pin_memory=False,
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    img_batch = batch["image"]
    target_batch = batch["target"]

    assert img_batch.shape == (
        BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Incorrect image batch shape: {img_batch.shape}"
    assert target_batch.shape == (
        BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect target batch shape: {target_batch.shape}"
    print("Data loading and batch structure verified.")

    # 3. Model Initialization
    print("\n--- Model Initialization ---")
    # Using ResNet18 as it's lighter than others
    model = get_model("resnet18", device=device)
    print("Model 'resnet18' instantiated and moved to device.")

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
        dummy_output = model(dummy_input)
        assert dummy_output.shape == (
            2,
            Config.NUM_CLASSES,
        ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {dummy_output.shape}"
    print("Model forward pass verified.")

    # 4. Loss Function Setup
    print("\n--- Loss Function Setup ---")
    # Calculate positive weights for class imbalance
    # We need the dataframe for this utility function
    df_train = pd.read_csv(Config.TRAIN_CSV)
    pos_weights = calculate_pos_weights(df_train, device=device)

    # Initialize WeightedBCE
    criterion = WeightedBCE(pos_weights=pos_weights)
    print("WeightedBCE criterion initialized.")

    # 5. Training Loop (Phase 1: Standard Training)
    print("\n--- Training Loop (Standard) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # No scheduler for this short demo
    scheduler = None

    # Run one epoch
    loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        mixup_alpha=0.0,  # Disable mixup for simple deterministic check
    )
    print(f"Epoch 1 Training Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss returned NaN"

    # 6. Validation Loop
    print("\n--- Validation Loop ---")
    # Use the same subset as validation for demonstration purposes
    val_dataset = BirdDataset(
        images=images_sub, labels=labels_sub, rec_ids=rec_ids_sub, mode="val"
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_loss, val_auc = valid_one_epoch(
        model=model, loader=val_loader, criterion=criterion, device=device
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    assert 0 <= val_auc <= 1, "AUC score out of range [0, 1]"

    # 7. TTA Inference
    print("\n--- TTA Inference ---")
    # Perform inference on the subset
    # TTA steps = 2 for speed (Config usually has 4)
    tta_preds = tta_inference(
        model=model,
        images=images_sub,
        rec_ids=rec_ids_sub,
        device=device,
        tta_steps=2,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    assert tta_preds.shape == (
        subset_size,
        Config.NUM_CLASSES,
    ), f"TTA predictions shape mismatch: {tta_preds.shape}"
    print("TTA Inference completed successfully.")

    # 8. Distillation Training (Phase 3: Born-Again)
    print("\n--- Distillation Training ---")
    # Simulate soft targets (e.g., from a teacher model or TTA)
    # In practice, these come from the previous phase. Here we use random probs.
    soft_targets_sub = np.random.rand(subset_size, Config.NUM_CLASSES).astype(
        np.float32
    )

    # Create dataset with soft targets
    distill_dataset = BirdDataset(
        images=images_sub,
        labels=labels_sub,
        soft_labels=soft_targets_sub,
        rec_ids=rec_ids_sub,
        mode="train",
    )
    distill_loader = DataLoader(distill_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize Distillation Loss
    distill_criterion = DistillationLoss(pos_weights=pos_weights, lambda_distill=0.5)

    # Run one epoch of distillation
    d_loss = distill_one_epoch(
        model=model,
        loader=distill_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=distill_criterion,
        device=device,
        mixup_alpha=0.0,
    )
    print(f"Distillation Epoch Loss: {d_loss:.4f}")
    assert not np.isnan(d_loss), "Distillation loss returned NaN"

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    main()
