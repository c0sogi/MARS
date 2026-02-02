import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.engine import train_one_epoch, validate


def main():
    # --- 1. Setup & Configuration ---
    print("Initializing demonstration...")

    # Set deterministic behavior
    seed_everything(Config.seed)

    # Override Config for rapid demonstration
    print("Overriding Config for speed...")
    Config.debug = True  # Use a small subset (100 samples)
    Config.epochs = 1  # Run only 1 epoch
    Config.batch_size = 8  # Small batch size
    Config.num_workers = 2  # Reduce workers for simple script

    device = Config.device
    print(f"Device: {device}")

    # --- 2. Data Loading Demonstration ---
    print("\n--- Demonstrating Data Loading ---")

    # Initialize Datasets
    # We use the pre-generated metadata files
    train_dataset = CatheterDataset(
        metadata_path=Config.train_metadata,
        transform=get_transforms("train"),
        debug=Config.debug,
    )

    val_dataset = CatheterDataset(
        metadata_path=Config.val_metadata,
        transform=get_transforms("val"),
        debug=Config.debug,
    )

    print(f"Train Dataset Size (Debug): {len(train_dataset)}")
    print(f"Val Dataset Size (Debug): {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Verify Batch Structure
    images, targets = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.batch_size,
        3,
        Config.image_size,
        Config.image_size,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Incorrect target shape: {targets.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"
    print("Data Loading verification passed.")

    # --- 3. Model Instantiation Demonstration ---
    print("\n--- Demonstrating Model Instantiation ---")

    model = CatheterModel()
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        # Move dummy batch to device
        dummy_input = images.to(device)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.batch_size,
        Config.num_classes,
    ), f"Model output shape mismatch. Expected {(Config.batch_size, Config.num_classes)}, got {output.shape}"
    print("Model verification passed.")

    # --- 4. Training Loop Demonstration ---
    print("\n--- Demonstrating Training Loop (1 Epoch) ---")

    # Setup Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.learning_rate,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=Config.pct_start,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor,
    )

    # Run Training
    # We pass None for ema_model to keep the demo simple, though the engine supports it
    avg_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_loader,
        device=device,
        epoch=1,
        ema_model=None,
    )

    print(f"Epoch 1 Training Loss: {avg_loss:.4f}")

    # Assertions
    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss > 0, "Training loss should be positive"
    print("Training loop verification passed.")

    # --- 5. Validation Demonstration ---
    print("\n--- Demonstrating Validation ---")

    val_loss, val_auc = validate(model=model, dataloader=val_loader, device=device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Assertions
    assert not np.isnan(val_loss), "Validation loss is NaN"
    # AUC can be 0.0 if the small debug batch has only one class present, but usually returns a float
    assert isinstance(val_auc, float), "AUC should be a float"
    print("Validation verification passed.")

    # --- 6. Inference/Test Demonstration ---
    print("\n--- Demonstrating Inference (Test Set) ---")

    # Initialize Test Dataset
    test_dataset = CatheterDataset(
        metadata_path=Config.test_metadata,
        transform=get_transforms("test"),  # No augmentation
        is_test=True,
        debug=True,  # Limit to 100 for demo
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
    )

    model.eval()
    predictions = []
    ids = []

    # Run inference on a few batches
    with torch.no_grad():
        for i, (images, study_uids) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            predictions.append(probs.cpu().numpy())
            ids.extend(study_uids)

            if i >= 2:  # Stop after a few batches for speed
                break

    predictions = np.concatenate(predictions, axis=0)

    print(f"Inference Predictions Shape: {predictions.shape}")
    print(f"Number of IDs: {len(ids)}")

    # Assertions
    assert predictions.shape[1] == Config.num_classes, "Prediction columns mismatch"
    assert len(predictions) == len(ids), "Number of predictions and IDs mismatch"
    assert (predictions >= 0).all() and (
        predictions <= 1
    ).all(), "Probabilities must be in [0, 1]"

    print("Inference verification passed.")

    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
