import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config, set_seed
from library.dataset import BirdDataset
from library.transforms import get_transforms
from library.model import BirdClassifier
from library.engine import train_one_epoch, evaluate
from library.utils import calculate_roc_auc


def main():
    # 1. Setup and Configuration Overrides for Demo
    print("Initializing Demo...")
    set_seed(Config.SEED)

    # Override Config for speed
    Config.DEBUG = True  # Limits dataset size
    Config.DEBUG_SAMPLES = 32  # Small number of samples
    Config.EPOCHS = 2  # Minimal epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading and Transforms
    print("\n--- Testing Dataset and Transforms ---")

    # Define resolutions based on Config
    width, height = Config.RES_HIGH[1], Config.RES_HIGH[0]

    # Get transforms
    train_transform = get_transforms("train", width, height)
    val_transform = get_transforms("val", width, height)

    # Initialize Datasets
    # Note: We use the pre-generated metadata files
    train_dataset = BirdDataset(
        csv_path=Config.TRAIN_CSV, transform=train_transform, debug=True, preload=True
    )
    val_dataset = BirdDataset(
        csv_path=Config.VAL_CSV, transform=val_transform, debug=True, preload=True
    )

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,  # Ensure consistent batch sizes for mixup
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check batch shapes
    images, labels, rec_ids = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Expected: (Batch, 3, H, W)
    print(f"Batch Label Shape: {labels.shape}")  # Expected: (Batch, Num_Classes)

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        height,
        width,
    ), "Incorrect image tensor shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"

    # 3. Model Initialization
    print("\n--- Testing Model Architecture ---")

    model_name = "resnet18"  # Using a lightweight backbone for demo
    model = BirdClassifier(
        model_name=model_name, pretrained=False
    )  # False for speed/offline
    model.to(device)

    # Verification: Forward pass
    with torch.no_grad():
        dummy_input = images.to(device)
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # 4. Training Loop
    print("\n--- Testing Training Loop (Engine) ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate simple pos_weight (usually done on full dataset, approximating here)
    # Just setting to ones for demo purposes to test the pipeline
    pos_weight = torch.ones(Config.NUM_CLASSES).to(device)

    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=train_loader,
            device=device,
            pos_weight=pos_weight,
        )
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_loss:.4f}")

        # Simple assertion to ensure loss is valid
        assert not np.isnan(avg_loss), "Training loss returned NaN"
        assert avg_loss > 0, "Training loss should be positive"

    # 5. Evaluation and Metrics
    print("\n--- Testing Evaluation and Metrics ---")

    val_loss, preds, targets = evaluate(
        model=model, dataloader=val_loader, device=device, pos_weight=pos_weight
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Predictions Shape: {preds.shape}")
    print(f"Targets Shape: {targets.shape}")

    # Verification: Predictions range
    assert (
        preds.min() >= 0 and preds.max() <= 1
    ), "Predictions must be probabilities [0, 1]"
    assert preds.shape == targets.shape, "Prediction and Target shapes must match"

    # Calculate Metric
    roc_auc = calculate_roc_auc(targets, preds)
    print(f"ROC AUC Score: {roc_auc:.4f}")

    assert 0.0 <= roc_auc <= 1.0, "ROC AUC must be between 0 and 1"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
