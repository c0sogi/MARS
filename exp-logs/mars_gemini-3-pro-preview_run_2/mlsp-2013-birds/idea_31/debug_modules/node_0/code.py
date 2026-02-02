import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import (
    BirdDataset,
    get_transforms,
    load_dataset_data,
    CyclicRoll,
    FixedCyclicRoll,
)
from library.models import BirdClassifier
from library.loss import DistillationLoss
from library.engine import train_one_epoch, validate, predict_tta, save_submission


def main():
    print("Starting Library Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # =========================================================================
    set_seed(42)

    # Override Config values for a quick demo run
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TTA_STEPS = 2  # Reduce TTA steps
    Config.MODEL_RESNET = "resnet18"  # Ensure we use the lightest model

    # Use a specific demo working directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print("Configuration updated for demo execution.")

    # 2. Data Loading
    # =========================================================================
    print("\n--- Data Loading ---")
    # Load training data (images and labels)
    # This uses the metadata CSVs and loads images from disk
    train_images, train_labels, train_ids = load_dataset_data(
        "train", load_cached_data=False
    )

    # Load validation data
    val_images, val_labels, val_ids = load_dataset_data("val", load_cached_data=False)

    # Check if we successfully loaded data
    if len(train_images) == 0:
        print(
            "No training images found. Creating dummy data for demonstration purposes."
        )
        # Create dummy data matching the expected dimensions (N, H, W, 3)
        # Real data is ~ 256x1246, but we resize in transforms anyway.
        train_images = np.random.randint(0, 255, (16, 256, 1246, 3), dtype=np.uint8)
        train_labels = np.random.randint(0, 2, (16, Config.NUM_CLASSES)).astype(
            np.float32
        )
        train_ids = np.arange(16, dtype=np.int32)

        val_images = np.random.randint(0, 255, (8, 256, 1246, 3), dtype=np.uint8)
        val_labels = np.random.randint(0, 2, (8, Config.NUM_CLASSES)).astype(np.float32)
        val_ids = np.arange(16, 24, dtype=np.int32)

    # Subset data for speed (use only 8 samples)
    subset_size = 8
    train_images = train_images[:subset_size]
    train_labels = train_labels[:subset_size]
    train_ids = train_ids[:subset_size]

    val_images = val_images[:subset_size]
    val_labels = val_labels[:subset_size]
    val_ids = val_ids[:subset_size]

    print(f"Subset Train Shape: {train_images.shape}")
    print(f"Subset Train Labels Shape: {train_labels.shape}")

    # Assertions
    assert len(train_images) == len(train_labels) == len(train_ids) == subset_size
    assert train_images.dtype == np.uint8

    # 3. Dataset and DataLoader
    # =========================================================================
    print("\n--- Dataset & DataLoader ---")

    # Get transforms
    train_transforms = get_transforms(phase="train")
    val_transforms = get_transforms(phase="val")

    # Create Dataset instances
    train_dataset = BirdDataset(train_images, train_labels, transforms=train_transforms)
    val_dataset = BirdDataset(val_images, val_labels, transforms=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script debug
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch
    batch = next(iter(train_loader))
    imgs = batch["image"]
    targets = batch["target"]

    print(f"Batch Image Shape: {imgs.shape}")  # Should be [B, 3, 224, 448]
    print(f"Batch Target Shape: {targets.shape}")  # Should be [B, 19]

    assert imgs.shape == (Config.BATCH_SIZE, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)
    assert targets.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    # 4. Model Instantiation
    # =========================================================================
    print("\n--- Model Initialization ---")

    model = BirdClassifier(
        model_name=Config.MODEL_RESNET,
        num_classes=Config.NUM_CLASSES,
        pretrained=False,  # False for speed in demo (no download)
        drop_rate=0.2,
        drop_samples=2,
    )
    model.to(Config.DEVICE)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(
            Config.DEVICE
        )
        logits = model(dummy_input)
        print(f"Model Output Shape: {logits.shape}")
        assert logits.shape == (2, Config.NUM_CLASSES)

    # 5. Loss Function
    # =========================================================================
    print("\n--- Loss Function ---")

    # Calculate positive weights (dummy calculation based on subset)
    pos_counts = torch.tensor(np.sum(train_labels, axis=0))
    neg_counts = len(train_labels) - pos_counts
    # Avoid division by zero
    pos_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)
    pos_weights = pos_weights.to(Config.DEVICE)

    criterion = DistillationLoss(pos_weight=pos_weights)

    # Test Loss Calculation
    dummy_logits = torch.randn(Config.BATCH_SIZE, Config.NUM_CLASSES).to(Config.DEVICE)
    dummy_targets = (
        torch.randint(0, 2, (Config.BATCH_SIZE, Config.NUM_CLASSES))
        .float()
        .to(Config.DEVICE)
    )
    loss_val = criterion(dummy_logits, dummy_targets)

    print(f"Calculated Loss: {loss_val.item():.4f}")
    assert not torch.isnan(loss_val)

    # 6. Training Loop (Engine)
    # =========================================================================
    print("\n--- Training Loop ---")

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)

    # Train one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE
    )
    print(f"Epoch 1 Train Loss: {train_loss:.4f}")

    # Validate
    val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC:  {val_auc:.4f}")

    # 7. TTA Inference
    # =========================================================================
    print("\n--- TTA Inference ---")

    # Use the validation images as 'test' images for demonstration
    test_images = val_images
    test_ids = val_ids

    # Predict using TTA
    probs = predict_tta(model, test_images, Config.DEVICE)

    print(f"Predictions Shape: {probs.shape}")
    print(f"Predictions Range: [{probs.min():.4f}, {probs.max():.4f}]")

    assert probs.shape == (len(test_images), Config.NUM_CLASSES)
    assert probs.min() >= 0.0 and probs.max() <= 1.0

    # 8. Submission Generation
    # =========================================================================
    print("\n--- Submission Generation ---")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")
    save_submission(test_ids, probs, submission_path)

    # Verify file content
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at: {submission_path}")
        print(f"Submission Rows: {len(df_sub)}")
        print(df_sub.head(3))

        # Expected rows = num_samples * num_classes
        expected_rows = len(test_ids) * Config.NUM_CLASSES
        assert len(df_sub) == expected_rows
        assert "Id" in df_sub.columns and "Probability" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nLibrary demonstration completed successfully.")


if __name__ == "__main__":
    main()
