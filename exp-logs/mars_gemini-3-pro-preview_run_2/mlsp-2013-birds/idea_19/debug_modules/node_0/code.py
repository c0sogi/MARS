import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Import provided library modules
from library.utils import set_seed
from library.dataset import BirdDataset, mixup_data
from library.models import get_model
from library.loss import AsymmetricLoss
from library.engine import train_one_epoch, validate, inference_fn


def main():
    # 1. Setup and Configuration
    print("1. Setting up configuration...")
    SEED = 42
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "demo_cache")

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")

    # Hyperparameters for demonstration
    BATCH_SIZE = 8
    NUM_CLASSES = 19
    HEIGHT = 224
    WIDTH = 448  # As defined in dataset.py default or we can override

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Demonstrate Dataset Loading
    print("\n2. Demonstrating BirdDataset...")

    # Initialize Dataset
    train_dataset = BirdDataset(
        csv_file=TRAIN_CSV,
        mode="train",
        load_cached_data=False,  # Force reload to test logic
        cache_dir=CACHE_DIR,
        height=HEIGHT,
        width=WIDTH,
    )

    print(f"   Training Dataset Size: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty."

    # Fetch one sample
    img, label = train_dataset[0]

    # Verify shapes
    # Image should be (3, H, W) after ToTensorV2 (albumentations)
    print(f"   Sample Image Shape: {img.shape}")
    print(f"   Sample Label Shape: {label.shape}")

    assert img.shape == (
        3,
        HEIGHT,
        WIDTH,
    ), f"Expected image shape (3, {HEIGHT}, {WIDTH}), got {img.shape}"
    assert label.shape == (
        NUM_CLASSES,
    ), f"Expected label shape ({NUM_CLASSES},), got {label.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a torch Tensor"
    assert isinstance(label, np.ndarray) or isinstance(
        label, torch.Tensor
    ), "Label should be numpy array or tensor"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Test Mixup (Simulated)
    print("   Testing Mixup logic...")
    dummy_imgs = torch.randn(BATCH_SIZE, 3, HEIGHT, WIDTH).to(device)
    dummy_lbls = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float().to(device)
    mixed_x, y_a, y_b, lam = mixup_data(
        dummy_imgs, dummy_lbls, alpha=0.4, device=device
    )

    assert mixed_x.shape == dummy_imgs.shape, "Mixed images shape mismatch"
    assert y_a.shape == dummy_lbls.shape, "Target A shape mismatch"
    assert 0 <= lam <= 1, "Lambda should be between 0 and 1"

    # 3. Demonstrate Model Architecture
    print("\n3. Demonstrating BirdModel (ResNet18)...")

    model = get_model("resnet18", num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)

    # Forward pass with dummy input
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, HEIGHT, WIDTH).to(device)
        logits = model(dummy_input)

    print(f"   Output Logits Shape: {logits.shape}")
    assert logits.shape == (
        2,
        NUM_CLASSES,
    ), f"Expected output shape (2, {NUM_CLASSES}), got {logits.shape}"

    # 4. Demonstrate Loss Function
    print("\n4. Demonstrating AsymmetricLoss...")

    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    # Create dummy predictions (logits) and targets
    dummy_logits = torch.randn(BATCH_SIZE, NUM_CLASSES).to(device)
    dummy_targets = torch.randint(0, 2, (BATCH_SIZE, NUM_CLASSES)).float().to(device)

    loss_val = criterion(dummy_logits, dummy_targets)
    print(f"   Calculated Loss: {loss_val.item():.4f}")

    assert not torch.isnan(loss_val), "Loss should not be NaN"
    assert loss_val.item() >= 0, "Loss should be non-negative"

    # 5. Demonstrate Engine (Training & Validation)
    print("\n5. Demonstrating Training and Validation Engine...")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train for 1 epoch
    print("   Running training for 1 epoch...")
    train_loss = train_one_epoch(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataloader=train_loader,
        device=device,
        criterion=criterion,
    )
    print(f"   Epoch Training Loss: {train_loss:.4f}")
    assert train_loss > 0, "Training loss should be positive"

    # Validation Setup
    val_dataset = BirdDataset(
        csv_file=VAL_CSV,
        mode="val",
        load_cached_data=False,
        cache_dir=CACHE_DIR,
        height=HEIGHT,
        width=WIDTH,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Validate
    print("   Running validation...")
    val_loss, val_auc = validate(
        model=model, dataloader=val_loader, criterion=criterion, device=device
    )
    # Note: AUC might be 0.5 if model hasn't learned anything in 1 epoch or classes are constant,
    # but the function should execute without error.
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation AUC: {val_auc:.4f}")

    # 6. Demonstrate Inference (TTA)
    print("\n6. Demonstrating Inference with TTA...")

    # Use validation loader as test loader for demonstration
    preds = inference_fn(model, val_loader, device)

    print(f"   Predictions Shape: {preds.shape}")
    assert preds.shape == (
        len(val_dataset),
        NUM_CLASSES,
    ), f"Expected predictions shape ({len(val_dataset)}, {NUM_CLASSES}), got {preds.shape}"

    # Check probability range
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions must be probabilities between 0 and 1"

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
