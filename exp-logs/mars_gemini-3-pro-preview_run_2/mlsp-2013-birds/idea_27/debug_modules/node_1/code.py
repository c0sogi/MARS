import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_pos_weights
from library.dataset import load_images, BirdDataset
from library.transforms import get_transforms
from library.models import BirdClassifier
from library.losses import WeightedDistillationLoss
from library.engine import Engine


def main():
    print("Starting Library Usage Demonstration...")

    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Set a specific cache directory for this demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds for reproducibility
    set_seed(42)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading (Subset)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading Data Subset...")

    # Load training metadata
    if not os.path.exists(Config.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Config.TRAIN_CSV}")

    df_full = pd.read_csv(Config.TRAIN_CSV)

    # Select a small subset for demonstration (16 samples)
    # This ensures the script runs quickly
    subset_size = 16
    df_train = df_full.head(subset_size).copy()
    df_val = df_full.iloc[subset_size : subset_size * 2].copy()

    print(f"Training subset size: {len(df_train)}")
    print(f"Validation subset size: {len(df_val)}")

    # Load and process images
    # This function handles resizing and converting 1-channel BMPs to 3-channel pseudo-RGB
    print("Loading images (this may take a moment for processing)...")
    train_images = load_images(df_train, cache_name="demo_train")
    val_images = load_images(df_val, cache_name="demo_val")

    # Verify image shapes
    # Expected: (N, Height, Width, Channels) -> (16, 224, 448, 3)
    assert train_images.shape == (subset_size, Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
    assert train_images.dtype == np.uint8
    print("Image loading and processing verification passed.")

    # 3. Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n[Step 3] Creating Datasets and Loaders...")

    # Get augmentations
    train_transforms = get_transforms(
        mode="train", img_height=Config.IMG_HEIGHT, img_width=Config.IMG_WIDTH
    )
    val_transforms = get_transforms(
        mode="val", img_height=Config.IMG_HEIGHT, img_width=Config.IMG_WIDTH
    )

    # Instantiate Datasets
    train_dataset = BirdDataset(train_images, df_train, transforms=train_transforms)
    val_dataset = BirdDataset(val_images, df_val, transforms=val_transforms)

    # Instantiate DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script execution
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Verify DataLoader output
    sample_batch = next(iter(train_loader))
    # Check keys
    assert "image" in sample_batch
    assert "target" in sample_batch
    assert "rec_id" in sample_batch
    # Check tensor shapes: Image should be (B, C, H, W)
    assert sample_batch["image"].shape == (
        batch_size,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    )
    assert sample_batch["target"].shape == (batch_size, Config.NUM_CLASSES)
    print("Dataset and DataLoader verification passed.")

    # 4. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[Step 4] Initializing Model...")

    # Use ResNet18 as it is lighter than DenseNet/EfficientNet
    model_name = "resnet18"
    model = BirdClassifier(
        backbone_name=model_name, pretrained=True, num_classes=Config.NUM_CLASSES
    )
    model.to(device)

    # Verify forward pass
    with torch.no_grad():
        dummy_input = sample_batch["image"].to(device)
        dummy_output = model(dummy_input)

    # Output shape should be (Batch, Num_Classes)
    assert dummy_output.shape == (batch_size, Config.NUM_CLASSES)
    print(f"Model {model_name} initialized and forward pass verified.")

    # 5. Loss Function Setup
    # -------------------------------------------------------------------------
    print("\n[Step 5] Setting up Loss Function...")

    # Extract labels from dataframe to calculate imbalance weights
    label_cols = [c for c in df_train.columns if c.startswith("species_")]
    y_train_np = df_train[label_cols].values

    # Calculate positive weights
    pos_weights = calculate_pos_weights(y_train_np)
    pos_weights = pos_weights.to(device)

    # Instantiate Loss
    loss_fn = WeightedDistillationLoss(pos_weight=pos_weights)

    # Verify Loss Calculation
    dummy_targets = sample_batch["target"].to(device)
    loss_val = loss_fn(dummy_output, dummy_targets)
    assert isinstance(loss_val.item(), float)
    assert loss_val.item() > 0
    print(f"Loss function initialized. Initial loss value: {loss_val.item():.4f}")

    # 6. Training Loop (Engine)
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Training Loop (1 Epoch)...")

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize Engine
    engine = Engine(
        model=model, device=device, optimizer=optimizer, scheduler=None, loss_fn=loss_fn
    )

    # Run fit for 1 epoch
    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    best_score = engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=1,
        patience=1,
        save_path=save_path,
    )

    print(f"Training complete. Best Val AUC: {best_score:.4f}")
    assert os.path.exists(save_path), "Model checkpoint was not created."

    # 7. Inference
    # -------------------------------------------------------------------------
    print("\n[Step 7] Running Inference...")

    preds = engine.predict(val_loader)

    # Verify predictions
    assert preds.shape == (len(df_val), Config.NUM_CLASSES)
    assert (preds >= 0).all() and (preds <= 1).all()

    print("Inference successful. Predictions shape:", preds.shape)
    print("Sample prediction probabilities (first row):")
    print(preds[0])

    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE: All components verified.")
    print("=" * 50)


if __name__ == "__main__":
    main()
