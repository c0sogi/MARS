import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure the library modules are accessible
sys.path.append("./")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.data import load_dataset_data, get_transforms, BirdDataset
from library.models import BirdModel
from library.losses import WeightedBCELoss, DistillationLoss, calculate_pos_weights
from library.engine import train_one_epoch, validate, inference


def main():
    print("Starting Library Usage Demonstration...")

    # --- 1. Setup and Configuration Overrides ---
    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.IMG_HEIGHT = 128  # Smaller size for speed
    Config.IMG_WIDTH = 256
    Config.IMG_SIZE = (Config.IMG_HEIGHT, Config.IMG_WIDTH)

    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    seed_everything(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 2. Data Loading ---
    print("\n[1/5] Testing Data Loading and Dataset...")

    # Load training data (metadata + images)
    # This uses library.data.load_dataset_data
    # Note: This might take a few seconds to process/cache the first time
    all_images, df_train = load_dataset_data(mode="train", load_cached_data=False)

    # Use a tiny subset for demonstration
    subset_size = 16
    images_subset = all_images[:subset_size]
    df_subset = df_train.iloc[:subset_size].reset_index(drop=True)

    print(f"Loaded subset: {len(images_subset)} images.")

    # Instantiate Dataset
    train_dataset = BirdDataset(
        images=images_subset,
        df=df_subset,
        transforms=get_transforms("train"),
        phase="train",
    )

    # Instantiate DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for simple main-thread execution
        drop_last=True,
    )

    # Verify Batch
    images_batch, targets_batch, soft_targets_batch = next(iter(train_loader))

    assert images_batch.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), f"Image batch shape mismatch: {images_batch.shape}"
    assert targets_batch.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Target batch shape mismatch: {targets_batch.shape}"
    assert soft_targets_batch.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Soft target batch shape mismatch: {soft_targets_batch.shape}"

    print("Dataset and DataLoader verified successfully.")

    # --- 3. Model Initialization ---
    print("\n[2/5] Testing Model Architecture...")

    # Initialize Model (ResNet18)
    # pretrained=False to avoid downloading weights during this demo
    model = BirdModel(model_name="resnet18", pretrained=False)
    model.to(device)

    # Test Forward Pass
    dummy_input = images_batch.to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch: {logits.shape}"

    print("Model instantiated and forward pass verified.")

    # --- 4. Loss Function Verification ---
    print("\n[3/5] Testing Loss Functions...")

    # Calculate positive weights
    pos_weights = calculate_pos_weights(df_subset[train_dataset.label_cols].values)
    pos_weights = pos_weights.to(device)

    # Test WeightedBCELoss
    bce_loss_fn = WeightedBCELoss(pos_weights=pos_weights)
    loss_val = bce_loss_fn(logits, targets_batch.to(device))

    assert isinstance(loss_val, torch.Tensor), "Loss should be a tensor"
    assert loss_val.ndim == 0, "Loss should be a scalar"
    print(f"WeightedBCELoss calculated: {loss_val.item():.4f}")

    # Test DistillationLoss
    # Create dummy teacher logits
    teacher_logits = torch.randn_like(logits).to(device)
    distill_loss_fn = DistillationLoss(pos_weights=pos_weights)

    # Distillation loss requires (student_logits, teacher_logits, hard_targets)
    d_loss_val = distill_loss_fn(logits, teacher_logits, targets_batch.to(device))

    assert d_loss_val.ndim == 0, "Distillation Loss should be a scalar"
    print(f"DistillationLoss calculated: {d_loss_val.item():.4f}")

    # --- 5. Training Loop Demonstration ---
    print("\n[4/5] Testing Training Loop (Engine)...")

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Run one epoch of training
    # Using mixup_alpha=0.0 to simplify verification (no random mixing)
    avg_train_loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        epoch=1,
        loss_fn=bce_loss_fn,
        mixup_alpha=0.0,
    )

    assert isinstance(
        avg_train_loss, float
    ), "Train function should return a float loss"
    print(f"Training loop complete. Avg Loss: {avg_train_loss:.4f}")

    # --- 6. Validation and Inference ---
    print("\n[5/5] Testing Validation and Inference...")

    # Create Validation Loader (using same subset for demo)
    val_dataset = BirdDataset(
        images=images_subset,
        df=df_subset,
        transforms=get_transforms("val"),
        phase="val",
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Run Validation
    val_loss, val_auc = validate(model, val_loader, device, bce_loss_fn)

    assert isinstance(val_loss, float), "Validation loss should be float"
    assert isinstance(val_auc, float), "Validation AUC should be float"
    print(f"Validation complete. Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # Run Inference (Test Time Augmentation)
    # Using val_loader as proxy for test_loader
    preds = inference(model, val_loader, device)

    assert preds.shape == (
        subset_size,
        Config.NUM_CLASSES,
    ), f"Inference output shape mismatch: {preds.shape}"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions should be probabilities between 0 and 1"

    print("Inference complete.")
    print("\nAll demonstrations passed successfully!")


if __name__ == "__main__":
    main()
