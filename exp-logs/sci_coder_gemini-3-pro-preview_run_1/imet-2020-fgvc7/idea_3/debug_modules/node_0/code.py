import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config, seed_everything
from library.dataset import load_metadata_df, get_transforms, ArtworkDataset
from library.model import ArtworkModel
from library.train import train_one_epoch, validate
from library.utils import calculate_f1, find_best_threshold, ModelEMA


def run_demonstration():
    print("=== Starting Artwork Attribute Labeling Demo ===\n")

    # 1. Setup & Configuration Overrides
    # We override specific Config values to make this demo run fast (seconds instead of hours)
    seed_everything(Config.SEED)

    Config.IMG_SIZE = 128  # Reduce image size for speed
    Config.BATCH_SIZE = 8  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data sample
    Config.MODEL_NAME = "resnet18"  # Use a lighter model than ConvNeXt for demo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    print(f"Model: {Config.MODEL_NAME}, Image Size: {Config.IMG_SIZE}")

    # 2. Data Loading & Verification
    print("\n[1/4] Verifying Dataset and DataLoader...")

    # Load a tiny subset of metadata for verification
    # load_cached_data=False ensures we read from the source CSVs in ./metadata
    train_df = load_metadata_df("train", load_cached_data=False, sample_size=32)
    val_df = load_metadata_df("val", load_cached_data=False, sample_size=16)

    print(
        f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples."
    )

    # Initialize Dataset and Transforms
    train_transforms = get_transforms("train", Config.IMG_SIZE)
    train_dataset = ArtworkDataset(train_df, mode="train", transforms=train_transforms)

    # Verify Dataset Length
    assert len(train_dataset) == 32, "Dataset length mismatch."

    # Verify Item Structure
    sample_img, sample_target = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target Shape: {sample_target.shape}")

    assert sample_img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image dimensions."
    assert sample_target.shape == (Config.NUM_CLASSES,), "Incorrect target dimensions."
    assert isinstance(sample_img, torch.Tensor), "Image is not a Tensor."
    assert isinstance(sample_target, torch.Tensor), "Target is not a Tensor."

    # Verify DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    batch_imgs, batch_targets = next(iter(train_loader))

    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert batch_targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch."
    print("Dataset and DataLoader verified successfully.")

    # 3. Model Verification
    print("\n[2/4] Verifying Model Architecture...")

    # Initialize Model (pretrained=False to avoid downloading weights during demo)
    model = ArtworkModel(model_name=Config.MODEL_NAME, pretrained=False)
    model.to(device)

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch."

    # Verify ModelEMA
    print("Verifying ModelEMA...")
    ema = ModelEMA(model, decay=0.99)
    # Perform a dummy update
    ema.update(model)
    # Check that EMA shadow model exists and has correct parameters
    assert len(list(ema.ema.parameters())) == len(
        list(model.parameters())
    ), "EMA parameter count mismatch."
    print("Model architecture verified successfully.")

    # 4. Training Loop Simulation
    print("\n[3/4] Verifying Training and Validation Loops...")

    # Setup Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # Use BCEWithLogitsLoss as per train.py
    pos_weight = torch.ones([Config.NUM_CLASSES], device=device) * Config.POS_WEIGHT
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(
        enabled=False
    )  # Disable AMP for simple demo stability, though supported

    # Run Training Step
    print("Running training step...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scaler, ema_model=ema
    )
    print(f"Train Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss returned NaN."
    assert train_loss > 0, "Training loss should be positive."

    # Run Validation Step
    print("Running validation step...")
    val_dataset = ArtworkDataset(
        val_df, mode="val", transforms=get_transforms("val", Config.IMG_SIZE)
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)

    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Predictions Shape: {val_preds.shape}")

    assert not np.isnan(val_loss), "Validation loss returned NaN."
    assert val_preds.shape == (
        len(val_df),
        Config.NUM_CLASSES,
    ), "Prediction shape mismatch."
    assert val_targets.shape == (
        len(val_df),
        Config.NUM_CLASSES,
    ), "Target shape mismatch."
    print("Training and validation loops verified successfully.")

    # 5. Metrics Verification
    print("\n[4/4] Verifying Metrics and Utilities...")

    # Mock data for F1 and Threshold Search
    # Scenario: 2 samples, 3 classes
    # Sample 1: Target [1, 0, 0], Pred [0.9, 0.1, 0.2] -> Correct
    # Sample 2: Target [0, 1, 1], Pred [0.1, 0.8, 0.4] -> Class 2 ambiguous

    mock_targets = np.array([[1, 0, 0], [0, 1, 1]])
    mock_preds = np.array([[0.9, 0.1, 0.2], [0.1, 0.8, 0.45]])

    # Check F1 Calculation (Threshold 0.5)
    # Preds > 0.5: [[1, 0, 0], [0, 1, 0]]
    # Sample 1 matches. Sample 2 misses class index 2.
    binary_preds = (mock_preds > 0.5).astype(int)
    f1 = calculate_f1(binary_preds, mock_targets)
    print(f"Calculated F1 (Thresh 0.5): {f1:.4f}")
    assert 0 <= f1 <= 1.0, "F1 score out of valid range."

    # Check Threshold Search
    # If we lower threshold to 0.4, the 0.45 prediction becomes 1, matching the target.
    # This should improve the score.
    best_score, best_thresh = find_best_threshold(mock_preds, mock_targets)
    print(f"Best F1: {best_score:.4f} at Threshold: {best_thresh:.2f}")

    assert best_score >= f1, "Optimized threshold should not yield lower score."
    # We expect the threshold to be lower than 0.45 to capture the third positive label
    # Note: Config search range starts at 0.01.

    print("Metrics verified successfully.")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    run_demonstration()
