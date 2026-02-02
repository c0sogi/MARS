import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# Import from provided library files
from library.config import (
    SEED,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    INPUT_DIR,
)
from library.utils import seed_everything, dice_coef
from library.dataset import ContrailDataset, get_transforms
from library.model import IsotropicConvNeXtUNet
from library.loss import HybridLoss
from library.engine import train_one_epoch, validate, inference


def perform_failure_analysis(model, dataloader, device):
    """
    Analyzes model performance on the validation set to find correlations
    between error magnitude and metadata features.
    """
    model.eval()

    # Collect per-sample metrics
    records = []
    dice_scores = []

    # Access the dataframe from the dataset
    val_df = dataloader.dataset.df.copy()

    # We need to ensure alignment. The dataloader is sequential (shuffle=False).
    # We will iterate and compute dice for each sample.

    # Pre-calculate metadata features for correlation
    # Timestamp is in seconds. Convert to hour.
    if "timestamp" in val_df.columns:
        val_df["datetime"] = pd.to_datetime(val_df["timestamp"], unit="s")
        val_df["hour"] = val_df["datetime"].dt.hour
    else:
        val_df["hour"] = 0

    print("\nStarting Failure Analysis...")

    current_idx = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            batch_size = images.size(0)

            for i in range(batch_size):
                # Compute Dice for this single sample
                y_true = masks[i].cpu().numpy()
                y_pred = preds[i].cpu().numpy()

                # Simple Dice calculation for single sample
                intersection = np.sum(y_true * y_pred)
                cardinality = np.sum(y_true) + np.sum(y_pred)

                if cardinality == 0:
                    score = 1.0
                else:
                    score = (2.0 * intersection) / (cardinality + 1e-6)

                dice_scores.append(score)
                current_idx += 1

    # Add scores to dataframe
    # Ensure lengths match (dataloader might drop last batch if drop_last=True, but usually validation doesn't)
    if len(dice_scores) != len(val_df):
        # Truncate df to match processed samples
        val_df = val_df.iloc[: len(dice_scores)]

    val_df["dice"] = dice_scores
    val_df["error_magnitude"] = 1.0 - val_df["dice"]

    # Calculate correlations
    features = ["row_min", "col_min", "hour"]
    correlations = {}

    print("-" * 30)
    print("Correlation between Error Magnitude (1-Dice) and Features:")
    for feat in features:
        if feat in val_df.columns:
            corr = val_df["error_magnitude"].corr(val_df[feat])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")
    print("-" * 30)


def main():
    # 1. Setup
    seed_everything(SEED)
    print(f"Device: {DEVICE}")

    # 2. Data Loading
    # Limit training samples for fast baseline
    TRAIN_SAMPLES = 5000
    print(f"Loading training data (limit: {TRAIN_SAMPLES})...")
    train_dataset = ContrailDataset(
        split="train",
        transform=get_transforms("train"),
        load_cached_data=True,
        max_samples=TRAIN_SAMPLES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Validation must use full set for correct metric
    print("Loading validation data (full set)...")
    val_dataset = ContrailDataset(
        split="validation",
        transform=get_transforms("validation"),
        load_cached_data=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing IsotropicConvNeXtUNet...")
    model = IsotropicConvNeXtUNet().to(DEVICE)

    criterion = HybridLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_dice = 0.0

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_dice = validate(model, val_loader, DEVICE)

        # Scheduler step
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.6f}"
        )

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  New best model saved! ({val_dice:.6f})")

    # 5. Final Reporting & Failure Analysis
    print("\n" + "=" * 40)
    print(f"Final Validation Metric: {best_dice}")
    print("=" * 40)

    # Load best model for analysis and inference
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    # Failure Analysis
    perform_failure_analysis(model, val_loader, DEVICE)

    # 6. Submission
    THRESHOLD = 0.5910660985501295

    if best_dice > THRESHOLD:
        print(
            f"\nValidation score ({best_dice:.6f}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )

        test_dataset = ContrailDataset(
            split="test", transform=get_transforms("test"), load_cached_data=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        inference(model, test_loader, DEVICE, SUBMISSION_PATH)
    else:
        print(
            f"\nValidation score ({best_dice:.6f}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
