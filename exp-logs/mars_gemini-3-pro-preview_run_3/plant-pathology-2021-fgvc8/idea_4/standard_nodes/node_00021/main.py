import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.dataset import AppleDataset, get_transforms
from library.model import AppleConvNeXt
from library.engine import train_model
from library.inference import run_inference


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates F1 score and correlations between error and image statistics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Store stats for correlation analysis
    errors = []
    brightness_vals = []
    contrast_vals = []

    print("\nStarting Failure Analysis on Validation Set...")

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Predictions for F1
            preds_bin = (probs > Config.CONF_THRESHOLD).float()
            all_preds.append(preds_bin.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Failure Analysis Metrics
            # Error: Mean Absolute Error per sample (averaged across classes)
            # Shape: (B, C) -> (B,)
            batch_errors = torch.abs(targets - probs).mean(dim=1).cpu().numpy()
            errors.extend(batch_errors)

            # Image Stats (from normalized tensor)
            # images: (B, 3, H, W)
            # Brightness: Mean value across channel, height, width
            batch_brightness = images.mean(dim=[1, 2, 3]).cpu().numpy()
            brightness_vals.extend(batch_brightness)

            # Contrast: Std value across channel, height, width
            batch_contrast = images.std(dim=[1, 2, 3]).cpu().numpy()
            contrast_vals.extend(batch_contrast)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate Final Metric
    final_f1 = f1_score(all_targets, all_preds, average="macro")

    # Calculate Correlations
    errors = np.array(errors)
    brightness_vals = np.array(brightness_vals)
    contrast_vals = np.array(contrast_vals)

    corr_brightness = np.corrcoef(errors, brightness_vals)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast_vals)[0, 1]

    print("-" * 40)
    print(f"Failure Analysis Report:")
    print(f"Correlation (Error vs Brightness): {corr_brightness}")
    print(f"Correlation (Error vs Contrast):   {corr_contrast}")
    print("-" * 40)

    return final_f1


def main():
    # 1. Setup
    # Override Config for Fast Baseline Execution while maintaining performance
    Config.EPOCHS = 15  # Sufficient for fine-tuning ConvNeXt-Small

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading Metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    print(f"Train samples: {len(df_train)}")
    print(f"Val samples:   {len(df_val)}")

    train_dataset = AppleDataset(df_train, transforms=get_transforms(data="train"))
    val_dataset = AppleDataset(df_val, transforms=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = AppleConvNeXt(pretrained=True)
    model.to(device)

    # Optimizer & Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training
    print("Starting Training...")
    # train_model handles the loop, validation, and saving best_model.pth
    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs=Config.EPOCHS,
    )

    # 5. Validation & Failure Analysis
    print("\nLoading best model for analysis...")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Re-initialize model to ensure clean state
    best_model = AppleConvNeXt(pretrained=False)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)

    # Run Analysis
    final_f1 = analyze_failures(best_model, val_loader, device)

    # REQUIRED: Print Final Metric
    print(f"Final Validation Metric: {final_f1}")

    # 6. Submission Logic
    # Threshold from instructions
    THRESHOLD = 0.9096474096681636

    if final_f1 > THRESHOLD:
        print(
            f"\nMetric ({final_f1}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference()
    else:
        print(f"\nMetric ({final_f1}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    main()
