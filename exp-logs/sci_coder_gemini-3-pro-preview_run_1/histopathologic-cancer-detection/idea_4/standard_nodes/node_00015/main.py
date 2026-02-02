import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import TumorDataset
from library.model import get_model
from library.train import train_one_epoch, validate
from library.inference import run_inference


def perform_failure_analysis(model, loader, device):
    """
    Analyzes the correlation between model error and image meta-features
    (brightness and contrast) on the validation set.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()

    errors = []
    brightness_vals = []
    contrast_vals = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Get predictions
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Calculate Error Magnitude
            batch_errors = torch.abs(preds - labels).cpu().numpy().flatten()
            errors.extend(batch_errors.tolist())

            # Calculate Image Stats (Meta-features)
            # We calculate on the normalized tensor, which is a linear proxy for the original
            # Shape: (B, C, H, W) -> Mean/Std across (C, H, W) or just (H, W)
            # We'll take mean across all dims per sample for brightness
            # and std across all dims per sample for contrast

            # Flatten spatial and channel dims for stats: (B, -1)
            flat_images = images.view(images.size(0), -1)

            batch_brightness = torch.mean(flat_images, dim=1).cpu().numpy()
            batch_contrast = torch.std(flat_images, dim=1).cpu().numpy()

            brightness_vals.extend(batch_brightness.tolist())
            contrast_vals.extend(batch_contrast.tolist())

    # Calculate Correlations
    if len(errors) > 0:
        r_bright, _ = stats.pearsonr(errors, brightness_vals)
        r_contrast, _ = stats.pearsonr(errors, contrast_vals)

        print(f"Correlation between Error and Brightness: {r_bright:.4f}")
        print(f"Correlation between Error and Contrast:   {r_contrast:.4f}")
    else:
        print("No validation data available for failure analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    # Initialize Datasets
    train_dataset = TumorDataset(train_df, split="train")
    val_dataset = TumorDataset(val_df, split="val")

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = get_model()
    model = model.to(Config.DEVICE)

    # 4. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} [{elapsed:.1f}s] - "
            f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                filepath=Config.CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # 6. Final Evaluation & Requirements
    print("-" * 30)
    # Required Output Format
    print(f"Final Validation Metric: {best_auc}")
    print("-" * 30)

    # Load best model for analysis and inference
    print(f"Loading best model from {Config.CHECKPOINT_PATH}")
    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # Failure Analysis
    perform_failure_analysis(model, val_loader, Config.DEVICE)

    # 7. Submission
    # Threshold check
    THRESHOLD = 0.9849192531860572

    if best_auc > THRESHOLD:
        print(
            f"\nValidation metric ({best_auc}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        run_inference(
            checkpoint_path=Config.CHECKPOINT_PATH,
            output_path=Config.SUBMISSION_PATH,
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
        )
    else:
        print(
            f"\nValidation metric ({best_auc}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
