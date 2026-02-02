import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_train_val_loaders, get_test_loader
from library.model import MultiTaskModel
from library.engine import fit, inference_and_submit, evaluate
from library.loss import MultiTaskLoss


def run_failure_analysis(model, val_loader, device):
    """
    Performs failure analysis on the validation set.
    Calculates the correlation between model error and input features.
    """
    print("\nRunning Failure Analysis...")
    model.eval()

    errors = []
    means = []
    stds = []

    # Define loss functions with reduction='none' to get per-sample error
    seg_criterion = nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            # Forward pass
            cls_logits, seg_logits = model(images)

            # --- Calculate Per-Sample Classification Loss ---
            # SoftTargetCrossEntropy formula: - sum(target * log_softmax(input))
            log_probs = F.log_softmax(cls_logits, dim=-1)
            # Sum over classes, result is (B,)
            cls_loss_per_sample = torch.sum(-labels * log_probs, dim=-1)

            # --- Calculate Per-Sample Segmentation Loss ---
            # BCE per pixel: (B, 1, H, W)
            seg_loss_pixel = seg_criterion(seg_logits, masks.float())
            # Average over spatial dims to get (B,)
            seg_loss_per_sample = seg_loss_pixel.mean(dim=(1, 2, 3))

            # Total weighted error magnitude
            total_error = (Config.LOSS_WEIGHTS["class"] * cls_loss_per_sample) + (
                Config.LOSS_WEIGHTS["seg"] * seg_loss_per_sample
            )

            # --- Collect Statistics ---
            # Move to CPU numpy
            batch_errors = total_error.cpu().numpy()

            # Image stats (images are normalized, but relative stats still valid)
            # images shape: (B, 3, H, W)
            batch_means = images.mean(dim=(1, 2, 3)).cpu().numpy()
            batch_stds = images.std(dim=(1, 2, 3)).cpu().numpy()

            errors.extend(batch_errors)
            means.extend(batch_means)
            stds.extend(batch_stds)

    errors = np.array(errors)
    means = np.array(means)
    stds = np.array(stds)

    # Calculate Correlations
    corr_mean, _ = pearsonr(errors, means)
    corr_std, _ = pearsonr(errors, stds)

    print(f"Correlation (Error vs Input Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Input Std Dev): {corr_std:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Adjust Config for Fast Baseline
    # 12 Epochs is sufficient for a baseline on this dataset size with A100
    Config.EPOCHS = 12

    print(f"Starting execution on device: {Config.DEVICE}")
    print(f"Training for {Config.EPOCHS} epochs.")

    # 2. Data Loading
    # Load cached data to save time
    train_loader, val_loader = get_train_val_loaders(load_cached_data=True)
    test_loader = get_test_loader(load_cached_data=True)

    # 3. Model Initialization
    model = MultiTaskModel(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(Config.DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training
    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        Config.DEVICE,
        Config.EPOCHS,
    )

    # 6. Evaluation
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH))

    # Compute Final Metric
    _, val_map = evaluate(model, val_loader, Config.DEVICE)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_map}")

    # 7. Failure Analysis
    run_failure_analysis(model, val_loader, Config.DEVICE)

    # 8. Submission
    threshold = 0.49944536565378
    if val_map > threshold:
        print(
            f"\nValidation metric ({val_map}) > threshold ({threshold}). Generating submission..."
        )
        inference_and_submit(model, test_loader, Config.DEVICE)
    else:
        print(
            f"\nValidation metric ({val_map}) <= threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
