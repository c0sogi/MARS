import sys
import os
import time
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DSPRNet
from library.train import LaplaceNLLLoss, train_one_epoch, validate, generate_submission


def main():
    # --- Configuration ---
    # Cite {solution_lesson_node_00100}: Increase epochs to allow convergence
    Config.EPOCHS = 20

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- Data Loading ---
    print("Loading Data...")
    # Using load_cached_data=True to utilize pre-processed data in ./working
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)
    stats = train_loader.dataset.stats

    # --- Model Setup ---
    print("Initializing DSPRNet...")
    model = DSPRNet().to(device)

    # Optimizer setup (Differential Learning Rates)
    # Separate backbone parameters from the rest
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceNLLLoss()

    # --- Training Loop ---
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device, stats)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        # Checkpoint
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {train_loss:.6f} | Val Metric: {val_metric:.8f} | Time: {elapsed:.1f}s"
        )

    print(f"Training finished. Best Validation Metric: {best_metric}")

    # --- Failure Analysis & Final Metrics ---
    print("\nRunning Failure Analysis on Best Model...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Re-run validation to gather data for analysis
    all_true = []
    all_pred = []
    all_features = []

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    with torch.no_grad():
        for images, features, targets in val_loader:
            images = images.to(device)
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass (returns normalized mean and sigma)
            # We only need mean for error magnitude analysis
            mu_norm, _ = model(images, features)

            # Inverse Transform
            pred_fvc = mu_norm.cpu().numpy() * fvc_std + fvc_mean
            true_fvc = targets.cpu().numpy() * fvc_std + fvc_mean

            all_true.extend(true_fvc)
            all_pred.extend(pred_fvc)
            all_features.extend(features.cpu().numpy())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_features = np.array(all_features)

    # Calculate Absolute Error
    errors = np.abs(all_true - all_pred)

    # Features vector structure: [BaseFVC_norm, Time, Age_norm, Sex, Smoke]
    # We correlate error with continuous features
    feat_base_fvc = all_features[:, 0]
    feat_time = all_features[:, 1]
    feat_age = all_features[:, 2]

    # Compute correlations
    corr_base = np.corrcoef(errors, feat_base_fvc)[0, 1]
    corr_time = np.corrcoef(errors, feat_time)[0, 1]
    corr_age = np.corrcoef(errors, feat_age)[0, 1]

    print(f"Correlation (Error vs Baseline FVC): {corr_base}")
    print(f"Correlation (Error vs Relative Time): {corr_time}")
    print(f"Correlation (Error vs Age): {corr_age}")

    # Print Final Metric as required
    print(f"Final Validation Metric: {best_metric}")

    # --- Submission ---
    threshold = -6.573619738753321
    if best_metric > threshold:
        print(
            f"Metric ({best_metric}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(model, test_loader, device, stats)
    else:
        print(f"Metric ({best_metric}) <= Threshold ({threshold}). Submission skipped.")


if __name__ == "__main__":
    main()
