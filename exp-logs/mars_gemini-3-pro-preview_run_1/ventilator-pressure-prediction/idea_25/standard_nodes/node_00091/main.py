import sys
import os
import torch
import numpy as np
import pandas as pd

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device, save_checkpoint, load_checkpoint
from library.data import get_dataloaders
from library.model import VentilatorModel, train_epoch, validate, generate_submission


def run():
    # --- Configuration ---
    # Override defaults for a fast baseline execution that fits within time limits
    Config.EPOCHS = 30  # Optimized for 3h runtime and convergence
    Config.BATCH_SIZE = 512  # Keep high for stable updates and speed on A100
    Config.DEBUG = False  # Use full data to ensure we meet the accuracy threshold

    # Set seeds for reproducibility
    set_seed(Config.SEED)
    device = get_device()

    print(f"Running Experiment: {Config.EXPERIMENT_ID}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Device: {device}")

    # --- Data Loading ---
    print("Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Determine input dimension
    input_dim = train_loader.dataset.X.shape[-1]
    print(f"Input Features: {input_dim}")

    # --- Model Initialization ---
    model = VentilatorModel(input_dim=input_dim).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # --- Training Loop ---
    print("Starting Training...")
    best_mae = float("inf")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # Validate (using provided approximate batch-avg function for monitoring)
        val_loss, val_mae = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MAE: {val_mae:.5f}"
        )

        # Checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_mae,
                },
                is_best=True,
                filename="model_best.pth",
            )

    print(f"Training Complete. Best Approximate Val MAE: {best_mae:.6f}")

    # --- Final Evaluation & Failure Analysis ---
    print("Loading Best Model for Final Evaluation...")
    load_checkpoint(model, filename="model_best.pth")
    model.eval()

    total_ae = 0.0
    total_count = 0

    # Store data for failure analysis
    # We'll accumulate errors and features on CPU to avoid OOM
    all_errors = []
    all_features = []

    print("Computing Final Metrics and Analyzing Failures...")
    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Predict
            pred, _ = model(X)  # pred: (Batch, 80, 1)

            # Flatten tensors
            pred = pred.view(-1)
            y = y.view(-1)
            u_out = u_out.view(-1)
            X_flat = X.view(-1, input_dim)

            # Mask for inspiratory phase (u_out == 0)
            mask = u_out == 0

            if mask.sum() > 0:
                # Calculate Absolute Error
                batch_errors = torch.abs(pred[mask] - y[mask])

                # Accumulate for Global MAE
                total_ae += batch_errors.sum().item()
                total_count += mask.sum().item()

                # Accumulate for Failure Analysis
                # Move to CPU numpy
                all_errors.append(batch_errors.cpu().numpy())
                all_features.append(X_flat[mask].cpu().numpy())

    # 1. Final Validation Metric
    final_metric = total_ae / total_count if total_count > 0 else float("inf")
    print(f"Final Validation Metric: {final_metric}")

    # 2. Failure Analysis
    print("Performing Failure Analysis (Correlation of Error Magnitude vs Features)...")
    if len(all_errors) > 0:
        errors_concat = np.concatenate(all_errors)
        features_concat = np.concatenate(all_features, axis=0)

        correlations = []
        for i in range(input_dim):
            feat_vals = features_concat[:, i]
            # Check variance to avoid division by zero
            if np.std(feat_vals) > 1e-9:
                corr = np.corrcoef(errors_concat, feat_vals)[0, 1]
                correlations.append((i, corr))
            else:
                correlations.append((i, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Top 10 Correlations (Feature Index: Correlation):")
        for idx, corr in correlations[:10]:
            print(f"  Feature {idx}: {corr:.4f}")
    else:
        print("No inspiratory samples found for failure analysis.")

    # --- Submission ---
    THRESHOLD = 0.2164510190486908
    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating Submission...")
        generate_submission(model, test_loader, device)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission Skipped.")


if __name__ == "__main__":
    run()
