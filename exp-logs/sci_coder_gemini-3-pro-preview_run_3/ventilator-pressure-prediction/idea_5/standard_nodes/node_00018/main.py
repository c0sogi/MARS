import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data import get_dataloaders
from library.model import CAPNet, train_epoch, validate, generate_submission


def run_failure_analysis(model, val_loader, device):
    """
    Performs inference on validation set to compute final metric and
    analyzes error correlation with features.
    """
    model.eval()
    preds = []
    targets = []
    u_outs = []
    inputs = []

    # Collect predictions and data
    with torch.no_grad():
        for x, u_out, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            y_pred = model(x)

            preds.append(y_pred.cpu())
            targets.append(y.cpu())
            u_outs.append(u_out.cpu())
            inputs.append(x.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    u_outs = torch.cat(u_outs)
    inputs = torch.cat(inputs)  # Shape: (N, Seq_Len, Features)

    # 1. Compute Final Metric
    final_mae = compute_metric(preds, targets, u_outs)
    print(f"Final Validation Metric: {final_mae}")

    # 2. Failure Analysis
    # Flatten for analysis
    preds_flat = preds.flatten().numpy()
    targets_flat = targets.flatten().numpy()
    u_outs_flat = u_outs.flatten().numpy()
    inputs_flat = inputs.reshape(-1, inputs.shape[-1]).numpy()

    # Filter for inspiratory phase only (where metric is computed)
    mask = u_outs_flat == 0

    if mask.sum() > 0:
        error = np.abs(preds_flat[mask] - targets_flat[mask])
        features_masked = inputs_flat[mask]

        # Feature columns as defined in library.data.get_dataloaders
        # Order: time_step, u_in, u_out, R, C, area, u_in_diff, R_u_in, vol_C
        feature_names = [
            "time_step",
            "u_in",
            "u_out",
            "R",
            "C",
            "area",
            "u_in_diff",
            "R_u_in",
            "vol_C",
        ]

        print("\n=== Failure Analysis (Error Correlation) ===")
        analysis_data = pd.DataFrame(features_masked, columns=feature_names)
        analysis_data["error"] = error

        correlations = analysis_data.corr()["error"].sort_values(ascending=False)
        print(correlations)
        print("============================================\n")

    return final_mae


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Ensure fresh start
    Config.CLEAN_START = True
    Config.LOAD_CACHE = True

    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    # Clean cache if requested
    if Config.CLEAN_START:
        print("Cleaning working directory...")
        for f in os.listdir(Config.WORKING_DIR):
            if f.endswith(".npy") or f.endswith(".pth"):
                try:
                    os.remove(os.path.join(Config.WORKING_DIR, f))
                except:
                    pass

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=Config.LOAD_CACHE
    )

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    model = CAPNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    best_mae = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_mae = validate(model, val_loader, device)

        # Scheduler
        scheduler.step(val_mae)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
        )

        # Checkpoint
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            early_stop_counter += 1

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # ---------------------------------------------------------
    # 5. Final Validation & Analysis
    # ---------------------------------------------------------
    print("Loading best model for analysis...")
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    final_metric = run_failure_analysis(model, val_loader, device)

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = 0.5943868160247803

    if final_metric < THRESHOLD:
        print(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")
        generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission generation.")


if __name__ == "__main__":
    main()
