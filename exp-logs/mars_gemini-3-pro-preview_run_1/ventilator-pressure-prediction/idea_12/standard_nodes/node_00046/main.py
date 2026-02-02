import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import get_data_loaders
from library.model import VentilatorModel
from library.train import train_fn, valid_fn, inference_fn
from library.utils import get_device


def analyze_failures(model, loader, device, feature_cols):
    """
    Computes absolute errors on the validation set (inspiratory phase)
    and correlates them with input features.
    """
    model.eval()
    all_inputs = []
    all_preds = []
    all_targets = []
    all_u_out = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)

            # Move to CPU
            all_inputs.append(x.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_u_out.append(u_out.cpu().numpy())

    # Concatenate
    # Shapes: (N_batches * Batch, Seq_Len, Features) or (N_batches * Batch, Seq_Len)
    inputs = np.concatenate(all_inputs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    u_out = np.concatenate(all_u_out, axis=0)

    # Flatten for correlation analysis
    inputs_flat = inputs.reshape(-1, inputs.shape[-1])
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    u_out_flat = u_out.flatten()

    # Filter for Inspiratory Phase (u_out == 0)
    mask = u_out_flat == 0

    inputs_insp = inputs_flat[mask]
    preds_insp = preds_flat[mask]
    targets_insp = targets_flat[mask]

    # Calculate Error Magnitude
    error_magnitude = np.abs(preds_insp - targets_insp)

    # Create DataFrame
    df = pd.DataFrame(inputs_insp, columns=feature_cols)
    df["error_magnitude"] = error_magnitude

    # Compute Correlation
    correlations = df.corr()["error_magnitude"].sort_values(ascending=False)

    print("\n=== Failure Analysis: Correlation with Error Magnitude ===")
    print(correlations)


def main():
    # 1. Configuration
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=True)

    # 3. Model Setup
    model = VentilatorModel()
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    total_steps = Config.EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # 4. Training
    best_mae = float("inf")
    model_save_path = os.path.join(Config.WORKING_DIR, "model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
        val_mae = valid_fn(model, val_loader, device)

        # Print progress
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.5f} | Val MAE: {val_mae:.5f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), model_save_path)

    # 5. Final Validation
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    final_metric = valid_fn(model, val_loader, device)

    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    analyze_failures(model, val_loader, device, Config.FEATURE_COLS)

    # 7. Submission
    THRESHOLD = 0.2164510190486908
    if final_metric < THRESHOLD:
        inference_fn(model, test_loader, device)
    else:
        print(f"Metric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
