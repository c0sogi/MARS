import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from library
from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric
from library.dataset import load_data, VentilatorDataset
from library.model import FPBC_BiLSTM
from library.inference import run_inference

# ==========================================
# Configuration Overrides for Fast Baseline
# ==========================================
# Limit epochs to ensure completion within the time limit while allowing convergence
Config.EPOCHS = 15
Config.T_MAX = 15


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    # Load cached data if available to save time
    train_dataset, val_dataset, test_dataset = load_data(load_cached_data=True)

    # Limit maximum number of training samples (Constraint Check)
    # We restrict training to 40,000 breaths to ensure a fast baseline execution
    MAX_TRAIN_BREATHS = 40000
    if len(train_dataset) > MAX_TRAIN_BREATHS:
        print(
            f"Limiting training data from {len(train_dataset)} to {MAX_TRAIN_BREATHS} breaths."
        )
        train_dataset.X = train_dataset.X[:MAX_TRAIN_BREATHS]
        train_dataset.u_out = train_dataset.u_out[:MAX_TRAIN_BREATHS]
        train_dataset.y = train_dataset.y[:MAX_TRAIN_BREATHS]
        train_dataset.num_breaths = MAX_TRAIN_BREATHS

    # Create DataLoaders
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
    print("Initializing model...")
    model = FPBC_BiLSTM().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = WeightedL1Loss()

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_val_mae = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        num_batches = 0

        for X, u_out, y in train_loader:
            X, u_out, y = X.to(device), u_out.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X, u_out)
            loss = criterion(preds, y, u_out)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)
            optimizer.step()

            train_loss_sum += loss.item()
            num_batches += 1

        avg_train_loss = train_loss_sum / num_batches

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X, u_out, y in val_loader:
                X, u_out, y = X.to(device), u_out.to(device), y.to(device)

                preds = model(X, u_out)
                loss = criterion(preds, y, u_out)
                mae = compute_metric(preds, y, u_out)

                val_loss_sum += loss.item()
                val_mae_sum += mae
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches
        avg_val_mae = val_mae_sum / val_batches

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val MAE: {avg_val_mae:.6f}"
        )

        # Save Best Model
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # 5. Final Reporting
    # Print the exact metric required
    print(f"Final Validation Metric: {best_val_mae}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis on Validation Set...")
    # Reload best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    model.eval()

    all_abs_errors = []
    all_inputs = []

    with torch.no_grad():
        for X, u_out, y in val_loader:
            X, u_out, y = X.to(device), u_out.to(device), y.to(device)
            preds = model(X, u_out)

            # Calculate Absolute Error
            abs_error = torch.abs(preds - y)

            # Filter for Inspiratory Phase (u_out == 0) for analysis
            mask = u_out == 0

            # Flatten and filter
            err_flat = abs_error[mask].cpu().numpy()
            inputs_flat = X[mask].cpu().numpy()

            all_abs_errors.append(err_flat)
            all_inputs.append(inputs_flat)

    # Concatenate
    all_abs_errors = np.concatenate(all_abs_errors)
    all_inputs = np.concatenate(all_inputs)

    # Compute Correlations
    print("Correlation between Absolute Error and Input Features (Inspiratory Phase):")
    feature_names = Config.ALL_FEATURES

    for i, feat_name in enumerate(feature_names):
        feat_values = all_inputs[:, i]
        # Check for constant values (std=0) to avoid NaN correlation
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr = np.corrcoef(all_abs_errors, feat_values)[0, 1]
        print(f"  {feat_name}: {corr:.4f}")

    # 7. Submission Logic
    # Threshold from instructions
    SUBMISSION_THRESHOLD = 0.1619843989610672

    if best_val_mae < SUBMISSION_THRESHOLD:
        print(
            f"\nValidation Metric ({best_val_mae}) is lower than threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission...")
        # Run inference using the provided library function
        # It handles loading the best model and generating the CSV
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation Metric ({best_val_mae}) is NOT lower than threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
