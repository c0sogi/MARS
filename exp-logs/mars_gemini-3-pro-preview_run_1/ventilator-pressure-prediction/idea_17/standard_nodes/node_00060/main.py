import sys
import os
import time
import numpy as np
import torch
import torch.optim as optim
import pandas as pd

# Append current directory to path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import prepare_data
from library.model import WideStateNet
from library.loss import CompositeMaskedL1Loss
from library.train import train_epoch, validate_epoch
from library.inference import predict


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration for Fast Baseline
    # --------------------------------------------------------------------------
    # Override Config to ensure fast execution
    Config.DEBUG = True
    Config.EPOCHS = 2

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")
    print(f"Fast Baseline Configuration: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}")

    # --------------------------------------------------------------------------
    # 2. Data Preparation (Training Phase)
    # --------------------------------------------------------------------------
    print("\n=== Preparing Data for Training (Subsampled) ===")
    # Force load_cached_data=False to ensure we generate a subsampled dataset
    # based on Config.DEBUG=True, ignoring any existing full cache.
    train_loader, val_loader_sub, _, feature_names = prepare_data(
        load_cached_data=False
    )

    input_dim = len(feature_names)
    print(f"Features: {feature_names}")
    print(f"Input Dimension: {input_dim}")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n=== Initializing Model ===")
    model = WideStateNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n=== Starting Training ===")
    criterion = CompositeMaskedL1Loss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
    )

    best_val_mae = float("inf")
    model_save_path = os.path.join(Config.WORKING_DIR, "model.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Validate (on subsample)
        val_mae = validate_epoch(model, val_loader_sub, criterion, device)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE (Sub): {val_mae:.6f} | Time: {duration:.2f}s"
        )

        # Save Best Model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Model saved (MAE: {val_mae:.6f})")

    # --------------------------------------------------------------------------
    # 5. Full Validation & Failure Analysis
    # --------------------------------------------------------------------------
    print("\n=== Preparing Data for Full Validation ===")
    # Switch to full mode
    Config.DEBUG = False

    # Reload data with load_cached_data=False to overwrite the subsampled cache with full data
    # This ensures we validate on the entire hold-out set as required.
    _, val_loader_full, _, _ = prepare_data(load_cached_data=False)

    # Load best model weights
    print("Loading best model for full validation...")
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Calculate Final Metric
    print("Computing metrics on full validation set...")
    final_val_mae = validate_epoch(model, val_loader_full, criterion, device)
    print(f"Final Validation Metric: {final_val_mae}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    model.eval()

    all_errors = []
    feature_data = {name: [] for name in feature_names}

    # Collect errors and feature values
    with torch.no_grad():
        for batch in val_loader_full:
            x = batch["x"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Predict
            final_pred, _ = model(x, u_out)

            # Calculate absolute error
            abs_err = torch.abs(final_pred - y)

            # Filter for inspiratory phase (u_out == 0)
            mask = u_out == 0

            if mask.sum() > 0:
                valid_err = abs_err[mask].cpu().numpy()
                valid_x = x[mask].cpu().numpy()

                all_errors.extend(valid_err)
                for i, name in enumerate(feature_names):
                    feature_data[name].extend(valid_x[:, i])

    # Compute Correlations
    errors_np = np.array(all_errors)
    print("Correlation between Error Magnitude and Features:")
    for name in feature_names:
        feats_np = np.array(feature_data[name])
        # Avoid correlation calculation if constant feature (std dev is 0)
        if len(feats_np) > 0 and np.std(feats_np) > 1e-9:
            corr = np.corrcoef(errors_np, feats_np)[0, 1]
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: NaN (Constant or Empty)")

    # --------------------------------------------------------------------------
    # 6. Submission
    # --------------------------------------------------------------------------
    threshold = 0.2164510190486908
    if final_val_mae < threshold:
        print(
            f"\nValidation MAE ({final_val_mae}) < Threshold ({threshold}). Generating submission..."
        )
        # We use load_cached_data=True because we just generated the full cache in Step 5
        predict(load_cached_data=True)
    else:
        print(
            f"\nValidation MAE ({final_val_mae}) >= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
