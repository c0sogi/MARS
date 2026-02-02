import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.dataset import prepare_data
from library.model import LGRHNet
from library.train import train_one_epoch, evaluate, predict_and_submit, MaskedL1Loss
from library.utils import seed_everything, get_device


def main():
    # 1. Configuration
    # We limit epochs to 5 to ensure execution within the 1-hour limit while using the full dataset.
    config = Config(epochs=5, debug=False, load_cached_data=True)

    # Set seeds for reproducibility
    seed_everything(config.seed)

    # Detect device
    device = get_device()
    print(f"Running on device: {device}")

    # 2. Data Preparation
    print("Preparing data...")
    train_loader, val_loader, test_loader = prepare_data(config)

    # 3. Model Initialization
    # Get input dimension from a sample batch
    sample_X, _, _ = next(iter(train_loader))
    input_dim = sample_X.shape[2]
    print(f"Input feature dimension: {input_dim}")

    model = LGRHNet(input_dim=input_dim, config=config)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.eta_min
    )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.max_grad_norm
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  -> New best model saved! Score: {best_val_loss:.6f}")

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation & Failure Analysis...")

    # Load best model
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []
    val_u_out = []
    val_features = []

    with torch.no_grad():
        for X, y, u_out in val_loader:
            X = X.to(device)
            y = y.to(device)
            u_out = u_out.to(device)

            # Predict
            pred = model(X).squeeze(-1)

            # Store data for analysis
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
            val_u_out.append(u_out.cpu().numpy())
            val_features.append(X.cpu().numpy())

    # Concatenate all batches
    val_preds = np.concatenate(val_preds, axis=0)  # (N_breaths, Seq_Len)
    val_targets = np.concatenate(val_targets, axis=0)  # (N_breaths, Seq_Len)
    val_u_out = np.concatenate(val_u_out, axis=0)  # (N_breaths, Seq_Len)
    val_features = np.concatenate(val_features, axis=0)  # (N_breaths, Seq_Len, N_feats)

    # Flatten arrays for metric calculation
    preds_flat = val_preds.flatten()
    targets_flat = val_targets.flatten()
    u_out_flat = val_u_out.flatten()
    features_flat = val_features.reshape(-1, input_dim)

    # Create mask (Inspiratory phase only: u_out == 0)
    # Note: u_out is float, so compare with epsilon or cast
    mask = u_out_flat < 0.5

    # Calculate masked errors
    errors = np.abs(preds_flat - targets_flat)
    masked_errors = errors[mask]

    # Compute Final Metric
    final_metric = np.mean(masked_errors)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation
    print("\nFailure Analysis - Feature Correlations with Error Magnitude:")
    masked_features = features_flat[mask]

    # Calculate correlation for each feature
    # We don't have feature names explicitly here, so we use indices
    for i in range(input_dim):
        feat_values = masked_features[:, i]
        # Handle constant features (std=0) to avoid NaN correlation
        if np.std(feat_values) > 1e-9:
            corr = np.corrcoef(feat_values, masked_errors)[0, 1]
            print(f"Feature Index {i}: Correlation {corr:.4f}")
        else:
            print(f"Feature Index {i}: Correlation NaN (Constant Feature)")

    # 7. Submission Logic
    THRESHOLD = 0.16391726930343686
    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_submit(model, test_loader, config, device)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
