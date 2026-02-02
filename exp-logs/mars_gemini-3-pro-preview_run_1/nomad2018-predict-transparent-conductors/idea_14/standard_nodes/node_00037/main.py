import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# Import from library
from library.config import Config, set_seed
from library.data_loader import get_data_loaders
from library.model import DCPDS_Model
from library.trainer import train_epoch, validate_epoch, generate_predictions


def calculate_validation_metrics(model, val_loader, device):
    """
    Computes predictions and targets in log-space to calculate
    Column-wise Root Mean Squared Logarithmic Error (RMSLE).
    """
    model.eval()
    preds_list = []
    targets_list = []
    global_feats_list = []

    with torch.no_grad():
        for batch in val_loader:
            atomic_feats, batch_idx, global_feats, targets, _ = batch

            atomic_feats = atomic_feats.to(device)
            batch_idx = batch_idx.to(device)
            global_feats = global_feats.to(device)

            outputs = model(atomic_feats, batch_idx, global_feats)

            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())
            global_feats_list.append(global_feats.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    global_feats = np.concatenate(global_feats_list, axis=0)

    # Calculate Column-wise RMSLE
    # Note: Model outputs and targets are already log1p transformed.
    # So RMSLE is simply RMSE of these values.
    mse_per_col = np.mean((preds - targets) ** 2, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)
    mean_rmsle = np.mean(rmsle_per_col)

    return mean_rmsle, preds, targets, global_feats


def perform_failure_analysis(preds, targets, global_feats):
    """
    Correlates prediction error with global features to identify failure modes.
    """
    # Calculate Mean Absolute Error per sample in log space
    errors = np.mean(np.abs(preds - targets), axis=1)

    # Global feature names based on data_loader construction:
    # Lat Len (3) + Lat Ang (3) + Vol (1) + Density (1) + Stoich (3) + N_atoms (1)
    feat_names = [
        "lv1",
        "lv2",
        "lv3",
        "alpha",
        "beta",
        "gamma",
        "volume",
        "density",
        "pct_Al",
        "pct_Ga",
        "pct_In",
        "num_atoms",
    ]

    print("\nFailure Analysis (Correlation between Error and Global Features):")
    print("-" * 60)
    print(f"{'Feature':<20} | {'Correlation':<12} | {'P-Value':<12}")
    print("-" * 60)

    correlations = []
    for i, name in enumerate(feat_names):
        if i < global_feats.shape[1]:
            feat_vals = global_feats[:, i]
            # Handle constant features (std=0)
            if np.std(feat_vals) > 1e-9:
                corr, p_val = pearsonr(feat_vals, errors)
                correlations.append((name, corr, p_val))
            else:
                correlations.append((name, 0.0, 1.0))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, corr, p_val in correlations:
        print(f"{name:<20} | {corr:<12.4f} | {p_val:<12.4e}")
    print("-" * 60)


def main():
    # 1. Setup
    # Override Config for fast baseline execution
    Config.NUM_EPOCHS = 50  # Limit epochs for speed
    Config.BATCH_SIZE = 128  # Increase batch size for speed

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 3. Model Initialization
    print("Initializing DCPDS Model...")
    model = DCPDS_Model(Config).to(device)

    # 4. Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    best_val_metric = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (Loss)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        # Scheduler
        scheduler.step(val_loss)

        # Checkpointing based on Loss (MSE)
        # Note: We use val_loss for early stopping, but we will compute the exact metric later
        if val_loss < best_val_metric:
            best_val_metric = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f} | Time: {time.time()-epoch_start:.1f}s"
            )

    print(f"Training finished in {time.time() - start_time:.1f}s")

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    mean_rmsle, val_preds, val_targets, val_global_feats = calculate_validation_metrics(
        model, val_loader, device
    )

    print(f"Final Validation Metric: {mean_rmsle}")

    # 6. Failure Analysis
    perform_failure_analysis(val_preds, val_targets, val_global_feats)

    # 7. Submission
    THRESHOLD = 0.05479004207787702

    if mean_rmsle < THRESHOLD:
        print(f"Validation metric {mean_rmsle} < {THRESHOLD}. Generating submission...")
        generate_predictions(model, test_loader, device, Config.SUBMISSION_PATH)
    else:
        print(
            f"Validation metric {mean_rmsle} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
