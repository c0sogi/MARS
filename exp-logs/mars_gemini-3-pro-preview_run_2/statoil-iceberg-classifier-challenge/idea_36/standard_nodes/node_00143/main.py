import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from library
import library.config as config
from library.utils import set_seed, EarlyStopping
from library.model import RDPWBN, load_and_process_data
from library.data_loader import get_dataloaders
from library.train import train_one_epoch, validate

# ==========================================
# CONFIGURATION OVERRIDES FOR FAST BASELINE
# ==========================================
# Limit epochs to ensure execution within time limits
config.NUM_EPOCHS = 20
THRESHOLD_METRIC = 0.15744295919935183


def run_pipeline():
    # 1. Setup
    set_seed(config.SEED)
    print(f"Running on device: {config.DEVICE}")

    # 2. Data Loading
    # We use get_dataloaders to respect the fixed metadata split (train.csv / val.csv)
    print("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE, load_cached_data=True
    )

    # We also need test_ids for submission
    # load_and_process_data caches results, so this is fast
    _, _, _, _, _, test_ids = load_and_process_data(load_cached_data=True)

    # 3. Model Initialization
    model = RDPWBN().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Early Stopping setup
    checkpoint_path = os.path.join(config.WORKING_DIR, "best_model_baseline.pth")
    early_stopping = EarlyStopping(patience=5, verbose=True, path=checkpoint_path)

    # 4. Training Loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        print(
            f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
        )

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # 5. Final Evaluation & Failure Analysis
    print("\nLoading best model for evaluation...")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Collect Validation Predictions and Features for Analysis
    val_probs = []
    val_targets = []
    val_errors = []

    # Features for correlation
    feat_inc_angle = []
    feat_b1_mean = []
    feat_b2_mean = []

    with torch.no_grad():
        for imgs, incs, labels in val_loader:
            imgs = imgs.to(config.DEVICE)
            incs = incs.to(config.DEVICE)

            outputs = model(imgs, incs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            targets = labels.numpy().flatten()

            val_probs.extend(probs)
            val_targets.extend(targets)

            # Calculate error
            batch_errors = np.abs(probs - targets)
            val_errors.extend(batch_errors)

            # Collect features
            # incs is (B,) or (B, 1)
            feat_inc_angle.extend(incs.cpu().numpy().flatten())

            # imgs is (B, 3, 75, 75). Channel 0 is Band 1, Channel 1 is Band 2
            # Calculate mean per image
            b1_means = imgs[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
            b2_means = imgs[:, 1, :, :].mean(dim=(1, 2)).cpu().numpy()
            feat_b1_mean.extend(b1_means)
            feat_b2_mean.extend(b2_means)

    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    val_errors = np.array(val_errors)

    # Compute Metric
    final_metric = log_loss(val_targets, val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis (Correlation with Absolute Error):")

    # Handle NaNs in inc_angle if any (though loader should have imputed them)
    inc_arr = np.array(feat_inc_angle)
    if np.isnan(inc_arr).any():
        inc_arr = np.nan_to_num(inc_arr, nan=np.nanmean(inc_arr))

    corr_inc, _ = pearsonr(val_errors, inc_arr)
    corr_b1, _ = pearsonr(val_errors, np.array(feat_b1_mean))
    corr_b2, _ = pearsonr(val_errors, np.array(feat_b2_mean))

    print(f"  Error vs Inc Angle: {corr_inc:.4f}")
    print(f"  Error vs Band 1 Mean: {corr_b1:.4f}")
    print(f"  Error vs Band 2 Mean: {corr_b2:.4f}")

    # 6. Submission
    if final_metric < THRESHOLD_METRIC:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({THRESHOLD_METRIC}). Generating submission..."
        )

        test_probs = []
        with torch.no_grad():
            for imgs, incs in test_loader:
                imgs = imgs.to(config.DEVICE)
                incs = incs.to(config.DEVICE)

                outputs = model(imgs, incs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                test_probs.extend(probs)

        test_probs = np.array(test_probs)

        # Create submission dataframe
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": test_probs})

        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({THRESHOLD_METRIC}). Skipping submission."
        )


if __name__ == "__main__":
    run_pipeline()
