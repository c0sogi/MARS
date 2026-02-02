import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Import provided library components
from library.config import Config
from library.utils import (
    seed_everything,
    save_checkpoint,
    load_checkpoint,
    laplace_log_likelihood_metric,
)
from library.data import get_dataloaders
from library.model import DynamicDepthGeMNet
from library.train import train_one_epoch, generate_submission, CustomLaplaceLoss


def main():
    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    # Modify Config for Fast Baseline execution
    Config.EPOCHS = 15
    Config.NUM_WORKERS = 2  # Adjust for environment

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Running on device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("Loading metadata and preparing dataloaders...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # get_dataloaders handles caching internally with load_cached_data=True
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # ---------------------------------------------------------
    # 3. Model Initialization
    # ---------------------------------------------------------
    print("Initializing model...")
    model = DynamicDepthGeMNet().to(device)

    criterion = CustomLaplaceLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    print("Starting training...")
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validation (Metric calculation)
        # We perform a quick validation check here to save best model
        model.eval()
        val_targets_list = []
        val_fvc_list = []
        val_sigma_list = []

        with torch.no_grad():
            for inputs, target in val_loader:
                for k, v in inputs.items():
                    inputs[k] = v.to(device)
                target = target.to(device)

                fvc_pred, sigma_pred = model(inputs)

                val_targets_list.append(target.cpu().numpy())
                val_fvc_list.append(fvc_pred.cpu().numpy())
                val_sigma_list.append(sigma_pred.cpu().numpy())

        val_targets_np = np.concatenate(val_targets_list).flatten()
        val_fvc_np = np.concatenate(val_fvc_list).flatten()
        val_sigma_np = np.concatenate(val_sigma_list).flatten()

        val_score = laplace_log_likelihood_metric(
            val_targets_np, val_fvc_np, val_sigma_np
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            save_checkpoint(
                {"state_dict": model.state_dict()},
                is_best=True,
                filename=os.path.join(Config.CACHE_DIR, f"checkpoint_ep{epoch+1}.pth"),
                best_filename=Config.MODEL_SAVE_PATH,
            )

    # ---------------------------------------------------------
    # 5. Final Validation & Failure Analysis
    # ---------------------------------------------------------
    print("\nLoading best model for final analysis...")
    load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    model.eval()

    val_targets = []
    val_fvc_preds = []
    val_sigma_preds = []
    val_meta = []

    with torch.no_grad():
        for inputs, target in val_loader:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            target = target.to(device)

            fvc_pred, sigma_pred = model(inputs)

            val_targets.extend(target.cpu().numpy().flatten())
            val_fvc_preds.extend(fvc_pred.cpu().numpy().flatten())
            val_sigma_preds.extend(sigma_pred.cpu().numpy().flatten())

            # Store metadata features for correlation analysis
            # inputs['tab'] shape (B, 5)
            val_meta.append(inputs["tab"].cpu().numpy())

    val_targets = np.array(val_targets)
    val_fvc_preds = np.array(val_fvc_preds)
    val_sigma_preds = np.array(val_sigma_preds)
    val_meta = np.concatenate(val_meta, axis=0)

    # Compute Final Metric
    final_metric = laplace_log_likelihood_metric(
        val_targets, val_fvc_preds, val_sigma_preds
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nFailure Analysis (Correlation with Error Magnitude):")
    # Calculate absolute error
    errors = np.abs(val_targets - val_fvc_preds)

    # Feature names corresponding to Config.TABULAR_COLS
    # ["Weeks", "Percent", "Age", "Sex", "SmokingStatus"]
    feature_names = [
        "Delta_Weeks",
        "Baseline_Percent",
        "Baseline_Age",
        "Sex",
        "SmokingStatus",
    ]

    for i, name in enumerate(feature_names):
        if i < val_meta.shape[1]:
            feat_values = val_meta[:, i]
            # Handle constant values (std=0) to avoid nan in correlation
            if np.std(feat_values) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(errors, feat_values)[0, 1]
            print(f"{name}: {corr:.4f}")

    # ---------------------------------------------------------
    # 6. Submission
    # ---------------------------------------------------------
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"\nMetric passed threshold ({final_metric} > {THRESHOLD}). Generating submission..."
        )
        generate_submission(test_loader, model, test_df, device)
    else:
        print(
            f"\nMetric failed threshold ({final_metric} <= {THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
