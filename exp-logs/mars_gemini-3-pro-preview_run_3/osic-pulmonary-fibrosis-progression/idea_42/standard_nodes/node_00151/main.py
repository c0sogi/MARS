import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    laplace_log_likelihood_metric,
    inverse_transform_predictions,
)
from library.data import get_dataloaders
from library.model import SPCRNet
from library.train import MetricAlignedLLLoss


def main():
    # --- 1. Setup & Configuration ---
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Configure for Fast Baseline
    Config.EPOCHS = 10
    Config.DEBUG = False  # Use full dataset for meaningful failure analysis

    # Override submission paths to match prompt requirements
    Config.SUBMISSION_DIR = "./submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(
        f"Initializing SPCR-Net Pipeline. Device: {Config.DEVICE}, Epochs: {Config.EPOCHS}"
    )

    # --- 2. Data Loading ---
    # load_cached_data=True is handled internally by OSICDataset
    train_loader, val_loader, test_loader = get_dataloaders()

    # --- 3. Model Initialization ---
    model = SPCRNet().to(Config.DEVICE)

    # --- 4. Optimizer Setup (Differential Learning Rates) ---
    # Identify backbone parameters for lower learning rate
    backbone_ids = list(map(id, model.visual_stream.backbone.parameters()))

    backbone_params = []
    head_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_ids:
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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = MetricAlignedLLLoss().to(Config.DEVICE)

    # --- 5. Training Loop ---
    best_score = -float("inf")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_losses = []

        for batch in train_loader:
            imgs, tabs, targets = batch
            imgs = imgs.to(Config.DEVICE)
            tabs = tabs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()

            # Forward pass
            final_out, aux_out = model(imgs, tabs)

            # Loss Calculation (Main + Auxiliary)
            loss_main = criterion(final_out, targets)
            loss_aux = criterion(aux_out, targets)
            loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation Step (Model Selection)
        model.eval()
        val_preds_mu = []
        val_preds_sigma = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, tabs, targets = batch
                imgs = imgs.to(Config.DEVICE)
                tabs = tabs.to(Config.DEVICE)
                targets = targets.to(Config.DEVICE)

                final_out, _ = model(imgs, tabs)

                # Extract and Transform Predictions
                mu_norm = final_out[:, 0]
                sigma_norm = F.softplus(final_out[:, 1])  # Enforce positivity

                mu_abs, sigma_abs = inverse_transform_predictions(mu_norm, sigma_norm)
                target_abs = targets * Config.TARGET_STD + Config.TARGET_MEAN

                val_preds_mu.extend(mu_abs.cpu().numpy())
                val_preds_sigma.extend(sigma_abs.cpu().numpy())
                val_targets.extend(target_abs.cpu().numpy())

        # Calculate Metric
        val_score = laplace_log_likelihood_metric(
            np.array(val_targets), np.array(val_preds_mu), np.array(val_preds_sigma)
        )

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {np.mean(train_losses):.4f} | Val Score: {val_score:.4f}"
        )

        # Save Best Model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Validation Score: {best_score}")

    # --- 6. Final Validation & Failure Analysis ---
    print("\n--- Performing Failure Analysis ---")
    # Load best model
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []
    val_features = []  # Store tabular features for correlation

    with torch.no_grad():
        for batch in val_loader:
            imgs, tabs, targets = batch
            imgs = imgs.to(Config.DEVICE)
            tabs = tabs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            final_out, _ = model(imgs, tabs)

            mu_norm = final_out[:, 0]
            sigma_norm = F.softplus(final_out[:, 1])

            mu_abs, sigma_abs = inverse_transform_predictions(mu_norm, sigma_norm)
            target_abs = targets * Config.TARGET_STD + Config.TARGET_MEAN

            val_preds_mu.extend(mu_abs.cpu().numpy())
            val_preds_sigma.extend(sigma_abs.cpu().numpy())
            val_targets.extend(target_abs.cpu().numpy())
            val_features.extend(tabs.cpu().numpy())

    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)
    val_features = np.array(val_features)  # [fvc_norm, age_norm, sex, smoke, time]

    # Calculate Final Metric
    final_metric = laplace_log_likelihood_metric(
        val_targets, val_preds_mu, val_preds_sigma
    )
    print(f"Final Validation Metric: {final_metric}")

    # Calculate Correlations
    absolute_errors = np.abs(val_targets - val_preds_mu)

    feature_names = ["Norm_Base_FVC", "Norm_Age", "Sex", "Smoking", "Relative_Time"]
    print("Correlation between Absolute Error and Input Features:")
    for i, name in enumerate(feature_names):
        feat_values = val_features[:, i]
        correlation = np.corrcoef(absolute_errors, feat_values)[0, 1]
        print(f"  {name}: {correlation:.4f}")

    # Also correlate with Target FVC magnitude
    target_corr = np.corrcoef(absolute_errors, val_targets)[0, 1]
    print(f"  Target_FVC_Magnitude: {target_corr:.4f}")

    # --- 7. Submission Generation ---
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")

        test_preds_mu = []
        test_preds_sigma = []

        with torch.no_grad():
            for batch in test_loader:
                imgs, tabs, _ = batch  # Ignore dummy targets
                imgs = imgs.to(Config.DEVICE)
                tabs = tabs.to(Config.DEVICE)

                final_out, _ = model(imgs, tabs)

                mu_norm = final_out[:, 0]
                sigma_norm = F.softplus(final_out[:, 1])

                mu_abs, sigma_abs = inverse_transform_predictions(mu_norm, sigma_norm)

                test_preds_mu.extend(mu_abs.cpu().numpy())
                test_preds_sigma.extend(sigma_abs.cpu().numpy())

        # Prepare Submission DataFrame
        test_df = test_loader.dataset.df
        submission = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": test_preds_mu,
                "Confidence": test_preds_sigma,
            }
        )

        # Apply Confidence Clipping (Rule: max(sigma, 70))
        submission["Confidence"] = submission["Confidence"].apply(
            lambda x: max(x, Config.SIGMA_CLIP)
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
