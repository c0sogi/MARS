import os
import sys
import numpy as np
import pandas as pd
import torch

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, prepare_inference_data
from library.model import MultiViewNet
from library.train import run_training


def main():
    # -------------------------------------------------------------------------
    # 1. Training Phase
    # -------------------------------------------------------------------------
    print("Initializing training pipeline...")
    # run_training handles seeding, device setup, training loop, and checkpointing
    run_training()

    # -------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStarting validation and failure analysis...")

    device = torch.device(Config.device)

    # Load the best model saved during training
    model = MultiViewNet().to(device)
    checkpoint_path = os.path.join(Config.checkpoint_dir, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"Critical Error: Checkpoint file not found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get validation dataloader
    _, val_loader = get_dataloaders(
        batch_size=Config.batch_size, num_workers=Config.num_workers
    )

    # Containers for analysis
    all_targets = []
    all_mus = []
    all_sigmas = []
    all_tabular = []

    # Inference loop on validation set
    with torch.no_grad():
        for images, tabular, targets in val_loader:
            images = images.to(device)
            tabular = tabular.to(device)

            # Forward pass
            mu_std, sigma_std = model(images, tabular)

            # Store results (move to CPU)
            all_targets.append(targets.cpu().numpy())
            all_mus.append(mu_std.cpu().numpy())
            all_sigmas.append(sigma_std.cpu().numpy())
            all_tabular.append(tabular.cpu().numpy())

    # Concatenate batches
    targets_std = np.concatenate(all_targets)
    mus_std = np.concatenate(all_mus)
    sigmas_std = np.concatenate(all_sigmas)
    tabular_data = np.concatenate(all_tabular)

    # Inverse Transformation (Standardized -> Real Units)
    # Target/Mean: val * std + mean
    targets_real = targets_std * Config.target_std + Config.target_mean
    mus_real = mus_std * Config.target_std + Config.target_mean
    # Sigma: val * std (scale only)
    sigmas_real = sigmas_std * Config.target_std

    # Calculate Final Metric
    final_metric = calculate_metric(targets_real, mus_real, sigmas_real)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis (Correlation with Absolute Error) ---")

    # Calculate absolute error per sample
    abs_errors = np.abs(targets_real - mus_real)

    # Feature names corresponding to the tabular input vector
    # Order in LungDataset: [age, sex, smoke, weeks, base_fvc]
    feature_names = ["Age", "Sex", "SmokingStatus", "Weeks", "Baseline_FVC"]

    for i, name in enumerate(feature_names):
        feat_values = tabular_data[:, i]

        # Check for constant values to avoid division by zero in correlation
        if np.std(feat_values) < 1e-9 or np.std(abs_errors) < 1e-9:
            print(f"  {name}: NaN (Constant values detected)")
        else:
            # Calculate Pearson correlation
            corr = np.corrcoef(feat_values, abs_errors)[0, 1]
            print(f"  {name}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 3. Submission Generation
    # -------------------------------------------------------------------------
    threshold = -6.6997912217

    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Prepare test data (expands test set to all required Patient_Weeks)
        test_loader, sub_df = prepare_inference_data()

        test_mus = []
        test_sigmas = []

        with torch.no_grad():
            for images, tabular in test_loader:
                images = images.to(device)
                tabular = tabular.to(device)

                mu_std, sigma_std = model(images, tabular)

                test_mus.append(mu_std.cpu().numpy())
                test_sigmas.append(sigma_std.cpu().numpy())

        # Concatenate results
        test_mus = np.concatenate(test_mus)
        test_sigmas = np.concatenate(test_sigmas)

        # Inverse Transform
        pred_fvc = test_mus * Config.target_std + Config.target_mean
        pred_sigma = test_sigmas * Config.target_std

        # Apply metric-specific clipping to confidence
        # "confidence values are clipped at 70 ml"
        pred_sigma = np.maximum(pred_sigma, 70.0)

        # Fill DataFrame
        sub_df["FVC"] = pred_fvc
        sub_df["Confidence"] = pred_sigma

        # Save Submission
        sub_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved successfully to {Config.submission_path}")
        print("First 5 rows of submission:")
        print(sub_df.head())

    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
