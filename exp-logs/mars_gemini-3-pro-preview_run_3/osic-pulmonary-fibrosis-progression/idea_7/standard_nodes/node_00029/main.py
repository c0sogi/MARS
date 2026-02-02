import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import DEVICE, BEST_MODEL_PATH, SEED
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import get_dataloaders
from library.model import SAPNet
from library.train import train_model, generate_submission


def main():
    # 1. Setup and Reproducibility
    seed_everything(SEED)

    # 2. Training
    # We reduce epochs to 20 for a fast baseline execution as requested.
    # The library function handles data loading, model init, and the training loop.
    print("--- Starting Training Pipeline ---")
    scaler = train_model(epochs=20, load_cached_data=True)

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load the best model saved during training
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU.")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model = SAPNet().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Ensure scaler statistics match the best model's training state
    scaler.means = checkpoint["scaler_means"]
    scaler.stds = checkpoint["scaler_stds"]
    scaler.fitted = True

    # Get validation dataloader
    # Note: We reload dataloaders to ensure clean state, though train_model already created them.
    _, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    all_mu = []
    all_sigma = []
    all_targets = []
    all_features = []  # To store tabular inputs for correlation analysis

    with torch.no_grad():
        for imgs, tab, targets in val_loader:
            imgs = imgs.to(DEVICE)
            tab_device = tab.to(DEVICE)

            # Inference
            mu_scaled, sigma_scaled = model(imgs, tab_device)

            # Inverse Transform to original scale
            mu = scaler.inverse_transform_target(mu_scaled.cpu().numpy())
            sigma = scaler.inverse_transform_sigma(sigma_scaled.cpu().numpy())
            targets_orig = scaler.inverse_transform_target(targets.numpy())

            all_mu.extend(mu)
            all_sigma.extend(sigma)
            all_targets.extend(targets_orig)
            all_features.extend(tab.numpy())

    all_mu = np.array(all_mu)
    all_sigma = np.array(all_sigma)
    all_targets = np.array(all_targets)
    all_features = np.array(
        all_features
    )  # Shape: (N, 5) -> [Baseline, Weeks, Age, Sex, Smoking]

    # Calculate Final Metric
    final_metric = metric_laplace_log_likelihood(all_targets, all_mu, all_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Error Magnitude and Features
    errors = np.abs(all_targets - all_mu)

    # Create DataFrame for analysis
    # Feature indices from library/data.py: 0:Baseline, 1:Weeks, 2:Age, 3:Sex, 4:Smoking
    analysis_df = pd.DataFrame(
        {
            "Error": errors,
            "Baseline_Scaled": all_features[:, 0],
            "Weeks_Scaled": all_features[:, 1],
            "Age_Scaled": all_features[:, 2],
            "Sex": all_features[:, 3],
            "Smoking": all_features[:, 4],
        }
    )

    print("\nFailure Analysis - Correlation with Absolute Error:")
    correlations = analysis_df.corr()["Error"].drop("Error")
    print(correlations)

    # 4. Submission Generation
    THRESHOLD = -6.57744688338769

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(scaler)
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
