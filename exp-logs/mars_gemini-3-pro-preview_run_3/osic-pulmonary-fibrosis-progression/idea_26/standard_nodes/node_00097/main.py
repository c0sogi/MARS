import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, STATS
from library.model import DSPRNet
from library.train import run_training
from library.predict import generate_submission


def main():
    # 1. Setup and Configuration
    # Enforce reproducibility
    seed_everything(Config.SEED)

    # Speed up training for baseline execution to meet time constraints
    # Increasing to 30 to ensure convergence with corrected metric monitoring
    Config.EPOCHS = 30

    print("--- Starting Pipeline ---")

    # 2. Training Phase
    # This will train the model and save 'best_model.pth' to Config.CHECKPOINT_DIR
    run_training()

    # 3. Validation Phase
    print("\n--- Starting Validation Analysis ---")
    device = Config.get_device()

    # Load the best model
    model = DSPRNet()
    model = model.to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Get validation data
    _, val_loader, _ = get_dataloaders(batch_size=Config.BATCH_SIZE)

    # Storage for analysis
    preds_mu_list = []
    preds_sigma_list = []
    targets_list = []
    tabular_list = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            # Inference
            out = model(images, tabular)

            # Store raw outputs (move to CPU for numpy processing)
            preds_mu_list.append(out[:, 0].cpu().numpy())
            preds_sigma_list.append(out[:, 1].cpu().numpy())
            targets_list.append(target.cpu().numpy())
            tabular_list.append(tabular.cpu().numpy())

    # Concatenate all batches
    preds_mu = np.concatenate(preds_mu_list)
    preds_sigma = np.concatenate(preds_sigma_list)
    targets = np.concatenate(targets_list)
    tabular_data = np.concatenate(tabular_list)

    # 4. Inverse Transformation
    # Recover real units (ml) for metric calculation and analysis
    # Predictions and Targets were Z-scored using these stats
    fvc_mean = STATS["fvc_mean"]
    fvc_std = STATS["fvc_std"]

    real_mu = preds_mu * fvc_std + fvc_mean
    real_sigma = preds_sigma * fvc_std
    real_target = targets * fvc_std + fvc_mean

    # 5. Metric Calculation
    # Metric: - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    # Constraints: sigma_clipped >= 70, delta = min(|true-pred|, 1000)

    sigma_clipped = np.maximum(real_sigma, 70.0)
    delta = np.abs(real_target - real_mu)
    delta_clipped = np.minimum(delta, 1000.0)

    metric_terms = -(np.sqrt(2) * delta_clipped) / sigma_clipped - np.log(
        np.sqrt(2) * sigma_clipped
    )
    final_metric = np.mean(metric_terms)

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate correlation between Error Magnitude (unclipped) and Input Features
    # Tabular features order in data.py:
    # [Base_FVC_Scaled, Base_Percent_Scaled, Relative_Weeks, Age_Scaled, Sex_Code, Smoking_Code]

    feature_names = [
        "Baseline_FVC",
        "Baseline_Percent",
        "Relative_Weeks",
        "Age",
        "Sex",
        "SmokingStatus",
    ]
    error_magnitude = delta  # Use unclipped absolute error for analysis

    print("Correlation (Pearson) between Absolute Error and Input Features:")
    for i, name in enumerate(feature_names):
        feature_vals = tabular_data[:, i]

        # Check for constant arrays to avoid warnings
        if np.std(feature_vals) == 0 or np.std(error_magnitude) == 0:
            print(f"  {name}: NaN (Constant variance)")
        else:
            corr, _ = pearsonr(feature_vals, error_magnitude)
            print(f"  {name}: {corr:.6f}")

    # 7. Submission Generation
    # Threshold: -6.573619738753321
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric {final_metric} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric {final_metric} does not exceed threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
