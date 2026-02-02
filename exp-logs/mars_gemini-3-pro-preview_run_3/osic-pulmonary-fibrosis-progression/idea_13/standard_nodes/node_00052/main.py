import sys
import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.data import get_dataloaders
from library.model import DSPRNet, generate_submission
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Training
    # We use Config.EPOCHS (35) to ensure convergence of uncertainty estimation (Cite Lesson 00011).
    # debug=False ensures we use the full dataset to meet the metric threshold.
    print("Starting training...")
    best_model_path = run_training(
        epochs=Config.EPOCHS, load_cached_data=True, debug=False
    )

    # 3. Validation Inference
    print("Loading best model for validation...")

    # Load validation data and scalers
    # We discard train_loader (index 0)
    _, val_loader, target_scaler, _ = get_dataloaders(load_cached_data=True)

    # Initialize model and load weights
    model = DSPRNet().to(device)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Containers for results
    true_fvcs = []
    pred_fvcs = []
    pred_sigmas = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            t_rel = batch["t_rel"].to(device)
            target_scaled = batch["target"].to(device).squeeze(-1)

            # Forward pass
            pred_mu_scaled, pred_sigma_scaled = model(img, tab, t_rel)

            # Inverse transform to get real units (ml)
            pred_mu, pred_sigma = target_scaler.inverse_transform(
                pred_mu_scaled, pred_sigma_scaled
            )
            true_fvc = target_scaler.inverse_transform(target_scaled)

            # Collect results
            true_fvcs.append(true_fvc.cpu().numpy())
            pred_fvcs.append(pred_mu.cpu().numpy())
            pred_sigmas.append(pred_sigma.cpu().numpy())

    # Concatenate batches
    true_fvcs = np.concatenate(true_fvcs)
    pred_fvcs = np.concatenate(pred_fvcs)
    pred_sigmas = np.concatenate(pred_sigmas)

    # 4. Metric Calculation
    # Metric: - (sqrt(2) * delta / sigma_clipped) - ln(sqrt(2) * sigma_clipped)
    # Constraints: sigma_clipped = max(sigma, 70), delta = min(|true - pred|, 1000)

    sigma_clipped = np.maximum(pred_sigmas, 70)
    delta = np.abs(true_fvcs - pred_fvcs)
    delta = np.minimum(delta, 1000)

    sqrt_2 = np.sqrt(2)
    metric_values = -(sqrt_2 * delta) / sigma_clipped - np.log(sqrt_2 * sigma_clipped)
    final_metric = np.mean(metric_values)

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing failure analysis...")

    # Calculate absolute errors (unclipped for analysis)
    abs_errors = np.abs(true_fvcs - pred_fvcs)

    # Get the validation dataframe to correlate with features
    val_df = val_loader.dataset.df

    # Ensure alignment
    if len(val_df) != len(abs_errors):
        print(
            f"Warning: Dataframe length ({len(val_df)}) does not match predictions ({len(abs_errors)})."
        )

    # Create analysis dataframe
    analysis_df = val_df.copy()
    analysis_df["Error"] = abs_errors

    # Features to correlate
    features_to_check = ["Age", "Weeks", "Baseline_FVC", "Percent"]

    print("Correlation between Absolute Error and Features:")
    for feat in features_to_check:
        if feat in analysis_df.columns:
            corr = analysis_df[feat].corr(analysis_df["Error"])
            print(f"  {feat}: {corr:.4f}")

    # 6. Submission Generation
    THRESHOLD = -6.57744688338769

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric}) is higher than threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"Metric ({final_metric}) did not beat threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
