import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import from provided libraries
from library.config import Config
from library.utils import (
    seed_everything,
    load_checkpoint,
    laplace_log_likelihood_metric,
)
from library.data import get_dataloaders
from library.model import PRTNet
from library.train import run_training


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Run Training
    # This function handles the training loop and saves the best model to Config.BEST_MODEL_PATH
    run_training()

    # 3. Load Best Model for Analysis and Inference
    print("\nLoading best model for analysis...")
    model = PRTNet().to(device)
    load_checkpoint(model, path=Config.BEST_MODEL_PATH)
    model.eval()

    # 4. Data Loading
    # We need the stats to perform inverse transformation
    _, val_loader, test_loader, stats = get_dataloaders(load_cached_data=True)

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    # 5. Validation Assessment & Failure Analysis
    print("\nRunning Validation Assessment...")
    val_preds_mu = []
    val_preds_sigma = []
    val_true = []
    val_errors = []

    # Store features for correlation analysis
    feat_baseline_fvc = []
    feat_age = []
    feat_rel_time = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            static = batch["static"].to(device)
            rel_time = batch["rel_time"].to(device)
            raw_fvc = batch["raw_fvc"].to(device)  # True values in ml

            # Forward pass
            mu_scaled, sigma_scaled = model(images, static, rel_time)

            # Inverse Transform
            mu_abs = mu_scaled * fvc_std + fvc_mean
            sigma_abs = sigma_scaled * fvc_std

            # Collect data
            val_preds_mu.append(mu_abs.cpu().numpy())
            val_preds_sigma.append(sigma_abs.cpu().numpy())
            val_true.append(raw_fvc.cpu().numpy())

            # Collect features for failure analysis
            # static: [Baseline_FVC, Age, Sex, Smoking]
            feat_baseline_fvc.append(static[:, 0].cpu().numpy())
            feat_age.append(static[:, 1].cpu().numpy())
            feat_rel_time.append(rel_time.cpu().numpy().flatten())

    # Concatenate all batches
    val_preds_mu = np.concatenate(val_preds_mu)
    val_preds_sigma = np.concatenate(val_preds_sigma)
    val_true = np.concatenate(val_true)

    feat_baseline_fvc = np.concatenate(feat_baseline_fvc)
    feat_age = np.concatenate(feat_age)
    feat_rel_time = np.concatenate(feat_rel_time)

    # Compute Metric
    final_metric = laplace_log_likelihood_metric(
        val_true, val_preds_mu, val_preds_sigma
    )
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    abs_errors = np.abs(val_true - val_preds_mu)

    # Create a DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "Abs_Error": abs_errors,
            "Baseline_FVC_Scaled": feat_baseline_fvc,
            "Age_Scaled": feat_age,
            "Rel_Time_Scaled": feat_rel_time,
        }
    )

    correlations = analysis_df.corr()["Abs_Error"].drop("Abs_Error")
    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 6. Submission Generation
    threshold = -6.57744688338769
    if final_metric > threshold:
        print(
            f"\nMetric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )

        test_ids = []
        test_preds_mu = []
        test_preds_sigma = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                static = batch["static"].to(device)
                rel_time = batch["rel_time"].to(device)
                patient_weeks = batch["patient_week"]

                # Forward pass
                mu_scaled, sigma_scaled = model(images, static, rel_time)

                # Inverse Transform
                mu_abs = mu_scaled * fvc_std + fvc_mean
                sigma_abs = sigma_scaled * fvc_std

                # Clip confidence as per submission requirement (min 70)
                sigma_clipped = torch.clamp(sigma_abs, min=70)

                test_ids.extend(patient_weeks)
                test_preds_mu.extend(mu_abs.cpu().numpy())
                test_preds_sigma.extend(sigma_clipped.cpu().numpy())

        # Create Submission DataFrame
        sub_df = pd.DataFrame(
            {
                "Patient_Week": test_ids,
                "FVC": test_preds_mu,
                "Confidence": test_preds_sigma,
            }
        )

        # Ensure FVC is integer? The sample submission shows integers, but floats are usually accepted.
        # However, FVC is physically a volume. Let's round to be safe and match sample format.
        # Confidence is also int in sample, but metric uses float.
        # We will keep them as is or round if strictly necessary.
        # The metric definition uses floats. We will save as is (pandas handles it).

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
        print(sub_df.head())

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({threshold}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
