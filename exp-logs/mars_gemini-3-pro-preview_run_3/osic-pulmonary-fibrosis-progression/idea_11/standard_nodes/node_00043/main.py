import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.train import Trainer
from library.data import get_dataloaders
from library.model import TCDSNet
from library.utils import seed_everything, metric_score, InverseScaler


def main():
    # ==========================================
    # 1. Setup and Configuration Overrides
    # ==========================================
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Setup directories
    Config.setup()

    # Override Config for Fast Baseline execution
    # Reducing epochs to 15 ensures completion within 2 hours while allowing convergence
    Config.EPOCHS = 15
    Config.DEBUG = False  # Use full dataset

    # ==========================================
    # 2. Training
    # ==========================================
    print("Initializing Trainer...")
    trainer = Trainer(debug=Config.DEBUG)

    print(f"Starting training for {Config.EPOCHS} epochs...")
    trainer.fit()

    # ==========================================
    # 3. Validation and Failure Analysis
    # ==========================================
    print("\nStarting Validation and Failure Analysis...")

    # Retrieve dataloaders (need to recall to get test_loader which Trainer discards)
    # Note: Trainer.fit() has already trained the model, we just need loaders for analysis/submission
    _, val_loader, test_loader, scalers = get_dataloaders(debug=Config.DEBUG)

    # Initialize Inverse Scaler
    inverse_scaler = InverseScaler(mean=scalers["fvc_mean"], std=scalers["fvc_std"])

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = TCDSNet().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("Error: Best model checkpoint not found.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Validation Inference
    val_mus = []
    val_sigmas = []
    val_targets = []
    val_tabular = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            # raw_target is used for metric calculation (unscaled)
            raw_target = batch["raw_target"].numpy().flatten()

            mu, sigma = model(images, tabular)

            # Inverse transform predictions
            mu_orig, sigma_orig = inverse_scaler(mu, sigma)

            val_mus.extend(mu_orig)
            val_sigmas.extend(sigma_orig)
            val_targets.extend(raw_target)
            val_tabular.extend(batch["tabular"].cpu().numpy())

    val_mus = np.array(val_mus)
    val_sigmas = np.array(val_sigmas)
    val_targets = np.array(val_targets)
    val_tabular = np.array(val_tabular)

    # Calculate Final Metric
    final_metric = metric_score(val_targets, val_mus, val_sigmas)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlation between Abs Error and Features
    abs_errors = np.abs(val_targets - val_mus)

    # Feature indices based on library.data.OSICDataset:
    # [Base_FVC_Scaled, Age_Scaled, Sex_Code, Smoke_Code, Relative_Time]
    feature_names = ["Baseline_FVC", "Age", "Sex", "SmokingStatus", "Relative_Time"]

    print(
        "\nFailure Analysis (Correlation between Error Magnitude and Input Features):"
    )
    for i, name in enumerate(feature_names):
        feat_values = val_tabular[:, i]
        if np.std(feat_values) < 1e-9:
            corr = 0.0
        else:
            corr, _ = pearsonr(abs_errors, feat_values)
        print(f"{name}: {corr:.4f}")

    # ==========================================
    # 4. Submission Generation
    # ==========================================
    threshold = -6.57744688338769

    if final_metric > threshold:
        print(
            f"\nValidation metric ({final_metric}) exceeds threshold ({threshold}). Generating submission..."
        )

        submission_results = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                patient_weeks = batch["patient_week"]

                mu, sigma = model(images, tabular)

                # Inverse transform
                mu_orig, sigma_orig = inverse_scaler(mu, sigma)

                # Post-processing: Clip confidence at 70ml
                sigma_final = np.maximum(sigma_orig, 70.0)

                for pw, fvc, conf in zip(patient_weeks, mu_orig, sigma_final):
                    submission_results.append(
                        {"Patient_Week": pw, "FVC": fvc, "Confidence": conf}
                    )

        # Create DataFrame
        sub_df = pd.DataFrame(submission_results)

        # Ensure correct column order
        sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

        # Save submission
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_metric}) did not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
