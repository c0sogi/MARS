import os
import sys
import warnings
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, InverseScaler, LaplaceMetric
from library.train import Trainer
from library.inference import generate_submission
from library.model import ZIMARNet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Configuration & Setup
    seed_everything(Config.SEED)

    # Configure for Fast Baseline
    # Reducing epochs allows the pipeline to complete quickly while using the full dataset
    # to ensure the model learns enough to pass the metric threshold.
    Config.EPOCHS = 30

    # Ensure necessary directories exist
    Config.setup()

    print("--- Starting ZIMAR-Net Fast Baseline ---")

    # 2. Training
    # Initialize Trainer with debug=False to use the full training set (1109 samples).
    # This is small enough to run quickly on GPU but large enough for valid training.
    trainer = Trainer(debug=False)
    trainer.fit()

    # 3. Final Validation & Failure Analysis
    print("\n--- Performing Final Validation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)

    # Load the best model checkpoint saved during training
    model = ZIMARNet().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    else:
        print("Error: Best model checkpoint not found. Using random initialization.")

    model.eval()

    # Retrieve the validation loader from the trainer instance
    val_loader = trainer.val_loader

    # Initialize utilities
    scaler = InverseScaler()
    metric_fn = LaplaceMetric()

    # Storage for failure analysis
    analysis_records = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            imgs = batch["image"].to(device)
            clinical = batch["clinical"].to(device)

            # Get raw targets (in ml) for accurate metric calculation
            targets_raw = batch["fvc_raw"].numpy()

            # Forward pass
            mu_scaled, sigma_scaled = model(imgs, clinical)

            # Move predictions to CPU
            mu_scaled = mu_scaled.cpu()
            sigma_scaled = sigma_scaled.cpu()

            # Inverse transform Z-scored predictions to original units
            mu_raw, sigma_raw = scaler(mu_scaled, sigma_scaled)

            # Update the metric accumulator
            metric_fn.update(mu_raw, sigma_raw, targets_raw)

            # Collect data for failure analysis
            # Clinical vector structure: [Baseline_FVC_scaled, Time_scaled, Age_scaled, Sex, Smoking]
            clinical_np = clinical.cpu().numpy()

            for i in range(len(targets_raw)):
                # Calculate absolute error
                error = np.abs(targets_raw[i] - mu_raw[i].item())

                analysis_records.append(
                    {
                        "Target": targets_raw[i],
                        "Prediction": mu_raw[i].item(),
                        "Confidence": sigma_raw[i].item(),
                        "Error": error,
                        "Baseline_FVC_Scaled": clinical_np[i, 0],
                        "Time_Scaled": clinical_np[i, 1],
                        "Age_Scaled": clinical_np[i, 2],
                    }
                )

    # Compute and print the final validation metric
    final_score = metric_fn.compute()
    print(f"Final Validation Metric: {final_score}")

    # Perform Failure Analysis
    if analysis_records:
        df_analysis = pd.DataFrame(analysis_records)
        print("\nFailure Analysis - Correlation with Absolute Error:")

        features_to_check = [
            "Confidence",
            "Baseline_FVC_Scaled",
            "Time_Scaled",
            "Age_Scaled",
        ]
        for feat in features_to_check:
            if feat in df_analysis.columns:
                # Calculate Pearson correlation (check for zero variance to avoid warnings)
                if df_analysis[feat].std() > 1e-9:
                    corr, _ = pearsonr(df_analysis["Error"], df_analysis[feat])
                    print(f"Correlation Error vs {feat}: {corr:.4f}")
                else:
                    print(f"Correlation Error vs {feat}: Undefined (Constant feature)")

    # 4. Submission Generation
    # Threshold defined in the task description
    THRESHOLD = -6.573619738753321

    if final_score > THRESHOLD:
        print(
            f"\nValidation Score ({final_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(debug=False)
    else:
        print(
            f"\nValidation Score ({final_score}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
