import os
import sys
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood
from library.data import get_dataloaders
from library.model import BBSLNet
from library.train import train_model, generate_submission


def main():
    # 1. Setup and Configuration
    seed_everything(Config.SEED)
    print(f"Initializing run for Idea: {Config.IDEA_ID}")

    # 2. Train the Model
    # We use debug=False because the dataset is small enough that 50 epochs
    # will complete quickly (approx 10-15 mins), satisfying the 'fast baseline'
    # requirement while ensuring a competitive score.
    print("Starting training pipeline...")
    best_model_path = train_model(debug=False)
    print(f"Training complete. Best model saved at: {best_model_path}")

    # 3. Validation Inference & Metric Calculation
    print("\nStarting Validation and Failure Analysis...")

    # Load the best model
    device = Config.DEVICE
    model = BBSLNet().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get validation loader
    _, val_loader, _ = get_dataloaders(debug=False)

    # Containers for analysis
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_features = []

    # Inference Loop (No Grad for speed)
    with torch.no_grad():
        for batch in val_loader:
            # Move data to GPU
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            # Metadata for reconstruction
            base_fvc = batch["meta"]["Base_FVC"].to(device).float()
            weeks = batch["meta"]["Weeks"].to(device).float()

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Reconstruct FVC and Confidence
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Store results (move to CPU to save GPU memory)
            all_targets.append(target.cpu())
            all_preds.append(fvc_pred.cpu())
            all_sigmas.append(sigma_pred.cpu())
            all_features.append(tabular.cpu())

    # Concatenate results
    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_preds)
    sigma = torch.cat(all_sigmas)
    features = torch.cat(all_features)

    # Calculate Final Metric
    # Note: laplace_log_likelihood handles clipping internally as per Config
    metric_score = laplace_log_likelihood(y_true, y_pred, sigma)
    final_metric = metric_score.item()

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    abs_errors = torch.abs(y_true - y_pred).numpy()

    # Map features to names based on Data Loader logic
    # Tabular structure: [Weeks, Age, Sex, Smoke0, Smoke1, Smoke2, BaseFVC, BasePct]
    feature_names = [
        "Weeks_Scaled",
        "Age_Scaled",
        "Sex_Encoded",
        "Smoke_Ex",
        "Smoke_Never",
        "Smoke_Current",
        "Base_FVC_Scaled",
        "Base_Percent_Scaled",
    ]

    # Create DataFrame for correlation analysis
    analysis_df = pd.DataFrame(features.numpy(), columns=feature_names)
    analysis_df["Abs_Error"] = abs_errors

    # Compute correlation
    correlations = analysis_df.corr()["Abs_Error"].sort_values(ascending=False)
    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 5. Conditional Submission
    # Threshold defined in task description
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(f"\nValidation Metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission file...")
        generate_submission(best_model_path, debug=False)
    else:
        print(
            f"\nValidation Metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
