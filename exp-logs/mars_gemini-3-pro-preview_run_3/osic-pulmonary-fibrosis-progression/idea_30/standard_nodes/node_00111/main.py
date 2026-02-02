import os
import sys
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, get_test_dataloader
from library.model import MAZR_DS
from library.train import run_training


def main():
    # 1. Setup and Reproducibility
    seed_everything(Config.SEED)
    Config.setup()

    # 2. Train the model
    # run_training handles the loop and saves 'best_model.pth' to Config.CHECKPOINTS_DIR
    print("Starting training process...")
    _ = run_training(load_cached_data=True)

    # 3. Load the best model for Analysis and Inference
    device = torch.device(Config.DEVICE)
    model = MAZR_DS().to(device)

    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 4. Validation & Metric Calculation
    # We need to reload the validation loader to get the data
    _, val_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    val_preds_mu = []
    val_preds_sigma = []
    val_targets = []

    # For Failure Analysis
    val_features = {"Baseline_FVC": [], "Relative_Weeks": [], "Age": []}

    print("Running validation inference...")
    with torch.no_grad():
        for imgs, tabular, targets in val_loader:
            imgs = imgs.to(device)
            tabular = tabular.to(device)

            # Forward pass
            mu, sigma = model(imgs, tabular)

            # Inverse transform predictions
            # mu_raw = mu_norm * std + mean
            mu_raw = mu.cpu().numpy() * fvc_std + fvc_mean
            # sigma_raw = sigma_norm * std
            sigma_raw = sigma.cpu().numpy() * fvc_std
            # target_raw = target_norm * std + mean
            target_raw = targets.numpy() * fvc_std + fvc_mean

            val_preds_mu.extend(mu_raw)
            val_preds_sigma.extend(sigma_raw)
            val_targets.extend(target_raw)

            # Extract features for analysis
            # tabular: [base_fvc_norm, t_rel, age_norm, sex, smoke]
            tab_np = tabular.cpu().numpy()

            # Reconstruct original feature scales approximately for correlation analysis
            # Base FVC
            base_fvc = tab_np[:, 0] * fvc_std + fvc_mean
            # Weeks (Relative) -> t_rel / scale
            weeks_rel = tab_np[:, 1] / Config.TIME_SCALE
            # Age
            age = tab_np[:, 2] * stats["age_std"] + stats["age_mean"]

            val_features["Baseline_FVC"].extend(base_fvc)
            val_features["Relative_Weeks"].extend(weeks_rel)
            val_features["Age"].extend(age)

    val_preds_mu = np.array(val_preds_mu)
    val_preds_sigma = np.array(val_preds_sigma)
    val_targets = np.array(val_targets)

    # Calculate Metric
    final_metric = calculate_metric(val_targets, val_preds_mu, val_preds_sigma)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    abs_errors = np.abs(val_targets - val_preds_mu)

    # Create a DataFrame for correlation
    analysis_df = pd.DataFrame(
        {
            "Error": abs_errors,
            "Baseline_FVC": val_features["Baseline_FVC"],
            "Relative_Weeks": val_features["Relative_Weeks"],
            "Age": val_features["Age"],
        }
    )

    # Compute correlations
    correlations = analysis_df.corr()["Error"].drop("Error")
    print("Correlation between Absolute Error and Input Features:")
    print(correlations)

    # 6. Submission Generation
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Get Test Loader
        test_loader, sub_df = get_test_dataloader(
            stats, batch_size=Config.BATCH_SIZE, load_cached_data=True
        )

        test_preds_mu = []
        test_preds_sigma = []

        with torch.no_grad():
            for imgs, tabular, _ in test_loader:
                imgs = imgs.to(device)
                tabular = tabular.to(device)

                mu, sigma = model(imgs, tabular)

                # Inverse transform
                mu_raw = mu.cpu().numpy() * fvc_std + fvc_mean
                sigma_raw = sigma.cpu().numpy() * fvc_std

                test_preds_mu.extend(mu_raw)
                test_preds_sigma.extend(sigma_raw)

        test_preds_mu = np.array(test_preds_mu)
        test_preds_sigma = np.array(test_preds_sigma)

        # Apply Post-Processing for Submission
        # 1. Clip Sigma at 70ml
        test_preds_sigma = np.maximum(test_preds_sigma, 70)

        # Update Submission DataFrame
        sub_df["FVC"] = test_preds_mu
        sub_df["Confidence"] = test_preds_sigma

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
