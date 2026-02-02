import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import get_dataloaders
from library.runner import Trainer


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Optimize for speed: Reduce epochs for a fast baseline execution
    # We update T_MAX as well to ensure the Cosine Annealing scheduler adapts correctly
    Config.N_EPOCHS = 25
    Config.T_MAX = 25

    print(f"Configuration updated: N_EPOCHS={Config.N_EPOCHS}")

    # ==========================================
    # 2. Model Training
    # ==========================================
    # Initialize Trainer (handles Model, Loss, Optimizer, Scheduler)
    trainer = Trainer()

    # Run training loop (handles Data Loading, Preprocessing, Early Stopping)
    trainer.train()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\nStarting Validation Analysis...")

    # We get the dataloaders again to access the validation set specifically for analysis
    # Note: Preprocessing is cached, so this is fast.
    _, val_loader, _ = get_dataloaders()

    # Set model to evaluation mode
    trainer.model.eval()
    device = trainer.device

    all_preds = []
    all_sigmas = []
    all_targets = []
    all_metas = []

    # Inference loop on Validation set
    with torch.no_grad():
        for batch in val_loader:
            axial = batch["axial"].to(device).float()
            coronal = batch["coronal"].to(device).float()
            fusion = batch["fusion"].to(device)
            anchor = batch["anchor"].to(device)
            meta = batch["meta"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = trainer.model(
                axial, coronal, fusion, anchor
            )

            # Reconstruct Predictions from Trajectory Parameters
            # FVC = Base + Slope * Diff
            base_fvc = meta[:, 0]
            week_diff = meta[:, 1]

            fvc_pred = base_fvc + alpha * week_diff

            # Sigma = Base + Growth * |Diff|
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

            # Collect results
            all_preds.extend(fvc_pred.cpu().numpy())
            all_sigmas.extend(sigma_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_metas.extend(meta.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    sigma = np.array(all_sigmas)

    # --- Calculate Final Metric ---
    metric_score = calculate_metric(y_true, y_pred, sigma)

    # PRINT REQUIRED METRIC (Full Precision)
    print(f"Final Validation Metric: {metric_score}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    abs_errors = np.abs(y_true - y_pred)

    # Extract features from the validation dataset DataFrame
    # The DataLoader is sequential (shuffle=False), so order aligns with predictions
    val_df = val_loader.dataset.df

    # Create a DataFrame for correlation analysis
    analysis_df = pd.DataFrame(
        {
            "Error": abs_errors,
            "Age": val_df["Age"].values,
            "Percent": val_df["Percent"].values,
            "Week_Diff": val_df["Week_Diff"].values,
            "Baseline_FVC": val_df["Baseline_FVC"].values,
        }
    )

    print("Correlation between Absolute Error and Input Features:")
    features_to_check = ["Age", "Percent", "Week_Diff", "Baseline_FVC"]

    for col in features_to_check:
        # Check if column exists (it should based on dataset.py)
        if col in analysis_df.columns:
            # Handle potential NaNs just in case, though preprocessing handles this
            valid_mask = ~analysis_df[col].isna() & ~analysis_df["Error"].isna()
            if valid_mask.sum() > 1:
                corr, _ = pearsonr(
                    analysis_df.loc[valid_mask, "Error"],
                    analysis_df.loc[valid_mask, col],
                )
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Insufficient data for correlation")

    # ==========================================
    # 4. Submission Logic
    # ==========================================
    THRESHOLD = -6.510164260864258

    print(f"\nChecking Threshold: {metric_score} vs {THRESHOLD}")

    if metric_score > THRESHOLD:
        print("Metric check passed. Generating submission...")
        trainer.predict()
    else:
        print("Metric check failed. Submission will not be generated.")


if __name__ == "__main__":
    main()
