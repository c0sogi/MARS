import os
import torch
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import DSPRNet
from library.train import train_model
from library.inference import predict_test


def run():
    # 1. Initialization
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Training
    # Cite solution_lesson_node_00100: Sync scheduler with epochs.
    print("--- Starting Training ---")
    train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    # 3. Validation & Failure Analysis
    print("\n--- Starting Validation & Failure Analysis ---")

    # Load validation data
    # We only need the val_loader here
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Load the best model saved during training
    model = DSPRNet().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: Best model checkpoint not found. Using random weights.")

    model.eval()

    # Storage for analysis
    all_targets = []
    all_mu = []
    all_sigma = []
    all_errors = []

    # Metadata storage for correlation analysis
    meta_weeks = []
    meta_base_fvc = []
    meta_ages_scaled = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(images, tabular)

            # Inverse Transformation to Real-World Units
            # FVC = mu_scaled * STD + MEAN
            mu_real = mu_scaled.cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN
            # Confidence = sigma_scaled * STD
            sigma_real = sigma_scaled.cpu().numpy() * Config.TARGET_STD

            # Targets
            targets_real = (
                batch["target"].squeeze(-1).numpy() * Config.TARGET_STD
                + Config.TARGET_MEAN
            )

            # Calculate Absolute Errors for this batch
            errors = np.abs(targets_real - mu_real)

            # Store predictions and targets for global metric calculation
            all_targets.extend(targets_real)
            all_mu.extend(mu_real)
            all_sigma.extend(sigma_real)
            all_errors.extend(errors)

            # Store metadata
            # 'weeks' and 'base_fvc' are directly available in batch dict
            meta_weeks.extend(batch["weeks"].numpy())
            meta_base_fvc.extend(batch["base_fvc"].numpy())
            # Age is the 3rd feature (index 2) in the tabular vector (Scaled_Age)
            meta_ages_scaled.extend(tabular[:, 2].cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_targets)
    y_pred = np.array(all_mu)
    sigma_pred = np.array(all_sigma)

    # Calculate Final Metric
    final_metric = calculate_metric(y_true, y_pred, sigma_pred)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis: Correlations
    # We correlate the absolute error with input features
    df_analysis = pd.DataFrame(
        {
            "Error": np.array(all_errors),
            "Weeks": np.array(meta_weeks),
            "Base_FVC": np.array(meta_base_fvc),
            "Age_Scaled": np.array(meta_ages_scaled),
        }
    )

    corr_weeks = df_analysis["Error"].corr(df_analysis["Weeks"])
    corr_base = df_analysis["Error"].corr(df_analysis["Base_FVC"])
    corr_age = df_analysis["Error"].corr(df_analysis["Age_Scaled"])

    print("\nFailure Analysis (Correlation with Absolute Error):")
    print(f"Correlation (Error vs Weeks): {corr_weeks:.4f}")
    print(f"Correlation (Error vs Base FVC): {corr_base:.4f}")
    print(f"Correlation (Error vs Age): {corr_age:.4f}")

    # 4. Submission Logic
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(f"\nValidation Metric ({final_metric}) exceeds threshold ({THRESHOLD}).")
        print("Generating submission...")
        predict_test()
    else:
        print(
            f"\nValidation Metric ({final_metric}) does not exceed threshold ({THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run()
