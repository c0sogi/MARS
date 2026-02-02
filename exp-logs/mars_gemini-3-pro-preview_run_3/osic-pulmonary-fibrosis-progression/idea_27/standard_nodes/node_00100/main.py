import sys
import os
import pandas as pd
import numpy as np
import torch
import warnings

# Import from library
from library.config import Config
from library.utils import seed_everything, calculate_competition_metric
from library.train import Trainer
from library.inference import generate_submission
from library.model import GMARNet

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Setup
    print("--- Starting Runfile Execution ---")
    seed_everything(Config.SEED)

    # Adjust Config for fast baseline execution
    # The dataset is small (1109 rows), so full data training is fast.
    # We reduce epochs to 25 to ensure it completes well within limits while learning enough.
    Config.EPOCHS = 25

    # 2. Training
    print("\n--- Phase 1: Training ---")
    # We use debug=False to use the full dataset for better performance
    # The dataset is small enough that full training is still very fast.
    trainer = Trainer(debug=False)
    trainer.train()

    # 3. Validation & Failure Analysis
    print("\n--- Phase 2: Validation & Failure Analysis ---")

    # Load the best model
    device = torch.device(Config.DEVICE)
    model = GMARNet().to(device)

    if not os.path.exists(Config.BEST_MODEL_PATH):
        print("Error: Best model not found.")
        return

    print(f"Loading best model from {Config.BEST_MODEL_PATH}...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Get Validation Loader and Scalers from trainer instance
    val_loader = trainer.val_loader
    fvc_scaler = trainer.fvc_scaler

    all_true = []
    all_pred_mu = []
    all_pred_sigma = []

    # Run Inference on Validation Set
    print("Running inference on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            clinical = batch["clinical"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(images, clinical)

            # Extract scaled predictions
            mu_scaled = preds[:, 0].cpu().numpy()
            sigma_scaled = preds[:, 1].cpu().numpy()
            target_scaled = targets.cpu().numpy()

            # Inverse Transform to original scale (ml)
            mu_orig = fvc_scaler.inverse_transform(mu_scaled)
            sigma_orig = fvc_scaler.inverse_transform_sigma(sigma_scaled)
            target_orig = fvc_scaler.inverse_transform(target_scaled)

            all_true.extend(target_orig)
            all_pred_mu.extend(mu_orig)
            all_pred_sigma.extend(sigma_orig)

    all_true = np.array(all_true)
    all_pred_mu = np.array(all_pred_mu)
    all_pred_sigma = np.array(all_pred_sigma)

    # Calculate Metric
    final_metric = calculate_competition_metric(all_true, all_pred_mu, all_pred_sigma)

    # REQUIRED PRINT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Load validation metadata to correlate features
    val_df = pd.read_csv(Config.VAL_CSV)

    # Calculate absolute error
    errors = np.abs(all_true - all_pred_mu)

    # Ensure lengths match (loader should preserve order as shuffle=False)
    if len(val_df) != len(errors):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(errors)}"
        )
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    # Add error to dataframe for correlation
    val_df["Abs_Error"] = errors

    # Features to analyze
    features = ["Weeks", "Percent", "Age", "FVC"]

    print("Correlation between Absolute Error and Input Features:")
    for feat in features:
        if feat in val_df.columns:
            corr = val_df[feat].corr(val_df["Abs_Error"])
            print(f"  {feat}: {corr:.4f}")

    # 4. Submission
    print("\n--- Phase 3: Submission ---")
    THRESHOLD = -6.573619738753321

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} > Threshold {THRESHOLD}. Generating submission file..."
        )
        generate_submission()
    else:
        print(f"Metric {final_metric} <= Threshold {THRESHOLD}. Submission skipped.")

    print("\n--- Runfile Execution Complete ---")


if __name__ == "__main__":
    main()
