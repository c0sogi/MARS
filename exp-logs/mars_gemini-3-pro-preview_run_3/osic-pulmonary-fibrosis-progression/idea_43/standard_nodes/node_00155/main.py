import sys
import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.train import train_model, generate_submission
from library.data import get_dataloaders
from library.model import DSPRNet


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    print("Initializing SB-PDS Net Pipeline...")

    # 2. Train the model
    # We use the default epochs (30) defined in Config.
    # Since the dataset is small (~1k rows), this is very fast on A100.
    print("Starting training...")
    best_model_path, scalers = train_model(
        epochs=Config.EPOCHS,
        max_train_samples=Config.MAX_TRAIN_SAMPLES,
        max_val_samples=Config.MAX_VAL_SAMPLES,
    )

    # 3. Validation & Failure Analysis
    print("\nStarting Validation and Failure Analysis...")
    device = torch.device(Config.DEVICE)

    # Load the best model
    model = DSPRNet().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get validation dataloader
    # We need the full validation set for the final metric calculation
    _, val_loader, _ = get_dataloaders(
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        max_train_samples=None,
        max_val_samples=None,
    )

    # Storage for analysis
    all_targets = []
    all_preds = []
    all_sigmas = []
    all_features = []

    # Scalers for inverse transformation
    fvc_mean = scalers["fvc_mean"]
    fvc_std = scalers["fvc_std"]

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"]  # Keep on CPU

            # Forward pass
            outputs = model(images, tabular)

            # Get normalized predictions
            pred_mean_norm = outputs["final_mean"].cpu().numpy()
            pred_sigma_norm = outputs["final_sigma"].cpu().numpy()

            # Inverse Transform to Raw Scale
            pred_mean_raw = pred_mean_norm * fvc_std + fvc_mean
            pred_sigma_raw = pred_sigma_norm * fvc_std

            # Store results
            all_targets.append(targets.numpy())
            all_preds.append(pred_mean_raw)
            all_sigmas.append(pred_sigma_raw)
            all_features.append(tabular.cpu().numpy())

    # Concatenate batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_sigmas = np.concatenate(all_sigmas)
    all_features = np.concatenate(all_features, axis=0)

    # Calculate Final Metric
    metric = calculate_metric(all_targets, all_preds, all_sigmas)
    print(f"Final Validation Metric: {metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(all_targets - all_preds)

    # Feature names based on data.py construction:
    # [base_fvc_norm, time_scaled, age, sex, smoke]
    feature_names = ["Baseline_FVC", "Time", "Age", "Sex", "Smoking"]

    analysis_df = pd.DataFrame(all_features, columns=feature_names)
    analysis_df["Error"] = errors
    analysis_df["Predicted_Sigma"] = all_sigmas

    # Calculate correlations
    correlations = analysis_df.corr()["Error"].sort_values(ascending=False)
    print("Correlation between Input Features and Absolute Error:")
    print(correlations)

    # 4. Submission
    # Threshold from instructions
    THRESHOLD = -6.573619738753321

    if metric > THRESHOLD:
        print(
            f"\nMetric ({metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        generate_submission(best_model_path, scalers)
    else:
        print(
            f"\nMetric ({metric}) does not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
