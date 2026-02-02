import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from provided library files
from library.config import Config, seed_everything
from library.model import get_extended_dataloaders, TSCGNet
from library.train import train_model
from library.predict import inference_fn
from library.utils import score_function
from library.data import prepare_dataframe


def main():
    # 1. Setup Environment
    seed_everything(Config.SEED)
    Config.setup()

    print("==================================================")
    print("Step 1: Training Fast Baseline")
    print("==================================================")

    # Train the model
    # Limiting epochs to 20 for a fast baseline execution as requested.
    # The dataset is small, so this allows for reasonable convergence.
    best_model_path = train_model(
        epochs=20, batch_size=Config.BATCH_SIZE, debug=False, patience=6
    )

    print("\n==================================================")
    print("Step 2: Validation & Failure Analysis")
    print("==================================================")

    device = torch.device(Config.DEVICE)

    # Load the best model for evaluation
    model = TSCGNet().to(device)
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Model file not found at: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Get Validation Data Loader
    # We use the extended loader to get Baseline_FVC and other necessary fields
    _, val_loader, _ = get_extended_dataloaders(batch_size=Config.BATCH_SIZE)

    # Run Inference on Validation Set
    all_targets = []
    all_preds = []
    all_sigmas = []

    print("Running validation inference...")
    with torch.no_grad():
        for batch in val_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            target = batch["target"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass to get parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate anchored trajectory predictions
            fvc_pred = baseline_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Collect results
            all_targets.append(target.cpu().numpy())
            all_preds.append(fvc_pred.cpu().numpy())
            all_sigmas.append(sigma_pred.cpu().numpy())

    # Concatenate all batches
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    sigma = np.concatenate(all_sigmas)

    # Compute Final Metric
    final_metric = score_function(y_true, y_pred, sigma)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    # Load validation metadata and process it to align with the loader (sorted by Patient, Weeks)
    val_meta = pd.read_csv(Config.VAL_CSV)
    val_df = prepare_dataframe(val_meta, is_train=True)

    # Calculate absolute prediction error
    abs_error = np.abs(y_true - y_pred)

    print("\nFailure Analysis - Correlation with Absolute Error:")
    analysis_features = ["Weeks", "Baseline_FVC", "Baseline_Percent", "Baseline_Age"]

    for feat in analysis_features:
        if feat in val_df.columns:
            feat_values = val_df[feat].values

            # Ensure dimensions match before correlation
            if len(feat_values) == len(abs_error):
                # Calculate Pearson correlation using numpy
                corr = np.corrcoef(feat_values, abs_error)[0, 1]
                print(f"  {feat}: Correlation = {corr:.4f}")
            else:
                print(
                    f"  {feat}: Dimension mismatch (Data: {len(feat_values)}, Preds: {len(abs_error)})"
                )
        else:
            print(f"  {feat}: Feature not found in dataframe")

    print("\n==================================================")
    print("Step 3: Submission Generation")
    print("==================================================")

    # Check metric against threshold
    THRESHOLD = -6.510164260864258

    if final_metric > THRESHOLD:
        print(
            f"Metric {final_metric} > Threshold {THRESHOLD}. Proceeding to submission..."
        )
        inference_fn(
            model_path=best_model_path, batch_size=Config.BATCH_SIZE, debug=False
        )
    else:
        print(f"Metric {final_metric} <= Threshold {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
