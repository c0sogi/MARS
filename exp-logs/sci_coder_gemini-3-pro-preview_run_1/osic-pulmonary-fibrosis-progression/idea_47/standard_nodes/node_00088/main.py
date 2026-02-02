import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.train import train_model, predict_and_submit
from library.model import SLHDAN
from library.data import get_dataloaders
from library.utils import laplace_log_likelihood_metric


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print("==========================================")
    print("       SLH-DAN FAST BASELINE RUN          ")
    print("==========================================")

    # 2. Training
    # We limit epochs to 15 for a fast baseline execution as requested.
    print("\n[Step 1] Starting Training...")
    train_model(epochs=15, debug=False)

    # 3. Validation Inference
    print("\n[Step 2] Performing Validation Inference...")

    # Load validation loader
    _, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Determine tabular dimension dynamically to initialize model
    sample_batch = next(iter(val_loader))
    tab_dim = sample_batch["tabular"].shape[1]

    # Load the best model
    model = SLHDAN(tabular_input_dim=tab_dim).to(device)
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Containers for results
    all_targets = []
    all_preds = []
    all_confidences = []

    with torch.no_grad():
        for batch in val_loader:
            # Move inputs to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            week = batch["week"].to(device)
            baseline_week = batch["baseline_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target_fvc = batch["target"].to(device)

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

            # Parametric Inference
            delta_t = week - baseline_week
            fvc_pred = baseline_fvc + alpha * delta_t
            confidence = sigma_base + sigma_growth * torch.abs(delta_t)

            # Store results
            all_targets.append(target_fvc.cpu())
            all_preds.append(fvc_pred.cpu())
            all_confidences.append(confidence.cpu())

    # Concatenate results
    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    sigma = torch.cat(all_confidences).numpy()

    # 4. Metric Calculation
    # Compute metric on the full validation set
    final_metric_tensor = laplace_log_likelihood_metric(y_true, y_pred, sigma)
    final_metric = final_metric_tensor.item()

    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n[Step 3] Failure Analysis on Validation Set...")

    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)

    # Load raw validation metadata to correlate with features
    # val_loader is shuffle=False, so it aligns with the CSV read sequentially
    val_df = pd.read_csv(Config.VAL_CSV)

    # Safety check for length alignment
    if len(val_df) != len(errors):
        print(
            f"Warning: Validation DataFrame length ({len(val_df)}) differs from predictions ({len(errors)}). Truncating to minimum."
        )
        min_len = min(len(val_df), len(errors))
        val_df = val_df.iloc[:min_len]
        errors = errors[:min_len]

    val_df["Error"] = errors

    # Features to analyze
    features_to_check = ["Age", "Percent", "Weeks"]

    # Encode categorical features for correlation
    if "Sex" in val_df.columns:
        val_df["Sex_Encoded"] = val_df["Sex"].astype("category").cat.codes
        features_to_check.append("Sex_Encoded")

    if "SmokingStatus" in val_df.columns:
        val_df["Smoking_Encoded"] = val_df["SmokingStatus"].astype("category").cat.codes
        features_to_check.append("Smoking_Encoded")

    print("Correlation between Absolute Error and Input Features:")
    for feat in features_to_check:
        if feat in val_df.columns:
            # Handle constant columns which cause PearsonR warnings
            if val_df[feat].std() == 0:
                print(f"  {feat}: N/A (Constant value)")
            else:
                corr, _ = pearsonr(val_df[feat], val_df["Error"])
                print(f"  {feat}: {corr:.4f}")

    # 6. Submission Logic
    threshold = -6.510164260864258
    print(f"\n[Step 4] Submission Check (Threshold: {threshold})")

    if final_metric > threshold:
        print("Metric check passed. Generating submission...")
        predict_and_submit()
    else:
        print("Metric check failed. Submission skipped.")


if __name__ == "__main__":
    main()
