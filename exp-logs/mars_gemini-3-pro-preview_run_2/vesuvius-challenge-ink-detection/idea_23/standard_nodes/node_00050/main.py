import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.utils import set_seed, fbeta_score
from library.train import train_specialist
from library.inference import predict_and_submit, load_models
from library.data import InkDataset


def run_pipeline():
    # --- Configuration Setup ---
    # Override Config for Fast Baseline and Requirements
    # Reducing epochs to 10 ensures the pipeline completes quickly while allowing convergence on the small dataset.
    Config.EPOCHS = 10

    # Create submission directory and update path as per Requirements
    os.makedirs("./submission", exist_ok=True)
    Config.SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure reproducibility
    set_seed(Config.SEED)

    print("Starting Matched-Depth Specialist Ensemble (MDSE) Pipeline...")

    # --- 1. Train Specialists ---
    # Train three independent models, each specializing in a specific Z-depth range.
    specialists = ["High", "Mid", "Low"]
    for spec in specialists:
        print(f"\n>>> Training Specialist: {spec}")
        # Set gating_threshold to 0.0 to ensure we always save the best model found,
        # guaranteeing that the inference stage has models to load.
        train_specialist(spec, gating_threshold=0.0)

    # --- 2. Ensemble Validation ---
    print("\n>>> Validating Ensemble on Hold-out Set")

    # Load Validation Metadata
    if not os.path.exists(Config.METADATA_VAL_PATH):
        print(f"Error: Validation metadata not found at {Config.METADATA_VAL_PATH}")
        return

    val_df = pd.read_csv(Config.METADATA_VAL_PATH)

    # Initialize Datasets for each view
    # We load them in parallel. Since they are based on the same metadata dataframe,
    # the i-th sample in each dataset corresponds to the same physical patch.
    ds_high = InkDataset(val_df, "High", mode="val")
    ds_mid = InkDataset(val_df, "Mid", mode="val")
    ds_low = InkDataset(val_df, "Low", mode="val")

    # DataLoaders
    # Use the config batch size. shuffle=False is critical for alignment across the three loaders.
    dl_high = DataLoader(
        ds_high,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    dl_mid = DataLoader(
        ds_mid,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    dl_low = DataLoader(
        ds_low,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Trained Models
    models = load_models()

    # Storage for analysis
    sample_scores = []
    sample_errors = []
    sample_intensities = []

    device = Config.DEVICE

    # Inference Loop
    with torch.no_grad():
        # Zip dataloaders to iterate simultaneously through the matched views
        for batch_h, batch_m, batch_l in zip(dl_high, dl_mid, dl_low):
            # Unpack batches
            # InkDataset returns (image, label) for mode='val'
            imgs_h, labels = batch_h
            imgs_m, _ = batch_m
            imgs_l, _ = batch_l

            # Move to device
            imgs_h = imgs_h.to(device)
            imgs_m = imgs_m.to(device)
            imgs_l = imgs_l.to(device)
            labels = labels.to(device)

            # Individual Predictions
            logits_h = models["High"](imgs_h)
            logits_m = models["Mid"](imgs_m)
            logits_l = models["Low"](imgs_l)

            probs_h = torch.sigmoid(logits_h)
            probs_m = torch.sigmoid(logits_m)
            probs_l = torch.sigmoid(logits_l)

            # Ensemble Fusion: Max Probability Projection
            # We take the maximum probability across the three specialists per pixel.
            # Stack: (3, B, 1, H, W) -> Max over dim 0 -> (B, 1, H, W)
            stacked_probs = torch.stack([probs_h, probs_m, probs_l], dim=0)
            fused_probs, _ = torch.max(stacked_probs, dim=0)

            # Compute Metrics per sample for analysis
            batch_size = imgs_h.size(0)
            for i in range(batch_size):
                # Extract single sample tensors
                p = fused_probs[i : i + 1]
                t = labels[i : i + 1]

                # F0.5 Score
                score = fbeta_score(
                    p, t, beta=Config.DICE_BETA, threshold=Config.THRESHOLD
                )

                # Input Intensity (Average of 3 views)
                # This represents the overall signal strength available to the ensemble
                intensity = (
                    imgs_h[i].mean() + imgs_m[i].mean() + imgs_l[i].mean()
                ) / 3.0

                sample_scores.append(score)
                sample_errors.append(1.0 - score)
                sample_intensities.append(intensity.item())

    # Compute Final Metric
    final_metric = np.mean(sample_scores)
    # Print exactly as required
    print(f"Final Validation Metric: {final_metric}")

    # --- 3. Failure Analysis ---
    print("\n>>> Performing Failure Analysis")
    if len(sample_errors) > 1:
        # Correlation between Error Magnitude and Input Intensity
        corr, p_val = pearsonr(sample_errors, sample_intensities)
        print(f"Correlation (Error vs Intensity): {corr:.6f} (p-value: {p_val:.6f})")

        if corr < -0.2:
            print(
                "Observation: Negative correlation. Brighter inputs tend to have lower error."
            )
        elif corr > 0.2:
            print(
                "Observation: Positive correlation. Brighter inputs tend to have higher error."
            )
        else:
            print("Observation: No strong linear correlation with intensity.")
    else:
        print("Insufficient data for failure analysis.")

    # --- 4. Submission ---
    threshold = 0.597622633
    if final_metric > threshold:
        print(f"\nValidation Metric ({final_metric}) exceeds threshold ({threshold}).")
        print("Generating submission file...")
        predict_and_submit()
    else:
        print(
            f"\nValidation Metric ({final_metric}) does not exceed threshold ({threshold})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    run_pipeline()
