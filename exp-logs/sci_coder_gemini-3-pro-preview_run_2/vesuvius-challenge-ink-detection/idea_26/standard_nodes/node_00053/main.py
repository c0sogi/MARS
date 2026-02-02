import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library components
from library.config import Config
from library.train import train_model
from library.inference import predict_and_submit
from library.model import InkSegFormer
from library.data import get_loaders
from library.utils import dice_coefficient


def main():
    # --- Configuration Override ---
    # Ensure submission is saved to the specific path required by the instructions
    Config.SUBMISSION_FILE = "./submission/submission.csv"
    os.makedirs(os.path.dirname(Config.SUBMISSION_FILE), exist_ok=True)

    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # --- 1. Training ---
    print("--- Starting Training ---")
    # Execute training (fast baseline with 15 epochs)
    # Using load_cached_data=True to leverage pre-processed volumes
    train_model(load_cached_data=True)

    # --- 2. Validation & Failure Analysis ---
    print("\n--- Starting Validation & Failure Analysis ---")

    device = torch.device(Config.DEVICE)
    model = InkSegFormer()
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    # Load the best model saved during training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Retrieve validation loader
    _, val_loader = get_loaders(load_cached_data=True)

    all_preds = []
    all_targets = []

    # Data collectors for failure analysis
    batch_errors = []
    batch_intensities = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Store predictions for global metric calculation
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

            # --- Failure Analysis ---
            # Calculate Mean Intensity of inputs (Input Feature)
            # images shape: (B, 3, H, W) -> mean over (1, 2, 3) gives (B,)
            means = images.mean(dim=(1, 2, 3)).cpu().numpy()

            # Calculate Error Magnitude (1 - F0.5) per sample
            for i in range(images.size(0)):
                p = probs[i]
                t = masks[i]

                # Compute F0.5 for the individual sample
                score = dice_coefficient(p, t, threshold=Config.THRESHOLD, beta=0.5)
                error = 1.0 - score

                batch_errors.append(error)
                batch_intensities.append(means[i])

    # --- Calculate Final Validation Metric ---
    full_preds = torch.cat(all_preds)
    full_targets = torch.cat(all_targets)

    # Calculate global F0.5 score on the full validation set
    final_metric = dice_coefficient(
        full_preds, full_targets, threshold=Config.THRESHOLD, beta=0.5
    )

    # Print metric in the required format with full precision
    print(f"Final Validation Metric: {final_metric}")

    # --- Calculate Correlation for Failure Analysis ---
    if len(batch_errors) > 1:
        corr, p_value = pearsonr(batch_intensities, batch_errors)
        print(
            f"Failure Analysis: Correlation between Input Intensity and Error Magnitude: {corr} (p-value: {p_value})"
        )
    else:
        print("Insufficient data for failure analysis.")

    # --- 3. Submission ---
    # Threshold defined in the task
    THRESHOLD = 0.597622633

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        predict_and_submit(load_cached_data=True)
    else:
        print(
            f"\nMetric ({final_metric}) does not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
