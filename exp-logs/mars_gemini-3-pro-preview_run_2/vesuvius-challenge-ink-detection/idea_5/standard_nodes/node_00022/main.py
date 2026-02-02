import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import warnings

# Import from library
from library.config import Config
from library.train import train_model, set_seed
from library.inference import generate_submission
from library.model import StratifiedSegFormer
from library.data import get_dataloaders

# Suppress warnings
warnings.filterwarnings("ignore")


def global_validate(model, loader, device):
    """
    Performs validation on the entire dataset to calculate the global F0.5 score
    and gathers statistics for failure analysis.
    """
    model.eval()

    # Metrics accumulators
    all_tp = 0
    all_fp = 0
    all_fn = 0

    # Failure analysis lists
    errors = []
    intensities = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Binarize predictions
            preds = (probs > Config.THRESHOLD).float()

            # --- Global Metric Accumulation ---
            # Flatten for calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            tp = (preds_flat * masks_flat).sum().item()
            fp = (preds_flat * (1 - masks_flat)).sum().item()
            fn = ((1 - preds_flat) * masks_flat).sum().item()

            all_tp += tp
            all_fp += fp
            all_fn += fn

            # --- Failure Analysis Data Collection ---
            # We analyze at the image level (batch sample level)
            # Calculate Mean Absolute Error per image
            # Shape: (B, 1, H, W) -> (B,)
            batch_mae = torch.abs(probs - masks).mean(dim=(1, 2, 3)).cpu().numpy()

            # Calculate Mean Intensity per image (across channels and pixels)
            # Shape: (B, C, H, W) -> (B,)
            batch_intensity = images.mean(dim=(1, 2, 3)).cpu().numpy()

            errors.extend(batch_mae)
            intensities.extend(batch_intensity)

    # Calculate Global F0.5 Score
    beta = 0.5
    beta_sq = beta**2
    numerator = (1 + beta_sq) * all_tp
    denominator = (1 + beta_sq) * all_tp + beta_sq * all_fn + all_fp

    # Avoid division by zero
    if denominator == 0:
        score = 0.0
    else:
        score = numerator / denominator

    return score, np.array(errors), np.array(intensities)


def run():
    # 1. Set Seed for Reproducibility
    set_seed(Config.SEED)

    # 2. Train Model
    # Uses the provided training loop which saves 'best_model.pth'
    # debug=False ensures we use the full provided metadata (which is already subsampled/curated)
    print("Starting Training...")
    train_model(debug=False)

    # 3. Load Best Model for Evaluation
    device = torch.device(Config.DEVICE)
    model = StratifiedSegFormer(pretrained=False)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. Final Validation & Failure Analysis
    print("Performing Final Validation and Failure Analysis...")

    # Get validation loader (re-using get_dataloaders)
    _, val_loader = get_dataloaders(load_cached_data=True, debug=False)

    f05_score, errors, intensities = global_validate(model, val_loader, device)

    # Print required metric format
    print(f"Final Validation Metric: {f05_score}")

    # Failure Analysis: Correlation between Input Intensity and Error
    if len(errors) > 1:
        # Check for constant input to avoid warnings
        if np.std(errors) > 0 and np.std(intensities) > 0:
            corr, p_val = pearsonr(intensities, errors)
            print(
                f"Failure Analysis - Correlation between Input Intensity and Error: {corr:.4f} (p={p_val:.4f})"
            )
        else:
            print("Failure Analysis - Variance too low to compute correlation.")
    else:
        print("Failure Analysis - Insufficient data.")

    # 5. Conditional Submission
    # Threshold defined in task
    THRESHOLD_SCORE = 0.4738558828830719

    if f05_score > THRESHOLD_SCORE:
        print(
            f"Validation score ({f05_score}) exceeds threshold ({THRESHOLD_SCORE}). Generating submission..."
        )
        generate_submission(load_cached_data=True)
    else:
        print(
            f"Validation score ({f05_score}) does not exceed threshold ({THRESHOLD_SCORE}). Skipping submission."
        )


if __name__ == "__main__":
    run()
