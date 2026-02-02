import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy import stats

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, fbeta_score
from library.model import HPUnet
from library.data import get_loaders
from library.train import train_model
from library.inference import run_inference


def main():
    # 1. Setup and Reproducibility
    set_seed(Config.SEED)

    # Adjust Config for Fast Baseline Execution
    # The dataset is small (412 patches), but we ensure it runs quickly.
    Config.EPOCHS = 10

    print(f"--- Starting Training (Epochs: {Config.EPOCHS}) ---")

    # 2. Train Model
    # This saves the best model to Config.CHECKPOINT_PATH
    train_model(load_cached_data=True)

    print("\n--- Starting Evaluation & Failure Analysis ---")

    # 3. Load Validation Data
    # We need the data loader to iterate through validation samples
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # We only need the validation loader here
    _, val_loader = get_loaders(train_df, val_df, load_cached_data=True)

    # 4. Load Best Model Checkpoint
    device = torch.device(Config.DEVICE)
    model = HPUnet(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)

    if not os.path.exists(Config.CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {Config.CHECKPOINT_PATH}")
        return

    checkpoint = torch.load(Config.CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 5. Run Validation Inference
    all_preds = []
    all_targets = []

    # Lists for Failure Analysis
    patch_errors = []
    patch_intensities = []  # Mean intensity of Channel 0 (Global MIP)

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Store predictions and targets for global metric calculation
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

            # --- Failure Analysis Data Collection ---
            # Calculate Mean Absolute Error per patch: |Pred - Target|
            # Shape: (B, 1, H, W)
            mae = torch.abs(probs - masks)
            # Average over spatial dimensions -> (B, 1)
            batch_mae = mae.mean(dim=[1, 2, 3])
            patch_errors.extend(batch_mae.cpu().numpy())

            # Calculate Mean Intensity of Channel 0 (Global MIP) per patch
            # images shape: (B, 4, H, W) -> Take channel 0
            batch_intensity = images[:, 0, :, :].mean(dim=[1, 2])
            patch_intensities.extend(batch_intensity.cpu().numpy())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # 6. Compute Global Metric
    # We compute the F0.5 score over the entire validation set (Micro-averaged)
    final_metric = fbeta_score(
        all_preds, all_targets, beta=0.5, threshold=Config.THRESHOLD
    )

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # 7. Perform Failure Analysis
    if len(patch_errors) > 1:
        # Calculate Pearson correlation between Error and Input Intensity
        corr, p_val = stats.pearsonr(patch_errors, patch_intensities)
        print(
            f"Failure Analysis - Correlation (Error vs Global MIP Intensity): {corr:.6f} (p-value: {p_val:.6f})"
        )

        if abs(corr) > 0.3:
            print(
                "Observation: Significant correlation detected. Model performance varies with ink/papyrus radiodensity."
            )
        else:
            print(
                "Observation: Low correlation detected. Errors may be driven by shape or texture complexity rather than intensity."
            )

    # 8. Conditional Submission
    BASELINE_SCORE = 0.4738558828830719

    if final_metric > BASELINE_SCORE:
        print(
            f"\nValidation Score ({final_metric}) exceeds baseline ({BASELINE_SCORE})."
        )
        print("Generating submission file...")
        run_inference(load_cached_data=True)
    else:
        print(
            f"\nValidation Score ({final_metric}) does not exceed baseline ({BASELINE_SCORE})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
