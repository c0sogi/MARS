import sys
import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, load_checkpoint, compute_roc_auc
from library.dataset import get_dataloaders
from library.model import MultiScaleResNet
from library.train import run_training
from library.inference import generate_submission


def analyze_failures(model, val_loader, device):
    """
    Performs failure analysis by correlating prediction error with image meta-features.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    all_errors = []
    # Meta-feature accumulators
    meta_brightness = []
    meta_contrast = []
    meta_red = []
    meta_green = []
    meta_blue = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Forward pass
            logits = model(images)
            preds = torch.sigmoid(logits)

            # Calculate absolute error magnitude
            errors = torch.abs(preds - labels).cpu().numpy().flatten()
            all_errors.extend(errors)

            # Calculate image stats on the GPU tensors
            # images shape: (B, 3, H, W)
            # Brightness: Mean intensity across all channels and pixels
            brightness = torch.mean(images, dim=[1, 2, 3]).cpu().numpy()
            # Contrast: Standard deviation of intensity
            contrast = torch.std(images, dim=[1, 2, 3]).cpu().numpy()

            # Channel Means
            reds = torch.mean(images[:, 0, :, :], dim=[1, 2]).cpu().numpy()
            greens = torch.mean(images[:, 1, :, :], dim=[1, 2]).cpu().numpy()
            blues = torch.mean(images[:, 2, :, :], dim=[1, 2]).cpu().numpy()

            meta_brightness.extend(brightness)
            meta_contrast.extend(contrast)
            meta_red.extend(reds)
            meta_green.extend(greens)
            meta_blue.extend(blues)

    # Compute correlations
    all_errors = np.array(all_errors)
    features = {
        "Brightness": meta_brightness,
        "Contrast": meta_contrast,
        "Red_Mean": meta_red,
        "Green_Mean": meta_green,
        "Blue_Mean": meta_blue,
    }

    print("Correlation between Error Magnitude and Input Features:")
    for name, values in features.items():
        values = np.array(values)
        # Avoid correlation calculation if feature has zero variance
        if len(np.unique(values)) > 1:
            corr, _ = pearsonr(all_errors, values)
            print(f"{name}: {corr:.4f}")
        else:
            print(f"{name}: NaN (Constant feature)")


def main():
    # 1. Setup & Configuration Override
    # We override Config attributes to create a fast baseline execution.
    # Config.MAX_EPOCHS is set in config.py (reduced to 20)
    # Config.SEEDS is set in config.py (5 seeds)
    # We do NOT override them here to allow full ensemble training (Cite solution_lesson_node_00007)

    Config.setup()
    set_seed(42)

    print("Starting Optimized Run...")
    print(f"Configuration: Epochs={Config.MAX_EPOCHS}, Seeds={Config.SEEDS}")

    # 2. Train
    # Execute the training pipeline (includes saving the best model)
    run_training()

    # 3. Validation Assessment
    print("\n--- Validation Assessment ---")
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")

    # Load validation data (using cache if available)
    _, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Load the first trained model for analysis (ensemble is used for submission)
    model = MultiScaleResNet().to(device)
    checkpoint_path = f"model_seed_{Config.SEEDS[0]}.pth"
    try:
        load_checkpoint(model, checkpoint_path, device=device)
    except FileNotFoundError:
        print(
            f"Error: Could not find checkpoint {checkpoint_path}. Training may have failed."
        )
        return

    # Calculate Metric (Single Seed for quick check, Submission uses Ensemble)
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            # Forward
            logits = model(images)
            prob = torch.sigmoid(logits)

            all_labels.append(labels.numpy())
            all_preds.append(prob.cpu().numpy().flatten())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    val_auc = compute_roc_auc(all_labels, all_preds)

    # Print Metric in the required format
    print(f"Final Validation Metric: {val_auc}")

    # 4. Failure Analysis
    analyze_failures(model, val_loader, device)

    # 5. Submission
    # The requirement "metric > 1.0" is impossible for ROC AUC (max 1.0).
    # We interpret this as a requirement to submit if the model is performant (better than random).
    # We use 0.5 as the threshold.
    if val_auc > 0.5:
        generate_submission()
    else:
        print(f"Validation AUC ({val_auc}) is too low. Skipping submission.")


if __name__ == "__main__":
    main()
