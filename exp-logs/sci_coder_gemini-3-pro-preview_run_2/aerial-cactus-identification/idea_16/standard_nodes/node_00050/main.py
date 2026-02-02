import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

# Import from provided library files
from library.utils import seed_everything, get_device, calculate_roc_auc
from library.dataset import get_dataloaders
from library.model import WideResNetMultiScale
from library.train import run_training


def analyze_failures(val_loader, val_preds):
    """
    Performs failure analysis by correlating prediction errors with image statistics.
    """
    print("\n--- Failure Analysis ---")

    targets = []
    stats = {
        "brightness": [],
        "contrast": [],
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
    }

    print("Computing image statistics for validation set...")

    # Iterate through loader to get images and calculate stats
    # val_loader returns images as (B, C, H, W) tensors in [0, 1]
    with torch.no_grad():
        for images, labels in val_loader:
            targets.extend(labels.numpy())

            # Compute stats per image
            # Mean over spatial dims (H, W) for each channel -> (B, 3)
            channel_means = images.mean(dim=[2, 3]).cpu().numpy()

            # Global mean (brightness) over (C, H, W) -> (B,)
            brightness = images.mean(dim=[1, 2, 3]).cpu().numpy()

            # Global std (contrast) over (C, H, W) -> (B,)
            contrast = images.std(dim=[1, 2, 3]).cpu().numpy()

            stats["brightness"].extend(brightness)
            stats["contrast"].extend(contrast)
            stats["red_mean"].extend(channel_means[:, 0])
            stats["green_mean"].extend(channel_means[:, 1])
            stats["blue_mean"].extend(channel_means[:, 2])

    targets = np.array(targets)
    val_preds = np.array(val_preds)

    # Calculate Error Magnitude: |y_true - y_pred|
    errors = np.abs(targets - val_preds)

    print(f"Average Absolute Error: {np.mean(errors):.6f}")

    # Calculate correlations
    print("\nCorrelation between Error Magnitude and Image Features:")
    for feature_name, values in stats.items():
        values = np.array(values)
        if len(values) != len(errors):
            continue

        corr, pval = pearsonr(values, errors)
        print(
            f"{feature_name.ljust(12)}: Correlation = {corr:.4f} (p-value = {pval:.4f})"
        )


def main():
    # 1. Configuration
    seed_everything(42)
    device = get_device()

    # Parameters for robust baseline
    N_FOLDS = 5
    EPOCHS = 20
    BATCH_SIZE = 64
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    print("Starting Runfile Execution...")

    # 2. Train Models and Generate Test Submission
    # run_training handles the loop over seeds, training, and generating submission.csv
    run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        n_folds=N_FOLDS,
        working_dir=WORKING_DIR,
        submission_dir=SUBMISSION_DIR,
        load_cached_data=True,
    )

    # 3. Ensemble Validation & Metric Calculation
    print("\n--- Performing Ensemble Validation ---")

    # Get DataLoaders (cached)
    dataloaders = get_dataloaders(batch_size=BATCH_SIZE, load_cached_data=True)
    val_loader = dataloaders["val"]

    # Prepare to collect predictions
    val_dataset_len = len(val_loader.dataset)
    ensemble_val_preds = np.zeros((val_dataset_len, N_FOLDS))
    val_targets = []

    # Collect ground truth labels once
    for _, labels in val_loader:
        val_targets.extend(labels.numpy())
    val_targets = np.array(val_targets)

    # Iterate over trained seeds to get validation predictions
    for seed in range(N_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found at {model_path}")
            continue

        # Load Model
        model = WideResNetMultiScale().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Predict on Validation Set
        seed_preds = []
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                seed_preds.append(probs)

        ensemble_val_preds[:, seed] = np.concatenate(seed_preds).flatten()

    # Average predictions across ensemble (Homogeneous Seed Averaging)
    avg_val_preds = np.mean(ensemble_val_preds, axis=1)

    # Calculate Final Metric
    final_auc = calculate_roc_auc(val_targets, avg_val_preds)

    # Print the required metric in the specific format
    print(f"Final Validation Metric: {final_auc}")

    # 4. Failure Analysis
    analyze_failures(val_loader, avg_val_preds)

    # 5. Submission Verification
    sub_file = os.path.join(SUBMISSION_DIR, "submission.csv")
    if os.path.exists(sub_file):
        print(f"Submission successfully generated at: {sub_file}")
    else:
        print("Error: Submission file was not found.")


if __name__ == "__main__":
    main()
