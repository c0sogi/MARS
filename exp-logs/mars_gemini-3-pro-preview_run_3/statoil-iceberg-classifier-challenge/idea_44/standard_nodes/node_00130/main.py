import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library import config, utils, data, model, train


def run_oof_validation(device):
    """
    Generates Out-Of-Fold predictions for the entire dataset using the 5 trained models.
    Returns:
        oof_df (pd.DataFrame): DataFrame containing ids, targets, predictions, and metadata.
        final_metric (float): The overall Log Loss.
    """
    print("Generating OOF predictions for validation...")

    all_targets = []
    all_preds = []
    all_angles = []
    all_b1_means = []
    all_b2_means = []

    # Iterate over each fold
    for fold_idx in range(config.NUM_FOLDS):
        # Load validation data for this fold
        _, val_loader = data.get_dataloaders(fold_idx, load_cached_data=True)

        # Load Model
        net = model.SPPCNN().to(device)
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        utils.load_checkpoint(net, checkpoint_path, device=device)
        net.eval()

        fold_targets = []
        fold_preds = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_t = angles.to(device)

                # Forward pass
                outputs = net(images, angles_t)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                # Store data
                fold_preds.extend(probs)
                fold_targets.extend(labels.numpy().flatten())
                all_angles.extend(angles.numpy().flatten())

                # Calculate simple image stats for failure analysis
                # images is (B, 3, 75, 75). Channel 0 is HH, 1 is HV.
                imgs_np = images.cpu().numpy()
                all_b1_means.extend(np.mean(imgs_np[:, 0, :, :], axis=(1, 2)))
                all_b2_means.extend(np.mean(imgs_np[:, 1, :, :], axis=(1, 2)))

        all_targets.extend(fold_targets)
        all_preds.extend(fold_preds)

    # Calculate Metric
    # Clip predictions to avoid log(0)
    y_pred = np.clip(all_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(all_targets, y_pred)

    print(f"Final Validation Metric: {final_metric}")

    # Create DataFrame for analysis
    oof_df = pd.DataFrame(
        {
            "target": all_targets,
            "prediction": all_preds,
            "inc_angle": all_angles,
            "b1_mean": all_b1_means,
            "b2_mean": all_b2_means,
        }
    )

    return oof_df, final_metric


def perform_failure_analysis(oof_df):
    """
    Analyzes the correlation between error magnitude and input features.
    """
    print("\n--- Failure Analysis ---")

    # Calculate Error Magnitude
    oof_df["error"] = np.abs(oof_df["target"] - oof_df["prediction"])

    # Features to correlate
    features = ["inc_angle", "b1_mean", "b2_mean"]

    print("Correlation between Error Magnitude and Features:")
    for feat in features:
        # Handle potential NaNs if any
        valid_df = oof_df.dropna(subset=[feat, "error"])
        if len(valid_df) > 1:
            corr, _ = pearsonr(valid_df[feat], valid_df["error"])
            print(f"  {feat}: {corr:.4f}")
        else:
            print(f"  {feat}: N/A (Insufficient data)")


def generate_submission(device):
    """
    Generates predictions for the test set using an ensemble of all 5 models.
    Saves to submission.csv.
    """
    print("\nGenerating submission for test set...")

    # Load Test Data
    test_loader, test_ids = data.get_test_dataloader(load_cached_data=True)

    # Initialize array to store sum of predictions
    avg_preds = np.zeros(len(test_ids))

    # Iterate over all folds
    for fold_idx in range(config.NUM_FOLDS):
        # Load Model
        net = model.SPPCNN().to(device)
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        utils.load_checkpoint(net, checkpoint_path, device=device)
        net.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = net(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_preds.extend(probs)

        avg_preds += np.array(fold_preds)

    # Average predictions
    avg_preds /= config.NUM_FOLDS

    # Create Submission DataFrame
    sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def main():
    # Set Seed
    utils.set_seed(config.SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Train
    print("Starting Training...")
    # This executes the training loop for all folds defined in config
    train.run_training()

    # 2. Validation & Failure Analysis
    oof_df, final_metric = run_oof_validation(device)
    perform_failure_analysis(oof_df)

    # 3. Submission
    THRESHOLD = 0.1806015565870406
    if final_metric < THRESHOLD:
        generate_submission(device)
    else:
        print(
            f"\nValidation metric ({final_metric}) did not meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
