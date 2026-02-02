import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from library.config import Config
from library.utils import set_seed, load_dataset
from library.data_loader import get_loaders, get_test_loader
from library.model import WA_IDPH_CNN
from library.train import run_fold


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Ensure Config is set up (directories created)
    Config.setup()

    print("Starting Fast Baseline Run...")

    # Lists to store results across folds
    fold_losses = []

    # Storage for Failure Analysis
    all_y_true = []
    all_y_pred = []
    all_angles = []
    all_b1_means = []
    all_b1_stds = []
    all_b2_means = []
    all_b2_stds = []

    device = torch.device(Config.DEVICE)

    # 1. Train and Validate across Folds
    for fold_idx in range(Config.N_FOLDS):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Train the fold and retrieve the best validation loss
        # run_fold handles training, early stopping, and checkpoint saving
        best_loss = run_fold(fold_idx)
        fold_losses.append(best_loss)

        # --- Post-Fold Analysis Data Collection ---
        # Reload the best model for this fold to get predictions and features
        model = WA_IDPH_CNN()
        model.to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        # Get validation loader for this fold
        _, val_loader = get_loaders(fold_idx)

        # Collect predictions
        fold_y_true = []
        fold_y_pred = []
        fold_angles = []
        fold_images = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Forward pass
                outputs = model(images, angles_gpu)
                probs = torch.sigmoid(outputs)

                # Store data (move to CPU)
                fold_y_true.append(labels.cpu().numpy())
                fold_y_pred.append(probs.cpu().numpy())
                fold_angles.append(angles.numpy())
                fold_images.append(images.cpu().numpy())

        # Concatenate fold data
        fold_y_true = np.concatenate(fold_y_true)
        fold_y_pred = np.concatenate(fold_y_pred)
        fold_angles = np.concatenate(fold_angles)
        fold_images = np.concatenate(fold_images)  # Shape: (N, 3, 75, 75)

        # Append to global lists
        all_y_true.append(fold_y_true)
        all_y_pred.append(fold_y_pred)
        all_angles.append(fold_angles)

        # Calculate Image Stats for Analysis (Band 1=0, Band 2=1)
        # Mean/Std per image across pixels
        all_b1_means.append(np.mean(fold_images[:, 0, :, :], axis=(1, 2)))
        all_b1_stds.append(np.std(fold_images[:, 0, :, :], axis=(1, 2)))
        all_b2_means.append(np.mean(fold_images[:, 1, :, :], axis=(1, 2)))
        all_b2_stds.append(np.std(fold_images[:, 1, :, :], axis=(1, 2)))

    # 2. Calculate Final Validation Metric
    # We use the average of the best fold losses as the CV score
    final_metric = np.mean(fold_losses)
    print(f"\nFinal Validation Metric: {final_metric:.10f}")

    # 3. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Flatten all lists
    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)
    angles_flat = np.concatenate(all_angles)
    b1_means_flat = np.concatenate(all_b1_means)
    b1_stds_flat = np.concatenate(all_b1_stds)
    b2_means_flat = np.concatenate(all_b2_means)
    b2_stds_flat = np.concatenate(all_b2_stds)

    # Calculate Error (Absolute difference)
    errors = np.abs(y_true_flat - y_pred_flat)

    # Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_flat,
            "b1_mean": b1_means_flat,
            "b1_std": b1_stds_flat,
            "b2_mean": b2_means_flat,
            "b2_std": b2_stds_flat,
        }
    )

    # Calculate Correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    # 4. Submission Generation
    threshold = 0.17174082291273365
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.6f}) is better than threshold ({threshold:.6f}). Generating submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_metric:.6f}) did not meet threshold ({threshold:.6f}). Skipping submission."
        )


def generate_submission():
    """
    Generates submission file by averaging predictions from all 5 fold models.
    """
    device = torch.device(Config.DEVICE)

    # Get Test Loader
    test_loader = get_test_loader()

    # Get Test IDs (load_dataset returns X, angles, y, ids)
    # We load cached test data to retrieve IDs quickly
    _, _, _, test_ids = load_dataset("test", load_cached_data=True)

    # Accumulate predictions
    avg_preds = np.zeros(len(test_ids))

    for fold_idx in range(Config.N_FOLDS):
        print(f"Inference with model from Fold {fold_idx}...")

        # Load Model
        model = WA_IDPH_CNN()
        model.to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth"
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)

                fold_preds.append(probs.cpu().numpy())

        # Concatenate and add to average
        fold_preds = np.concatenate(fold_preds)
        avg_preds += fold_preds

    # Average
    avg_preds /= Config.N_FOLDS

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    main()
