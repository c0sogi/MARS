import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.data import get_fold_loaders, get_test_loader
from library.model import CCTICNN
from library.train import run_fold

# --- Configuration Overrides for Fast Execution ---
# Reduce epochs to ensure completion within the time limit.
# The dataset is small, so 25 epochs is sufficient for convergence.
Config.NUM_EPOCHS = 25


def analyze_failures(df_results):
    """
    Performs failure analysis by correlating error with features.
    """
    # Calculate absolute error
    df_results["error"] = (df_results["target"] - df_results["pred"]).abs()

    # Calculate correlations
    correlations = {}
    features = ["inc_angle", "signal_mean", "signal_std"]

    print("\n--- Failure Analysis (Correlation with Error) ---")
    for feat in features:
        if feat in df_results.columns:
            corr = df_results["error"].corr(df_results[feat])
            correlations[feat] = corr
            print(f"{feat}: {corr:.4f}")

    return correlations


def main():
    # Set global seed
    set_seed(Config.SEED)

    # Ensure directories exist
    Config.create_directories()

    device = torch.device(Config.DEVICE)

    # -------------------------------------------------------------------------
    # 1. Training Loop (5-Fold CV)
    # -------------------------------------------------------------------------
    print("Starting 5-Fold Cross-Validation Training...")

    for fold_idx in range(Config.NUM_FOLDS):
        # run_fold handles training, logging, and saving the best checkpoint
        run_fold(fold_idx)

    # -------------------------------------------------------------------------
    # 2. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\nStarting Validation and Failure Analysis...")

    all_preds = []
    all_targets = []
    all_ids = []
    all_angles = []
    all_signal_means = []
    all_signal_stds = []

    # Re-load models and predict on validation sets to gather full stats
    criterion = nn.BCEWithLogitsLoss()

    for fold_idx in range(Config.NUM_FOLDS):
        # Get validation loader for this fold
        _, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

        # Load Model
        model = CCTICNN()
        model.to(device)

        ckpt_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for (images, angles), targets, ids in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Forward
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                # Collect data
                all_preds.extend(probs)
                all_targets.extend(targets.numpy().flatten())
                all_ids.extend(ids)
                all_angles.extend(angles.numpy().flatten())

                # Compute simple image stats for failure analysis (on CPU)
                # images is (B, 3, 75, 75). Channel 0 is HH, 1 is HV.
                imgs_np = images.cpu().numpy()
                # Mean of HH and HV channels per image
                means = np.mean(imgs_np[:, 0:2, :, :], axis=(1, 2, 3))
                stds = np.std(imgs_np[:, 0:2, :, :], axis=(1, 2, 3))

                all_signal_means.extend(means)
                all_signal_stds.extend(stds)

    # Create DataFrame for analysis
    df_val = pd.DataFrame(
        {
            "id": all_ids,
            "target": all_targets,
            "pred": all_preds,
            "inc_angle": all_angles,
            "signal_mean": all_signal_means,
            "signal_std": all_signal_stds,
        }
    )

    # Calculate Metric
    final_log_loss = log_loss(df_val["target"], df_val["pred"])
    print(f"Final Validation Metric: {final_log_loss}")

    # Failure Analysis
    analyze_failures(df_val)

    # -------------------------------------------------------------------------
    # 3. Submission Generation
    # -------------------------------------------------------------------------
    threshold = 0.17174082291273365

    if final_log_loss < threshold:
        print(
            f"\nMetric ({final_log_loss}) is better than threshold ({threshold}). Generating submission..."
        )

        # Load Test Data
        test_loader = get_test_loader(load_cached_data=True)

        # Initialize accumulator
        # We need to map IDs to predictions properly.
        # The test loader returns batches. We will accumulate predictions in a list
        # and match them with IDs.

        # We will store the sum of probabilities for each sample
        test_probs_sum = None
        test_ids = []

        # Iterate over folds
        for fold_idx in range(Config.NUM_FOLDS):
            print(f"Predicting with model fold {fold_idx}...")

            # Load Model
            model = CCTICNN()
            model.to(device)
            ckpt_path = os.path.join(
                Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            fold_preds = []
            current_ids = []

            with torch.no_grad():
                for (images, angles), _, ids in test_loader:
                    images = images.to(device)
                    angles_gpu = angles.to(device)

                    logits = model(images, angles_gpu)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

                    fold_preds.extend(probs)
                    if fold_idx == 0:
                        current_ids.extend(ids)

            fold_preds = np.array(fold_preds)

            if test_probs_sum is None:
                test_probs_sum = fold_preds
                test_ids = current_ids
            else:
                test_probs_sum += fold_preds

        # Average predictions
        avg_probs = test_probs_sum / Config.NUM_FOLDS

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_probs})

        # Save
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_log_loss}) did not meet threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
