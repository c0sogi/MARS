"""
Iceberg Classifier - End-to-End Orchestration Script
Implements Idea 10: Ensembled Robust Micro-ResNet (ERM-ResNet)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library.config import Config, set_seed
from library.utils import load_checkpoint
from library.model import MicroResNet
from library.data_loader import get_loaders
from library.train_eval import train_fold


def analyze_failures(preds, targets, angles, image_stats):
    """
    Performs failure analysis on the validation set.
    Calculates correlations between error magnitude and input features.
    """
    # Calculate absolute error
    preds = np.array(preds)
    targets = np.array(targets)
    errors = np.abs(preds - targets)

    # Prepare features
    angles = np.array(angles)
    image_stats = np.array(image_stats)
    b1_means = image_stats[:, 0]
    b2_means = image_stats[:, 1]

    # Handle NaNs in angles (though they should be imputed)
    valid_mask = ~np.isnan(angles)

    if np.sum(valid_mask) > 1:
        corr_angle, _ = pearsonr(errors[valid_mask], angles[valid_mask])
    else:
        corr_angle = 0.0

    corr_b1, _ = pearsonr(errors, b1_means)
    corr_b2, _ = pearsonr(errors, b2_means)

    print(f"Correlation of Error with Incidence Angle: {corr_angle:.4f}")
    print(f"Correlation of Error with Band 1 Mean: {corr_b1:.4f}")
    print(f"Correlation of Error with Band 2 Mean: {corr_b2:.4f}")

    # Top failures
    sorted_indices = np.argsort(errors)[::-1]
    print("\nTop 5 High Confidence Errors:")
    for i in range(min(5, len(errors))):
        idx = sorted_indices[i]
        print(
            f"  Idx: {idx} | True: {targets[idx]:.0f} | Pred: {preds[idx]:.4f} | Error: {errors[idx]:.4f} | Angle: {angles[idx]:.1f}"
        )


def run_pipeline():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Ensure submission directory exists
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)

    # 2. Cross-Validation Loop
    num_folds = Config.NUM_FOLDS
    print(f"[{Config.PROJECT_NAME}] Starting {num_folds}-Fold Cross-Validation...")

    # Storage for Global OOF
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_stats = []

    # Storage for Test Predictions (Sum for averaging)
    test_preds_sum = None
    test_ids = None

    for fold in range(num_folds):
        print(f"\n--- Processing Fold {fold} ---")

        # Get DataLoaders
        # load_cached_data=True ensures we use the pre-processed numpy arrays
        train_loader, val_loader, test_loader = get_loaders(
            fold=fold, load_cached_data=True
        )

        # Train
        train_fold(fold, train_loader, val_loader)

        # Load Best Model for Inference
        model = MicroResNet().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return

        load_checkpoint(model, filename=checkpoint_path)
        model.eval()

        # Validation Inference (OOF)
        fold_preds = []
        fold_targets = []
        fold_angles = []
        fold_stats = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)

                # Forward
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                fold_preds.extend(probs)
                fold_targets.extend(targets.cpu().numpy().flatten())
                fold_angles.extend(angles.cpu().numpy().flatten())

                # Collect stats for failure analysis
                # images: (B, 3, 75, 75). Channel 0=HH, 1=HV
                imgs_np = images.cpu().numpy()
                for img in imgs_np:
                    fold_stats.append([np.mean(img[0]), np.mean(img[1])])

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)
        oof_angles.extend(fold_angles)
        oof_stats.extend(fold_stats)

        # Test Inference
        fold_test_preds = []
        fold_test_ids = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                outputs = model(images, angles)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                fold_test_preds.extend(probs)
                fold_test_ids.extend(ids)

        # Accumulate Test Predictions
        if test_preds_sum is None:
            test_preds_sum = np.zeros(len(fold_test_preds))
            test_ids = fold_test_ids

        test_preds_sum += np.array(fold_test_preds)

    # 3. Validation Evaluation
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Clip for numerical stability
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)

    final_metric = log_loss(oof_targets, oof_preds_clipped)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    analyze_failures(oof_preds, oof_targets, oof_angles, oof_stats)

    # 5. Submission
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric:.6f}) < Threshold ({threshold:.6f}). Generating submission..."
        )

        avg_test_preds = test_preds_sum / num_folds

        sub_df = pd.DataFrame({"id": test_ids, "is_iceberg": avg_test_preds})

        sub_path = os.path.join(submission_dir, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric ({final_metric:.6f}) >= Threshold ({threshold:.6f}). Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
