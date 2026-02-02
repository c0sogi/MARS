import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, save_submission
from library.data_loader import get_dataloaders
from library.model import WideSEResNet
from library.engine import train_fold


def extract_features_and_predict(model, loader, device):
    """
    Runs inference on a loader and extracts image statistics for failure analysis.
    Returns: predictions, targets, feature_dataframe
    """
    model.eval()
    preds = []
    targets = []

    # Features for analysis
    feat_inc_angle = []
    feat_b1_mean = []
    feat_b1_std = []
    feat_b2_mean = []
    feat_b2_std = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles_gpu = angles.to(device)

            # Forward pass
            logits = model(images, angles_gpu)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.extend(probs)

            # Handle labels (might be IDs for test set, but we use this for validation here)
            if isinstance(labels, torch.Tensor):
                targets.extend(labels.cpu().numpy())
            else:
                targets.extend(labels)  # IDs

            # Extract features from CPU images for analysis
            # images shape: (B, 3, 75, 75) -> Band 0 is HH, Band 1 is HV
            imgs_np = images.cpu().numpy()

            # Flatten spatial dims for stat calculation: (B, 3, 5625)
            imgs_flat = imgs_np.reshape(imgs_np.shape[0], 3, -1)

            b1 = imgs_flat[:, 0, :]
            b2 = imgs_flat[:, 1, :]

            feat_b1_mean.extend(np.mean(b1, axis=1))
            feat_b1_std.extend(np.std(b1, axis=1))
            feat_b2_mean.extend(np.mean(b2, axis=1))
            feat_b2_std.extend(np.std(b2, axis=1))
            feat_inc_angle.extend(angles.numpy())

    features = pd.DataFrame(
        {
            "inc_angle": feat_inc_angle,
            "b1_mean": feat_b1_mean,
            "b1_std": feat_b1_std,
            "b2_mean": feat_b2_mean,
            "b2_std": feat_b2_std,
        }
    )

    return np.array(preds), np.array(targets), features


def predict_test(model, loader, device):
    """
    Simple inference for test set (returns probs and ids).
    """
    model.eval()
    preds = []
    ids_list = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy()

            preds.extend(probs)
            ids_list.extend(ids)

    return np.array(preds), ids_list


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}")

    # Storage for OOF and Test predictions
    oof_preds = []
    oof_targets = []
    oof_features = []

    test_preds_fold = []
    test_ids = None

    # 2. Training Loop (5 Folds)
    for fold_idx in range(Config.NUM_FOLDS):
        # Train the fold
        train_fold(fold_idx)

        # Load the best model for this fold
        model = WideSEResNet().to(device)
        checkpoint_path = Config.get_checkpoint_path(fold_idx)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # Get DataLoaders
        # We need val_loader for OOF analysis and test_loader for submission
        _, val_loader, test_loader = get_dataloaders(
            fold_idx=fold_idx, load_cached_data=True
        )

        print(f"Generating analysis data for Fold {fold_idx}...")

        # Validation Inference
        val_p, val_t, val_f = extract_features_and_predict(model, val_loader, device)
        oof_preds.append(val_p)
        oof_targets.append(val_t)
        oof_features.append(val_f)

        # Test Inference
        test_p, ids = predict_test(model, test_loader, device)
        test_preds_fold.append(test_p)

        if test_ids is None:
            test_ids = ids

    # 3. Global Evaluation
    # Concatenate OOF results
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_targets = np.concatenate(oof_targets)
    all_oof_features = pd.concat(oof_features, ignore_index=True)

    # Calculate Metric
    final_metric = log_loss(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    # For binary classification, error can be defined as abs(y_true - y_pred)
    errors = np.abs(all_oof_targets - all_oof_preds)

    # Correlate errors with features
    analysis_df = all_oof_features.copy()
    analysis_df["error"] = errors

    correlations = {}
    for col in all_oof_features.columns:
        corr, _ = pearsonr(analysis_df[col], analysis_df["error"])
        correlations[col] = corr

    print("Correlation between Error Magnitude and Input Features:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        print(f"  {feat}: {corr:.4f}")

    # 5. Submission Logic
    THRESHOLD = 0.18145903282502943

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Average test predictions across folds
        avg_test_preds = np.mean(test_preds_fold, axis=0)

        # Save
        save_submission(test_ids, avg_test_preds, Config.SUBMISSION_FILE)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
