import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, check_initial_loss
from library.data import get_loaders
from library.model import AppleResNet34
from library.engine import fit, predict


def main():
    # 1. Setup
    seed_everything(42)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # ==========================================
    # Phase 1: Calibration (5-Fold CV)
    # ==========================================
    print("\n==== Phase 1: Calibration (5-Fold CV) ====")

    oof_preds = []
    oof_targets = []
    best_epochs = []

    # Containers for failure analysis
    fa_image_paths = []
    fa_errors = []

    for fold in range(Config.N_FOLDS):
        print(f"\n--- Fold {fold + 1}/{Config.N_FOLDS} ---")

        # Get DataLoaders
        train_loader, val_loader = get_loaders(fold_idx=fold, phase="calibration")

        # Initialize Model
        model = AppleResNet34().to(device)

        # Initial Loss Check (Sanity Check) - Only on first fold
        if fold == 0:
            print("Performing Initial Loss Check...")
            criterion = torch.nn.CrossEntropyLoss()
            check_initial_loss(model, train_loader, criterion, device)

        # Train (Calibration)
        # fit() handles Early Stopping and loads the best model state before returning
        history, best_epoch = fit(
            model, train_loader, val_loader, epochs=Config.MAX_EPOCHS, device=device
        )
        best_epochs.append(best_epoch)
        print(f"Fold {fold + 1} Best Epoch: {best_epoch}")

        # Generate OOF Predictions
        model.eval()
        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                fold_preds.append(probs.cpu().numpy())
                fold_targets.append(targets.numpy())

        fold_preds = np.concatenate(fold_preds)
        fold_targets = np.concatenate(fold_targets)

        oof_preds.append(fold_preds)
        oof_targets.append(fold_targets)

        # Collect data for Failure Analysis
        # Get file paths from the validation dataset
        fold_paths = val_loader.dataset.file_paths
        fa_image_paths.extend(fold_paths)

        # Calculate Error Magnitude: 1.0 - Probability of the Ground Truth Class
        # Assuming one-hot encoded targets or similar distribution
        true_indices = np.argmax(fold_targets, axis=1)
        true_class_probs = fold_preds[np.arange(len(fold_preds)), true_indices]
        errors = 1.0 - true_class_probs
        fa_errors.extend(errors)

    # Aggregate OOF Results
    all_oof_preds = np.concatenate(oof_preds)
    all_oof_targets = np.concatenate(oof_targets)

    # Calculate Final Validation Metric
    final_metric = calculate_roc_auc(all_oof_targets, all_oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Determine Optimal Epoch (E_opt)
    avg_best_epoch = int(np.round(np.mean(best_epochs)))
    print(f"Global Optimal Epoch (E_opt): {avg_best_epoch}")

    # ==========================================
    # Failure Analysis
    # ==========================================
    print("\n==== Failure Analysis ====")

    # We correlate error magnitude with basic image features (Intensity, Aspect Ratio)
    # Need to read images to get these stats.

    intensities = []
    aspect_ratios = []

    # Iterate through validated images
    # Note: fa_image_paths are relative paths like "images/Train_0.jpg"
    for rel_path in fa_image_paths:
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        img = cv2.imread(full_path)

        if img is not None:
            h, w, c = img.shape
            # Calculate mean intensity (normalized)
            mean_intensity = img.mean() / 255.0
            ar = w / float(h)

            intensities.append(mean_intensity)
            aspect_ratios.append(ar)
        else:
            # Fallback if image read fails (should not happen)
            intensities.append(0.5)
            aspect_ratios.append(1.0)

    # Calculate Correlations
    if len(fa_errors) > 1:
        corr_intensity, _ = pearsonr(fa_errors, intensities)
        corr_ar, _ = pearsonr(fa_errors, aspect_ratios)

        print(f"Correlation (Error vs Intensity): {corr_intensity:.6f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.6f}")
    else:
        print("Insufficient data for correlation analysis.")

    # ==========================================
    # Phase 2: Production (Seed Ensemble)
    # ==========================================
    THRESHOLD = 0.9871488489626378

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Proceeding to Phase 2..."
        )

        # Load Test Loader
        test_loader = get_loaders(phase="test")

        ensemble_preds = []

        for i, seed in enumerate(Config.SEEDS):
            print(
                f"\nTraining Production Model {i+1}/{len(Config.SEEDS)} (Seed: {seed})"
            )

            # Set Seed for this run
            seed_everything(seed)

            # Get Full Data Loader (100% Data)
            # Note: seed passed to get_loaders might not be used if shuffle=True is set in loader
            # but seed_everything handles the global random state for shuffle.
            train_loader, _ = get_loaders(phase="production", seed=seed)

            # Initialize fresh model
            model = AppleResNet34().to(device)

            # Train for fixed E_opt epochs (no validation)
            fit(
                model,
                train_loader,
                val_loader=None,
                epochs=avg_best_epoch,
                device=device,
            )

            # Predict on Test Set
            preds = predict(model, test_loader, device)
            ensemble_preds.append(preds)

        # Average Predictions (Ensemble)
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Generate Submission
        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        submission = pd.DataFrame(avg_preds, columns=Config.TARGET_COLS)
        submission.insert(0, "image_id", df_test["image_id"])

        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()
