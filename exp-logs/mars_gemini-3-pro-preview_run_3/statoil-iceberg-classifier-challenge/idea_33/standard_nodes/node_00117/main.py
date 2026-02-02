import os
import sys
import numpy as np
import pandas as pd
import torch

# Import from library
from library.config import Config
from library.train import run_fold
from library.data import get_loaders
from library.model import NBHACNN
from library.utils import set_seed, calculate_log_loss


def main():
    # 1. Setup
    # Using debug=False to ensure full training for best performance
    config = Config(debug=False)
    set_seed(config.seed)

    print(f"Starting execution with device: {config.device}")

    # Storage for OOF analysis
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_b1_means = []
    oof_b2_means = []

    # 2. Training Loop (5 Folds)
    for fold_idx in range(config.n_folds):
        print(f"\n--- Processing Fold {fold_idx} ---")

        # Train the model for this fold
        # This function handles the training loop and saves the best model checkpoint
        run_fold(config, fold_idx)

        # Load the best model for this fold to generate OOF predictions
        print(f"Loading best model for Fold {fold_idx}...")
        model = NBHACNN(config)
        ckpt_path = config.get_checkpoint_path(fold_idx)
        model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
        model.to(config.device)
        model.eval()

        # Get validation loader for this fold
        # Note: get_loaders with fold_idx returns (train, val, test)
        _, val_loader, _ = get_loaders(config, fold_idx=fold_idx)

        # Inference on Validation Set
        fold_probs = []
        fold_y = []
        fold_a = []
        fold_b1 = []
        fold_b2 = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                # Move to device
                images = images.to(config.device)
                angles_gpu = angles.to(config.device)

                # Forward
                logits = model(images, angles_gpu)
                probs = torch.sigmoid(logits)

                # Store results
                fold_probs.append(probs.cpu().numpy())
                fold_y.append(targets.cpu().numpy())
                fold_a.append(angles.numpy())

                # Calculate Image Stats for Failure Analysis
                # images is (B, 3, 75, 75). Channel 0 = HH, Channel 1 = HV
                # Calculate mean per image
                b1_mean = images[:, 0, :, :].mean(dim=(1, 2)).cpu().numpy()
                b2_mean = images[:, 1, :, :].mean(dim=(1, 2)).cpu().numpy()

                fold_b1.append(b1_mean)
                fold_b2.append(b2_mean)

        # Concatenate fold results
        fold_probs = np.concatenate(fold_probs)
        fold_y = np.concatenate(fold_y)
        fold_a = np.concatenate(fold_a)
        fold_b1 = np.concatenate(fold_b1)
        fold_b2 = np.concatenate(fold_b2)

        # Store for global analysis
        oof_preds.append(fold_probs)
        oof_targets.append(fold_y)
        oof_angles.append(fold_a)
        oof_b1_means.append(fold_b1)
        oof_b2_means.append(fold_b2)

        # Fold Metric
        fold_loss = calculate_log_loss(fold_y, fold_probs)
        print(f"Fold {fold_idx} Validation Log Loss: {fold_loss}")

    # 3. Global Validation & Failure Analysis
    print("\n--- Global Validation Results ---")
    all_preds = np.concatenate(oof_preds)
    all_targets = np.concatenate(oof_targets)
    all_angles = np.concatenate(oof_angles)
    all_b1 = np.concatenate(oof_b1_means)
    all_b2 = np.concatenate(oof_b2_means)

    final_metric = calculate_log_loss(all_targets, all_preds)
    # MUST PRINT EXACTLY THIS FORMAT
    print(f"Final Validation Metric: {final_metric}")

    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(all_targets.flatten() - all_preds.flatten())

    # 1. Correlation with Incidence Angle
    # Flatten angles
    flat_angles = all_angles.flatten()
    if np.std(flat_angles) > 0:
        corr_angle = np.corrcoef(errors, flat_angles)[0, 1]
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (Constant)")

    # 2. Correlation with Band 1 Mean (HH)
    corr_b1 = np.corrcoef(errors, all_b1.flatten())[0, 1]
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")

    # 3. Correlation with Band 2 Mean (HV)
    corr_b2 = np.corrcoef(errors, all_b2.flatten())[0, 1]
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.4f}")

    # 4. Submission Generation
    threshold = 0.1806015565870406
    if final_metric < threshold:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({threshold}). Generating submission..."
        )

        # Get Test Loader (Full test set)
        _, _, test_loader = get_loaders(config, fold_idx=None)

        # Ensemble Predictions
        ensemble_preds = None
        test_ids = []

        # Iterate over all 5 trained models
        for fold_idx in range(config.n_folds):
            print(f"Inference with Model Fold {fold_idx}...")

            # Load Model
            model = NBHACNN(config)
            ckpt_path = config.get_checkpoint_path(fold_idx)
            model.load_state_dict(torch.load(ckpt_path, map_location=config.device))
            model.to(config.device)
            model.eval()

            fold_preds = []
            current_ids = []

            with torch.no_grad():
                for images, angles, ids in test_loader:
                    images = images.to(config.device)
                    angles_gpu = angles.to(config.device)

                    logits = model(images, angles_gpu)
                    probs = torch.sigmoid(logits)

                    fold_preds.append(probs.cpu().numpy())

                    # Collect IDs only on the first pass
                    if fold_idx == 0:
                        current_ids.extend(ids)

            fold_preds = np.concatenate(fold_preds)

            # Accumulate
            if ensemble_preds is None:
                ensemble_preds = fold_preds
                test_ids = current_ids
            else:
                ensemble_preds += fold_preds

        # Average
        avg_preds = ensemble_preds / config.n_folds

        # Flatten
        avg_preds = avg_preds.flatten()

        # Save Submission
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

        submission.to_csv(config.submission_path, index=False)
        print(f"Submission saved to {config.submission_path}")

    else:
        print(
            f"\nMetric ({final_metric}) did not beat threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
