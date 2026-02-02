import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import get_loaders, get_test_loader
from library.model import RTICNN
from library.train import run_fold


def main():
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)

    # Adjust epochs for a fast but effective baseline run
    Config.NUM_EPOCHS = 50

    device = torch.device(Config.DEVICE)
    logger = setup_logger("Main", os.path.join(Config.WORKING_DIR, "main.log"))
    logger.info("Starting RTI-CNN Pipeline")
    Config.print_config()

    # 2. Training Loop (5 Folds)
    logger.info("--- Starting Cross-Validation Training ---")
    for fold_idx in range(Config.N_FOLDS):
        logger.info(f"Running Fold {fold_idx}")
        run_fold(fold_idx)

    # 3. Global Validation & Failure Analysis
    logger.info("--- Starting Global Validation & Failure Analysis ---")

    oof_preds = []
    oof_targets = []

    # Features for failure analysis
    feat_angles = []
    feat_b1_mean = []
    feat_b2_mean = []

    # Iterate folds to reconstruct validation sets and predict
    for fold_idx in range(Config.N_FOLDS):
        # Load validation data for this fold
        _, val_loader = get_loaders(fold_idx, load_cached_data=True)

        # Load best model
        model = RTICNN().to(device)
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles_gpu = angles.to(device)

                # Inference
                outputs = model(images, angles_gpu)
                probs = torch.sigmoid(outputs).cpu().numpy()

                fold_preds.extend(probs)
                fold_targets.extend(targets.numpy())

                # Collect features for analysis
                # images is (B, 3, 75, 75). Ch0=HH, Ch1=HV
                imgs_np = images.cpu().numpy()
                b1_means = np.mean(imgs_np[:, 0, :, :], axis=(1, 2))
                b2_means = np.mean(imgs_np[:, 1, :, :], axis=(1, 2))

                feat_angles.extend(angles.numpy())
                feat_b1_mean.extend(b1_means)
                feat_b2_mean.extend(b2_means)

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

    # Convert to arrays
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)
    feat_angles = np.array(feat_angles)
    feat_b1_mean = np.array(feat_b1_mean)
    feat_b2_mean = np.array(feat_b2_mean)

    # Calculate Metric
    final_metric = log_loss(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    logger.info("--- Failure Analysis ---")
    errors = np.abs(oof_targets - oof_preds)

    # Handle NaNs in angles if any remain (though loader imputes them)
    valid_angle_mask = ~np.isnan(feat_angles)

    if np.sum(valid_angle_mask) > 1:
        corr_angle, _ = pearsonr(
            errors[valid_angle_mask], feat_angles[valid_angle_mask]
        )
    else:
        corr_angle = 0.0

    corr_b1, _ = pearsonr(errors, feat_b1_mean)
    corr_b2, _ = pearsonr(errors, feat_b2_mean)

    print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    print(f"Correlation (Error vs Band 1 Mean): {corr_b1:.4f}")
    print(f"Correlation (Error vs Band 2 Mean): {corr_b2:.4f}")

    # 4. Submission
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        logger.info(f"Metric {final_metric} < {THRESHOLD}. Generating submission...")

        test_loader = get_test_loader(load_cached_data=True)
        test_ids = test_loader.dataset.ids
        num_test = len(test_ids)

        # Accumulator for ensemble predictions
        ensemble_preds = np.zeros(num_test, dtype=np.float32)

        # Predict with each fold's model
        for fold_idx in range(Config.N_FOLDS):
            logger.info(f"Inference with model fold {fold_idx}...")
            model = RTICNN().to(device)
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth"
            )
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            fold_test_preds = []
            with torch.no_grad():
                for images, angles, _ in test_loader:
                    images = images.to(device)
                    angles = angles.to(device)
                    outputs = model(images, angles)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    fold_test_preds.extend(probs)

            ensemble_preds += np.array(fold_test_preds)

        # Average
        ensemble_preds /= Config.N_FOLDS

        # Save
        submission_df = pd.DataFrame({"id": test_ids, "is_iceberg": ensemble_preds})

        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        logger.info(f"Submission saved to {save_path}")

    else:
        logger.info(f"Metric {final_metric} >= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
