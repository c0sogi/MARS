import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.metrics import log_loss

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, calculate_metric, Logger
from library.dataset import _process_data
from library.training import run_regime_a, run_regime_b
from library.stacking import Stacker


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Initialize Logger
    logger = Logger(os.path.join(Config.WORK_DIR, "runfile.log"))
    logger.log("Starting Runfile Execution...")

    # 2. Data Processing
    # Ensure data is cached and ready
    logger.log("Processing Data...")
    _process_data(load_cached_data=True)

    # 3. Training Loop
    # We train both experts for all folds.
    # The A100 GPU allows this to complete within the allocated time.

    n_folds = Config.N_FOLDS

    for fold_idx in range(n_folds):
        logger.log(f"\n=== Training Fold {fold_idx}/{n_folds-1} ===")

        # Train Expert A (ConvNeXt - Regime A)
        # Checkpoint check to avoid retraining if interrupted/re-run
        if not os.path.exists(
            os.path.join(Config.WORK_DIR, f"convnext_base_fold_{fold_idx}.pth")
        ):
            logger.log(f"Training Expert A (ConvNeXt) Fold {fold_idx}...")
            run_regime_a(fold_idx, debug=False)
        else:
            logger.log(f"Expert A Fold {fold_idx} already exists. Skipping.")

        # Train Expert B (Swin - Regime B)
        if not os.path.exists(
            os.path.join(Config.WORK_DIR, f"swin_base_fold_{fold_idx}.pth")
        ):
            logger.log(f"Training Expert B (Swin) Fold {fold_idx}...")
            run_regime_b(fold_idx, debug=False)
        else:
            logger.log(f"Expert B Fold {fold_idx} already exists. Skipping.")

    # 4. Stacking & Meta-Learning
    logger.log("\n=== Stacking & Meta-Learning ===")
    stacker = Stacker(debug=False)

    # Generate/Load OOF and Test Predictions
    # This aggregates predictions from all 5 folds using TTA
    stacking_data = stacker.get_data(load_cached_data=True)

    # Train Meta-Learner (Logistic Regression)
    meta_model = stacker.train_meta_learner(stacking_data)

    # 5. Validation Assessment
    # Calculate Final Metric on OOF Data (Proxy for Validation Set)
    X_oof = np.hstack([stacking_data["oof_preds_a"], stacking_data["oof_preds_b"]])
    y_oof = stacking_data["oof_targets"]

    oof_probs = meta_model.predict_proba(X_oof)
    final_metric = calculate_metric(y_oof, oof_probs)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")
    logger.log(f"Final Validation Metric: {final_metric}")

    # 6. Failure Analysis
    logger.log("\n=== Failure Analysis ===")

    # Calculate error magnitude per sample (Log Loss contribution)
    # Error = -log(P_true_class)
    epsilon = 1e-15
    oof_probs_clipped = np.clip(oof_probs, epsilon, 1 - epsilon)

    rows = np.arange(len(y_oof))
    true_class_probs = oof_probs_clipped[rows, y_oof]
    error_magnitudes = -np.log(true_class_probs)

    # Map OOF IDs to file paths to retrieve image metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA)
    val_meta = pd.read_csv(Config.VAL_METADATA)
    full_meta = pd.concat([train_meta, val_meta], ignore_index=True)

    id_to_path = pd.Series(full_meta.file_path.values, index=full_meta.id).to_dict()

    # Collect image stats for OOF samples
    widths = []
    heights = []
    aspect_ratios = []

    oof_ids = stacking_data["oof_ids"]
    logger.log("Computing image statistics for failure analysis...")

    for img_id in oof_ids:
        rel_path = id_to_path.get(img_id)
        if rel_path:
            full_path = os.path.join(Config.INPUT_DIR, rel_path)
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
            else:
                widths.append(0)
                heights.append(0)
                aspect_ratios.append(0)
        else:
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0)

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)

    # Calculate Correlations
    valid_mask = widths > 0

    if np.sum(valid_mask) > 0:
        # Use numpy for correlation to avoid extra dependencies
        corr_w = np.corrcoef(error_magnitudes[valid_mask], widths[valid_mask])[0, 1]
        corr_h = np.corrcoef(error_magnitudes[valid_mask], heights[valid_mask])[0, 1]
        corr_ar = np.corrcoef(error_magnitudes[valid_mask], aspect_ratios[valid_mask])[
            0, 1
        ]

        print("Failure Analysis: Correlation with Error Magnitude")
        print(f"Width: {corr_w:.4f}")
        print(f"Height: {corr_h:.4f}")
        print(f"Aspect Ratio: {corr_ar:.4f}")

        logger.log(f"Correlation Error vs Width: {corr_w:.4f}")
        logger.log(f"Correlation Error vs Height: {corr_h:.4f}")
        logger.log(f"Correlation Error vs Aspect Ratio: {corr_ar:.4f}")
    else:
        logger.log("Could not compute correlations due to missing images.")

    # 7. Submission
    # Strict threshold check
    threshold = 0.12970461086690332
    if final_metric < threshold:
        logger.log(f"Metric {final_metric} < {threshold}. Generating submission...")
        stacker.predict_and_submit(meta_model, stacking_data)
    else:
        logger.log(f"Metric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
