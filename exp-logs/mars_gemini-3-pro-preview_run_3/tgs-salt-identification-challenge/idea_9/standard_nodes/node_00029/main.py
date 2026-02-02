import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

# Import library modules
from library.config import (
    GeneralConfig,
    TrainConfig,
    PathConfig,
    DataConfig,
    seed_everything,
)
from library.trainer import train_fold
from library.inference import Evaluator
from library.dataset import load_data
from library.utils import calculate_iou_map


def main():
    # 1. Setup and Configuration
    seed_everything(GeneralConfig.SEED)

    # Ensure directories exist
    PathConfig.create_directories()

    print("Configuration:")
    print(f"  Device: {GeneralConfig.DEVICE}")
    print(f"  Epochs: {TrainConfig.EPOCHS}")
    print(f"  Batch Size: {DataConfig.BATCH_SIZE}")

    # 2. Training Loop (Stratified 5-Fold)
    model_paths = []

    # We need to track validation indices to align metadata for failure analysis
    # Load all training data to reconstruct splits
    print("\nLoading data for split reconstruction...")
    images, masks, depths, ids, coverage_classes = load_data(
        mode="train", load_cached_data=True
    )

    skf = StratifiedKFold(
        n_splits=DataConfig.NUM_FOLDS, shuffle=True, random_state=GeneralConfig.SEED
    )
    splits = list(skf.split(images, coverage_classes))

    # Containers for global evaluation
    global_preds = []
    global_targets = []
    global_depths = []
    global_coverages = []

    evaluator = Evaluator()

    # Cite solution_lesson_node_00024: Depth Over Width
    # Train only Fold 0 to allow full convergence within time limits
    for fold_idx in range(1):
        print(f"\n{'='*20} Processing Fold {fold_idx} {'='*20}")

        # A. Train
        # train_fold returns the path to the best checkpoint
        ckpt_path = train_fold(fold_idx, debug=False)
        model_paths.append(ckpt_path)

        # B. Predict (OOF)
        # We set load_cached_data=False for predictions to ensure we use the newly trained model
        # The data loader inside will still use cached images/masks if available
        preds, targets = evaluator.predict_fold(
            fold_idx, ckpt_path, load_cached_data=False, debug=False
        )

        # C. Collect Metadata for this fold
        # Get the validation indices for this fold to extract corresponding metadata
        _, val_idx = splits[fold_idx]

        # Verify alignment
        if len(preds) != len(val_idx):
            print(
                f"Warning: Prediction count {len(preds)} != Validation set size {len(val_idx)}"
            )

        global_preds.append(preds)
        global_targets.append(targets)
        global_depths.append(depths[val_idx])
        global_coverages.append(coverage_classes[val_idx])

    # 3. Global Evaluation
    print(f"\n{'='*20} Global Evaluation {'='*20}")

    # Concatenate all folds
    full_preds = np.concatenate(global_preds, axis=0)
    full_targets = np.concatenate(global_targets, axis=0)
    full_depths = np.concatenate(global_depths, axis=0)
    full_coverages = np.concatenate(global_coverages, axis=0)

    # Optimize Threshold
    # Finds the best probability threshold to maximize mAP
    best_threshold = evaluator.optimize_threshold(full_preds, full_targets)

    # Calculate Final Metric at Best Threshold
    # The metric is Mean Average Precision over IoU thresholds 0.5:0.95
    final_metric = calculate_iou_map(full_preds, full_targets, threshold=best_threshold)

    print(f"Final Validation Metric: {final_metric:.10f}")

    # 4. Failure Analysis
    print(f"\n{'='*20} Failure Analysis {'='*20}")

    # Calculate per-image score (mAP) to correlate with metadata
    bin_preds = (full_preds > best_threshold).astype(np.uint8)
    bin_targets = (full_targets > 0.5).astype(np.uint8)

    # Flatten spatial dims for IoU calculation
    N = bin_preds.shape[0]
    scores = []

    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(N):
        p = bin_preds[i].flatten()
        t = bin_targets[i].flatten()

        if np.sum(p) == 0 and np.sum(t) == 0:
            iou = 1.0
        elif np.sum(p) > 0 and np.sum(t) > 0:
            intersection = np.logical_and(p, t).sum()
            union = np.logical_or(p, t).sum()
            iou = intersection / union
        else:
            iou = 0.0

        matches = iou > iou_thresholds
        scores.append(np.mean(matches))

    scores = np.array(scores)
    errors = 1.0 - scores

    # Correlations
    # Depth vs Error
    corr_depth, _ = pearsonr(full_depths, errors)
    # Coverage Class vs Error
    corr_cov, _ = pearsonr(full_coverages, errors)

    print(f"Correlation (Depth vs Error): {corr_depth:.4f}")
    print(f"Correlation (Salt Coverage Class vs Error): {corr_cov:.4f}")

    if abs(corr_depth) > 0.1:
        print(">> Observation: Performance varies significantly with depth.")
    if abs(corr_cov) > 0.1:
        print(
            ">> Observation: Performance varies significantly with salt coverage amount."
        )

    # 5. Submission
    print(f"\n{'='*20} Submission Generation {'='*20}")

    if final_metric > 0.827:
        print(f"Metric {final_metric:.4f} > 0.827. Generating submission...")
        evaluator.generate_submission(
            model_paths, threshold=best_threshold, debug=False
        )
    else:
        print(f"Metric {final_metric:.4f} <= 0.827. Skipping submission.")


if __name__ == "__main__":
    main()
