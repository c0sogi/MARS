import os
import sys
import numpy as np
import pandas as pd
import torch
from library.config import Config
from library.pipeline import Pipeline
from library.data import process_and_cache_data, SaltDataset, get_transforms
from library.utils import do_kaggle_metric
from library.models import GeneralistStudent


def main():
    # =========================================================================
    # 1. Configuration Setup for Fast Baseline
    # =========================================================================
    Config.setup()

    # Constrain training data to the 'train.csv' portion only (2400 samples).
    # The data loader logic concatenates train+val, so slicing at 2400 ensures
    # that the 600 samples from 'val.csv' are never seen by the model during training.
    Config.MAX_TRAIN_SAMPLES = 2400

    # Adjust hyperparameters for a fast but effective run within time limits
    Config.STAGE1_FOLDS = 2  # Train only 2 folds for the teacher ensemble
    Config.STAGE1_EPOCHS = 10  # Reduced epochs for speed
    Config.STAGE3_EPOCHS = 10  # Reduced epochs for speed
    Config.STAGE1_GATING_THRESHOLD = (
        0.4  # Lower threshold to ensure models pass gating in short run
    )
    Config.BATCH_SIZE = 32  # Safe batch size for A100

    # =========================================================================
    # 2. Pipeline Execution
    # =========================================================================
    pipeline = Pipeline()

    print("=== Starting Pipeline Execution ===")

    # Stage 1: Train Specialist Teacher Ensemble
    # Returns list of valid checkpoint paths
    teacher_paths = pipeline.run_teacher_ensemble()

    # Stage 2: Generate Marginalized Soft Pseudo-Labels
    # Returns dict {id: mask_probability}
    pseudo_labels = pipeline.generate_marginalized_labels(teacher_paths)

    # Stage 3: Train Generalist Student
    # Returns path to the best student model checkpoint
    student_path = pipeline.train_student_distillation(
        pseudo_labels, epochs=Config.STAGE3_EPOCHS
    )

    # Optimization: Find best binarization threshold
    # Uses the internal validation subset (part of train) for threshold tuning
    best_threshold = pipeline.optimize_threshold(student_path)

    # =========================================================================
    # 3. Final Validation & Failure Analysis
    # =========================================================================
    print("\n=== Final Validation & Failure Analysis ===")

    # Load the pure holdout validation set (metadata/val.csv)
    # We use a custom cache prefix "holdout_val" to avoid conflict with pipeline loaders
    val_data = process_and_cache_data(
        Config.VAL_METADATA_PATH, "holdout_val", load_cached_data=True
    )
    val_ds = SaltDataset(val_data, mode="val", transform=get_transforms("val"))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Student Model
    model = GeneralistStudent().to(Config.DEVICE)
    checkpoint = torch.load(student_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_depths = []

    print("Running inference on holdout validation set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(Config.DEVICE)
            masks = batch["mask"].to(Config.DEVICE)
            depths = batch["depth"]  # Keep on CPU for analysis

            # Inference with Test-Time Augmentation (TTA)
            # 1. Original
            logits, _ = model(images)
            probs = torch.sigmoid(logits)

            # 2. Horizontal Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip, _ = model(images_flip)
            probs_flip = torch.flip(torch.sigmoid(logits_flip), dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
            all_depths.append(depths.numpy())

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_depths = np.concatenate(all_depths, axis=0).flatten()

    # Calculate Final Metric
    val_score = do_kaggle_metric(all_preds, all_targets, threshold=best_threshold)
    print(f"Final Validation Metric: {val_score:.10f}")

    # Failure Analysis
    # Calculate IoU per image at the best threshold
    preds_bin = (all_preds > best_threshold).astype(np.uint8)
    targets_bin = (all_targets > 0.5).astype(np.uint8)

    # Flatten spatial dimensions for IoU calculation: (N, H*W)
    preds_flat = preds_bin.reshape(preds_bin.shape[0], -1)
    targets_flat = targets_bin.reshape(targets_bin.shape[0], -1)

    intersection = (preds_flat & targets_flat).sum(axis=1)
    union = preds_flat.sum(axis=1) + targets_flat.sum(axis=1) - intersection

    # IoU = Intersection / Union (Handle division by zero for empty-empty case)
    ious = np.ones(len(intersection), dtype=np.float32)
    non_empty = union > 0
    ious[non_empty] = intersection[non_empty] / union[non_empty]

    # Error is defined as 1 - IoU
    errors = 1.0 - ious

    # Calculate Correlation between Error and Depth
    if len(errors) > 1:
        corr_matrix = np.corrcoef(errors, all_depths)
        corr = corr_matrix[0, 1]
        print(f"Correlation between Error (1-IoU) and Depth: {corr:.10f}")
    else:
        print("Not enough samples for correlation analysis.")

    # =========================================================================
    # 4. Submission Generation
    # =========================================================================
    if val_score > 0.7985:
        print("Validation metric exceeds threshold. Generating submission...")
        pipeline.generate_submission(student_path, best_threshold)
    else:
        print(f"Validation metric {val_score:.4f} <= 0.7985. Submission skipped.")


if __name__ == "__main__":
    main()
