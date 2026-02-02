import sys
import os
import numpy as np
import torch
import pandas as pd
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Configuration Patching (Must be done before importing solver)
import library.config as config

# Fast Baseline Settings to ensure runtime < 2 hours
config.EPOCHS = 3  # Reduced from 20 to 3
config.BATCH_SIZE = 4  # Safe batch size for A100
config.MAX_PILLARS_TRAIN = 12000  # Reduce max pillars to speed up PFN
config.NUM_WORKERS = 2

# Import Solver and Utils after config patch
from library.solver import Solver
from library.utils import box_iou_3d_pair
from torch.utils.data import DataLoader
from library.dataset import LidarDataset


def set_seed(seed=42):
    """Sets fixed random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_predictions(pred_str):
    """
    Parses prediction string into list of dicts.
    Format: conf x y z w l h yaw class
    """
    if pd.isna(pred_str) or pred_str == "":
        return []

    parts = pred_str.strip().split()
    stride = 9
    preds = []
    num_preds = len(parts) // stride
    for i in range(num_preds):
        off = i * stride
        try:
            p = {
                "conf": float(parts[off]),
                "box": [
                    float(parts[off + 1]),
                    float(parts[off + 2]),
                    float(parts[off + 3]),
                    float(parts[off + 4]),
                    float(parts[off + 5]),
                    float(parts[off + 6]),
                    float(parts[off + 7]),
                ],  # [x, y, z, w, l, h, yaw]
                "class": parts[off + 8],
            }
            preds.append(p)
        except:
            continue
    return preds


def parse_gt(label_str):
    """
    Parses GT string into list of dicts.
    Format: x y z w l h yaw class
    """
    if pd.isna(label_str) or label_str == "":
        return []
    parts = label_str.strip().split()
    stride = 8
    gts = []
    num_objs = len(parts) // stride
    for i in range(num_objs):
        off = i * stride
        try:
            g = {
                "box": [
                    float(parts[off]),
                    float(parts[off + 1]),
                    float(parts[off + 2]),
                    float(parts[off + 3]),
                    float(parts[off + 4]),
                    float(parts[off + 5]),
                    float(parts[off + 6]),
                ],  # [x, y, z, w, l, h, yaw]
                "class": parts[off + 7],
            }
            gts.append(g)
        except:
            continue
    return gts


def calculate_image_metric(preds, gts, thresholds):
    """
    Calculates the average precision for a single image across IoU thresholds.
    Metric formula: Mean over thresholds of (TP / (TP + FP + FN))
    """
    # Sort predictions by confidence (descending)
    preds.sort(key=lambda x: x["conf"], reverse=True)

    # Pre-calculate IoU matrix to optimize speed
    # Rows: Preds, Cols: GTs
    n_p = len(preds)
    n_g = len(gts)
    iou_mat = np.zeros((n_p, n_g))

    for i in range(n_p):
        for j in range(n_g):
            # Enforce class matching
            if preds[i]["class"] == gts[j]["class"]:
                iou_mat[i, j] = box_iou_3d_pair(preds[i]["box"], gts[j]["box"])
            else:
                iou_mat[i, j] = 0.0

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0
        matched_gt_indices = set()

        # Greedy matching
        for i in range(n_p):
            best_iou = -1.0
            best_gt_idx = -1

            # Find best matching available GT
            for j in range(n_g):
                if j in matched_gt_indices:
                    continue
                if iou_mat[i, j] > best_iou:
                    best_iou = iou_mat[i, j]
                    best_gt_idx = j

            if best_iou > t:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = n_g - len(matched_gt_indices)
        denom = tp + fp + fn

        if denom == 0:
            # No GT and No Preds -> Perfect score (1.0)
            score = 1.0
        else:
            score = tp / denom

        precisions.append(score)

    return np.mean(precisions)


def main():
    set_seed(42)

    # 1. Initialize Solver
    solver = Solver()

    # 2. Downsample Training Data (Fast Baseline)
    # Reducing dataset size to ensure completion within 2 hours
    full_train_meta = solver.train_dataset.metadata
    target_samples = 2000
    if len(full_train_meta) > target_samples:
        solver.train_dataset.metadata = full_train_meta.sample(
            n=target_samples, random_state=42
        ).reset_index(drop=True)
        # Re-initialize the loader with the smaller dataset
        solver.train_loader = DataLoader(
            solver.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            collate_fn=LidarDataset.collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    # 3. Train
    solver.fit()

    # 4. Validation Assessment
    print("Running Validation Assessment...")
    solver.model.eval()

    val_meta = solver.val_dataset.metadata
    # Create lookup for GT labels
    token_to_gt = dict(zip(val_meta["sample_token"], val_meta["label"]))

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    image_scores = []

    # Data for failure analysis
    fa_data = []

    with torch.no_grad():
        for batch_dict in solver.val_loader:
            batch_dict = solver._to_device(batch_dict)

            # Get model predictions (list of strings)
            pred_strs = solver.model(batch_dict)

            sample_tokens = batch_dict["sample_tokens"]
            num_points_batch = batch_dict["num_points"]

            for i, s_token in enumerate(sample_tokens):
                p_str = pred_strs[i]
                gt_str = token_to_gt.get(s_token, "")

                preds = parse_predictions(p_str)
                gts = parse_gt(gt_str)

                # Calculate Metric
                score = calculate_image_metric(preds, gts, thresholds)
                image_scores.append(score)

                # Collect stats for failure analysis
                fa_data.append(
                    {
                        "error": 1.0 - score,
                        "num_points": num_points_batch[i].item(),
                        "num_gt": len(gts),
                    }
                )

    final_metric = np.mean(image_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    df_fa = pd.DataFrame(fa_data)
    if not df_fa.empty:
        # Avoid NaN correlation if variance is 0
        if df_fa["error"].std() > 0 and df_fa["num_points"].std() > 0:
            corr_points = df_fa["error"].corr(df_fa["num_points"])
        else:
            corr_points = 0.0

        if df_fa["error"].std() > 0 and df_fa["num_gt"].std() > 0:
            corr_gt = df_fa["error"].corr(df_fa["num_gt"])
        else:
            corr_gt = 0.0

        print(f"Correlation (Error vs Num Points): {corr_points:.4f}")
        print(f"Correlation (Error vs Num GT Objects): {corr_gt:.4f}")

    # 6. Submission
    if final_metric > 0.0024:
        print("Metric threshold passed. Generating submission...")
        solver.inference()
    else:
        print("Metric too low. Skipping submission.")


if __name__ == "__main__":
    main()
