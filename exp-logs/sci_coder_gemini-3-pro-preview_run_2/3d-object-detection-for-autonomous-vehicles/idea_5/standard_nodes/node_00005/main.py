import os
import sys
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import pearsonr

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import TrainConfig, DataConfig, ModelConfig, VoxelConfig, set_seeds
from library.trainer import Trainer
from library.dataset import LidarDataset, collate_fn
from library.utils import BoxUtils


# ==========================================
# 1. Configuration and Setup
# ==========================================
def setup_config():
    # Modify configs for a fast baseline run
    # We use a subset of data and fewer epochs to finish within 2 hours
    TrainConfig.epochs = 3
    TrainConfig.debug_subset_size = (
        6000  # Enough data to learn, small enough to be fast
    )
    TrainConfig.batch_size = 4

    # Ensure directories exist
    os.makedirs(TrainConfig.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(TrainConfig.submission_path), exist_ok=True)

    set_seeds(TrainConfig.seed)


# ==========================================
# 2. Metric Calculation Utilities
# ==========================================
def poly_area(x, y):
    """Calculate polygon area using Shoelace formula"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box3d_iou(pred_box, gt_box):
    """
    Calculate 3D IoU as defined in the task:
    IoU = (Ground Intersection * Height Intersection) / Union
    Boxes: [x, y, z, w, l, h, yaw]
    """
    # 1. Height Intersection
    # z is center, h is height
    p_z_min = pred_box[2] - pred_box[5] / 2
    p_z_max = pred_box[2] + pred_box[5] / 2
    g_z_min = gt_box[2] - gt_box[5] / 2
    g_z_max = gt_box[2] + gt_box[5] / 2

    inter_h = max(0, min(p_z_max, g_z_max) - max(p_z_min, g_z_min))
    if inter_h == 0:
        return 0.0

    union_h = max(p_z_max, g_z_max) - min(p_z_min, g_z_min)

    # 2. Ground Intersection (Rotated Rectangles)
    # cv2.RotatedRect format: ((cx, cy), (w, h), angle_deg)
    # Note: BoxUtils defines w=y-dim, l=x-dim.
    # cv2 angle is clockwise? We need to be careful.
    # Simple approach: Get corners, use convex hull intersection if needed,
    # or use cv2.rotatedRectangleIntersection

    # Convert yaw to degrees
    p_angle = np.degrees(
        -pred_box[6]
    )  # cv2 uses clockwise for positive? standard is counter-clockwise.
    g_angle = np.degrees(-gt_box[6])

    # BoxUtils: x_corners = l/2, y_corners = w/2.
    # So dimension along X is l, along Y is w.
    # cv2 size format is (width, height). Let's map (l, w).
    rect_p = ((pred_box[0], pred_box[1]), (pred_box[4], pred_box[3]), p_angle)
    rect_g = ((gt_box[0], gt_box[1]), (gt_box[4], gt_box[3]), g_angle)

    try:
        int_type, int_pts = cv2.rotatedRectangleIntersection(rect_p, rect_g)

        if int_type == cv2.INTERSECT_NONE:
            inter_area = 0.0
        else:
            # Calculate area of intersection polygon
            if int_pts is not None:
                # int_pts is (N, 1, 2)
                order_pts = cv2.convexHull(int_pts, returnPoints=True)
                order_pts = order_pts.reshape(-1, 2)
                inter_area = cv2.contourArea(order_pts)
            else:
                inter_area = 0.0
    except:
        inter_area = 0.0

    p_area = pred_box[3] * pred_box[4]
    g_area = gt_box[3] * gt_box[4]

    # 3. Combine
    # Union Volume = VolA + VolB - Intersection Volume
    # Intersection Volume = inter_area * inter_h
    # Note: Task says IoU = (Ground Inter * Height Inter) / Union Boxes
    # Usually Union is Volume Union.
    # "The IoU is then the intersection of the ground bounding boxes * the intersection of the height differences, divided by the union of the bounding boxes."

    inter_vol = inter_area * inter_h
    p_vol = p_area * pred_box[5]
    g_vol = g_area * gt_box[5]

    union_vol = p_vol + g_vol - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol


def calculate_map(predictions, ground_truths, thresholds=None):
    """
    Calculate mAP over IoU thresholds.
    predictions: dict {token: list of boxes [x, y, z, w, l, h, yaw, score, class_idx]}
    ground_truths: dict {token: list of boxes [x, y, z, w, l, h, yaw, class_name]}
    """
    if thresholds is None:
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    aps = []

    # Pre-process GT to match class indices or just match spatially first?
    # Task implies class matching is required ("matches a ground truth object").
    # We will assume class matching is strict.

    # Flatten all samples to calculate global mAP or per-sample AP?
    # Task: "The average precision of a single image is calculated... mean average precision"
    # Usually mAP is mean over classes, or mean over samples?
    # Task: "The average precision of a single image is calculated as the mean of the above precision values at each IoU threshold"
    # "If there are no ground truth objects... image receiving a score of zero, and being included in the mean average precision."
    # This implies we calculate AP per image, then mean over images.

    sample_aps = []

    for token in ground_truths.keys():
        gts = ground_truths[token]
        preds = predictions.get(token, [])

        # Sort preds by confidence
        if len(preds) > 0:
            preds = sorted(preds, key=lambda x: x[7], reverse=True)

        # Per-threshold precision
        precisions = []

        for t in thresholds:
            tp = 0
            fp = 0
            # fn is implicit in the denominator (TP + FP + FN = Num_Preds + Num_Unmatched_GT)
            # Actually denominator TP + FP + FN = Num_Preds (TP+FP) + Num_Missed_GT (FN)
            # = Num_Preds + (Total_GT - TP) = Num_Preds + Total_GT - TP

            matched_gt = set()

            for p in preds:
                p_box = p[:7]
                p_cls_idx = int(p[8])
                p_cls = ModelConfig.class_names[p_cls_idx]

                best_iou = 0
                best_gt_idx = -1

                for i, g in enumerate(gts):
                    if i in matched_gt:
                        continue

                    # Class check
                    if g["class_name"] != p_cls:
                        continue

                    # GT Box: center_x, center_y, center_z, width, length, height, yaw
                    g_box = [
                        g["center_x"],
                        g["center_y"],
                        g["center_z"],
                        g["width"],
                        g["length"],
                        g["height"],
                        g["yaw"],
                    ]

                    iou = box3d_iou(p_box, g_box)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou > t:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gts) - len(matched_gt)

            denom = tp + fp + fn
            if denom == 0:
                prec = 0.0  # Should be 1.0 if both empty?
                # If no GT and no Preds -> Precision is technically undefined or 1.
                # Task: "If there are no ground truth objects... ANY number of predictions (false positives) will result in the image receiving a score of zero"
                # If no GT and no Preds, TP=0, FP=0, FN=0.
                if len(gts) == 0 and len(preds) == 0:
                    prec = 1.0
                else:
                    prec = 0.0
            else:
                prec = tp / denom

            precisions.append(prec)

        sample_aps.append(np.mean(precisions))

    return np.mean(sample_aps)


# ==========================================
# 3. Main Execution Flow
# ==========================================
def main():
    setup_config()

    # --------------------------------------
    # A. Training
    # --------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer()

    print("Starting Training (Fast Baseline)...")
    trainer.train()

    # --------------------------------------
    # B. Validation & Metric
    # --------------------------------------
    print("Starting Validation...")

    # Load Validation Metadata for GT
    val_df = pd.read_csv(DataConfig.val_metadata_path)
    val_df["annotations"] = val_df["annotations"].apply(json.loads)
    val_df["file_paths"] = val_df["file_paths"].apply(json.loads)

    # Limit validation to subset if configured
    if TrainConfig.debug_subset_size:
        val_df = val_df.iloc[: TrainConfig.debug_subset_size]

    gt_map = {row["token"]: row["annotations"] for _, row in val_df.iterrows()}

    # Run Inference on Validation Set
    trainer.model.eval()
    val_loader = trainer.val_loader
    device = trainer.device

    predictions = {}

    # For Failure Analysis
    analysis_data = []  # (error_dist, dist_from_ego)

    with torch.no_grad():
        for batch in val_loader:
            pillar_features = batch["pillar_features"].to(device)
            pillar_coords = batch["pillar_coords"].to(device)
            tokens = batch["tokens"]
            matrices = batch["matrices"].numpy()

            batched_inputs = {
                "pillar_features": pillar_features,
                "pillar_coords": pillar_coords,
                "batch_size": batch["batch_size"],
            }

            preds = trainer.model(batched_inputs)
            batch_boxes = trainer._decode_predictions(preds, score_thresh=0.1)

            for i, boxes in enumerate(batch_boxes):
                token = tokens[i]
                matrix = matrices[i]

                # Transform to Global
                try:
                    sens_to_global = np.linalg.inv(matrix)
                except:
                    sens_to_global = np.eye(4)

                global_boxes = []

                for box in boxes:
                    x, y, z, w, l, h, yaw, score, cls_idx = box

                    # Center
                    c_s = np.array([x, y, z, 1.0])
                    c_g = sens_to_global @ c_s

                    # Yaw
                    v_s = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
                    v_g = sens_to_global @ v_s
                    yaw_g = np.arctan2(v_g[1], v_g[0])

                    # [x, y, z, w, l, h, yaw, score, cls_idx]
                    g_box = [c_g[0], c_g[1], c_g[2], w, l, h, yaw_g, score, cls_idx]
                    global_boxes.append(g_box)

                    # --- Failure Analysis Data Collection ---
                    # Simple matching for analysis: Find closest GT of same class
                    cls_name = ModelConfig.class_names[int(cls_idx)]
                    gts = gt_map.get(token, [])
                    min_dist = float("inf")

                    for g in gts:
                        if g["class_name"] == cls_name:
                            d = np.sqrt(
                                (g["center_x"] - c_g[0]) ** 2
                                + (g["center_y"] - c_g[1]) ** 2
                            )
                            if d < min_dist:
                                min_dist = d

                    if min_dist < 2.0:  # Only consider "matched" for localization error
                        # Distance of object from ego (approx origin in sensor frame, but we have global)
                        # We need ego location. In sensor frame, ego is near 0,0,0.
                        # So distance is approx sqrt(x^2 + y^2) in sensor frame
                        dist_ego = np.sqrt(x**2 + y**2)
                        analysis_data.append((min_dist, dist_ego))

                predictions[token] = global_boxes

    # Calculate Metric
    final_metric = calculate_map(predictions, gt_map)
    print(f"Final Validation Metric: {final_metric:.10f}")

    # --------------------------------------
    # C. Failure Analysis
    # --------------------------------------
    print("\n=== Failure Analysis ===")
    if len(analysis_data) > 10:
        errors, dists = zip(*analysis_data)
        corr, _ = pearsonr(errors, dists)
        print(
            f"Correlation between Localization Error and Distance from Ego: {corr:.4f}"
        )
        print(f"Mean Localization Error: {np.mean(errors):.4f} m")
    else:
        print("Not enough matched predictions for correlation analysis.")

    # --------------------------------------
    # D. Submission
    # --------------------------------------
    if final_metric > 0.0:
        print("\nGenerating Submission...")
        trainer.generate_submission()
    else:
        print("\nValidation metric is 0.0. Skipping submission generation.")


if __name__ == "__main__":
    main()
