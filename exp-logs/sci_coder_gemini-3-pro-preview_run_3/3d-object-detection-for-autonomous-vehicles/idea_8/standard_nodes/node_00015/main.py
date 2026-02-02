import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import time
from torch.utils.data import DataLoader

# Import from provided library
from library.config import Config
from library.dataset import LidarDataset
from library.detector import TwoStagePointPillars
from library.trainer import run_training
from library.inference import generate_submission


# ==============================================================================
# 1. Configuration & Setup
# ==============================================================================
def setup_config():
    """
    Adjusts configuration for a fast baseline run within the time limit.
    """
    # Limit epochs for fast execution within 2 hours
    # 1 epoch takes roughly 20-30 mins depending on dataloader speed.
    # We set to 3 epochs to ensure completion and sufficient convergence.
    Config.NUM_EPOCHS = 3

    # Ensure batch size is safe
    Config.BATCH_SIZE = 4

    # Set seed
    Config.set_seed(Config.SEED)

    print(f"Configuration Set: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}")


# ==============================================================================
# 2. Metric Implementation (IoU & mAP)
# ==============================================================================
def rotated_rect_area(w, l, angle_rad):
    """Calculates area of a rotated rectangle."""
    return w * l


def polygon_intersection_area(box1, box2):
    """
    Calculates intersection area of two rotated rectangles (BEV).
    box: [x, y, w, l, yaw]
    """
    # Convert to cv2 Box2D format: ((x, y), (w, l), angle_deg)
    # cv2 angle is degrees clockwise? No, it's specific.
    # We use cv2.rotatedRectangleIntersection

    def get_rect(b):
        x, y, w, l, yaw = b
        # Convert yaw (radians) to degrees
        angle = np.degrees(yaw)
        return ((x, y), (w, l), angle)

    rect1 = get_rect(box1)
    rect2 = get_rect(box2)

    try:
        ret, points = cv2.rotatedRectangleIntersection(rect1, rect2)
        if ret == cv2.INTERSECT_NONE:
            return 0.0
        elif ret == cv2.INTERSECT_FULL:
            # One is inside the other, return area of smaller
            return min(box1[2] * box1[3], box2[2] * box2[3])
        elif ret == cv2.INTERSECT_PARTIAL and points is not None:
            # points is (N, 1, 2)
            # Convex hull order is needed for contourArea?
            # rotatedRectangleIntersection returns vertices in order.
            return cv2.contourArea(points)
        else:
            return 0.0
    except:
        return 0.0


def calculate_iou_3d(box1, box2):
    """
    Calculates 3D IoU.
    box: [x, y, z, w, l, h, yaw]
    """
    # 1. Height Intersection
    z1, h1 = box1[2], box1[5]
    z2, h2 = box2[2], box2[5]

    z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
    z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2

    h_int = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

    if h_int == 0:
        return 0.0

    # 2. BEV Intersection (Area)
    # box subset: [x, y, w, l, yaw]
    bev1 = [box1[0], box1[1], box1[3], box1[4], box1[6]]
    bev2 = [box2[0], box2[1], box2[3], box2[4], box2[6]]

    area_int = polygon_intersection_area(bev1, bev2)

    if area_int == 0:
        return 0.0

    # 3. Volume Calculation
    vol_int = area_int * h_int

    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]

    vol_union = vol1 + vol2 - vol_int

    return vol_int / (vol_union + 1e-6)


def calculate_sample_precision(pred_boxes, pred_scores, gt_boxes, thresholds):
    """
    Calculates precision for a single sample across thresholds.
    Metric: TP(t) / (TP(t) + FP(t) + FN(t))
    """
    # Sort predictions by score descending
    if len(pred_boxes) > 0:
        sorted_idx = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_idx]

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0

        matched_gt_indices = set()

        # Greedy matching
        for i in range(len(pred_boxes)):
            p_box = pred_boxes[i]

            best_iou = -1.0
            best_gt_idx = -1

            # Find best match among unmatched GTs
            for j in range(len(gt_boxes)):
                if j in matched_gt_indices:
                    continue

                iou = calculate_iou_3d(p_box, gt_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou > t:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt_indices)

        # Precision (CSI)
        denom = tp + fp + fn
        if denom == 0:
            score = 0.0  # Should be 1.0 if both empty? Task says "no ground truth... ANY predictions... score of zero".
            # If both empty, TP=0, FP=0, FN=0. Usually 1.0.
            # But if GT is empty and Preds > 0 -> FP > 0 -> 0/(0+FP+0) = 0.
            # If GT > 0 and Preds empty -> FN > 0 -> 0/(0+0+FN) = 0.
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                score = 1.0
            else:
                score = 0.0
        else:
            score = tp / denom

        precisions.append(score)

    return np.mean(precisions)


def evaluate_validation_set(model, val_loader, device):
    """
    Runs inference on validation set and calculates the official metric.
    """
    print("\nStarting Validation Evaluation...")
    model.eval()

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    sample_precisions = []

    # For failure analysis
    analysis_data = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            # Inference
            # Returns lists of tensors
            pred_boxes_list, pred_scores_list, _ = model(batch, mode="test")

            gt_boxes_list = batch["gt_boxes"]  # list of tensors

            # Iterate batch
            for i in range(len(pred_boxes_list)):
                # Get Preds (Lidar Frame)
                p_boxes = pred_boxes_list[i].cpu().numpy()
                p_scores = pred_scores_list[i].cpu().numpy()

                # Get GT (Lidar Frame)
                g_boxes = gt_boxes_list[i].cpu().numpy()

                # Filter low confidence for metric calculation to reduce FP
                # Task says "evaluated in order of confidence".
                # Including very low confidence boxes increases FP in the CSI metric.
                # We apply a reasonable threshold.
                mask = p_scores > 0.1
                p_boxes = p_boxes[mask]
                p_scores = p_scores[mask]

                # Calculate Metric
                ap = calculate_sample_precision(p_boxes, p_scores, g_boxes, thresholds)
                sample_precisions.append(ap)

                # Collect Data for Failure Analysis (using IoU@0.5 matches)
                if len(p_boxes) > 0 and len(g_boxes) > 0:
                    # Sort
                    sorted_idx = np.argsort(p_scores)[::-1]
                    p_boxes_sorted = p_boxes[sorted_idx]
                    p_scores_sorted = p_scores[sorted_idx]

                    matched_gt = set()
                    for j in range(len(p_boxes_sorted)):
                        pb = p_boxes_sorted[j]
                        best_iou = 0
                        best_gt_idx = -1
                        for k in range(len(g_boxes)):
                            if k in matched_gt:
                                continue
                            iou = calculate_iou_3d(pb, g_boxes[k])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = k

                        if best_iou > 0.5:
                            matched_gt.add(best_gt_idx)
                            gb = g_boxes[best_gt_idx]

                            # Error Magnitude: 1 - IoU
                            err_mag = 1.0 - best_iou

                            # Features
                            # Range (dist from 0,0)
                            rng = np.sqrt(pb[0] ** 2 + pb[1] ** 2)
                            # Volume
                            vol = pb[3] * pb[4] * pb[5]

                            analysis_data.append(
                                {
                                    "error": err_mag,
                                    "range": rng,
                                    "volume": vol,
                                    "score": p_scores_sorted[j],
                                }
                            )

    mean_ap = np.mean(sample_precisions)
    return mean_ap, pd.DataFrame(analysis_data)


# ==============================================================================
# 3. Failure Analysis
# ==============================================================================
def perform_failure_analysis(df):
    """
    Calculates correlation between error and features.
    """
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    if len(df) < 10:
        print("Insufficient data for analysis.")
        return

    # Correlations
    corr_range = df["error"].corr(df["range"])
    corr_vol = df["error"].corr(df["volume"])
    corr_conf = df["error"].corr(df["score"])

    print(f"Correlation (Error vs Range):  {corr_range:.4f}")
    print(f"Correlation (Error vs Volume): {corr_vol:.4f}")
    print(f"Correlation (Error vs Score):  {corr_conf:.4f}")

    print("\nInterpretation:")
    if abs(corr_range) > 0.3:
        print(
            f" - Significant relationship between object distance and error ({corr_range:.2f})."
        )
    if abs(corr_vol) > 0.3:
        print(
            f" - Significant relationship between object size and error ({corr_vol:.2f})."
        )


# ==============================================================================
# 4. Main Execution
# ==============================================================================
def main():
    # 1. Setup
    setup_config()
    device = Config.DEVICE

    # 2. Train
    print("\n--- Phase 1: Training ---")
    # run_training handles dataloading, model init, loop, and saving checkpoint
    run_training(max_epochs=Config.NUM_EPOCHS)

    # 3. Validate & Metric
    print("\n--- Phase 2: Validation & Metrics ---")

    # Load Validation Data
    val_dataset = LidarDataset(split="val", load_cached_data=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=LidarDataset.collate_fn,
        pin_memory=True,
    )

    # Load Model
    model = TwoStagePointPillars()
    model.to(device)

    checkpoint_path = Config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        print(f"Loading best checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Warning: Checkpoint not found. Using current model state.")

    # Evaluate
    final_metric, analysis_df = evaluate_validation_set(model, val_loader, device)

    # PRINT FINAL METRIC (REQUIRED FORMAT)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    perform_failure_analysis(analysis_df)

    # 5. Submission
    print("\n--- Phase 3: Submission ---")
    threshold = 0.031193465694278867

    if final_metric > threshold:
        print(
            f"Metric ({final_metric}) > Threshold ({threshold}). Generating submission..."
        )
        generate_submission(
            checkpoint_path=Config.CHECKPOINT_PATH,
            output_path="./submission/submission.csv",
        )
    else:
        print(
            f"Metric ({final_metric}) <= Threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
