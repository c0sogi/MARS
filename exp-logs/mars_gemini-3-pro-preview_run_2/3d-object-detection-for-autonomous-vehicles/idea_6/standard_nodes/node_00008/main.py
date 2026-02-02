import os
import sys
import time
import numpy as np
import torch
import cv2
import pandas as pd
from torch.utils.data import DataLoader

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PillarUNet3D
from library.train import train_model
from library.inference import decode_predictions, generate_submission
from library.utils import transform_box_to_global

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
FAST_RUN_SAMPLES = 3000  # Limit training samples for speed
FAST_RUN_EPOCHS = 5  # Limit epochs for speed
METRIC_THRESHOLD = 0.0821688096
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. Metric Implementation
# ==============================================================================
def calculate_bev_iou(box_a, box_b):
    """
    Calculates BEV Intersection over Union for two rotated rectangles.
    Box format: [x, y, z, w, l, h, yaw]
    """
    # Create RotatedRect for cv2
    # cv2.RotatedRect takes ((x, y), (w, h), angle_deg)
    # Note: box[3] is width (x-axis size in local), box[4] is length (y-axis size)
    # Yaw is radians. Convert to degrees.

    rect_a = ((box_a[0], box_a[1]), (box_a[3], box_a[4]), np.degrees(box_a[6]))
    rect_b = ((box_b[0], box_b[1]), (box_b[3], box_b[4]), np.degrees(box_b[6]))

    # Intersection
    try:
        intersection_type, poly = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        if intersection_type == cv2.INTERSECT_NONE:
            return 0.0, 0.0
        elif intersection_type == cv2.INTERSECT_FULL:
            # One is inside the other, area is the smaller one
            area_a = box_a[3] * box_a[4]
            area_b = box_b[3] * box_b[4]
            intersection_area = min(area_a, area_b)
        else:
            # Partial intersection
            # poly is a set of vertices
            if poly is not None:
                intersection_area = cv2.contourArea(poly)
            else:
                intersection_area = 0.0
    except Exception:
        return 0.0, 0.0

    return intersection_area


def calculate_3d_iou(box_a, box_b):
    """
    Calculates 3D IoU as per task description.
    IoU = (Intersect_BEV * Intersect_Height) / Union_Vol
    """
    # 1. BEV Intersection
    inter_area = calculate_bev_iou(box_a, box_b)
    if inter_area == 0:
        return 0.0

    # 2. Height Intersection
    # Box A z-range
    za_min = box_a[2] - box_a[5] / 2.0
    za_max = box_a[2] + box_a[5] / 2.0
    # Box B z-range
    zb_min = box_b[2] - box_b[5] / 2.0
    zb_max = box_b[2] + box_b[5] / 2.0

    z_inter_min = max(za_min, zb_min)
    z_inter_max = min(za_max, zb_max)
    inter_h = max(0.0, z_inter_max - z_inter_min)

    if inter_h == 0:
        return 0.0

    intersection_vol = inter_area * inter_h

    # 3. Union
    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]

    union_vol = vol_a + vol_b - intersection_vol

    if union_vol <= 0:
        return 0.0

    return intersection_vol / union_vol


def evaluate_sample_metric(pred_boxes, gt_boxes, thresholds):
    """
    Calculates the average precision for a single sample across thresholds.
    Metric: Mean of (TP / (TP + FP + FN)) at each threshold.
    """
    if len(gt_boxes) == 0:
        # If no ground truth, any prediction is FP.
        # If predictions exist, score is 0. If no predictions, score is 1 (perfect match of nothing)?
        # Task says: "If there are no ground truth objects at all for a given image,
        # ANY number of predictions (false positives) will result in the image receiving a score of zero"
        if len(pred_boxes) > 0:
            return 0.0
        else:
            # No GT, No Pred -> Perfect? Or undefined?
            # Usually in detection, empty image with empty pred is 1.0.
            # Based on formula: TP=0, FP=0, FN=0. 0/0 is undefined.
            # Assuming 1.0 for correct empty prediction.
            return 1.0

    # Pre-calculate IoU matrix to save time
    # Matrix shape: (num_preds, num_gts)
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_preds == 0:
        # TP=0, FP=0, FN=num_gts. Precision = 0 / (0 + 0 + num_gts) = 0
        return 0.0

    iou_matrix = np.zeros((num_preds, num_gts))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_3d_iou(p["box"], g)

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0

        # Greedy matching based on confidence (preds are already sorted)
        gt_matched = np.zeros(num_gts, dtype=bool)
        pred_matched = np.zeros(num_preds, dtype=bool)

        for i in range(num_preds):
            # Find best match for this prediction
            best_iou = -1.0
            best_gt_idx = -1

            for j in range(num_gts):
                if not gt_matched[j]:
                    iou = iou_matrix[i, j]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou > t:
                # Hit
                tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[i] = True
            else:
                fp += 1

        # False Negatives: GTs not matched
        fn = np.sum(~gt_matched)

        # Note: FP is also count of unmatched preds, which we calculated iteratively
        # Double check FP calculation:
        # In the loop, if matched, we don't increment FP. If not matched, we increment FP.
        # This is correct for the logic "A false positive indicates a predicted object had no associated ground truth object"

        denom = tp + fp + fn
        if denom == 0:
            prec = 0.0  # Should not happen if len(gt) > 0
        else:
            prec = tp / denom

        precisions.append(prec)

    return np.mean(precisions)


def run_validation_and_analysis(model, dataloader, device):
    print("Running Validation and Failure Analysis...")
    model.eval()

    metric_scores = []

    # Failure Analysis Data
    # List of (error_magnitude, distance, volume)
    fa_data = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            batch["voxels"] = batch["voxels"].to(device)
            batch["num_points"] = batch["num_points"].to(device)
            batch["coordinates"] = batch["coordinates"].to(device)

            # Predict
            preds = model(batch)
            decoded_batch = decode_predictions(preds, Config)

            # Get GT (in Ego frame for comparison)
            # The dataset collate_fn puts gt_boxes in a list of arrays
            gt_batch = batch["gt_boxes"]

            for i, pred_boxes in enumerate(decoded_batch):
                gt_boxes = gt_batch[i]  # Array (N, 8)

                # Sort preds by score descending
                pred_boxes.sort(key=lambda x: x["score"], reverse=True)

                # 1. Calculate Metric
                score = evaluate_sample_metric(pred_boxes, gt_boxes, IOU_THRESHOLDS)
                metric_scores.append(score)

                # 2. Failure Analysis
                # For each GT, find best matching prediction (max IoU)
                # GT format: [x, y, z, w, l, h, yaw, class]
                if len(gt_boxes) > 0:
                    for gt in gt_boxes:
                        gt_box_params = gt[:7]

                        best_iou = 0.0
                        for p in pred_boxes:
                            iou = calculate_3d_iou(p["box"], gt_box_params)
                            if iou > best_iou:
                                best_iou = iou

                        error_mag = 1.0 - best_iou

                        # Features
                        # Distance: sqrt(x^2 + y^2)
                        dist = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
                        # Volume: w * l * h
                        vol = gt[3] * gt[4] * gt[5]

                        fa_data.append(
                            {"error": error_mag, "distance": dist, "volume": vol}
                        )

    final_metric = np.mean(metric_scores) if metric_scores else 0.0

    # Failure Analysis Correlation
    if fa_data:
        df_fa = pd.DataFrame(fa_data)
        corr_dist = df_fa["error"].corr(df_fa["distance"])
        corr_vol = df_fa["error"].corr(df_fa["volume"])
    else:
        corr_dist = 0.0
        corr_vol = 0.0

    return final_metric, corr_dist, corr_vol


# ==============================================================================
# 3. Main Execution
# ==============================================================================
def main():
    set_seed(Config.SEED)

    # 1. Train
    print("=== Starting Training Phase ===")
    # Using the library function, but we rely on it saving 'best_model.pth'
    # Cite debug_lesson_1: Invalidate Persistent Caches When Modifying Data Generation Logic
    train_model(
        num_epochs=FAST_RUN_EPOCHS,
        debug_limit=FAST_RUN_SAMPLES,
        load_cached_data=False,
        patience=3,
    )

    # 2. Validation
    print("\n=== Starting Validation Phase ===")
    device = Config.DEVICE

    # Load Model
    model = PillarUNet3D().to(device)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(
            "Checkpoint not found, using initialized model (performance will be poor)."
        )
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = (
            checkpoint["model_state_dict"]
            if "model_state_dict" in checkpoint
            else checkpoint
        )
        # Fix module keys if needed
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

    # Load Val Dataset
    val_dataset = NuScenesDataset(is_train=False, load_cached_data=True)
    # Limit validation size if needed, but metric should be on full set for accuracy
    # However, for speed in this task, we might limit it slightly if it takes too long
    # But requirement says "entire hold-out validation set".

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
    )

    final_metric, corr_dist, corr_vol = run_validation_and_analysis(
        model, val_loader, device
    )

    print(f"Final Validation Metric: {final_metric:.10f}")

    print("\n=== Failure Analysis ===")
    print(f"Correlation between Error and Distance: {corr_dist:.4f}")
    print(f"Correlation between Error and Volume: {corr_vol:.4f}")

    # 3. Submission
    if final_metric > METRIC_THRESHOLD:
        print("\n=== Generating Submission ===")
        generate_submission(checkpoint_path, Config.SUBMISSION_PATH)
    else:
        print(
            f"\nMetric {final_metric:.10f} did not exceed threshold {METRIC_THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
