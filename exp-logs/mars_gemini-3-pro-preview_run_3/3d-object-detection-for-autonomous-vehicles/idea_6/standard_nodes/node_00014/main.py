import os
import sys
import time
import numpy as np
import torch
import pandas as pd
import cv2
import shutil
import math
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.trainer import Trainer
from library.model import TemporalPointPillars
from library.dataset import NuScenesLidarDataset
from library.utils import box3d_to_corners

# -----------------------------------------------------------------------------
# Configuration Overrides for Fast Baseline
# -----------------------------------------------------------------------------
# Limit training to ensure completion within time limits
Config.EPOCHS = 3
Config.SUBSET_SIZE = 2000  # Train on a subset
Config.USE_SUBSET = True
Config.BATCH_SIZE = 4
Config.NUM_WORKERS = 4
Config.SUBMISSION_PATH = "./working/temp_submission.csv"

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs(Config.WORKING_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Metric Calculation Utilities
# -----------------------------------------------------------------------------
def rotated_iou_3d(box1, box2):
    """
    Calculates 3D IoU between two boxes.
    Box format: [x, y, z, w, l, h, yaw]
    """
    # 1. Height Intersection
    z1_min = box1[2] - box1[5] / 2
    z1_max = box1[2] + box1[5] / 2
    z2_min = box2[2] - box2[5] / 2
    z2_max = box2[2] + box2[5] / 2

    inter_h = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    if inter_h == 0:
        return 0.0

    # 2. BEV Area Intersection
    # cv2.RotatedRect format: ((cx, cy), (width, height), angle_deg)
    # Our box: l is along x (0 deg), w is along y.
    # Convert yaw to degrees
    yaw1_deg = np.degrees(box1[6])
    yaw2_deg = np.degrees(box2[6])

    # Note: cv2 RotatedRect size is (width, height).
    # We map our (l, w) to this.
    rect1 = ((box1[0], box1[1]), (box1[4], box1[3]), yaw1_deg)
    rect2 = ((box2[0], box2[1]), (box2[4], box2[3]), yaw2_deg)

    try:
        intersection_type, points = cv2.rotatedRectangleIntersection(rect1, rect2)
        if intersection_type == cv2.INTERSECT_NONE:
            inter_area = 0.0
        elif intersection_type == cv2.INTERSECT_FULL:
            # One is inside the other, area is the smaller one
            area1 = box1[3] * box1[4]
            area2 = box2[3] * box2[4]
            inter_area = min(area1, area2)
        else:
            if points is not None and len(points) > 2:
                # Need to organize points for contourArea?
                # rotatedRectangleIntersection returns vertices in order usually
                inter_area = cv2.contourArea(points)
            else:
                inter_area = 0.0
    except Exception:
        inter_area = 0.0

    # 3. 3D IoU
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]

    inter_vol = inter_area * inter_h
    union_vol = vol1 + vol2 - inter_vol

    if union_vol <= 0:
        return 0.0

    return inter_vol / union_vol


def calculate_image_precision(pred_boxes, pred_scores, gt_boxes, thresholds):
    """
    Calculates the average precision for a single image across thresholds.
    Precision(t) = TP(t) / (TP(t) + FP(t) + FN(t))
    """
    if len(gt_boxes) == 0:
        # If no GT, any prediction is FP.
        # If preds > 0: TP=0, FP>0, FN=0 -> 0 / (0+FP+0) = 0
        # If preds == 0: TP=0, FP=0, FN=0 -> Undefined, usually 1.0 (perfect rejection)
        # Task says: "If there are no ground truth objects... ANY number of predictions... score of zero"
        if len(pred_boxes) > 0:
            return 0.0
        else:
            # Assuming 1.0 for perfect empty prediction matches empty GT
            return 1.0

    if len(pred_boxes) == 0:
        # GT exists but no preds. TP=0, FP=0, FN=len(GT). Score = 0.
        return 0.0

    # Sort preds by score
    sort_idx = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sort_idx]

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0

        # Track matched GTs
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(pred_boxes)):
            p_box = pred_boxes[i]

            # Find best matching GT
            best_iou = -1
            best_gt_idx = -1

            for j in range(len(gt_boxes)):
                if not gt_matched[j]:
                    iou = rotated_iou_3d(p_box, gt_boxes[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_gt_idx != -1 and best_iou > t:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        fn = len(gt_boxes) - np.sum(gt_matched)

        denominator = tp + fp + fn
        if denominator == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / denominator)

    return np.mean(precisions)


def validate_and_analyze(model, val_loader, device):
    """
    Runs validation, calculates metric, and performs failure analysis.
    """
    model.eval()
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    image_scores = []

    # Failure Analysis Data
    fa_data = []  # (gt_distance, gt_volume, max_iou)

    print("Running validation and analysis...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move to device
            voxels = batch["voxels"].to(device)
            num_points = batch["num_points"].to(device)
            coordinates = batch["coordinates"].to(device)
            sample_tokens = batch["sample_tokens"]
            metadata_list = batch["metadata"]

            batch_size = len(sample_tokens)
            preds = model(voxels, num_points, coordinates, batch_size=batch_size)

            # Decode
            # Use Trainer's decode logic via a helper or reimplement
            # Reimplementing minimal decode here to avoid instantiating Trainer again
            # Actually, let's borrow the decode method from Trainer class dynamically or copy logic
            # For simplicity, I'll use the logic I implemented in Trainer._decode_detections
            # But since I can't easily import the method unbound, I'll replicate the essential parts.

            # --- DECODE START ---
            k = Config.POST_MAX_OBJECTS
            hm = torch.sigmoid(preds["heatmap"])
            hm = torch.clamp(hm, min=1e-4, max=1 - 1e-4)
            # Max pool
            hm_pool = torch.nn.functional.max_pool2d(
                hm, kernel_size=3, stride=1, padding=1
            )
            mask = (hm_pool == hm).float()
            hm = hm * mask

            # Top K
            B, C, H, W = hm.shape
            scores, inds = torch.topk(hm.view(B, -1), k)

            # Gather regression
            # Helper
            def gather(feat, ind):
                dim = feat.size(1)
                feat = feat.view(B, dim, -1).permute(0, 2, 1)
                ind_exp = ind.unsqueeze(2).expand(B, k, dim)
                return feat.gather(1, ind_exp)

            ys = (inds // W).long()
            xs = (inds % W).long()
            inds = inds % (H * W)  # Fix index for gather if flattened differently?
            # Note: inds from topk on (B, C*H*W) are global indices.
            # We need spatial indices for regression gather.
            # Regression maps are (B, RegC, H, W). Flattened to (B, RegC, H*W).
            # The topk indices include channel offset. We need to mod by H*W.
            spatial_inds = inds % (H * W)

            reg_offset = gather(preds["offset"], spatial_inds)
            reg_z = gather(preds["height"], spatial_inds)
            reg_dim = gather(preds["dim"], spatial_inds)
            reg_rot = gather(preds["rot"], spatial_inds)

            # Decode coords
            xs = xs.float().view(B, k, 1) + reg_offset[:, :, 0:1]
            ys = ys.float().view(B, k, 1) + reg_offset[:, :, 1:2]

            # World coords
            xs = xs * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0]
            ys = ys * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1]
            zs = reg_z

            # Dims: exp
            dims = torch.exp(reg_dim)
            ls, ws, hs = dims[:, :, 0:1], dims[:, :, 1:2], dims[:, :, 2:3]

            # Rot
            yaws = torch.atan2(reg_rot[:, :, 0:1], reg_rot[:, :, 1:2])

            # [x, y, z, w, l, h, yaw]
            final_boxes = torch.cat([xs, ys, zs, ws, ls, hs, yaws], dim=2).cpu().numpy()
            scores = scores.cpu().numpy()
            # --- DECODE END ---

            # Process each sample in batch
            for i in range(B):
                # Get Predictions
                p_boxes = final_boxes[i]
                p_scores = scores[i]

                # Filter by score threshold
                mask_score = p_scores > Config.POST_SCORE_THRESHOLD
                p_boxes = p_boxes[mask_score]
                p_scores = p_scores[mask_score]

                # Get Ground Truth
                # Parse label string from metadata
                label_str = metadata_list[i].get("label", "")
                if pd.isna(label_str):
                    label_str = ""

                parts = str(label_str).strip().split()
                gt_boxes = []
                if len(parts) > 0 and len(parts) % 8 == 0:
                    num_objs = len(parts) // 8
                    for obj_i in range(num_objs):
                        off = obj_i * 8
                        # x, y, z, w, l, h, yaw
                        box = [float(parts[off + j]) for j in range(7)]
                        gt_boxes.append(box)
                gt_boxes = np.array(gt_boxes)

                # Calculate Metric
                score = calculate_image_precision(
                    p_boxes, p_scores, gt_boxes, thresholds
                )
                image_scores.append(score)

                # Failure Analysis Data Collection
                # For each GT, find max IoU with any prediction
                if len(gt_boxes) > 0:
                    for gt in gt_boxes:
                        # gt: [x, y, z, w, l, h, yaw]
                        dist = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
                        vol = gt[3] * gt[4] * gt[5]

                        max_iou = 0.0
                        if len(p_boxes) > 0:
                            for pb in p_boxes:
                                iou = rotated_iou_3d(pb, gt)
                                if iou > max_iou:
                                    max_iou = iou

                        fa_data.append([dist, vol, max_iou])

    # Final Metric
    final_metric = np.mean(image_scores) if len(image_scores) > 0 else 0.0

    # Failure Analysis Correlation
    fa_data = np.array(fa_data)
    corr_dist = 0.0
    corr_vol = 0.0
    if len(fa_data) > 1:
        # Correlation between Max IoU and Distance
        corr_dist = np.corrcoef(fa_data[:, 0], fa_data[:, 2])[0, 1]
        # Correlation between Max IoU and Volume
        corr_vol = np.corrcoef(fa_data[:, 1], fa_data[:, 2])[0, 1]

    return final_metric, corr_dist, corr_vol


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("Initializing Run...")

    # 1. Train
    print("Starting Training Phase...")
    trainer = Trainer(load_cached_data=True, subset_size=Config.SUBSET_SIZE)
    trainer.fit()

    # 2. Validate
    print("Starting Validation Phase...")
    # Load best model
    device = torch.device(Config.DEVICE)
    model = TemporalPointPillars().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Error: Model checkpoint not found.")
        return

    # Load validation data (use a larger subset or full for metric calculation)
    # Using 1000 samples for validation speed while maintaining representativeness
    val_dataset = NuScenesLidarDataset(
        mode="val", subset_size=1000, load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesLidarDataset.collate_fn,
    )

    metric, corr_dist, corr_vol = validate_and_analyze(model, val_loader, device)

    # 3. Report Results
    print(f"Final Validation Metric: {metric}")
    print(f"Failure Analysis - Correlation (IoU vs Distance): {corr_dist:.4f}")
    print(f"Failure Analysis - Correlation (IoU vs Volume): {corr_vol:.4f}")

    # 4. Submission
    threshold = 0.031193465694278867
    if metric > threshold:
        print(
            f"Metric ({metric}) exceeds threshold ({threshold}). Generating submission..."
        )
        trainer.predict_and_submit()

        # Move submission to required path
        target_path = "./submission/submission.csv"
        if os.path.exists(Config.SUBMISSION_PATH):
            shutil.move(Config.SUBMISSION_PATH, target_path)
            print(f"Submission moved to {target_path}")
        else:
            print("Error: Submission file generation failed.")
    else:
        print(
            f"Metric ({metric}) did not exceed threshold ({threshold}). Skipping submission."
        )


if __name__ == "__main__":
    main()
