import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2

# Import from provided library
from library.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    SUBMISSION_PATH,
    WORKING_DIR,
    CLASS_NAMES,
    POINT_CLOUD_RANGE,
    VOXEL_SIZE,
    OUT_SIZE_FACTOR,
    set_deterministic,
)
from library.dataset import LidarDataset, collate_fn
from library.model import CenterPointNet
from library.engine import train_one_epoch, LossWrapper
from library.utils import decode_predictions

# ==============================================================================
# Configuration for Fast Baseline
# ==============================================================================
# Override defaults for speed within 2-hour limit
FAST_TRAIN_SAMPLES = 2000
FAST_VAL_SAMPLES = 500
FAST_EPOCHS = 5
CONFIDENCE_THRESHOLD = 0.2
IOU_THRESHOLDS = np.arange(0.5, 0.96, 0.05)
METRIC_PASS_THRESHOLD = 0.0024

# Ensure reproducibility
set_deterministic(42)

# ==============================================================================
# Helper Functions: Geometric & Metric
# ==============================================================================


def get_corners_2d(x, y, w, l, yaw):
    """
    Returns the 4 corners of a rotated rectangle in 2D.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Corners relative to center (w is along x-axis in box frame, l along y)
    # Note: Dataset defines w, l, h. Usually w is width (y-axis in ego?), l is length (x-axis in ego?).
    # Based on config, w corresponds to x-size in grid if aligned?
    # Let's assume standard definition: w is dimension along local x, l along local y.

    dx = w / 2.0
    dy = l / 2.0

    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])

    # Rotate
    # x' = x cos - y sin
    # y' = x sin + y cos
    rotated_corners = np.zeros_like(corners)
    rotated_corners[:, 0] = corners[:, 0] * cos_yaw - corners[:, 1] * sin_yaw
    rotated_corners[:, 1] = corners[:, 0] * sin_yaw + corners[:, 1] * cos_yaw

    # Translate
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y

    return rotated_corners.astype(np.float32)


def compute_3d_iou(box_a, box_b):
    """
    Calculates 3D IoU between two boxes.
    Box format: [x, y, z, w, l, h, yaw]
    """
    # 1. Height Intersection
    za_min = box_a[2] - box_a[5] / 2.0
    za_max = box_a[2] + box_a[5] / 2.0
    zb_min = box_b[2] - box_b[5] / 2.0
    zb_max = box_b[2] + box_b[5] / 2.0

    h_intersect = max(0, min(za_max, zb_max) - max(za_min, zb_min))
    h_union = max(za_max, zb_max) - min(
        za_min, zb_min
    )  # This is height of union bounding box, not union of heights

    # Correct 3D IoU definition: Intersection Volume / Union Volume
    # VolA = AreaA * Ha
    # VolB = AreaB * Hb
    # VolInt = AreaInt * h_intersect
    # VolUnion = VolA + VolB - VolInt

    if h_intersect <= 0:
        return 0.0

    # 2. BEV Intersection (Rotated Rectangles)
    # Quick center distance check
    dist = np.sqrt((box_a[0] - box_b[0]) ** 2 + (box_a[1] - box_b[1]) ** 2)
    max_radius = (
        np.sqrt(box_a[3] ** 2 + box_a[4] ** 2) + np.sqrt(box_b[3] ** 2 + box_b[4] ** 2)
    ) / 2.0
    if dist > max_radius:
        return 0.0

    rect_a = get_corners_2d(box_a[0], box_a[1], box_a[3], box_a[4], box_a[6])
    rect_b = get_corners_2d(box_b[0], box_b[1], box_b[3], box_b[4], box_b[6])

    # Use OpenCV for intersection area
    try:
        # intersectConvexConvex expects contours
        intersection_area, intersection_contour = cv2.intersectConvexConvex(
            rect_a, rect_b
        )
    except:
        intersection_area = 0.0

    if intersection_area <= 0:
        return 0.0

    area_a = box_a[3] * box_a[4]
    area_b = box_b[3] * box_b[4]

    vol_a = area_a * box_a[5]
    vol_b = area_b * box_b[5]
    vol_int = intersection_area * h_intersect

    iou = vol_int / (vol_a + vol_b - vol_int + 1e-6)
    return iou


def calculate_sample_metric(pred_boxes, gt_boxes, thresholds):
    """
    Calculates the average precision for a single sample across thresholds.
    pred_boxes: list of [x, y, z, w, l, h, yaw, score, class_id]
    gt_boxes: list of [x, y, z, w, l, h, yaw, class_id]
    """
    # Filter by class? The metric description implies generic object matching or per-class?
    # "A true positive is counted when a single predicted object matches a ground truth object..."
    # Usually 3D detection is per-class. However, the metric formula provided doesn't explicitly loop over classes.
    # It says "comparing the predicted object to all ground truth objects".
    # We will assume class matching is required for a valid match.

    if len(gt_boxes) == 0:
        return (
            0.0 if len(pred_boxes) > 0 else 1.0
        )  # If no GT and no Preds -> Perfect? Formula says: "If no GT... ANY predictions... score of zero". If no preds either, undefined? Assume 1.0 or 0.0. Let's assume 0 if no GT.
        # Actually: "If there are no ground truth objects at all for a given image, ANY number of predictions (false positives) will result in the image receiving a score of zero"
        # Implies if 0 preds, score might be 1? But usually AP is 0 if no GT.
        # Let's stick to the formula: sum(TP/(TP+FP+FN)). If GT=0, FN=0. If Pred=0, TP=0, FP=0. 0/0.
        # We'll return 0.0 if GT is empty.

    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by confidence
    pred_boxes = sorted(pred_boxes, key=lambda x: x[7], reverse=True)

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0

        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # Greedy matching
        for p_idx, pred in enumerate(pred_boxes):
            best_iou = -1.0
            best_gt_idx = -1

            p_geom = pred[:7]
            p_cls = int(pred[8])

            for g_idx, gt in enumerate(gt_boxes):
                if gt_matched[g_idx]:
                    continue

                g_cls = int(gt[7])
                if p_cls != g_cls:
                    continue

                iou = compute_3d_iou(p_geom, gt[:7])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou > t:
                tp += 1
                gt_matched[best_gt_idx] = True
                pred_matched[p_idx] = True
            else:
                fp += 1

        # False Negatives: GTs not matched
        fn = len(gt_boxes) - sum(gt_matched)

        denom = tp + fp + fn
        if denom == 0:
            precisions.append(0.0)
        else:
            precisions.append(tp / denom)

    return np.mean(precisions)


# ==============================================================================
# Main Execution Flow
# ==============================================================================


def main():
    print("Starting Runfile Execution...")

    # --------------------------------------------------------------------------
    # 1. Dataset & Dataloader Setup
    # --------------------------------------------------------------------------
    print("Initializing Datasets...")
    train_dataset = LidarDataset(
        TRAIN_METADATA_PATH, mode="train", num_samples=FAST_TRAIN_SAMPLES
    )
    val_dataset = LidarDataset(
        VAL_METADATA_PATH, mode="val", num_samples=FAST_VAL_SAMPLES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # --------------------------------------------------------------------------
    # 2. Model Training
    # --------------------------------------------------------------------------
    print("Setting up Model...")
    model = CenterPointNet().to(DEVICE)
    criterion = LossWrapper().to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    total_steps = len(train_loader) * FAST_EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )

    print(f"Training for {FAST_EPOCHS} epochs on {len(train_dataset)} samples...")
    model.train()

    for epoch in range(FAST_EPOCHS):
        # Reuse engine's train_one_epoch logic but we call it directly
        # Note: train_one_epoch prints output, which is fine.
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, DEVICE, epoch
        )

    # Save model state
    model_path = os.path.join(WORKING_DIR, "fast_model.pth")
    torch.save(model.state_dict(), model_path)
    print("Training Complete.")

    # --------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Running Validation...")
    model.eval()

    sample_scores = []
    all_predictions = []  # For failure analysis
    all_ground_truths = []

    with torch.no_grad():
        for batch in val_loader:
            points = [p.to(DEVICE) for p in batch["points"]]
            sample_tokens = batch["sample_tokens"]

            # Forward
            preds = model({"points": points})

            # Decode
            detections = decode_predictions(
                preds["heatmap"],
                preds["dim"],
                preds["rot"],
                preds["reg"],
                preds["z_map"],
                K=50,
            )
            detections = detections.cpu().numpy()

            # Get GT for this batch
            # We need to retrieve GT boxes from dataset or pass them through collate
            # LidarDataset returns 'gt_boxes' in __getitem__, but collate_fn doesn't stack them nicely
            # because they are variable length. It's not in the batch dict passed to model.
            # We need to re-fetch or modify collate?
            # Actually, collate_fn does not return gt_boxes in the batch dict!
            # We must fetch them from the dataset using sample_tokens or indices.
            # Since shuffle=False for Val, we can map indices, but safer to use tokens.
            # The dataset object has a df. We can lookup by token.

            for i, token in enumerate(sample_tokens):
                # Get Preds
                sample_dets = detections[i]
                valid_preds = sample_dets[sample_dets[:, 7] > CONFIDENCE_THRESHOLD]

                # Get GT
                # Parse from dataframe in dataset
                # We need to replicate the parsing logic from dataset.__getitem__ to get Ego-frame boxes
                # Or easier: modify dataset to return raw GT in collate? No, cannot modify library.
                # We must use the dataset's internal methods or re-parse.

                # Re-parsing logic:
                row = val_dataset.df[val_dataset.df["sample_token"] == token].iloc[0]
                # We need transformation matrices to convert Global GT to Ego GT
                # The dataset caches these.
                M_global_to_ego, M_ego_to_sensor = (
                    val_dataset.get_transformation_matrices(token)
                )

                gt_boxes_ego = []
                if pd.notna(row["label"]):
                    label_str = str(row["label"]).split()
                    stride = 8
                    num_objs = len(label_str) // stride
                    for j in range(num_objs):
                        base = j * stride
                        cx, cy, cz = (
                            float(label_str[base]),
                            float(label_str[base + 1]),
                            float(label_str[base + 2]),
                        )
                        w, l, h = (
                            float(label_str[base + 3]),
                            float(label_str[base + 4]),
                            float(label_str[base + 5]),
                        )
                        yaw = float(label_str[base + 6])
                        cls_name = label_str[base + 7]

                        if cls_name in CLASS_NAMES:
                            cls_id = CLASS_NAMES.index(cls_name)

                            # Transform to Ego
                            center_global = np.array([cx, cy, cz, 1.0])
                            center_ego = M_global_to_ego @ center_global
                            # Note: Model outputs are in Ego-Sensor frame (based on config POINT_CLOUD_RANGE).
                            # Dataset __getitem__ transforms Global -> Ego -> Sensor.
                            # So we must compare in Sensor frame.
                            center_sensor = M_ego_to_sensor @ center_ego

                            # Transform Yaw
                            R_box_global = np.array(
                                [
                                    [np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1],
                                ]
                            )
                            R_global_to_ego_3x3 = M_global_to_ego[:3, :3]
                            R_ego_to_sensor_3x3 = M_ego_to_sensor[:3, :3]
                            R_box_sensor = (
                                R_ego_to_sensor_3x3 @ R_global_to_ego_3x3 @ R_box_global
                            )
                            new_yaw = np.arctan2(R_box_sensor[1, 0], R_box_sensor[0, 0])

                            gt_boxes_ego.append(
                                [
                                    center_sensor[0],
                                    center_sensor[1],
                                    center_sensor[2],
                                    w,
                                    l,
                                    h,
                                    new_yaw,
                                    cls_id,
                                ]
                            )

                gt_boxes_ego = np.array(gt_boxes_ego)

                # Calculate Metric for this sample
                score = calculate_sample_metric(
                    valid_preds, gt_boxes_ego, IOU_THRESHOLDS
                )
                sample_scores.append(score)

                # Store for failure analysis
                all_predictions.append(valid_preds)
                all_ground_truths.append(gt_boxes_ego)

    final_metric = np.mean(sample_scores)
    print(f"Final Validation Metric: {final_metric}")

    # --------------------------------------------------------------------------
    # 4. Failure Analysis
    # --------------------------------------------------------------------------
    print("Performing Failure Analysis...")

    error_magnitudes = []  # 1 - IoU for TPs
    input_features = []  # Distance from ego

    for i in range(len(all_predictions)):
        preds = all_predictions[i]
        gts = all_ground_truths[i]

        if len(preds) == 0 or len(gts) == 0:
            continue

        # Find TPs at 0.5 IoU
        preds = sorted(preds, key=lambda x: x[7], reverse=True)
        gt_matched = [False] * len(gts)

        for p in preds:
            p_geom = p[:7]
            p_cls = int(p[8])

            best_iou = -1.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(gts):
                if gt_matched[g_idx] or int(gt[7]) != p_cls:
                    continue
                iou = compute_3d_iou(p_geom, gt[:7])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou > 0.1:  # Loose match for analysis
                gt_matched[best_gt_idx] = True

                # Feature: Distance from Ego (0,0,0)
                dist = np.sqrt(p[0] ** 2 + p[1] ** 2)

                # Error: 1 - IoU
                error = 1.0 - best_iou

                error_magnitudes.append(error)
                input_features.append(dist)

    if len(error_magnitudes) > 1:
        correlation = np.corrcoef(error_magnitudes, input_features)[0, 1]
        print(
            f"Correlation between Error (1-IoU) and Distance from Ego: {correlation:.4f}"
        )
    else:
        print("Not enough matches for correlation analysis.")

    # --------------------------------------------------------------------------
    # 5. Submission Generation
    # --------------------------------------------------------------------------
    if final_metric > METRIC_PASS_THRESHOLD:
        print("Metric passed threshold. Generating Submission...")

        test_dataset = LidarDataset(TEST_METADATA_PATH, mode="test", num_samples=None)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

        results = []
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                points = [p.to(DEVICE) for p in batch["points"]]
                sample_tokens = batch["sample_tokens"]

                preds = model({"points": points})

                detections = decode_predictions(
                    preds["heatmap"],
                    preds["dim"],
                    preds["rot"],
                    preds["reg"],
                    preds["z_map"],
                    K=50,
                )
                detections = detections.cpu().numpy()

                for i, token in enumerate(sample_tokens):
                    sample_dets = detections[i]
                    prediction_strings = []

                    for det in sample_dets:
                        x, y, z, w, l, h, yaw, score, cls_id = det

                        if score < CONFIDENCE_THRESHOLD:
                            continue

                        cls_name = CLASS_NAMES[int(cls_id)]
                        # Format: score x y z w l h yaw class_name
                        pred_str = f"{score:.4f} {x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {yaw:.4f} {cls_name}"
                        prediction_strings.append(pred_str)

                    full_pred_str = " ".join(prediction_strings)
                    results.append({"Id": token, "PredictionString": full_pred_str})

        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(f"Metric {final_metric} <= {METRIC_PASS_THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
