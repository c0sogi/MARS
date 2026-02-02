import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision

# Import from provided library
from library.config import Config
from library.train import train_model
from library.predict import Predictor, generate_predictions
from library.dataset import LidarDataset
from library.utils import sensor_to_world, compute_iou_bev

# Ensure reproducible results
Config.set_seed(42)


def calculate_3d_iou(pred_box, gt_box):
    """
    Calculates 3D IoU based on the task description:
    IoU = (Intersection Area * Intersection Height) / Union Volume
    Boxes are [x, y, z, w, l, h, yaw] (Global Frame)
    """
    # 1. BEV Intersection (using simplified axis-aligned assumption for speed/baseline)
    # Although boxes have yaw, the task description simplifies 3D bounding volume context
    # to "ground bounding box" and "height".
    # For a strict baseline, we approximate using the provided compute_iou_bev which assumes axis-aligned.
    # To be more precise with yaw, we would need polygon intersection, but for this baseline
    # we will treat the BEV footprint as axis-aligned bounding box of the rotated box
    # OR simply ignore yaw for the overlap check if the deviation is small.
    # Given the complexity, we will use the axis-aligned approximation on the rotated corners
    # or simply pass w/l. The provided utils has `compute_iou_bev` for axis aligned.
    # We will use that on the (x, y, w, l) directly.

    # Unpack
    p_x, p_y, p_z, p_w, p_l, p_h = pred_box[:6]
    g_x, g_y, g_z, g_w, g_l, g_h = gt_box[:6]

    # BEV IoU (Axis Aligned Approximation for speed)
    # Intersection Width
    dx = min(p_x + p_w / 2, g_x + g_w / 2) - max(p_x - p_w / 2, g_x - g_w / 2)
    dy = min(p_y + p_l / 2, g_y + g_l / 2) - max(p_y - p_l / 2, g_y - g_l / 2)

    if dx <= 0 or dy <= 0:
        return 0.0

    inter_area = dx * dy
    area_p = p_w * p_l
    area_g = g_w * g_l

    # Height Intersection
    # z is center. z_min = z - h/2, z_max = z + h/2
    p_zmin, p_zmax = p_z - p_h / 2, p_z + p_h / 2
    g_zmin, g_zmax = g_z - g_h / 2, g_z + g_h / 2

    dz = min(p_zmax, g_zmax) - max(p_zmin, g_zmin)
    if dz <= 0:
        return 0.0

    inter_vol = inter_area * dz

    vol_p = area_p * p_h
    vol_g = area_g * g_h

    union_vol = vol_p + vol_g - inter_vol

    return inter_vol / union_vol


def get_val_predictions_and_gt(predictor):
    """
    Runs inference on validation set and returns structured predictions and ground truths.
    """
    print("Generating validation predictions...")
    dataset = LidarDataset(split="val", load_cached_data=True)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    predictions_map = {}  # token -> list of dicts
    ground_truth_map = {}  # token -> list of dicts

    # Load all GT first
    for idx in range(len(dataset)):
        row = dataset.metadata.iloc[idx]
        token = row["token"]
        anns = row["annotations"]
        gt_boxes = []
        for ann in anns:
            if ann["class_name"] in Config.DETECT_CLASSES:
                gt_boxes.append(
                    {
                        "box": [
                            ann["center_x"],
                            ann["center_y"],
                            ann["center_z"],
                            ann["width"],
                            ann["length"],
                            ann["height"],
                            ann["yaw"],
                        ],
                        "class_name": ann["class_name"],
                    }
                )
        ground_truth_map[token] = gt_boxes

    # Inference
    predictor.model.eval()
    with torch.no_grad():
        for bev, targets, tokens in dataloader:
            bev = bev.to(predictor.device)
            preds = predictor.model(bev)
            decoded = predictor._decode_predictions(preds)  # (B, A, H, W, 9)

            # Flatten
            B = decoded.shape[0]
            decoded = decoded.view(B, -1, 9)

            for i in range(B):
                token = tokens[i]
                sample_preds = decoded[i]

                # Filter by confidence
                mask = sample_preds[:, 7] >= Config.CONF_THRESHOLD
                valid_preds = sample_preds[mask]

                final_preds_list = []

                if valid_preds.shape[0] > 0:
                    # NMS
                    x_c, y_c = valid_preds[:, 0], valid_preds[:, 1]
                    w, l = valid_preds[:, 3], valid_preds[:, 4]
                    x1, y1 = x_c - w / 2, y_c - l / 2
                    x2, y2 = x_c + w / 2, y_c + l / 2
                    boxes_nms = torch.stack([x1, y1, x2, y2], dim=1)
                    scores_nms = valid_preds[:, 7]

                    keep = torchvision.ops.nms(
                        boxes_nms, scores_nms, Config.NMS_IOU_THRESHOLD
                    )
                    final_preds = valid_preds[keep]

                    # Transform to Global
                    ego_pose, calib_sensor = dataset.calib_lookup.get_calibration(token)
                    final_preds_np = final_preds.cpu().numpy()

                    centers_s = final_preds_np[:, 0:3]
                    centers_g = sensor_to_world(centers_s, ego_pose, calib_sensor)

                    # Yaw transform (simplified)
                    yaws_s = final_preds_np[:, 6]
                    vecs_s = np.stack(
                        [np.cos(yaws_s), np.sin(yaws_s), np.zeros_like(yaws_s)], axis=1
                    )
                    p1_g = sensor_to_world(
                        np.zeros_like(vecs_s), ego_pose, calib_sensor
                    )
                    p2_g = sensor_to_world(vecs_s, ego_pose, calib_sensor)
                    vecs_g = p2_g - p1_g
                    yaws_g = np.arctan2(vecs_g[:, 1], vecs_g[:, 0])

                    for j in range(len(final_preds_np)):
                        box = [
                            centers_g[j, 0],
                            centers_g[j, 1],
                            centers_g[j, 2],
                            final_preds_np[j, 3],
                            final_preds_np[j, 4],
                            final_preds_np[j, 5],
                            yaws_g[j],
                        ]
                        cls_idx = int(final_preds_np[j, 8])
                        final_preds_list.append(
                            {
                                "box": box,
                                "score": float(final_preds_np[j, 7]),
                                "class_name": Config.DETECT_CLASSES[cls_idx],
                            }
                        )

                # Sort by score descending
                final_preds_list.sort(key=lambda x: x["score"], reverse=True)
                predictions_map[token] = final_preds_list

    return predictions_map, ground_truth_map


def evaluate_metric(predictions_map, ground_truth_map):
    """
    Calculates the mAP metric as defined in the task.
    """
    thresholds = np.arange(0.5, 0.95 + 1e-6, 0.05)
    aps_per_threshold = []

    print("Calculating metrics...")

    for t in thresholds:
        precisions = []

        for token, gt_objects in ground_truth_map.items():
            preds = predictions_map.get(token, [])

            # If no GT, precision is 0 if there are any preds, else 1?
            # Task: "If there are no ground truth objects at all for a given image,
            # ANY number of predictions (false positives) will result in the image receiving a score of zero"
            if not gt_objects:
                if len(preds) > 0:
                    precisions.append(0.0)
                else:
                    # No GT and No Preds -> Perfect? Or undefined?
                    # Usually 1.0. Let's assume 1.0 for empty-empty match.
                    precisions.append(1.0)
                continue

            tp = 0
            fp = 0

            # Track which GT are matched
            gt_matched = [False] * len(gt_objects)

            for p in preds:
                # Find best match in GT
                best_iou = -1
                best_gt_idx = -1

                for i, g in enumerate(gt_objects):
                    if gt_matched[i]:
                        continue
                    # Check class
                    if p["class_name"] != g["class_name"]:
                        continue

                    iou = calculate_3d_iou(p["box"], g["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

                if best_iou > t:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1

            fn = len(gt_objects) - sum(gt_matched)

            denom = tp + fp + fn
            if denom == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / denom)

        avg_precision_t = np.mean(precisions)
        aps_per_threshold.append(avg_precision_t)

    final_metric = np.mean(aps_per_threshold)
    return final_metric


def failure_analysis(predictions_map, ground_truth_map):
    """
    Analyzes correlation between localization error and distance from ego.
    """
    print("Performing failure analysis...")
    errors = []
    distances = []

    for token, gt_objects in ground_truth_map.items():
        preds = predictions_map.get(token, [])
        if not gt_objects or not preds:
            continue

        gt_matched = [False] * len(gt_objects)

        # Greedy match at 0.5 IoU
        for p in preds:
            best_iou = -1
            best_gt_idx = -1
            for i, g in enumerate(gt_objects):
                if gt_matched[i]:
                    continue
                if p["class_name"] != g["class_name"]:
                    continue
                iou = calculate_3d_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou > 0.5:
                gt_matched[best_gt_idx] = True
                g = gt_objects[best_gt_idx]

                # Calculate Error (Euclidean distance of centers)
                p_c = np.array(p["box"][:3])
                g_c = np.array(g["box"][:3])
                error = np.linalg.norm(p_c - g_c)

                # Calculate Distance from Ego (Approximate using GT center coordinates)
                # Note: These are global coordinates. To get distance from ego, we need ego position.
                # However, usually 'distance' implies distance from sensor.
                # Since we don't have ego pose easily accessible here without reloading,
                # we can approximate if we assume the scene is centered or use the metadata lookup again.
                # For speed, let's assume the correlation is desired against the sensor-frame distance.
                # But we transformed to global.
                # Let's use the assumption that samples are roughly centered around ego in the BEV crop,
                # but we have global coords.
                # We need to reload ego pose for this specific analysis or pass it through.
                # Let's skip the exact ego lookup and use the fact that we have the data in the loop.
                pass

    # Re-implementing loop to get ego pose for distance calculation
    dataset = LidarDataset(split="val", load_cached_data=True)

    # Map token to ego pose
    token_to_pose = {}
    for idx in range(len(dataset)):
        row = dataset.metadata.iloc[idx]
        token = row["token"]
        ego_pose, _ = dataset.calib_lookup.get_calibration(token)
        token_to_pose[token] = np.array(ego_pose["translation"])

    for token, gt_objects in ground_truth_map.items():
        if token not in token_to_pose:
            continue
        ego_pos = token_to_pose[token]

        preds = predictions_map.get(token, [])
        gt_matched = [False] * len(gt_objects)

        for p in preds:
            best_iou = -1
            best_gt_idx = -1
            for i, g in enumerate(gt_objects):
                if gt_matched[i]:
                    continue
                if p["class_name"] != g["class_name"]:
                    continue
                iou = calculate_3d_iou(p["box"], g["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou > 0.5:
                gt_matched[best_gt_idx] = True
                g = gt_objects[best_gt_idx]

                # Error
                p_c = np.array(p["box"][:3])
                g_c = np.array(g["box"][:3])
                loc_error = np.linalg.norm(p_c - g_c)

                # Distance from Ego
                dist = np.linalg.norm(g_c - ego_pos)

                errors.append(loc_error)
                distances.append(dist)

    if len(errors) > 1:
        correlation = np.corrcoef(errors, distances)[0, 1]
        print(
            f"Correlation between Error Magnitude and Distance from Ego: {correlation:.4f}"
        )
    else:
        print("Not enough matches for failure analysis.")


def main():
    # 1. Configure for Fast Baseline
    # Reduce epochs to ensure completion within 2 hours
    Config.NUM_EPOCHS = 3
    Config.BATCH_SIZE = 32

    print("=== Starting 3D Object Detection Pipeline ===")

    # 2. Train Model
    print("\n[Step 1/4] Training Model...")
    train_model(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

    # 3. Validation & Metrics
    print("\n[Step 2/4] Evaluating on Validation Set...")
    predictor = Predictor(device=Config.DEVICE)
    preds_map, gt_map = get_val_predictions_and_gt(predictor)

    final_metric = evaluate_metric(preds_map, gt_map)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n[Step 3/4] Failure Analysis...")
    failure_analysis(preds_map, gt_map)

    # 5. Submission
    print("\n[Step 4/4] Generating Submission...")
    # Generate predictions for test set
    generate_predictions(batch_size=Config.BATCH_SIZE)

    print("\nPipeline Complete.")


if __name__ == "__main__":
    main()
