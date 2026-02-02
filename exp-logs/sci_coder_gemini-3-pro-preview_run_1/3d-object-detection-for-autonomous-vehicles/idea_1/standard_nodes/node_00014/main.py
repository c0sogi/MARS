import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr

# Import provided library modules
import library.config as config
import library.utils as utils
from library.data_interface import DataInterface
from library.dataset import BEVDataset
from library.model import BEVDetector
from library.train import train_model
from library.inference import (
    generate_submission,
    decode_predictions,
    transform_predictions_to_world,
)


def calculate_iou_matrix(gt_boxes, pred_boxes):
    """
    Calculates the IoU matrix between ground truth and predicted boxes.
    gt_boxes: (N, 7)
    pred_boxes: (M, 7)
    Returns: (N, M) matrix
    """
    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)
    iou_matrix = np.zeros((num_gt, num_pred))

    for i in range(num_gt):
        for j in range(num_pred):
            # utils.box_3d_iou expects list/array of 7 elements
            iou_matrix[i, j] = utils.box_3d_iou(gt_boxes[i], pred_boxes[j])

    return iou_matrix


def compute_map(predictions, gt_annotations, iou_thresholds):
    """
    Computes Mean Average Precision according to the task description.
    """
    aps_per_image = []

    # Align predictions and GT by sample_token
    sample_tokens = list(gt_annotations.keys())

    for token in tqdm(sample_tokens, disable=True):
        gt = np.array(gt_annotations[token])  # (N, 7)
        preds = predictions.get(token, [])

        # If preds is a list of dicts/arrays, extract boxes and scores
        if len(preds) > 0:
            # preds is a dict with 'bboxes', 'scores', 'labels'
            p_boxes = preds["bboxes"]
            p_scores = preds["scores"]

            # Sort by confidence descending
            sort_idx = np.argsort(p_scores)[::-1]
            p_boxes = p_boxes[sort_idx]
            p_scores = p_scores[sort_idx]
        else:
            p_boxes = np.empty((0, 7))
            p_scores = np.array([])

        # Handle empty cases defined in metric
        if len(gt) == 0:
            # If no GT, any prediction is FP.
            # Formula: TP / (TP + FP + FN).
            # If preds > 0: TP=0, FP>0, FN=0 -> Precision = 0.
            # If preds == 0: TP=0, FP=0, FN=0 -> Undefined, usually 1.0 or 0.0.
            # Task says: "ANY number of predictions (false positives) will result in the image receiving a score of zero"
            # It implies if preds > 0, score is 0. If preds == 0, it's a perfect match for empty image.
            if len(p_boxes) > 0:
                aps_per_image.append(0.0)
            else:
                aps_per_image.append(1.0)
            continue

        if len(p_boxes) == 0:
            # GT exists but no preds -> Precision = 0
            aps_per_image.append(0.0)
            continue

        # Calculate IoU Matrix once
        iou_mat = calculate_iou_matrix(gt, p_boxes)  # (Num_GT, Num_Pred)

        precisions = []

        for t in iou_thresholds:
            tp = 0
            fp = 0

            # Greedy matching
            gt_matched = np.zeros(len(gt), dtype=bool)
            pred_matched = np.zeros(len(p_boxes), dtype=bool)

            # Iterate through predictions (already sorted by conf)
            for j in range(len(p_boxes)):
                # Find best matching GT that is not yet matched
                best_iou = -1
                best_gt_idx = -1

                for i in range(len(gt)):
                    if not gt_matched[i]:
                        iou = iou_mat[i, j]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                if best_gt_idx != -1 and best_iou > t:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                    pred_matched[j] = True
                else:
                    fp += 1

            fn = len(gt) - np.sum(gt_matched)

            denom = tp + fp + fn
            if denom == 0:
                precisions.append(1.0)
            else:
                precisions.append(tp / denom)

        # Average precision for this image over thresholds
        aps_per_image.append(np.mean(precisions))

    return np.mean(aps_per_image)


def run_failure_analysis(predictions, gt_annotations, data_interface):
    """
    Correlates error magnitude with input features.
    Focuses on True Positives at IoU=0.5.
    """
    errors = []
    features = []  # Distance from ego

    sample_tokens = list(gt_annotations.keys())

    for token in sample_tokens:
        gt = np.array(gt_annotations[token])
        preds = predictions.get(token, [])

        if len(gt) == 0 or len(preds) == 0 or len(preds["bboxes"]) == 0:
            continue

        p_boxes = preds["bboxes"]
        p_scores = preds["scores"]

        # Sort
        sort_idx = np.argsort(p_scores)[::-1]
        p_boxes = p_boxes[sort_idx]

        iou_mat = calculate_iou_matrix(gt, p_boxes)

        gt_matched = np.zeros(len(gt), dtype=bool)

        for j in range(len(p_boxes)):
            best_iou = -1
            best_gt_idx = -1
            for i in range(len(gt)):
                if not gt_matched[i]:
                    iou = iou_mat[i, j]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            # Check match at loose threshold for analysis
            if best_gt_idx != -1 and best_iou > 0.5:
                gt_matched[best_gt_idx] = True

                # Calculate Error: L2 distance between centers
                # Boxes are in World Frame
                gt_box = gt[best_gt_idx]
                pred_box = p_boxes[j]

                dist_error = np.linalg.norm(gt_box[:3] - pred_box[:3])

                # Feature: Distance of GT from Ego
                # Need to transform GT back to sensor frame to get range, or just use World coords if ego is origin?
                # Ego moves, so we need relative distance.
                # We can approximate using the sensor frame predictions since they matched.
                # Or use data_interface to get ego pose.
                # Faster: Use the predicted box's distance from sensor origin (since we transformed pred from sensor to world, we know the sensor-relative coords roughly or can re-compute).
                # Actually, let's just use the sensor-frame prediction logic.
                # But here we have world frame boxes.
                # Let's inverse transform the GT box to sensor frame for accurate range.
                try:
                    world_to_sensor = data_interface.get_transform_matrix(token)
                    gt_center_hom = np.append(gt_box[:3], 1)
                    gt_center_sensor = (world_to_sensor @ gt_center_hom)[:3]
                    range_val = np.linalg.norm(gt_center_sensor)

                    errors.append(dist_error)
                    features.append(range_val)
                except:
                    pass

    if len(errors) > 1:
        corr, _ = pearsonr(errors, features)
        print(f"Correlation between Distance from Ego and Center Error: {corr:.4f}")
    else:
        print("Not enough matched samples for failure analysis.")


def main():
    # 1. Setup
    config.set_seed(config.SEED)
    device = config.get_device()

    # Override Config for Fast Baseline
    # Limit epochs to ensure completion within 2 hours
    # Typically 1 epoch takes ~10-15 mins on this data size with ResNet18
    # Updated to use config value (15) for better convergence
    TRAIN_EPOCHS = config.NUM_EPOCHS

    # 2. Train Model
    print("Starting Training...")
    model = train_model(
        num_epochs=TRAIN_EPOCHS,
        batch_size=config.BATCH_SIZE,
        load_cached_data=True,
        num_workers=config.NUM_WORKERS,
    )

    # 3. Validation Inference
    print("Starting Validation...")
    data_interface = DataInterface(load_cached_data=True)
    val_dataset = BEVDataset(
        split="val", data_interface=data_interface, load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    model.eval()
    val_predictions = {}
    gt_annotations = {}

    # Load GT from metadata directly for speed
    val_df = pd.read_csv(config.VAL_METADATA)
    val_df["annotations"] = val_df["annotations"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    # Build GT Dict: sample_token -> list of [x, y, z, w, l, h, yaw]
    for _, row in val_df.iterrows():
        token = row["sample_token"]
        anns = row["annotations"]
        boxes = []
        for ann in anns:
            # Filter by class if needed, but metric evaluates all provided GT
            # We only trained on specific classes, so we should only evaluate on those?
            # The task implies general detection. We stick to config classes.
            if ann["class_name"] in config.CLASS_NAMES:
                boxes.append(
                    [
                        ann["center_x"],
                        ann["center_y"],
                        ann["center_z"],
                        ann["width"],
                        ann["length"],
                        ann["height"],
                        ann["yaw"],
                    ]
                )
        gt_annotations[token] = boxes

    # Inference Loop
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["input"].to(device)
            sample_tokens = batch["sample_token"]

            hm, reg = model(inputs)

            # Decode in Sensor Frame
            batch_preds_sensor = decode_predictions(hm, reg, threshold=0.1)

            # Transform to World Frame
            batch_preds_world = transform_predictions_to_world(
                batch_preds_sensor, sample_tokens, data_interface
            )

            for i, token in enumerate(sample_tokens):
                val_predictions[token] = batch_preds_world[i]

    # 4. Metric Calculation
    print("Calculating Metrics...")
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    final_metric = compute_map(val_predictions, gt_annotations, iou_thresholds)

    print(f"Final Validation Metric: {final_metric:.10f}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    run_failure_analysis(val_predictions, gt_annotations, data_interface)

    # 6. Submission Generation
    # Only generate if metric improved over baseline
    BASELINE_METRIC = 0.0876104272
    if final_metric > BASELINE_METRIC:
        print(
            f"Metric {final_metric:.10f} > Baseline {BASELINE_METRIC:.10f}. Generating Submission..."
        )
        best_model_path = os.path.join(config.CACHE_DIR, "best_model.pth")
        generate_submission(
            model_path=best_model_path,
            batch_size=config.BATCH_SIZE,
            load_cached_data=True,
            num_workers=config.NUM_WORKERS,
            threshold=0.1,
        )
    else:
        print(
            f"Metric {final_metric:.10f} did not improve over baseline {BASELINE_METRIC:.10f}. Skipping submission."
        )

    print("Process Complete.")


if __name__ == "__main__":
    main()
