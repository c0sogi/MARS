import sys
import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from library
from library.config import Config
from library.engine import Trainer
from library.submission import SubmissionGenerator
from library.dataset import LyftDataset, collate_fn
from library.model import PointPillars, AnchorGenerator
from library.utils import decode_boxes, nms_3d, iou3d_cpu, setup_logger

# Set seeds for reproducibility
Config.set_seed()


def calculate_metric_per_image(pred_boxes, pred_scores, gt_boxes, thresholds):
    """
    Calculates the custom mean AP metric for a single image.

    Args:
        pred_boxes: (N, 7) numpy array
        pred_scores: (N,) numpy array
        gt_boxes: (M, 7) numpy array
        thresholds: list of floats (IoU thresholds)

    Returns:
        float: The average precision score for this image.
    """
    # Sort predictions by confidence (descending)
    if len(pred_boxes) > 0:
        sort_idx = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]

    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    # Handle edge cases as per task description
    if n_gt == 0:
        if n_pred > 0:
            return 0.0  # False positives present -> 0 score
        else:
            return 1.0  # Perfect empty prediction -> 1 score (assumed perfect recall/precision)

    if n_pred == 0:
        return 0.0  # False negatives present -> 0 score

    # Calculate 3D IoU Matrix: (N_pred, N_gt)
    # Row i is prediction i, Col j is GT j
    iou_matrix = iou3d_cpu(pred_boxes, gt_boxes)

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0
        # Track which GTs have been matched for this threshold
        gt_matched = np.zeros(n_gt, dtype=bool)

        # Greedy matching in order of confidence
        for i in range(n_pred):
            best_iou = -1.0
            best_gt_idx = -1

            # Find the best matching unmatched GT
            for j in range(n_gt):
                if not gt_matched[j]:
                    iou = iou_matrix[i, j]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # Check if match meets threshold
            if best_iou > t:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1

        # Count False Negatives (unmatched GTs)
        fn = np.sum(~gt_matched)

        # Calculate Precision/Score for this threshold: TP / (TP + FP + FN)
        denom = tp + fp + fn
        if denom == 0:
            score = 0.0
        else:
            score = tp / denom

        precisions.append(score)

    # Average over all thresholds
    return np.mean(precisions)


def run_validation_and_analysis(model_path, device):
    print("Starting Validation and Failure Analysis...")

    # Load Model
    model = PointPillars().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    model.eval()

    # Load Validation Dataset
    val_ds = LyftDataset(Config.VAL_METADATA_PATH, mode="val", load_cached_data=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # Metric Thresholds: 0.5 to 0.95 step 0.05
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Initialize Anchor Generator
    anchor_generator = AnchorGenerator()
    anchors = anchor_generator.get_anchors().to(device)

    image_scores = []

    # Features for failure analysis
    feat_num_gt = []
    feat_mean_vol = []
    feat_mean_dist = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pillars = batch["pillars"].to(device)
            coords = batch["pillar_coords"].to(device)
            num_points = batch["num_points"].to(device)
            gt_boxes_list = batch["gt_boxes"]  # List of tensors

            # Forward Pass
            cls_preds, box_preds, dir_preds = model(pillars, coords, num_points)

            batch_size = cls_preds.shape[0]

            for b in range(batch_size):
                # Get Ground Truth for this sample
                gts = gt_boxes_list[b].numpy()  # (M, 7)

                # --- Feature Extraction for Failure Analysis ---
                n_gt = len(gts)
                if n_gt > 0:
                    # Volume = w * l * h
                    mean_vol = np.mean(gts[:, 3] * gts[:, 4] * gts[:, 5])
                    # Distance = sqrt(x^2 + y^2)
                    mean_dist = np.mean(np.sqrt(gts[:, 0] ** 2 + gts[:, 1] ** 2))
                else:
                    mean_vol = 0.0
                    mean_dist = 0.0

                feat_num_gt.append(n_gt)
                feat_mean_vol.append(mean_vol)
                feat_mean_dist.append(mean_dist)

                # --- Prediction Decoding ---
                scores = torch.sigmoid(cls_preds[b])
                max_scores, labels = scores.max(dim=1)

                # Filter by score threshold to reduce computation
                mask = max_scores > Config.SCORE_THRESHOLD

                if not mask.any():
                    # No predictions
                    score = calculate_metric_per_image(
                        np.empty((0, 7)), np.empty((0,)), gts, thresholds
                    )
                    image_scores.append(score)
                    continue

                valid_box_preds = box_preds[b][mask]
                valid_anchors = anchors[mask]
                valid_scores = max_scores[mask]

                decoded = decode_boxes(valid_box_preds, valid_anchors)

                # NMS
                boxes_np = decoded.cpu().numpy()
                scores_np = valid_scores.cpu().numpy()

                keep = nms_3d(
                    boxes_np,
                    scores_np,
                    threshold=Config.NMS_IOU_THRESHOLD,
                    max_detections=Config.MAX_DETECTIONS,
                )

                final_boxes = boxes_np[keep]
                final_scores = scores_np[keep]

                # --- Metric Calculation ---
                score = calculate_metric_per_image(
                    final_boxes, final_scores, gts, thresholds
                )
                image_scores.append(score)

    final_metric = np.mean(image_scores)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Error is 1.0 - Score (higher score is better)
    errors = 1.0 - np.array(image_scores)

    df = pd.DataFrame(
        {
            "error": errors,
            "num_gt": feat_num_gt,
            "mean_vol": feat_mean_vol,
            "mean_dist": feat_mean_dist,
        }
    )

    correlations = df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Input Features:")
    print(correlations)

    return final_metric


def main():
    # 1. Configure for Fast Baseline
    # We modify the Config class directly to tune for a 2-hour runtime limit
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.MAX_PILLARS = 12000

    # 2. Train
    print("Initializing Trainer...")
    trainer = Trainer(load_cached_data=True)

    # Explicitly pass epochs to ensure modification is respected
    trainer.fit(epochs=Config.EPOCHS)

    # 3. Validation & Analysis
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = run_validation_and_analysis(Config.MODEL_SAVE_PATH, device)

    # 4. Submission
    if metric > 0.0024:
        print("\nMetric exceeds threshold. Generating submission...")
        generator = SubmissionGenerator(model_path=Config.MODEL_SAVE_PATH)
        generator.generate()
    else:
        print(f"\nMetric {metric} is below threshold 0.0024. Skipping submission.")


if __name__ == "__main__":
    main()
