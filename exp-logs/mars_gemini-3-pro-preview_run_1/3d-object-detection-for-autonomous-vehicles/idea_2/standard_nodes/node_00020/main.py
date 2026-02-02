import sys
import os
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import pearsonr

# Add current directory to path
sys.path.append(os.getcwd())

from library.config import Config
from library.train import Trainer
from library.inference import Predictor
from library.utils import calc_iou_3d, get_logger, set_seed
from library.dataset import BEVDataset
from library.model import DLASeg

# Initialize Logger
logger = get_logger()


def calculate_iou_metric(pred_boxes, gt_boxes):
    """
    Calculates the average precision for a single image across IoU thresholds 0.5-0.95.

    Args:
        pred_boxes: List of lists [x, y, z, w, l, h, yaw]
        gt_boxes: List of lists [x, y, z, w, l, h, yaw]

    Returns:
        avg_precision: Float
        matches: List of tuples (pred_idx, gt_idx, iou) for TP at lowest threshold (0.5) for analysis
    """
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    precisions = []

    # Pre-calculate IoU matrix to save time
    # Matrix shape: (num_preds, num_gts)
    num_preds = len(pred_boxes)
    num_gts = len(gt_boxes)

    if num_gts == 0:
        if num_preds == 0:
            return 1.0, []  # Empty image, no predictions -> Perfect
        else:
            return 0.0, []  # Empty image, but predictions -> False Positives -> 0 score

    if num_preds == 0:
        return 0.0, []  # GTs exist, but no predictions -> 0 score

    iou_matrix = np.zeros((num_preds, num_gts))
    for i, p in enumerate(pred_boxes):
        for j, g in enumerate(gt_boxes):
            iou_matrix[i, j] = calc_iou_3d(p, g)

    # Calculate Precision at each threshold
    for t in thresholds:
        tp = 0
        fp = 0
        matched_gt_indices = set()

        # Greedy matching based on confidence (preds are already sorted by confidence)
        for i in range(num_preds):
            best_iou = -1
            best_gt_idx = -1

            # Find best matching GT that hasn't been matched yet
            for j in range(num_gts):
                if j in matched_gt_indices:
                    continue

                iou = iou_matrix[i, j]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou > t:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = num_gts - len(matched_gt_indices)

        # Precision = TP / (TP + FP + FN)
        denom = tp + fp + fn
        if denom > 0:
            precisions.append(tp / denom)
        else:
            precisions.append(0.0)

    # Collect matches at threshold 0.5 for failure analysis
    matches_05 = []
    matched_gt_indices_05 = set()
    for i in range(num_preds):
        best_iou = -1
        best_gt_idx = -1
        for j in range(num_gts):
            if j in matched_gt_indices_05:
                continue
            if iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_gt_idx = j

        if best_iou > 0.5:
            matches_05.append((i, best_gt_idx, best_iou))
            matched_gt_indices_05.add(best_gt_idx)

    return np.mean(precisions), matches_05


def evaluate_and_analyze(model, val_loader, device):
    """
    Runs inference on validation set, calculates metric, and performs failure analysis.
    """
    model.eval()

    total_score = 0.0
    num_samples = 0

    # For Failure Analysis
    errors_distance = []
    gt_distances = []
    gt_volumes = []

    # Predictor helper for decoding
    # We use a temporary instance to access decode_predictions
    predictor = Predictor(checkpoint_path=None)
    predictor.model = model  # Swap model

    logger.info("Starting Validation and Analysis...")

    with torch.no_grad():
        for batch in tqdm(val_loader, disable=True):  # Disable tqdm for clean output
            inputs = batch["input"].to(device)

            # Forward
            outputs = model(inputs)

            # Decode
            sensor_to_global = batch["sensor_to_global"].to(device)
            yaw_bias = batch["yaw_bias"].to(device)

            batch_detections = predictor.decode_predictions(
                outputs,
                sensor_to_global,
                yaw_bias,
                K=Config.MAX_DETECTIONS,
                score_thresh=Config.SCORE_THRESHOLD,
            )

            # Process each sample in batch
            for i in range(len(batch_detections)):
                # Get Predictions
                preds = batch_detections[i]  # List of dicts
                # Sort by score descending
                preds.sort(key=lambda x: x["score"], reverse=True)

                pred_boxes = [
                    [
                        d["center_x"],
                        d["center_y"],
                        d["center_z"],
                        d["width"],
                        d["length"],
                        d["height"],
                        d["yaw"],
                    ]
                    for d in preds
                ]

                # Get Ground Truth
                # Reconstruct GT from batch targets is hard because they are heatmaps.
                # We should use the raw annotations from the dataset.
                # Since val_loader is shuffled=False (usually), we can rely on order if careful.
                # But safer to use the 'ind' and 'reg' to reconstruct or just load from metadata.
                # However, the batch doesn't contain raw annotations easily.
                # Strategy: The dataset __getitem__ returns targets.
                # To get raw GT, we should look up by sample_token if available,
                # but BEVDataset train/val mode doesn't return sample_token by default.
                # Let's assume we can reconstruct from the targets provided in batch? No, too lossy.

                # ALTERNATIVE: Modify BEVDataset to return raw annotations or sample_token?
                # I cannot modify library files.
                # But I can access the dataset object via val_loader.dataset.samples
                # The loader is sequential (shuffle=False).

                sample_idx = num_samples + i
                if sample_idx >= len(val_loader.dataset):
                    break

                sample_info = val_loader.dataset.samples[sample_idx]
                anns = sample_info["annotations"]

                gt_boxes = [
                    [
                        a["center_x"],
                        a["center_y"],
                        a["center_z"],
                        a["width"],
                        a["length"],
                        a["height"],
                        a["yaw"],
                    ]
                    for a in anns
                ]

                # Calculate Metric
                score, matches = calculate_iou_metric(pred_boxes, gt_boxes)
                total_score += score

                # Failure Analysis Data Collection
                for pred_idx, gt_idx, iou in matches:
                    p = pred_boxes[pred_idx]
                    g = gt_boxes[gt_idx]

                    # Error: Euclidean distance between centers
                    dist_error = np.sqrt(
                        (p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2 + (p[2] - g[2]) ** 2
                    )

                    # Feature: GT Distance from origin
                    gt_dist = np.sqrt(g[0] ** 2 + g[1] ** 2 + g[2] ** 2)

                    # Feature: GT Volume
                    gt_vol = g[3] * g[4] * g[5]

                    errors_distance.append(dist_error)
                    gt_distances.append(gt_dist)
                    gt_volumes.append(gt_vol)

            num_samples += len(batch_detections)

    final_metric = total_score / num_samples if num_samples > 0 else 0.0

    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis Calculation
    if len(errors_distance) > 1:
        corr_dist, _ = pearsonr(errors_distance, gt_distances)
        corr_vol, _ = pearsonr(errors_distance, gt_volumes)

        print("Failure Analysis:")
        print(f"Correlation (Error vs Distance): {corr_dist:.4f}")
        print(f"Correlation (Error vs Volume): {corr_vol:.4f}")
    else:
        print("Failure Analysis: Not enough matches to calculate correlation.")

    return final_metric


def main():
    # 1. Configuration
    # Set cache dir to current working dir to generate new cache with correct dimensions
    Config.CACHE_DIR = "./working/idea_3"
    # Ensure working dir is correct for saving model
    Config.WORKING_DIR = "./working/idea_3"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "dla34_best_model.pth")

    # Fast training settings
    TRAIN_SAMPLE_SIZE = 5000  # Subset of 14k
    Config.NUM_EPOCHS = 25

    set_seed(Config.SEED)

    # 2. Training
    logger.info("Initializing Trainer...")
    # We pass sample_size to limit data loading
    trainer = Trainer(sample_size=TRAIN_SAMPLE_SIZE)

    logger.info("Starting Training...")
    trainer.train()

    # 3. Validation & Analysis
    logger.info("Starting Evaluation...")
    # Load best model
    model = DLASeg().to(trainer.device)
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=trainer.device)
    )

    # Use the validation loader from trainer (it has full val set if we didn't limit it too much)
    # Note: Trainer init limited val set by sample_size too.
    # Ideally we want full validation. Let's reload full val dataset for accurate metric.
    val_dataset = BEVDataset(split="val", load_cached_data=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    metric = evaluate_and_analyze(model, val_loader, trainer.device)

    # 4. Submission
    THRESHOLD = 0.0876104272
    if metric > THRESHOLD:
        logger.info(f"Metric {metric} > {THRESHOLD}. Generating submission...")
        predictor = Predictor(checkpoint_path=Config.MODEL_SAVE_PATH)
        predictor.run_inference()
    else:
        logger.info(f"Metric {metric} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
