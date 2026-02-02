import os
import sys
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import get_model
from library.engine import train_model, predict_and_submit
from library.utils import get_device, set_seed


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Box format: [xmin, ymin, xmax, ymax]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_ap_voc2010(recalls, precisions):
    """
    Compute Average Precision using PASCAL VOC 2010 method (all points interpolation).
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, data_loader, device, iou_threshold=0.4):
    """
    Evaluates mAP @ IoU > 0.4 on the validation set.
    Also returns data for failure analysis.
    """
    model.eval()

    # Store all predictions and targets
    # Structure: class_id -> list of {score, box, image_id}
    preds_by_class = defaultdict(list)
    # Structure: class_id -> dict of {image_id -> list of {box, used}}
    gt_by_class = defaultdict(lambda: defaultdict(list))

    # For failure analysis
    image_stats = defaultdict(
        lambda: {"fp": 0, "fn": 0, "tp": 0, "gt_count": 0, "gt_area_sum": 0}
    )

    print("Running validation inference for mAP...")

    with torch.no_grad():
        for images, targets, image_ids in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = image_ids[i]

                # Process Targets (Ground Truth)
                gt_boxes = targets[i]["boxes"].cpu().numpy()
                gt_labels = targets[i]["labels"].cpu().numpy()

                # Update image stats for failure analysis
                image_stats[img_id]["gt_count"] = len(gt_boxes)
                if len(gt_boxes) > 0:
                    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
                        gt_boxes[:, 3] - gt_boxes[:, 1]
                    )
                    image_stats[img_id]["gt_area_sum"] = np.sum(areas)

                for box, label in zip(gt_boxes, gt_labels):
                    # Model class 1-14 corresponds to Dataset class 0-13
                    # We evaluate on dataset classes 0-13.
                    # Label 1 (Model) -> 0 (Dataset)
                    class_id = int(label) - 1
                    if 0 <= class_id <= 13:
                        gt_by_class[class_id][img_id].append(
                            {"box": box, "used": False}
                        )

                # Process Predictions
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    class_id = int(label) - 1
                    if 0 <= class_id <= 13:
                        preds_by_class[class_id].append(
                            {"score": score, "box": box, "image_id": img_id}
                        )

    # Calculate AP for each class
    aps = []
    print("Calculating AP per class...")

    for class_id in range(14):  # 0 to 13
        # Get ground truths for this class
        class_gts = gt_by_class[class_id]
        n_pos = sum(len(v) for v in class_gts.values())

        # Get predictions for this class
        class_preds = preds_by_class[class_id]
        # Sort by confidence descending
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            pred_box = pred["box"]

            best_iou = 0.0
            best_gt_idx = -1

            # Find best matching GT
            if img_id in class_gts:
                gts = class_gts[img_id]
                for idx, gt in enumerate(gts):
                    iou = calculate_iou(pred_box, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou > iou_threshold:
                if not class_gts[img_id][best_gt_idx]["used"]:
                    tp[i] = 1.0
                    class_gts[img_id][best_gt_idx]["used"] = True
                    image_stats[img_id]["tp"] += 1
                else:
                    fp[i] = 1.0  # Duplicate detection
                    image_stats[img_id]["fp"] += 1
            else:
                fp[i] = 1.0
                image_stats[img_id]["fp"] += 1

        # Calculate FN for failure analysis (Total GT - TP for this class)
        # Note: This is approximate per image aggregation later

        # Compute Precision/Recall
        acc_fp = np.cumsum(fp)
        acc_tp = np.cumsum(tp)
        rec = acc_tp / n_pos if n_pos > 0 else np.zeros_like(acc_tp)
        prec = acc_tp / (acc_tp + acc_fp) if len(acc_tp) > 0 else np.zeros_like(acc_tp)

        ap = compute_ap_voc2010(rec, prec)
        aps.append(ap)
        # print(f"Class {class_id} AP: {ap:.4f}")

    # Calculate FN for image stats
    for img_id, stats in image_stats.items():
        stats["fn"] = stats["gt_count"] - stats["tp"]

    mean_ap = np.mean(aps) if aps else 0.0
    return mean_ap, image_stats


def perform_failure_analysis(image_stats):
    """
    Analyzes correlation between error magnitude and image features.
    """
    print("\n=== Failure Analysis ===")

    errors = []
    num_annotations = []
    mean_areas = []

    for img_id, stats in image_stats.items():
        # Error Magnitude: Total mistakes (False Positives + False Negatives)
        error_mag = stats["fp"] + stats["fn"]

        errors.append(error_mag)
        num_annotations.append(stats["gt_count"])

        avg_area = (
            stats["gt_area_sum"] / stats["gt_count"] if stats["gt_count"] > 0 else 0
        )
        mean_areas.append(avg_area)

    errors = np.array(errors)
    num_annotations = np.array(num_annotations)
    mean_areas = np.array(mean_areas)

    # Correlation: Error vs Num Annotations
    if len(errors) > 1:
        corr_num, _ = pearsonr(errors, num_annotations)
        print(f"Correlation (Error vs Num Annotations): {corr_num:.4f}")

        corr_area, _ = pearsonr(errors, mean_areas)
        print(f"Correlation (Error vs Mean Box Area): {corr_area:.4f}")
    else:
        print("Not enough data for correlation analysis.")


def main():
    # --- 1. Configuration ---
    print(f"Starting Run (Epochs: {Config.EPOCHS})")

    # --- 2. Training ---
    # Train the model and get the path to the best checkpoint
    best_model_path = train_model(debug=False)
    print(f"Training complete. Best model saved at: {best_model_path}")

    # --- 3. Validation & Metric Calculation ---
    device = get_device()

    # Load the best model
    model = get_model(pretrained=False)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Error: Best model path not found!")
        sys.exit(1)

    model.to(device)

    # Get Validation DataLoader
    # We use the standard getter but only need the validation part
    _, val_loader = get_dataloaders(debug=False)

    # Calculate mAP
    mean_ap, image_stats = evaluate_map(model, val_loader, device, iou_threshold=0.4)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {mean_ap}")

    # --- 4. Failure Analysis ---
    perform_failure_analysis(image_stats)

    # --- 5. Submission ---
    if mean_ap > 0.0:
        print("\nGenerating submission...")
        predict_and_submit(model_path=best_model_path)
    else:
        print("\nValidation mAP is 0.0. Skipping submission to prevent scoring errors.")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
