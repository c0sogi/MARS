import os
import sys
import torch
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from tqdm import tqdm

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import AnatomicalCenterNet
from library.loss import CenterNetLoss
from library.train import train_one_epoch, validate
from library.inference import predict_and_format, decode_predictions
from library.utils import get_original_dimensions, rescale_boxes

# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
Config.NUM_EPOCHS = 8  # Reduced epochs for speed
Config.TRAIN_SUBSET_SIZE = 15000  # Subsample training data
Config.BATCH_SIZE = 32


# =============================================================================
# Metric Calculation Utilities (mAP @ IoU > 0.4)
# =============================================================================
def calculate_iou(box1, box2):
    """
    Calculate IoU between two sets of boxes.
    box1: (N, 4) [x1, y1, x2, y2]
    box2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoUs
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def calculate_ap(rec, prec):
    """
    Compute AP given precision and recall.
    Uses PASCAL VOC 2010+ method (all-point interpolation).
    """
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, loader, val_df, device, iou_thresh=0.4):
    """
    Evaluates mAP on the validation set.
    """
    model.eval()

    # 1. Get Original Dimensions for rescaling
    orig_dims_map = get_original_dimensions(val_df, load_cached_data=True)

    # 2. Collect Predictions and Ground Truths
    # Structure: class_id -> list of {image_id, box, score}
    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(lambda: defaultdict(list))

    # Pre-process GT
    # Filter out "No finding" (class 14) for detection mAP
    gt_df = val_df[val_df["class_id"] != 14]
    for _, row in gt_df.iterrows():
        img_id = str(row["image_id"])
        cls_id = int(row["class_id"])
        box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        gts_by_class[cls_id][img_id].append(box)

    # Run Inference
    with torch.no_grad():
        for images, _, image_ids in tqdm(loader, desc="Evaluating mAP", disable=True):
            images = images.to(device)
            outputs = model(images)

            # Decode
            hm = outputs["hm"]
            wh = outputs["wh"]
            reg = outputs["reg"]

            # Use lower threshold to get more candidates for PR curve
            bboxes, scores, clses = decode_predictions(
                hm, wh, reg, K=100, output_stride=4
            )

            # Move to CPU
            bboxes = bboxes.cpu().numpy()
            scores = scores.cpu().numpy()
            clses = clses.cpu().numpy()

            for i in range(len(image_ids)):
                img_id = str(image_ids[i])

                # Rescale to original dimensions
                orig_w, orig_h = orig_dims_map.get(img_id, (1024, 1024))

                # Filter valid detections
                valid_mask = scores[i] > 0.01
                img_boxes = bboxes[i][valid_mask]
                img_scores = scores[i][valid_mask]
                img_clses = clses[i][valid_mask]

                if len(img_boxes) > 0:
                    scaled_boxes = rescale_boxes(
                        img_boxes,
                        current_shape=Config.IMAGE_SIZE,
                        original_shape=(orig_h, orig_w),
                    )

                    for box, score, cls in zip(scaled_boxes, img_scores, img_clses):
                        cls = int(cls)
                        if cls != 14:  # Ignore No Finding predictions for mAP
                            preds_by_class[cls].append(
                                {"image_id": img_id, "bbox": box, "score": score}
                            )

    # 3. Calculate AP per class
    aps = []
    class_aps = {}

    # Evaluate classes 0-13
    for cls_id in range(14):
        cls_preds = preds_by_class[cls_id]
        # Sort by score descending
        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        # Get all GTs for this class
        cls_gts = gts_by_class[cls_id]  # dict: img_id -> list of boxes

        # Count total positives
        n_pos = sum(len(boxes) for boxes in cls_gts.values())

        if n_pos == 0:
            continue

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        # Track detected GTs to avoid double counting
        gt_detected = {
            img_id: [False] * len(boxes) for img_id, boxes in cls_gts.items()
        }

        for idx, pred in enumerate(cls_preds):
            img_id = pred["image_id"]
            pred_box = np.array([pred["bbox"]])

            if img_id in cls_gts:
                gt_boxes = np.array(cls_gts[img_id])
                ious = calculate_iou(pred_box, gt_boxes)[0]  # (M_gt,)

                max_iou = -1
                max_idx = -1

                if len(ious) > 0:
                    max_iou = np.max(ious)
                    max_idx = np.argmax(ious)

                if max_iou > iou_thresh:
                    if not gt_detected[img_id][max_idx]:
                        tp[idx] = 1.0
                        gt_detected[img_id][max_idx] = True
                    else:
                        fp[idx] = 1.0
                else:
                    fp[idx] = 1.0
            else:
                fp[idx] = 1.0

        # Compute Precision/Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        rec = tp_cumsum / n_pos
        prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = calculate_ap(rec, prec)
        aps.append(ap)
        class_aps[cls_id] = ap

    mAP = np.mean(aps) if aps else 0.0
    return mAP, class_aps


# =============================================================================
# Main Execution
# =============================================================================
def main():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on {device}")

    # 1. Load Data
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    # Subsample training data for fast baseline
    if len(train_df) > Config.TRAIN_SUBSET_SIZE:
        print(f"Subsampling training data to {Config.TRAIN_SUBSET_SIZE} samples...")
        # Sample by image_id to keep objects together
        unique_img_ids = train_df["image_id"].unique()
        selected_ids = np.random.choice(
            unique_img_ids, size=min(len(unique_img_ids), 10000), replace=False
        )  # Roughly map to subset size
        train_df = train_df[train_df["image_id"].isin(selected_ids)]
        # Trim to exact row count if needed, but keeping image integrity is better

    # Dataloaders
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, test_df=None)

    # 2. Setup Model
    print("Initializing model...")
    model = AnatomicalCenterNet().to(device)
    criterion = CenterNetLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate (Loss)
        val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}: Train Loss {train_metrics['loss']:.4f}, Val Loss {val_metrics['loss']:.4f}"
        )

        # Save Best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), best_model_path)

    # 4. Final Evaluation (mAP)
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Calculating Validation mAP...")
    mAP, class_aps = evaluate_map(model, val_loader, val_df, device, iou_thresh=0.4)

    print(f"Final Validation Metric: {mAP}")

    # 5. Failure Analysis
    print("Performing Failure Analysis...")
    # Correlate Class AP with Class Frequency and Mean Box Area
    class_stats = []

    # Calculate stats from val_df
    val_findings = val_df[val_df["class_id"] != 14]

    for cls_id in range(14):
        cls_data = val_findings[val_findings["class_id"] == cls_id]
        count = len(cls_data)
        if count > 0:
            area = (
                (cls_data["x_max"] - cls_data["x_min"])
                * (cls_data["y_max"] - cls_data["y_min"])
            ).mean()
        else:
            area = 0

        ap = class_aps.get(cls_id, 0.0)
        class_stats.append({"class_id": cls_id, "count": count, "area": area, "ap": ap})

    stats_df = pd.DataFrame(class_stats)

    if len(stats_df) > 1:
        corr_count = stats_df["ap"].corr(stats_df["count"])
        corr_area = stats_df["ap"].corr(stats_df["area"])

        print(f"Correlation (AP vs Frequency): {corr_count:.4f}")
        print(f"Correlation (AP vs BBox Area): {corr_area:.4f}")
        print(
            "Analysis: "
            + (
                "Larger objects detected better."
                if corr_area > 0
                else "Smaller objects detected better."
            )
        )
    else:
        print("Insufficient class diversity for correlation analysis.")

    # 6. Submission
    THRESHOLD = 0.1783551866
    if mAP > THRESHOLD:
        print(
            f"Validation metric {mAP} exceeds threshold {THRESHOLD}. Generating submission..."
        )
        predict_and_format(checkpoint_path=best_model_path)
    else:
        print(
            f"Validation metric {mAP} does not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
