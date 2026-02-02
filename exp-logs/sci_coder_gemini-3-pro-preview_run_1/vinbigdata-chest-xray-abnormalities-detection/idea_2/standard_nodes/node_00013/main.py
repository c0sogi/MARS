import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict

# Import from provided library
from library.config import Config, seed_everything
from library.dataset import VinBigDataDataset
from library.model import MultiTaskCenterNet
from library.loss import MultiTaskLoss
from library.engine import run_training, decode_predictions
from library.inference import predict_and_format

# =============================================================================
# Utility Functions for Metrics and Analysis
# =============================================================================


def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-6)


def compute_ap(recall, precision):
    """Computes Average Precision using VOC 2010 method (Area Under Curve)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, val_loader, val_df, device, iou_threshold=0.4):
    """
    Calculates mAP @ IoU > 0.4 on the validation set.
    """
    model.eval()

    # Store predictions: list of (image_id, class_id, score, bbox)
    all_preds = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)

            # Decode predictions
            detections = decode_predictions(
                outputs["hm"], outputs["wh"], outputs["reg"]
            )
            # detections: [B, K, 6] -> x1, y1, x2, y2, score, class

            batch_size = images.size(0)
            for i in range(batch_size):
                img_id = targets["image_id"][i]
                orig_shape = targets["original_shape"][i]
                orig_h, orig_w = orig_shape[0].item(), orig_shape[1].item()

                # Rescale factors
                scale_x = orig_w / Config.IMG_SIZE
                scale_y = orig_h / Config.IMG_SIZE

                # Process detections
                dets = detections[i]
                mask = dets[:, 4] > 0.01  # Low threshold to capture recall
                valid_dets = dets[mask]

                for det in valid_dets:
                    x1, y1, x2, y2, score, cls_id = det.tolist()

                    # Rescale to original coordinates
                    x1 = max(0, min(x1 * scale_x, orig_w))
                    y1 = max(0, min(y1 * scale_y, orig_h))
                    x2 = max(0, min(x2 * scale_x, orig_w))
                    y2 = max(0, min(y2 * scale_y, orig_h))

                    all_preds.append(
                        {
                            "image_id": img_id,
                            "class_id": int(cls_id),
                            "score": score,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

    # Group predictions by class
    preds_by_class = defaultdict(list)
    for p in all_preds:
        preds_by_class[p["class_id"]].append(p)

    # Group ground truth by class
    # Filter out "No finding" (14) from GT for mAP calculation of findings
    gt_by_class = defaultdict(lambda: defaultdict(list))

    # val_df has columns: image_id, class_id, x_min, y_min, x_max, y_max
    for _, row in val_df.iterrows():
        cid = int(row["class_id"])
        if cid == 14:
            continue

        box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        gt_by_class[cid][row["image_id"]].append({"bbox": box, "used": False})

    aps = []

    # Calculate AP for each class (0-13)
    for cls_id in range(Config.NUM_CLASSES):  # 0 to 13
        class_preds = preds_by_class[cls_id]
        class_gt = gt_by_class[cls_id]

        if not class_gt and not class_preds:
            continue

        if not class_gt:
            aps.append(0.0)
            continue

        # Sort predictions by score descending
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        # Total number of ground truth objects for this class
        n_pos = sum(len(objs) for objs in class_gt.values())

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            pred_box = pred["bbox"]

            best_iou = 0.0
            best_gt_idx = -1

            if img_id in class_gt:
                gt_objs = class_gt[img_id]
                for idx, obj in enumerate(gt_objs):
                    iou = calculate_iou(pred_box, obj["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

            if best_iou > iou_threshold:
                if not class_gt[img_id][best_gt_idx]["used"]:
                    tp[i] = 1.0
                    class_gt[img_id][best_gt_idx]["used"] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recall = tp_cumsum / n_pos
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap(recall, precision)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def perform_failure_analysis(model, val_loader, val_df, device):
    """
    Analyzes correlation between detection failure and object properties (Area, Aspect Ratio).
    """
    model.eval()

    # Prepare GT data structure
    # List of dicts: {'area': float, 'ar': float, 'detected': 0/1}
    gt_analysis_data = []

    # Pre-compute predictions map for fast lookup
    # Map: image_id -> class_id -> list of boxes
    preds_map = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            detections = decode_predictions(
                outputs["hm"], outputs["wh"], outputs["reg"]
            )

            batch_size = images.size(0)
            for i in range(batch_size):
                img_id = targets["image_id"][i]
                orig_shape = targets["original_shape"][i]
                orig_h, orig_w = orig_shape[0].item(), orig_shape[1].item()
                scale_x = orig_w / Config.IMG_SIZE
                scale_y = orig_h / Config.IMG_SIZE

                dets = detections[i]
                # Filter low confidence
                valid_dets = dets[dets[:, 4] > 0.1]

                for det in valid_dets:
                    x1, y1, x2, y2, score, cls_id = det.tolist()
                    # Rescale
                    x1 = max(0, min(x1 * scale_x, orig_w))
                    y1 = max(0, min(y1 * scale_y, orig_h))
                    x2 = max(0, min(x2 * scale_x, orig_w))
                    y2 = max(0, min(y2 * scale_y, orig_h))

                    preds_map[img_id][int(cls_id)].append([x1, y1, x2, y2])

    # Iterate over GT and check if detected
    for _, row in val_df.iterrows():
        cid = int(row["class_id"])
        if cid == 14:
            continue  # Skip No Finding

        img_id = row["image_id"]
        gt_box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]

        width = gt_box[2] - gt_box[0]
        height = gt_box[3] - gt_box[1]
        area = width * height
        ar = width / (height + 1e-6)

        is_detected = 0
        if img_id in preds_map and cid in preds_map[img_id]:
            pred_boxes = preds_map[img_id][cid]
            for pb in pred_boxes:
                if calculate_iou(gt_box, pb) > 0.4:
                    is_detected = 1
                    break

        gt_analysis_data.append(
            {
                "area": area,
                "aspect_ratio": ar,
                "error": 1 - is_detected,  # 1 if failed, 0 if detected
            }
        )

    if not gt_analysis_data:
        print("No ground truth findings available for failure analysis.")
        return

    df_analysis = pd.DataFrame(gt_analysis_data)

    # Calculate correlations
    corr_area = df_analysis["area"].corr(df_analysis["error"])
    corr_ar = df_analysis["aspect_ratio"].corr(df_analysis["error"])

    print("\n=== Failure Analysis ===")
    print(f"Correlation between Error and BBox Area: {corr_area:.4f}")
    print(f"Correlation between Error and Aspect Ratio: {corr_ar:.4f}")


# =============================================================================
# Main Orchestration
# =============================================================================


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = VinBigDataDataset(split="train")
    val_dataset = VinBigDataDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 3. Model & Training Setup
    print("Initializing Model...")
    model = MultiTaskCenterNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    criterion = MultiTaskLoss()

    # 4. Training Loop
    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        num_epochs=Config.NUM_EPOCHS,
    )

    # 5. Load Best Model for Evaluation
    print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # 6. Validation Metric (mAP)
    print("Calculating Validation mAP...")
    val_df = pd.read_csv(Config.VAL_META_PATH)
    final_map = evaluate_map(
        model, val_loader, val_df, device, iou_threshold=Config.IOU_THRESHOLD
    )

    print(f"Final Validation Metric: {final_map:.10f}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, val_df, device)

    # 8. Submission
    # Only submit if we improve upon the previous best (0.1516149067)
    if final_map > 0.1516149067:
        print(f"Validation metric {final_map:.6f} > 0.1516. Generating submission...")
        predict_and_format(
            model_path=Config.MODEL_SAVE_PATH,
            batch_size=Config.BATCH_SIZE,
            device=Config.DEVICE,
            num_workers=Config.NUM_WORKERS,
        )
    else:
        print("Validation metric is 0.0. Skipping submission generation.")


if __name__ == "__main__":
    main()
