import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

# Import provided library modules
from library.config import Config
from library.dataset import VinBigDataset
from library.model import SpatiallyAwareCenterNet
from library.inference import predict, post_process, decode_predictions
from library import train
from library import inference
from library.utils import seed_everything, get_logger

# Setup Logger
logger = get_logger("RunFile")


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes (x1, y1, x2, y2).
    box1: [N, 4]
    box2: [M, 4]
    Returns: [N, M]
    """
    # Expand dims to support broadcasting
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1[:, None], b2_x1)
    inter_y1 = np.maximum(b1_y1[:, None], b2_y1)
    inter_x2 = np.minimum(b1_x2[:, None], b2_x2)
    inter_y2 = np.minimum(b1_y2[:, None], b2_y2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area[:, None] + b2_area - inter_area

    # Avoid division by zero
    return inter_area / (union_area + 1e-6)


def calculate_map_voc2010(pred_df, gt_df, iou_thresh=0.4, num_classes=14):
    """
    Calculates mAP using PASCAL VOC 2010 method (Area under Precision-Recall curve).
    """
    average_precisions = []

    for c in range(num_classes):
        # Filter data for current class
        c_preds = (
            pred_df[pred_df["class_id"] == c]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
        c_gts = gt_df[gt_df["class_id"] == c].reset_index(drop=True)

        n_pos = len(c_gts)
        if n_pos == 0:
            continue

        tp = np.zeros(len(c_preds))
        fp = np.zeros(len(c_preds))

        # Keep track of detected GTs to avoid double counting
        gt_detected = np.zeros(len(c_gts))

        if len(c_preds) > 0:
            # Vectorized IoU calculation
            pred_boxes = c_preds[["x_min", "y_min", "x_max", "y_max"]].values
            gt_boxes = c_gts[["x_min", "y_min", "x_max", "y_max"]].values

            ious = calculate_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]

            for i in range(len(c_preds)):
                # Find best matching GT
                best_iou = 0
                best_gt_idx = -1

                if len(c_gts) > 0:
                    best_gt_idx = np.argmax(ious[i])
                    best_iou = ious[i, best_gt_idx]

                if best_iou > iou_thresh:
                    if gt_detected[best_gt_idx] == 0:
                        tp[i] = 1
                        gt_detected[best_gt_idx] = 1
                    else:
                        fp[i] = 1  # Duplicate detection
                else:
                    fp[i] = 1

        # Compute precision and recall
        acc_tp = np.cumsum(tp)
        acc_fp = np.cumsum(fp)
        rec = acc_tp / n_pos
        prec = acc_tp / (acc_tp + acc_fp + 1e-6)

        # VOC 2010 Average Precision (Area Under Curve)
        # Compute mean precision at each unique recall level
        # Append sentinel values
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return np.mean(average_precisions)


def perform_failure_analysis(pred_df, gt_df):
    """
    Correlates error (Missed Detection) with Bounding Box Area.
    """
    logger.info("Performing Failure Analysis...")

    # We analyze per Ground Truth object
    # For each GT, did we detect it?

    gt_data = []

    for c in range(Config.NUM_CLASSES):
        c_preds = pred_df[pred_df["class_id"] == c]
        c_gts = gt_df[gt_df["class_id"] == c]

        if len(c_gts) == 0:
            continue

        if len(c_preds) > 0:
            pred_boxes = c_preds[["x_min", "y_min", "x_max", "y_max"]].values
            gt_boxes = c_gts[["x_min", "y_min", "x_max", "y_max"]].values
            ious = calculate_iou(gt_boxes, pred_boxes)  # [N_gt, N_pred]

            # Max IoU for each GT
            max_ious = (
                np.max(ious, axis=1) if ious.shape[1] > 0 else np.zeros(len(c_gts))
            )
        else:
            max_ious = np.zeros(len(c_gts))

        for idx, (_, row) in enumerate(c_gts.iterrows()):
            area = (row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"])
            detected = 1 if max_ious[idx] > 0.4 else 0
            gt_data.append({"area": area, "detected": detected})

    if not gt_data:
        logger.warning("No GT data for analysis.")
        return

    df_analysis = pd.DataFrame(gt_data)

    # Correlation between Area and Detected status
    # We expect positive correlation (larger objects easier to detect)
    # Error Magnitude proxy = 1 - detected (0 for correct, 1 for error)
    df_analysis["error"] = 1 - df_analysis["detected"]

    correlation = df_analysis["error"].corr(df_analysis["area"])

    print(
        f"Correlation between Error (Missed Detection) and BBox Area: {correlation:.4f}"
    )


def main():
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    logger.info("Configuring for Fast Baseline execution...")
    Config.EPOCHS = 5

    # Create a subset of training data for speed
    df_train = pd.read_csv(Config.TRAIN_META)
    # Filter out "No finding" for training density or just sample
    # Sampling 5000 images is reasonable for A100 in < 2 hours
    subset_size = 5000
    if len(df_train) > subset_size:
        df_train_subset = df_train.sample(n=subset_size, random_state=Config.SEED)
        subset_path = os.path.join(Config.WORK_DIR, "train_subset.csv")
        df_train_subset.to_csv(subset_path, index=False)
        Config.TRAIN_META = subset_path
        logger.info(f"Subsampled training data to {subset_size} rows.")

    # -------------------------------------------------------------------------
    # 2. Training
    # -------------------------------------------------------------------------
    logger.info("Starting Training Phase...")
    train.run_training()

    # -------------------------------------------------------------------------
    # 3. Validation & Metric Calculation
    # -------------------------------------------------------------------------
    logger.info("Starting Validation Phase...")

    # Load Validation Data
    val_dataset = VinBigDataset(
        csv_path=Config.VAL_META, mode="val", load_cached_data=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = SpatiallyAwareCenterNet(pretrained=False)
    model.to(device)

    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Run Inference on Validation Set
    raw_results = predict(model, val_loader, device)

    # Process predictions for Metric Calculation
    # We need to scale boxes to original dimensions to match val_meta.csv
    pred_rows = []
    for res in raw_results:
        img_id = res["image_id"]
        detections = res["detections"]  # [x1, y1, x2, y2, score, cls] (640 scale)
        orig_h, orig_w = res["orig_h"], res["orig_w"]

        scale_x = orig_w / Config.IMG_SIZE
        scale_y = orig_h / Config.IMG_SIZE

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det

            # Filter low confidence
            if score < 0.01:
                continue

            # Rescale
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y

            pred_rows.append(
                {
                    "image_id": img_id,
                    "class_id": int(cls_id),
                    "score": float(score),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
            )

    df_pred = pd.DataFrame(pred_rows)
    df_gt = pd.read_csv(Config.VAL_META)

    # Filter GT: Exclude Class 14 (No finding) for object detection mAP
    df_gt = df_gt[df_gt["class_id"] != 14]

    # Calculate mAP
    if len(df_pred) == 0:
        final_map = 0.0
    else:
        final_map = calculate_map_voc2010(
            df_pred, df_gt, iou_thresh=0.4, num_classes=14
        )

    print(f"Final Validation Metric: {final_map}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    if len(df_pred) > 0 and len(df_gt) > 0:
        perform_failure_analysis(df_pred, df_gt)

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 0.1783551866

    if final_map > THRESHOLD:
        logger.info(
            f"Metric ({final_map}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        inference.run_inference()
    else:
        logger.info(
            f"Metric ({final_map}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
