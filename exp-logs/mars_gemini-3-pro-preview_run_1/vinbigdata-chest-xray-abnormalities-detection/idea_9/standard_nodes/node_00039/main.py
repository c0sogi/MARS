import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import defaultdict

# Import from provided libraries
from library.config import (
    DEVICE,
    NUM_CLASSES,
    VAL_META_PATH,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    SUBMISSION_FILE,
    SEED,
)
from library.utils import seed_everything
from library.data import create_dataloaders
from library.model import EfficientDetDecoupled
from library.loss import ThoracicLoss
from library.engine import fit
from library.inference import predict_and_format

# =============================================================================
# METRIC CALCULATION UTILS (PASCAL VOC mAP)
# =============================================================================


def calculate_iou(box1, box2):
    """Calculates IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def compute_ap_voc2010(rec, prec):
    """Computes AP using VOC 2010 interpolation."""
    rec = np.concatenate(([0.0], rec, [1.0]))
    prec = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope
    for i in range(prec.size - 1, 0, -1):
        prec[i - 1] = np.maximum(prec[i - 1], prec[i])

    # Integrate area under curve
    i = np.where(rec[1:] != rec[:-1])[0]
    ap = np.sum((rec[i + 1] - rec[i]) * prec[i + 1])
    return ap


def evaluate_map(pred_df, gt_df, iou_thresh=0.4):
    """
    Calculates mAP @ IoU > 0.4.
    pred_df: DataFrame with [image_id, class_id, score, x_min, y_min, x_max, y_max]
    gt_df: DataFrame with [image_id, class_id, x_min, y_min, x_max, y_max]
    """
    aps = []

    # Process each class
    # Classes 0-13 are findings. Class 14 is "No finding" (ignored for mAP usually, or handled as empty)
    # The task metric implies detection of findings.

    valid_classes = gt_df["class_id"].unique()
    valid_classes = [c for c in valid_classes if c != 14]

    for cls_id in valid_classes:
        cls_preds = pred_df[pred_df["class_id"] == cls_id].sort_values(
            "score", ascending=False
        )
        cls_gts = gt_df[gt_df["class_id"] == cls_id]

        # Group GT by image
        gt_by_image = defaultdict(list)
        for _, row in cls_gts.iterrows():
            gt_by_image[row["image_id"]].append(
                {
                    "bbox": [row["x_min"], row["y_min"], row["x_max"], row["y_max"]],
                    "used": False,
                }
            )

        n_pos = len(cls_gts)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, row in enumerate(cls_preds.itertuples()):
            img_id = row.image_id
            bb = [row.x_min, row.y_min, row.x_max, row.y_max]

            max_iou = -1
            max_idx = -1

            if img_id in gt_by_image:
                gt_boxes = gt_by_image[img_id]
                for j, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(bb, gt_box["bbox"])
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j

            if max_iou > iou_thresh:
                if not gt_by_image[img_id][max_idx]["used"]:
                    tp[i] = 1.0
                    gt_by_image[img_id][max_idx]["used"] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision/recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        rec = tp_cumsum / n_pos if n_pos > 0 else np.zeros_like(tp_cumsum)
        prec = (
            tp_cumsum / (tp_cumsum + fp_cumsum)
            if len(tp_cumsum) > 0
            else np.zeros_like(tp_cumsum)
        )

        if n_pos > 0:
            ap = compute_ap_voc2010(rec, prec)
            aps.append(ap)

    return np.mean(aps) if aps else 0.0


def parse_predictions(submission_rows):
    """Parses prediction strings into a DataFrame."""
    parsed = []
    for img_id, pred_str in submission_rows:
        if isinstance(pred_str, str) and len(pred_str.strip()) > 0:
            parts = pred_str.strip().split()
            # Format: class score x1 y1 x2 y2
            for i in range(0, len(parts), 6):
                if i + 5 < len(parts):
                    cls_id = int(parts[i])
                    # Skip 'No finding' class 14 for detection metric
                    if cls_id == 14:
                        continue
                    score = float(parts[i + 1])
                    x1 = float(parts[i + 2])
                    y1 = float(parts[i + 3])
                    x2 = float(parts[i + 4])
                    y2 = float(parts[i + 5])
                    parsed.append([img_id, cls_id, score, x1, y1, x2, y2])

    return pd.DataFrame(
        parsed,
        columns=["image_id", "class_id", "score", "x_min", "y_min", "x_max", "y_max"],
    )


# =============================================================================
# FAILURE ANALYSIS UTILS
# =============================================================================


def perform_failure_analysis(pred_df, gt_df):
    """
    Correlates error magnitude (1 - F1) with input features.
    """
    print("\nPerforming Failure Analysis...")

    # 1. Calculate per-image Error (1 - F1 at IoU 0.4)
    image_ids = gt_df["image_id"].unique()
    errors = []
    features = []

    # Pre-group for speed
    gt_grouped = gt_df[gt_df["class_id"] != 14].groupby("image_id")
    pred_grouped = pred_df.groupby("image_id")

    for img_id in image_ids:
        # Get GT
        if img_id in gt_grouped.groups:
            gts = gt_grouped.get_group(img_id)
            n_gts = len(gts)
            mean_area = (
                (gts["x_max"] - gts["x_min"]) * (gts["y_max"] - gts["y_min"])
            ).mean()
        else:
            n_gts = 0
            mean_area = 0

        # Get Preds
        if img_id in pred_grouped.groups:
            preds = pred_grouped.get_group(img_id)
        else:
            preds = pd.DataFrame()

        # Calculate F1
        tp = 0
        fp = 0
        fn = 0

        # Simple matching for F1 estimation
        matched_gt = set()

        if not preds.empty and n_gts > 0:
            for _, p in preds.iterrows():
                p_box = [p["x_min"], p["y_min"], p["x_max"], p["y_max"]]
                best_iou = 0
                best_gt_idx = -1

                for idx, g in gts.iterrows():
                    if idx in matched_gt:
                        continue
                    if g["class_id"] != p["class_id"]:
                        continue

                    g_box = [g["x_min"], g["y_min"], g["x_max"], g["y_max"]]
                    iou = calculate_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou > 0.4:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = n_gts - len(matched_gt)

        elif preds.empty and n_gts > 0:
            fn = n_gts
        elif not preds.empty and n_gts == 0:
            fp = len(preds)
        # else: both 0 -> perfect (tp=0, fp=0, fn=0), F1=1? No, accuracy=1.

        if tp + fp + fn == 0:
            f1 = 1.0  # Correctly predicted nothing
        else:
            f1 = (2 * tp) / (2 * tp + fp + fn)

        error = 1.0 - f1

        errors.append(error)
        features.append({"num_boxes": n_gts, "mean_box_area": mean_area})

    df_analysis = pd.DataFrame(features)
    df_analysis["error"] = errors

    # Calculate Correlation
    corr = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude (1-F1) and Input Features:")
    print(corr)
    return corr


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    seed_everything(SEED)

    # 1. Data Loading
    print("Initializing Data Loaders...")
    train_loader, val_loader, test_loader = create_dataloaders()

    # 2. Model Setup
    print("Initializing Model...")
    model = EfficientDetDecoupled(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = ThoracicLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)  # 5 Epochs

    # 3. Training
    # Override epochs to 5 for fast baseline
    EPOCHS_OVERRIDE = 5
    print(f"Starting Training for {EPOCHS_OVERRIDE} epochs...")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=EPOCHS_OVERRIDE,
        device=DEVICE,
        patience=3,
        save_path=CHECKPOINT_DIR,
    )

    # 4. Validation & Metric Calculation
    print("Running Validation Inference...")
    # Get predictions on validation set
    val_preds_raw = predict_and_format(
        model, val_loader, DEVICE, threshold=0.05, gate_threshold=0.8
    )

    # Load Ground Truth
    val_gt_df = pd.read_csv(VAL_META_PATH)

    # Parse Predictions
    val_pred_df = parse_predictions(val_preds_raw)

    # Calculate mAP
    if val_pred_df.empty:
        final_map = 0.0
    else:
        final_map = evaluate_map(val_pred_df, val_gt_df, iou_thresh=0.4)

    print(f"Final Validation Metric: {final_map}")

    # 5. Failure Analysis
    perform_failure_analysis(val_pred_df, val_gt_df)

    # 6. Submission
    THRESHOLD_SCORE = 0.1783551866

    if final_map > THRESHOLD_SCORE:
        print(
            f"Validation metric {final_map} > {THRESHOLD_SCORE}. Generating submission..."
        )

        # Run inference on test set
        test_preds_raw = predict_and_format(
            model, test_loader, DEVICE, threshold=0.2, gate_threshold=0.8
        )

        # Save to CSV
        df_sub = pd.DataFrame(test_preds_raw, columns=["image_id", "PredictionString"])
        os.makedirs(SUBMISSION_DIR, exist_ok=True)
        df_sub.to_csv(SUBMISSION_FILE, index=False)
        print(f"Submission saved to {SUBMISSION_FILE}")
    else:
        print(
            f"Validation metric {final_map} <= {THRESHOLD_SCORE}. Skipping submission."
        )


if __name__ == "__main__":
    main()
