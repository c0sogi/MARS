import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import cv2
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# Import from provided library
from library.config import Config
from library.dataset import ThoracicDataset
from library.model import ThoracicModel
from library.loss import ThoracicLoss
from library.engine import (
    train_one_epoch,
    validate,
    generate_submission,
    decode_predictions,
)
from library.utils import get_image_and_dimensions


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes [x_min, y_min, x_max, y_max].
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


def calculate_map(predictions, ground_truths, iou_threshold=0.4, num_classes=14):
    """
    Calculates mAP @ IoU > 0.4 using PASCAL VOC 2010 method (11-point interpolation or all-point).
    Actually, VOC 2010 uses all-point interpolation.

    predictions: List of dicts {'image_id': str, 'boxes': [[x1, y1, x2, y2, score, class_id], ...]}
    ground_truths: Dict mapping image_id to list of [[x1, y1, x2, y2, class_id], ...]
    """
    average_precisions = []

    # Process each class
    for class_id in range(num_classes):
        class_preds = []
        class_gts = {}
        n_pos = 0

        # Organize Ground Truths
        for img_id, boxes in ground_truths.items():
            # Filter boxes for this class
            cls_boxes = [b[:4] for b in boxes if int(b[4]) == class_id]
            class_gts[img_id] = {
                "boxes": np.array(cls_boxes),
                "det": [False] * len(cls_boxes),
            }
            n_pos += len(cls_boxes)

        # Organize Predictions
        for pred in predictions:
            img_id = pred["image_id"]
            # Parse prediction string or list
            # Here we assume predictions are already parsed into list of lists
            # But the engine returns PredictionString. We need to parse it.
            # Wait, we will parse the prediction strings before calling this.

            for box in pred["boxes"]:
                if int(box[5]) == class_id:
                    class_preds.append(
                        {"image_id": img_id, "bbox": box[:4], "score": box[4]}
                    )

        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        TP = np.zeros(len(class_preds))
        FP = np.zeros(len(class_preds))

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            bbox = pred["bbox"]

            if img_id not in class_gts:
                FP[i] = 1
                continue

            gt_data = class_gts[img_id]
            gt_boxes = gt_data["boxes"]

            max_iou = -1
            max_idx = -1

            if len(gt_boxes) > 0:
                # Vectorized IoU would be faster, but loop is fine for this scale
                for j, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(bbox, gt_box)
                    if iou > max_iou:
                        max_iou = iou
                        max_idx = j

            if max_iou >= iou_threshold:
                if not gt_data["det"][max_idx]:
                    TP[i] = 1
                    gt_data["det"][max_idx] = True
                else:
                    FP[i] = 1  # Duplicate detection
            else:
                FP[i] = 1

        # Compute Precision and Recall
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        rec = acc_TP / n_pos if n_pos > 0 else np.zeros_like(acc_TP)
        prec = acc_TP / (acc_TP + acc_FP)

        # Compute AP (VOC 2010 - Area under Precision-Recall Curve)
        # Add sentinel values
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        average_precisions.append(ap)

    return np.mean(average_precisions)


def parse_prediction_string(pred_str):
    """Parses 'class_id conf xmin ymin xmax ymax ...' into list of lists."""
    parts = pred_str.strip().split()
    boxes = []
    if len(parts) % 6 != 0:
        return boxes

    for i in range(0, len(parts), 6):
        cid = int(parts[i])
        conf = float(parts[i + 1])
        xmin = float(parts[i + 2])
        ymin = float(parts[i + 3])
        xmax = float(parts[i + 4])
        ymax = float(parts[i + 5])

        # Filter out "No finding" class 14 for mAP calculation
        if cid != 14:
            boxes.append([xmin, ymin, xmax, ymax, conf, cid])

    return boxes


def perform_failure_analysis(model, val_loader, device):
    """
    Calculates correlation between loss and image statistics.
    """
    print("\nPerforming Failure Analysis...")
    model.eval()
    criterion = ThoracicLoss()

    losses = []
    means = []
    stds = []

    # We iterate with batch_size=1 ideally, but DataLoader is already set.
    # We will compute loss per batch and average, or if we want per-sample correlation,
    # we need to iterate carefully.
    # To be precise and fast, let's just take a subset of validation for this analysis
    # and run with batch_size=1.

    analysis_subset = Subset(
        val_loader.dataset, indices=range(min(len(val_loader.dataset), 500))
    )
    analysis_loader = DataLoader(
        analysis_subset, batch_size=1, shuffle=False, num_workers=4
    )

    with torch.no_grad():
        for images, targets, _ in analysis_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)

            # Calculate loss (scalar for this single sample)
            loss, _ = criterion(outputs, targets)

            # Image stats (on CPU)
            img_np = images[0].cpu().numpy()
            # img is (3, H, W), normalized.

            losses.append(loss.item())
            means.append(np.mean(img_np))
            stds.append(np.std(img_np))

    # Calculate correlation
    if len(losses) > 1:
        corr_mean = np.corrcoef(losses, means)[0, 1]
        corr_std = np.corrcoef(losses, stds)[0, 1]

        print(
            f"Correlation between Error (Loss) and Image Mean Intensity: {corr_mean:.4f}"
        )
        print(
            f"Correlation between Error (Loss) and Image Contrast (Std): {corr_std:.4f}"
        )
    else:
        print("Not enough samples for failure analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Project: {Config.PROJECT_NAME}")
    print(f"Device: {device}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # 2. Data Loading
    # Train Set
    train_dataset = ThoracicDataset(mode="train")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Val Set (Full validation set for accurate metric)
    val_dataset = ThoracicDataset(mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model
    model = ThoracicModel()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    print("\nStarting Training...")
    best_val_loss = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss = validate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model (optional, but good practice)
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

    # 5. Metric Calculation (mAP)
    print("\nCalculating Validation mAP...")
    model.load_state_dict(
        torch.load(
            os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"), map_location=device
        )
    )
    model.eval()

    # Generate predictions on validation set
    val_preds_raw = []
    with torch.no_grad():
        for images, _, image_ids in val_loader:
            images = images.to(device)
            outputs = model(images)
            # Use is_test=False to assume train/val folder structure
            preds = decode_predictions(outputs, image_ids, device, is_test=False)
            val_preds_raw.extend(preds)

    # Parse predictions
    parsed_preds = []
    for p in val_preds_raw:
        parsed_preds.append(
            {
                "image_id": p["image_id"],
                "boxes": parse_prediction_string(p["PredictionString"]),
            }
        )

    # Load Ground Truths
    df_val = pd.read_csv(Config.VAL_META_PATH)
    ground_truths = defaultdict(list)
    for _, row in df_val.iterrows():
        if row["class_id"] != 14:
            ground_truths[row["image_id"]].append(
                [
                    row["x_min"],
                    row["y_min"],
                    row["x_max"],
                    row["y_max"],
                    row["class_id"],
                ]
            )

    # Calculate mAP
    final_map = calculate_map(
        parsed_preds, ground_truths, iou_threshold=0.4, num_classes=14
    )
    print(f"Final Validation Metric: {final_map}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 7. Submission
    THRESHOLD = 0.1783551866
    if final_map > THRESHOLD:
        print(
            f"\nMetric ({final_map}) > Threshold ({THRESHOLD}). Generating Submission..."
        )

        test_dataset = ThoracicDataset(mode="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_map}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
