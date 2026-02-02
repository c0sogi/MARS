import os
import sys
import torch
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader
from library import config, utils, dataset, model, loss, train, predict

# -----------------------------------------------------------------------------
# Helper Functions for Metrics
# -----------------------------------------------------------------------------


def calculate_iou(box1, box2):
    """
    Calculates IoU between two boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def calculate_global_map(all_preds, all_gts):
    """
    Calculates PASCAL VOC 2010 mAP @ IoU 0.5.

    Args:
        all_preds: List of [image_id, x1, y1, x2, y2, score]
        all_gts: Dict mapping image_id -> List of [x1, y1, x2, y2]
    """
    if not all_preds:
        return 0.0

    # Sort predictions by confidence score descending
    all_preds.sort(key=lambda x: x[5], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track matched GT boxes to avoid double counting
    gt_matched = {img_id: [False] * len(boxes) for img_id, boxes in all_gts.items()}

    for i, pred in enumerate(all_preds):
        img_id = pred[0]
        box_pred = pred[1:5]

        gts = all_gts.get(img_id, [])

        max_iou = 0.0
        max_idx = -1

        for j, gt in enumerate(gts):
            iou = calculate_iou(box_pred, gt)
            if iou > max_iou:
                max_iou = iou
                max_idx = j

        if max_iou >= 0.5:
            if not gt_matched[img_id][max_idx]:
                tp[i] = 1.0
                gt_matched[img_id][max_idx] = True
            else:
                fp[i] = 1.0  # Duplicate detection
        else:
            fp[i] = 1.0  # False positive

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    n_pos = sum(len(boxes) for boxes in all_gts.values())
    if n_pos == 0:
        return 0.0

    rec = cum_tp / n_pos
    prec = cum_tp / (cum_tp + cum_fp + 1e-6)

    # VOC 2010 AP Calculation (Area under interpolated curve)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------


def main():
    # 1. Setup
    utils.seed_everything(config.SEED)
    device = config.DEVICE

    # Fast Baseline Parameters
    EPOCHS = 3
    TRAIN_SAMPLES = 2000

    # 2. Data Loading
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Subsample training data for speed
    if len(df_train) > TRAIN_SAMPLES:
        df_train = df_train.sample(
            n=TRAIN_SAMPLES, random_state=config.SEED
        ).reset_index(drop=True)

    print(f"Training on {len(df_train)} samples. Validating on {len(df_val)} samples.")

    train_dataset = dataset.SIIMDataset(df_train, split="train", load_cached_data=True)
    val_dataset = dataset.SIIMDataset(df_val, split="val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    net = model.MultiTaskUNet(pretrained=True)
    net = net.to(device)

    criterion = loss.MultiTaskLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    best_val_loss = float("inf")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        # Train
        train_metrics = train.train_one_epoch(
            net, train_loader, criterion, optimizer, device
        )

        # Validate (Loss monitoring)
        val_metrics = train.validate(net, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(net.state_dict(), config.CHECKPOINT_PATH)

    print("Training complete.")

    # 5. Validation Assessment & Failure Analysis
    print("Running validation inference for Metric and Failure Analysis...")
    net.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    net.eval()

    all_preds = []  # For mAP
    all_gts = {}  # For mAP
    error_data = []  # For Failure Analysis

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)  # GT Masks (512x512)
            image_ids = batch["image_id"]

            seg_logits, class_logits = net(images)
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            class_probs = torch.softmax(class_logits, dim=1).cpu().numpy()

            # Process batch
            for i in range(len(images)):
                img_id = image_ids[i]
                mask_prob = seg_probs[i, 0]
                c_prob = class_probs[i]

                # Get Metadata for this image
                row = df_val[df_val["image_id"] == img_id].iloc[0]
                orig_h, orig_w = row["height"], row["width"]

                # --- 1. Prepare Data for mAP ---

                # Parse GT Boxes
                gt_boxes = []
                if pd.notna(row["boxes"]):
                    try:
                        boxes_list = ast.literal_eval(row["boxes"])
                        for b in boxes_list:
                            gt_boxes.append(
                                [
                                    b["x"],
                                    b["y"],
                                    b["x"] + b["width"],
                                    b["y"] + b["height"],
                                ]
                            )
                    except:
                        pass
                all_gts[img_id] = gt_boxes

                # Generate Pred Boxes
                pred_class_idx = np.argmax(c_prob)
                if pred_class_idx != 0:  # If not Negative
                    # Get boxes in 512x512
                    boxes_512 = utils.mask_to_boxes(
                        mask_prob, threshold=config.MASK_THRESHOLD
                    )

                    # Scale to original dimensions
                    scale_x = orig_w / config.IMG_SIZE
                    scale_y = orig_h / config.IMG_SIZE

                    for b in boxes_512:
                        px1, py1, px2, py2, score = b
                        all_preds.append(
                            [
                                img_id,
                                px1 * scale_x,
                                py1 * scale_y,
                                px2 * scale_x,
                                py2 * scale_y,
                                score,
                            ]
                        )

                # --- 2. Failure Analysis Data ---
                # Calculate Dice Score for this specific image
                # GT Mask is in batch['mask'][i] (1, 512, 512)
                gt_mask_np = masks[i, 0].cpu().numpy()
                pred_mask_np = (mask_prob > config.MASK_THRESHOLD).astype(np.float32)

                intersection = np.sum(gt_mask_np * pred_mask_np)
                dice = (2.0 * intersection + 1.0) / (
                    np.sum(gt_mask_np) + np.sum(pred_mask_np) + 1.0
                )

                error_data.append(
                    {
                        "error": 1.0 - dice,
                        "width": orig_w,
                        "height": orig_h,
                        "aspect_ratio": orig_w / orig_h,
                    }
                )

    # Calculate and Print Metric
    final_map = calculate_global_map(all_preds, all_gts)
    print(f"Final Validation Metric: {final_map}")

    # Perform Failure Analysis
    print("Performing failure analysis...")
    df_error = pd.DataFrame(error_data)
    if not df_error.empty:
        correlations = df_error.corr()["error"].drop("error")
        print("Correlation between Error (1-Dice) and Input Features:")
        print(correlations)

    # 6. Submission
    print("Generating submission file...")
    predict.generate_submission(load_cached_data=True)


if __name__ == "__main__":
    main()
