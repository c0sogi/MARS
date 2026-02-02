import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import rasterio
import cv2

# Import provided libraries
from library.config import Config
from library.trainer import Trainer, seed_everything
from library.dataset import VinBigDataDataset
from library.inference import decode_predictions, rescale_bboxes
from library.model import CenterNet

# =============================================================================
# Metric Calculation Functions (mAP IoU > 0.4)
# =============================================================================


def calculate_iou_batch(boxes1, boxes2):
    """
    Computes IoU between two sets of boxes.
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) matrix of IoUs
    """
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def compute_ap_voc2010(rec, prec):
    """
    Compute VOC2010 AP (Area under Precision-Recall Curve with interpolation)
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope (interpolation)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(predictions, ground_truths, iou_threshold=0.4, num_classes=14):
    """
    Calculates mAP @ IoU > 0.4.
    predictions: List of dicts {'image_id': str, 'boxes': [[x1, y1, x2, y2, score, class], ...]}
    ground_truths: DataFrame with columns [image_id, class_id, x_min, y_min, x_max, y_max]
    """
    aps = []

    # 1. Organize Ground Truth
    gt_by_class = defaultdict(list)
    n_pos = defaultdict(int)

    for cls in range(num_classes):
        cls_df = ground_truths[ground_truths["class_id"] == cls]
        for img_id, group in cls_df.groupby("image_id"):
            boxes = group[["x_min", "y_min", "x_max", "y_max"]].values
            detected = np.zeros(len(boxes), dtype=bool)
            gt_by_class[(cls, img_id)] = {"boxes": boxes, "detected": detected}
            n_pos[cls] += len(boxes)

    # 2. Organize Predictions
    preds_by_class = defaultdict(list)
    for pred in predictions:
        img_id = pred["image_id"]
        for box in pred["boxes"]:
            # box: x1, y1, x2, y2, score, class
            c = int(box[5])
            if c < num_classes:  # Ignore "No finding" class 14
                preds_by_class[c].append(
                    {"score": box[4], "image_id": img_id, "box": box[:4]}
                )

    # 3. Calculate AP per class
    for cls in range(num_classes):
        cls_preds = preds_by_class[cls]

        # If no ground truth and no predictions, AP is 0 (or undefined, usually 0 in competitions)
        # If ground truth exists but no predictions, AP is 0.
        if n_pos[cls] == 0:
            continue

        if len(cls_preds) == 0:
            aps.append(0.0)
            continue

        # Sort by confidence descending
        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, p in enumerate(cls_preds):
            img_id = p["image_id"]
            pred_box = np.array([p["box"]])  # (1, 4)

            gt_data = gt_by_class.get((cls, img_id))

            if gt_data is None:
                fp[i] = 1
                continue

            gt_boxes = gt_data["boxes"]
            gt_detected = gt_data["detected"]

            # Calculate IoU
            ious = calculate_iou_batch(pred_box, gt_boxes)[0]  # (M,)

            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]

                if max_iou > iou_threshold:
                    if not gt_detected[max_iou_idx]:
                        tp[i] = 1
                        gt_detected[max_iou_idx] = True
                    else:
                        fp[i] = 1  # Duplicate detection
                else:
                    fp[i] = 1
            else:
                fp[i] = 1

        # Compute Precision/Recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recall = tp_cumsum / n_pos[cls]
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap_voc2010(recall, precision)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


# =============================================================================
# Main Orchestration
# =============================================================================


def main():
    # --- 1. Configuration ---
    # Override defaults for a fast but effective baseline
    Config.NUM_EPOCHS = 15
    Config.DEBUG = False

    print(f"Starting Runfile. Device: {Config.DEVICE}, Epochs: {Config.NUM_EPOCHS}")
    seed_everything(Config.SEED)

    # --- 2. Training ---
    # Initialize trainer with caching enabled to speed up data loading
    trainer = Trainer(load_cached_data=True)
    trainer.train()

    # --- 3. Validation ---
    print("\nStarting Validation Inference...")

    # Load metadata
    val_meta = pd.read_csv(Config.VAL_META_PATH)
    img_id_to_path = dict(zip(val_meta["image_id"], val_meta["file_path"]))

    # Load best model
    model = CenterNet(pretrained=False)
    model.load_state_dict(
        torch.load(trainer.model_save_path, map_location=Config.DEVICE)
    )
    model.to(Config.DEVICE)
    model.eval()

    # Setup DataLoader
    val_dataset = VinBigDataDataset(
        split="val", debug=Config.DEBUG, load_cached_data=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    all_predictions = []
    failure_data = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            imgs = batch["image"].to(Config.DEVICE)
            img_ids = batch["image_id"]

            # Inference
            outputs = model(imgs)
            detections = decode_predictions(
                outputs["hm"], outputs["wh"], outputs["reg"], K=Config.TOP_K
            )

            # Retrieve Original Shapes for Rescaling
            original_shapes = []
            for img_id in img_ids:
                path = img_id_to_path.get(img_id)
                full_path = os.path.join(Config.INPUT_DIR, path)
                h, w = Config.IMG_SIZE, Config.IMG_SIZE

                # Try reading dimensions using rasterio (robust) or cv2
                try:
                    with rasterio.open(full_path) as src:
                        h, w = src.height, src.width
                except:
                    try:
                        img_tmp = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                        if img_tmp is not None:
                            h, w = img_tmp.shape[:2]
                    except:
                        pass  # Default to IMG_SIZE if all fails

                original_shapes.append((h, w))

            # Rescale predictions to original image coordinates
            rescaled_dets = rescale_bboxes(detections, original_shapes)

            # Store predictions and collect failure analysis data
            for i, img_id in enumerate(img_ids):
                # Save for mAP calculation
                all_predictions.append({"image_id": img_id, "boxes": rescaled_dets[i]})

                # Failure Analysis: Compare with GT
                gt_df = val_meta[
                    (val_meta["image_id"] == img_id) & (val_meta["class_id"] != 14)
                ]
                if not gt_df.empty:
                    gt_boxes = gt_df[["x_min", "y_min", "x_max", "y_max"]].values
                    gt_classes = gt_df["class_id"].values

                    pred_boxes = rescaled_dets[i]

                    for j in range(len(gt_boxes)):
                        gt_box = gt_boxes[j]
                        gt_cls = gt_classes[j]

                        # Calculate geometric properties
                        w_gt = gt_box[2] - gt_box[0]
                        h_gt = gt_box[3] - gt_box[1]
                        area = w_gt * h_gt
                        aspect_ratio = w_gt / (h_gt + 1e-6)

                        # Find best IoU match
                        max_iou = 0.0
                        cls_mask = pred_boxes[:, 5] == gt_cls
                        cls_preds = pred_boxes[cls_mask]

                        if len(cls_preds) > 0:
                            ious = calculate_iou_batch(
                                np.array([gt_box]), cls_preds[:, :4]
                            )[0]
                            max_iou = np.max(ious)

                        failure_data.append(
                            {
                                "area": area,
                                "aspect_ratio": aspect_ratio,
                                "error": 1.0 - max_iou,
                            }
                        )

    # --- 4. Metrics & Analysis ---
    map_score = evaluate_map(all_predictions, val_meta, iou_threshold=0.4)
    print(f"Final Validation Metric: {map_score}")

    print("\n--- Failure Analysis ---")
    if failure_data:
        df_fail = pd.DataFrame(failure_data)
        corr_area = df_fail["error"].corr(df_fail["area"])
        corr_ar = df_fail["error"].corr(df_fail["aspect_ratio"])
        print(f"Correlation between Error (1-IoU) and BBox Area: {corr_area:.6f}")
        print(f"Correlation between Error (1-IoU) and Aspect Ratio: {corr_ar:.6f}")
    else:
        print("No ground truth objects available for failure analysis.")

    # --- 5. Submission ---
    if map_score > 0.0:
        print("\nGenerating Submission for Test Set...")
        trainer.predict()
    else:
        print("\nSkipping submission generation due to 0.0 mAP.")

    print("Process Completed.")


if __name__ == "__main__":
    main()
