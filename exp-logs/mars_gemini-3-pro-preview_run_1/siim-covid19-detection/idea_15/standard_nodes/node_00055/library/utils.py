import os
import sys
import random
import numpy as np
import torch
import cv2
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_xray(path, size=None, fix_monochrome=True):
    """
    Reads a DICOM image, applies VOI LUT, handles monochrome inversion,
    normalizes to 0-255, and optionally resizes.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT if available
        if hasattr(dicom, "VOI_LUT_Sequence"):
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle Monochrome inversion
        # X-rays should have bones as white (high intensity) and air as black (low intensity).
        # MONOCHROME1: 0 is white. MONOCHROME2: 0 is black.
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data.astype(np.float32)
        data_min = np.min(data)
        data_max = np.max(data)

        if data_max - data_min > 0:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

        data = (data * 255).astype(np.uint8)

        # Resize if requested
        if size is not None:
            # OpenCV uses (width, height)
            if isinstance(size, int):
                size = (size, size)
            data = cv2.resize(data, size, interpolation=cv2.INTER_LINEAR)

        # Convert to 3 channels for model compatibility if needed,
        # but usually models take 3 channels.
        # Here we return 2D array (H, W). The dataset class usually handles channel duplication.
        return data

    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        # Return a black image of target size or default 512
        s = size if size is not None else (512, 512)
        if isinstance(s, int):
            s = (s, s)
        return np.zeros((s[1], s[0]), dtype=np.uint8)


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [xmin, ymin, xmax, ymax].
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    w_i = max(0, x2_i - x1_i)
    h_i = max(0, y2_i - y1_i)
    intersection = w_i * h_i

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_ap_voc2010(precisions, recalls):
    """
    Computes Average Precision using the PASCAL VOC 2010 method (all-point interpolation).
    """
    # Append sentinel values at both ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under the curve
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (Recall_i+1 - Recall_i) * Precision_i+1
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_prediction_string(pred_str):
    """
    Parses a prediction string into a list of [label, confidence, xmin, ymin, xmax, ymax].
    Format: "label conf xmin ymin xmax ymax label conf ..."
    """
    if not isinstance(pred_str, str) or pred_str.strip() == "":
        return []

    parts = pred_str.strip().split()
    # Each prediction has 6 elements
    num_preds = len(parts) // 6
    parsed = []

    for i in range(num_preds):
        idx = i * 6
        label = parts[idx]
        conf = float(parts[idx + 1])
        xmin = float(parts[idx + 2])
        ymin = float(parts[idx + 3])
        xmax = float(parts[idx + 4])
        ymax = float(parts[idx + 5])
        parsed.append({"label": label, "score": conf, "bbox": [xmin, ymin, xmax, ymax]})

    return parsed


def prepare_gt_from_metadata(
    metadata_df, load_cached_data=True, cache_dir="./working/idea_15"
):
    """
    Converts the metadata DataFrame (train.csv/val.csv) into a Ground Truth DataFrame
    compatible with the submission format (Id, PredictionString).

    Implements caching as requested for deterministic data processing.
    """
    cache_path = os.path.join(cache_dir, "gt_dataframe_cache.parquet")

    # 1. Try to load cache
    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Fallback to computation

    # 2. Compute
    records = []

    # Study Level Mapping
    study_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    # Process Study IDs
    # Group by study_id to handle potential duplicates if any (though metadata usually 1 row per study)
    # Metadata has one row per image, so we need to deduplicate studies
    study_df = metadata_df.drop_duplicates(subset=["study_id"])

    for _, row in study_df.iterrows():
        study_id = f"{row['study_id']}_study"
        labels = []
        for col, label_name in study_map.items():
            if row[col] == 1:
                # Format: class 1 0 0 1 1
                labels.append(f"{label_name} 1 0 0 1 1")

        prediction_string = " ".join(labels)
        records.append({"Id": study_id, "PredictionString": prediction_string})

    # Process Image IDs
    # Metadata has 'label' column which is already the GT string for images
    # e.g. "opacity 1 ... ... ... ..." or "none 1 0 0 1 1"
    for _, row in metadata_df.iterrows():
        image_id = f"{row['image_id']}_image"
        # The 'label' column in train_image_level.csv (and thus metadata) is the GT string
        # Note: In the provided metadata, the column is named 'label'.
        prediction_string = row["label"]
        records.append({"Id": image_id, "PredictionString": prediction_string})

    gt_df = pd.DataFrame(records)

    # 3. Save Cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        gt_df.to_parquet(cache_path)

    return gt_df


def calculate_map(pred_df, gt_df, iou_threshold=0.5):
    """
    Calculates the mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        pred_df: DataFrame with ['Id', 'PredictionString']
        gt_df: DataFrame with ['Id', 'PredictionString']
        iou_threshold: IoU threshold for considering a detection positive.

    Returns:
        float: mAP score
    """
    # Create dictionaries for fast access
    # gt_data[id] = list of {'label': str, 'bbox': [x1, y1, x2, y2]}
    gt_data = {}
    for _, row in gt_df.iterrows():
        gt_data[row["Id"]] = parse_prediction_string(row["PredictionString"])

    # pred_data[id] = list of {'label': str, 'score': float, 'bbox': [x1, y1, x2, y2]}
    pred_data = {}
    for _, row in pred_df.iterrows():
        pred_data[row["Id"]] = parse_prediction_string(row["PredictionString"])

    # Identify all unique classes
    all_classes = set()
    for items in gt_data.values():
        for item in items:
            all_classes.add(item["label"])
    for items in pred_data.values():
        for item in items:
            all_classes.add(item["label"])

    average_precisions = {}

    for cls in all_classes:
        # Collect all predictions and ground truths for this class
        class_preds = []  # list of (score, image_id, bbox)
        class_gts = {}  # image_id -> list of [bbox, used_flag]
        n_pos = 0

        # Populate GTs
        for img_id, items in gt_data.items():
            bboxes = [item["bbox"] for item in items if item["label"] == cls]
            # Each bbox needs a 'used' flag
            class_gts[img_id] = {"bboxes": bboxes, "used": [False] * len(bboxes)}
            n_pos += len(bboxes)

        # Populate Preds
        for img_id, items in pred_data.items():
            for item in items:
                if item["label"] == cls:
                    class_preds.append((item["score"], img_id, item["bbox"]))

        # Sort predictions by score descending
        class_preds.sort(key=lambda x: x[0], reverse=True)

        TP = np.zeros(len(class_preds))
        FP = np.zeros(len(class_preds))

        for i, (score, img_id, pred_bbox) in enumerate(class_preds):
            if img_id not in class_gts:
                # No GT for this image at all (should not happen if IDs match)
                FP[i] = 1
                continue

            gt_info = class_gts[img_id]
            gt_bboxes = gt_info["bboxes"]
            gt_used = gt_info["used"]

            best_iou = -1
            best_idx = -1

            for j, gt_bbox in enumerate(gt_bboxes):
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold:
                if not gt_used[best_idx]:
                    TP[i] = 1
                    gt_used[best_idx] = True
                else:
                    FP[i] = 1  # Double detection
            else:
                FP[i] = 1

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(TP)
        fp_cumsum = np.cumsum(FP)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / (n_pos + 1e-10)

        if n_pos == 0:
            ap = 0.0
        else:
            ap = compute_ap_voc2010(precisions, recalls)

        average_precisions[cls] = ap

    # Compute mAP
    if not average_precisions:
        return 0.0

    mAP = sum(average_precisions.values()) / len(average_precisions)

    # Print per-class AP for debugging/info
    print("\nPer-class Average Precision:")
    for cls, ap in average_precisions.items():
        print(f"  {cls}: {ap}")
    print(f"Mean Average Precision (mAP): {mAP}")

    return mAP
