import os
import random
import numpy as np
import torch
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import ast
from library.config import Config


def seed_everything(seed=42):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path, fix_monochrome=True):
    """
    Reads a DICOM file, applies VOI LUT if available, handles monochrome inversion,
    and normalizes pixel values to 8-bit (0-255).
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT (Value of Interest Look-Up Table) if available
        # This transforms raw pixel data to "human-friendly" view
        if "VOILUTSequence" in dicom or "WindowCenter" in dicom:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Handle MONOCHROME1 (where 0 is white). We want 0 to be black.
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data.astype(np.float32)
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)

        return data
    except Exception as e:
        print(f"Error reading DICOM {path}: {e}")
        # Return a blank 512x512 image in case of corruption to maintain pipeline stability
        return np.zeros((512, 512), dtype=np.uint8)


def process_dataset(
    df, cache_dir, image_size=512, load_cached_data=True, split_name="train"
):
    """
    Processes the dataset: reads DICOMs, resizes, generates masks and labels.
    Implements strict caching logic: loads from disk if available, otherwise computes and saves.

    Returns:
        images: (N, H, W, 3) uint8 array
        masks: (N, H, W, 1) float32 array
        labels: (N, 4) float32 array
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache file paths
    images_path = os.path.join(cache_dir, f"{split_name}_images.npy")
    masks_path = os.path.join(cache_dir, f"{split_name}_masks.npy")
    labels_path = os.path.join(cache_dir, f"{split_name}_labels.npy")

    # 1. Try to load cached data
    if load_cached_data:
        if (
            os.path.exists(images_path)
            and os.path.exists(masks_path)
            and os.path.exists(labels_path)
        ):
            print(f"Loading cached {split_name} data from {cache_dir}...")
            images = np.load(images_path)
            masks = np.load(masks_path)
            labels = np.load(labels_path)
            return images, masks, labels
        else:
            print(
                f"Cache miss for {split_name} in {cache_dir}. Processing from scratch..."
            )

    # 2. Process data from scratch
    print(f"Processing {split_name} dataset ({len(df)} images)...")

    img_list = []
    mask_list = []
    label_list = []

    # Study level columns
    study_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    for idx, row in df.iterrows():
        # Construct full path
        full_path = os.path.join(Config.INPUT_DIR, row["file_path"])

        # Read and Preprocess Image
        img = read_dicom(full_path)

        # Resize
        img_resized = cv2.resize(img, (image_size, image_size))

        # Convert to RGB (3 channels) for ResNet backbone compatibility
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_list.append(img_rgb)

        # Process Masks
        # Initialize empty mask
        mask = np.zeros((image_size, image_size), dtype=np.float32)

        if "boxes" in row and pd.notna(row["boxes"]):
            try:
                boxes = ast.literal_eval(row["boxes"])
                # Original dimensions for scaling
                orig_h, orig_w = img.shape[:2]

                for box in boxes:
                    x, y, w, h = box["x"], box["y"], box["width"], box["height"]

                    # Scale coordinates to new image size
                    x_s = int(x * image_size / orig_w)
                    y_s = int(y * image_size / orig_h)
                    w_s = int(w * image_size / orig_w)
                    h_s = int(h * image_size / orig_h)

                    # Fill rectangle on mask
                    cv2.rectangle(mask, (x_s, y_s), (x_s + w_s, y_s + h_s), 1.0, -1)
            except:
                pass  # Parse error or empty

        mask_list.append(mask)

        # Process Labels
        if all(c in row for c in study_cols):
            lbl = row[study_cols].values.astype(np.float32)
            label_list.append(lbl)
        else:
            # For test set or missing labels
            label_list.append(np.zeros(4, dtype=np.float32))

    # Stack into arrays
    images = np.array(img_list, dtype=np.uint8)
    masks = np.array(mask_list, dtype=np.float32)
    # Add channel dimension to masks: (N, H, W) -> (N, H, W, 1)
    masks = np.expand_dims(masks, axis=-1)
    labels = np.array(label_list, dtype=np.float32)

    # 3. Save to cache
    print(f"Saving processed {split_name} data to {cache_dir}...")
    np.save(images_path, images)
    np.save(masks_path, masks)
    np.save(labels_path, labels)

    return images, masks, labels


def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes [x1, y1, x2, y2].
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


def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using all-point interpolation (standard for PASCAL VOC).
    """
    # Append sentinel values at ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve
    # Look for points where X axis (recall) changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * precision
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(pred_df, gt_df, iou_threshold=0.5):
    """
    Calculates mAP @ IoU > 0.5 for the competition task.
    Handles both Study-level (classification) and Image-level (detection) predictions.

    Args:
        pred_df: DataFrame with columns ['id', 'PredictionString']
        gt_df: DataFrame with columns ['id', 'PredictionString']
    Returns:
        float: mean Average Precision
    """

    # Helper to parse "class conf x1 y1 x2 y2 ..." string
    def parse_prediction_string(s):
        if pd.isna(s):
            return []
        parts = str(s).split()
        res = []
        # Iterate in chunks of 6
        for i in range(0, len(parts), 6):
            if i + 5 >= len(parts):
                break
            cls = parts[i]
            try:
                conf = float(parts[i + 1])
                x1 = float(parts[i + 2])
                y1 = float(parts[i + 3])
                x2 = float(parts[i + 4])
                y2 = float(parts[i + 5])
                res.append({"class": cls, "conf": conf, "box": [x1, y1, x2, y2]})
            except ValueError:
                continue
        return res

    # Parse Ground Truth
    gt_data = {}
    for _, row in gt_df.iterrows():
        gt_data[row["id"]] = parse_prediction_string(row["PredictionString"])

    # Parse Predictions
    pred_data = {}
    for _, row in pred_df.iterrows():
        pred_data[row["id"]] = parse_prediction_string(row["PredictionString"])

    # Define all classes involved in the metric
    classes = [
        "negative",
        "typical",
        "indeterminate",
        "atypical",  # Study classes
        "opacity",
        "none",  # Image classes
    ]

    aps = []

    for cls in classes:
        class_preds = []  # List of (conf, id, box)
        class_gts = {}  # Map id -> {boxes: [], used: []}
        n_pos = 0

        # Union of all IDs to handle potential mismatches
        all_ids = set(gt_data.keys()) | set(pred_data.keys())

        for img_id in all_ids:
            # Extract GT boxes for this class
            gts = [x["box"] for x in gt_data.get(img_id, []) if x["class"] == cls]
            class_gts[img_id] = {
                "boxes": np.array(gts),
                "used": np.zeros(len(gts), dtype=bool),
            }
            n_pos += len(gts)

            # Extract Predictions for this class
            preds = [x for x in pred_data.get(img_id, []) if x["class"] == cls]
            for p in preds:
                class_preds.append((p["conf"], img_id, p["box"]))

        # If no ground truth for this class, AP is 0 (unless no preds either, but usually handled)
        if n_pos == 0:
            if len(class_preds) > 0:
                aps.append(0.0)  # False positives exist but no positives
            # If no preds and no GT, usually undefined or 1.0?
            # In Pascal VOC, if a class is not present in test set, it's excluded.
            # Here we assume classes exist in the dataset.
            continue

        # Sort predictions by confidence descending
        class_preds.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))

        for i, (conf, img_id, pred_box) in enumerate(class_preds):
            gt_info = class_gts[img_id]
            gt_boxes = gt_info["boxes"]
            gt_used = gt_info["used"]

            best_iou = -1
            best_idx = -1

            if len(gt_boxes) > 0:
                # Find best matching GT box
                for j, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j

            # Check if match is valid
            if best_iou >= iou_threshold:
                if not gt_used[best_idx]:
                    tp[i] = 1
                    gt_used[best_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection (already matched)
            else:
                fp[i] = 1  # False positive

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_pos
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)

    if not aps:
        return 0.0

    return np.mean(aps)
