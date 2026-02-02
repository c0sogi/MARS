import os
import random
import struct
import numpy as np
import pandas as pd
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_binary(path, fix_monochrome=True):
    """
    Reads a DICOM file as binary, extracts pixel data, and performs semantic normalization.

    Args:
        path (str): Path to the .dicom file.
        fix_monochrome (bool): If True, inverts MONOCHROME1 images so 0=Black.

    Returns:
        np.ndarray: The image data as a 2D or 3D numpy array.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        # Return a placeholder if file is missing (robustness)
        return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 1), dtype=np.uint8)

    # Helper to find tag value in Little Endian Explicit VR (common) or Implicit
    # We search for the tag bytes.
    # Rows: (0028, 0010) -> b'\x28\x00\x10\x00'
    # Cols: (0028, 0011) -> b'\x28\x00\x11\x00'
    # Photo: (0028, 0004) -> b'\x28\x00\x04\x00'
    # Bits: (0028, 0100) -> b'\x28\x00\x00\x01' (Bits Allocated)
    # Pixel: (7FE0, 0010) -> b'\xE0\x7F\x10\x00'

    def find_tag_value(tag_bytes, data, value_type="short"):
        start = data.find(tag_bytes)
        if start == -1:
            return None

        # Skip Tag (4)
        idx = start + 4

        # Check VR (Explicit)
        vr = data[idx : idx + 2]
        # Common VRs for these tags: US (Unsigned Short), CS (Code String)
        if vr in [b"US", b"SS", b"OW", b"OB"]:
            idx += 2  # Skip VR
            length = int.from_bytes(data[idx : idx + 2], "little")
            idx += 2
        elif vr in [b"CS", b"LO", b"SH"]:
            idx += 2  # Skip VR
            length = int.from_bytes(data[idx : idx + 2], "little")
            idx += 2
        else:
            # Implicit VR? Length is 4 bytes directly
            length = int.from_bytes(data[idx : idx + 4], "little")
            idx += 4

        value_bytes = data[idx : idx + length]

        if value_type == "short":
            return int.from_bytes(value_bytes, "little")
        elif value_type == "string":
            return value_bytes.decode("utf-8", errors="ignore").strip().strip("\x00")
        return None

    # Parse Metadata
    rows = find_tag_value(b"\x28\x00\x10\x00", data, "short")
    cols = find_tag_value(b"\x28\x00\x11\x00", data, "short")
    bits = find_tag_value(b"\x28\x00\x00\x01", data, "short")  # Bits Allocated
    photo = find_tag_value(b"\x28\x00\x04\x00", data, "string")

    # Defaults if parsing fails
    if rows is None:
        rows = Config.IMG_SIZE
    if cols is None:
        cols = Config.IMG_SIZE
    if bits is None:
        bits = 8

    # Find Pixel Data
    pixel_tag = b"\xe0\x7f\x10\x00"
    start = data.find(pixel_tag)

    img = None
    if start != -1:
        idx = start + 4
        # Determine length of pixel data
        # Explicit VR (OB/OW) has reserved bytes
        vr = data[idx : idx + 2]
        if vr in [b"OB", b"OW"]:
            idx += 4  # VR + Reserved
            length = int.from_bytes(data[idx : idx + 4], "little")
            idx += 4
        else:
            # Implicit or other
            length = int.from_bytes(data[idx : idx + 4], "little")
            idx += 4

        # Extract Raw Bytes
        # If length is undefined (0xFFFFFFFF), scan until Sequence Delimiter (not handled here for simplicity)
        # Assuming defined length which is standard for single frame CXR
        raw_bytes = data[idx : idx + length]

        expected_size = rows * cols * (bits // 8)

        # Safety check on size
        if len(raw_bytes) >= expected_size:
            dtype = np.uint8 if bits <= 8 else np.uint16
            img = np.frombuffer(raw_bytes[:expected_size], dtype=dtype)
            img = img.reshape((rows, cols))

    # Fallback if binary parsing failed
    if img is None:
        return np.zeros((rows, cols, 1), dtype=np.uint8)

    # Semantic Normalization
    if fix_monochrome and photo == "MONOCHROME1":
        # Invert: max_val - pixel
        max_val = (2**bits) - 1
        img = max_val - img

    # Normalize to 0-255 for consistency
    if img.max() > 0:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0

    img = img.astype(np.uint8)

    # Expand dims to H, W, 1
    img = np.expand_dims(img, axis=-1)

    return img


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.
    Boxes are [xmin, ymin, xmax, ymax].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_map(pred_df, gt_df, iou_threshold=0.4):
    """
    Calculates PASCAL VOC 2010 mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        pred_df (pd.DataFrame): Predictions with cols [image_id, class_id, confidence, x_min, y_min, x_max, y_max]
        gt_df (pd.DataFrame): Ground truth with cols [image_id, class_id, x_min, y_min, x_max, y_max]
        iou_threshold (float): IoU threshold for a positive match.

    Returns:
        float: The mAP score.
    """
    average_precisions = []

    # Get all unique classes present in GT
    classes = sorted(gt_df["class_id"].unique())

    # If Class 14 (No finding) is in GT, we include it.
    # Note: Usually detection metrics exclude the background class, but here
    # "No finding" is treated as an explicit prediction task with a 1x1 box.

    for class_id in classes:
        class_preds = pred_df[pred_df["class_id"] == class_id].copy()
        class_gts = gt_df[gt_df["class_id"] == class_id].copy()

        if len(class_gts) == 0:
            continue

        # Sort predictions by confidence descending
        class_preds = class_preds.sort_values(
            "confidence", ascending=False
        ).reset_index(drop=True)

        n_gt = len(class_gts)
        n_pred = len(class_preds)

        tp = np.zeros(n_pred)
        fp = np.zeros(n_pred)

        # Track which GT boxes have been matched
        gt_matched = {
            img_id: np.zeros(len(group))
            for img_id, group in class_gts.groupby("image_id")
        }

        # Group GT by image for faster access
        gt_by_image = {img_id: group for img_id, group in class_gts.groupby("image_id")}

        for i in range(n_pred):
            pred_box = class_preds.iloc[i][["x_min", "y_min", "x_max", "y_max"]].values
            img_id = class_preds.iloc[i]["image_id"]

            best_iou = -1.0
            best_gt_idx = -1

            if img_id in gt_by_image:
                gt_boxes = gt_by_image[img_id][
                    ["x_min", "y_min", "x_max", "y_max"]
                ].values

                # Find best matching GT box
                for j, gt_box in enumerate(gt_boxes):
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            # Check match
            if best_iou >= iou_threshold:
                # Check if this GT is already matched
                if gt_matched[img_id][best_gt_idx] == 0:
                    tp[i] = 1
                    gt_matched[img_id][best_gt_idx] = 1
                else:
                    fp[i] = 1  # Double detection
            else:
                fp[i] = 1  # False positive

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # PASCAL VOC 2010 AP Calculation (Integration)
        # Prepend 0 to recall and 0 to precision (or 1? VOC usually starts p=1 at r=0 if first is TP)
        # Standard approach:
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Compute maximum precision for any recall >= r (Smoothing)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Integrate area under curve
        # Find indices where recall changes
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return np.mean(average_precisions)
