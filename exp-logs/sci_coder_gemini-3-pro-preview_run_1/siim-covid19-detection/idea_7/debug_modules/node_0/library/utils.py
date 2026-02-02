import os
import cv2
import ast
import torch
import pydicom
import numpy as np
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config, seed_everything


def read_dicom(path, size=None, fix_monochrome=True):
    """
    Reads a DICOM file, applies VOI LUT if available, fixes monochrome interpretation,
    and resizes the image.

    Args:
        path (str): Path to the DICOM file.
        size (int, optional): Target size for resizing (square).
        fix_monochrome (bool): Whether to invert MONOCHROME1 images.

    Returns:
        np.ndarray: The processed image (H, W, 3) in uint8 format.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT if available (handles windowing)
        data = apply_voi_lut(dicom.pixel_array, dicom)

        # Handle Monochrome
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data.astype(np.float32)
        data = data - np.min(data)
        data = data / (np.max(data) + 1e-6)
        data = (data * 255).astype(np.uint8)

        # Resize if requested
        if size is not None:
            data = cv2.resize(data, (size, size))

        # Convert to RGB (ResNet expects 3 channels)
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)

        return data
    except Exception as e:
        # Return a black image of correct size in case of error
        s = size if size is not None else 512
        return np.zeros((s, s, 3), dtype=np.uint8)


def create_mask_from_boxes(boxes_str, width, height):
    """
    Parses the boxes string and creates a binary mask.

    Args:
        boxes_str (str): String representation of list of dicts.
        width (int): Original image width.
        height (int): Original image height.

    Returns:
        np.ndarray: Binary mask (H, W).
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if pd.isna(boxes_str) or boxes_str == "":
        return mask

    try:
        boxes = ast.literal_eval(boxes_str)
        for box in boxes:
            x = int(box["x"])
            y = int(box["y"])
            w = int(box["width"])
            h = int(box["height"])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 1, -1)
    except:
        pass

    return mask


def get_bbox_from_mask(mask):
    """
    Extracts bounding boxes from a binary mask using contour detection.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        list: List of [xmin, ymin, xmax, ymax].
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter extremely small artifacts
        if w > 1 and h > 1:
            bboxes.append([x, y, x + w, y + h])
    return bboxes


def compute_iou(box1, box2):
    """
    Computes IoU between two boxes [x1, y1, x2, y2].
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


def calculate_map(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Calculates PASCAL VOC 2010 Mean Average Precision (mAP) for a single class.

    Args:
        pred_boxes: List of List of boxes for each image [[x1, y1, x2, y2], ...].
        pred_scores: List of List of scores for each image.
        gt_boxes: List of List of ground truth boxes for each image.
        iou_threshold: IoU threshold for TP.

    Returns:
        float: Average Precision score.
    """
    # Flatten all predictions
    all_preds = []
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box, score in zip(boxes, scores):
            all_preds.append({"img_idx": img_idx, "box": box, "score": score})

    # Sort by confidence descending
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track which GT boxes have been matched
    gt_matched = {i: [False] * len(boxes) for i, boxes in enumerate(gt_boxes)}
    n_pos = sum(len(boxes) for boxes in gt_boxes)

    if n_pos == 0:
        return 0.0

    for i, pred in enumerate(all_preds):
        img_idx = pred["img_idx"]
        pred_box = pred["box"]

        best_iou = 0.0
        best_gt_idx = -1

        # Find best matching GT box
        current_gt_boxes = gt_boxes[img_idx]
        for gt_idx, gt_box in enumerate(current_gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            if not gt_matched[img_idx][best_gt_idx]:
                tp[i] = 1
                gt_matched[img_idx][best_gt_idx] = True
            else:
                fp[i] = 1  # Already matched (Duplicate)
        else:
            fp[i] = 1  # False Positive

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / n_pos
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # PASCAL VOC 2010+ uses all-point interpolation
    # Append sentinel values
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))

    # Smooth precision curve (make it monotonically decreasing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return float(ap)


def format_prediction_string(label, conf, boxes):
    """
    Formats predictions into the competition string format.

    Args:
        label (str): Class label (e.g., 'opacity', 'none').
        conf (float): Confidence score.
        boxes (list): List of [xmin, ymin, xmax, ymax].

    Returns:
        str: Formatted prediction string.
    """
    if label == "none":
        return "none 1 0 0 1 1"

    pred_strings = []
    # If single box passed as list of coords
    if boxes and isinstance(boxes[0], (int, float)):
        boxes = [boxes]

    for box in boxes:
        pred_strings.append(
            f"{label} {conf:.4f} {int(box[0])} {int(box[1])} {int(box[2])} {int(box[3])}"
        )

    if not pred_strings:
        return "none 1 0 0 1 1"

    return " ".join(pred_strings)


def process_dataset(
    df, subset_name, load_cached_data=True, img_size=Config.IMG_SIZE, sample_size=None
):
    """
    Processes the dataset: reads images, creates masks, extracts labels.
    Implements caching using .npy files.

    Args:
        df (pd.DataFrame): Metadata dataframe.
        subset_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to try loading from cache.
        img_size (int): Target image size.
        sample_size (int, optional): Limit number of samples for debugging.

    Returns:
        tuple: (images, masks, labels) as numpy arrays.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Define cache paths
    cache_prefix = f"{subset_name}"
    if sample_size:
        cache_prefix += f"_{sample_size}"

    img_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_images.npy")
    mask_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_masks.npy")
    label_cache_path = os.path.join(Config.WORKING_DIR, f"{cache_prefix}_labels.npy")

    # Check cache
    if load_cached_data:
        if (
            os.path.exists(img_cache_path)
            and os.path.exists(mask_cache_path)
            and os.path.exists(label_cache_path)
        ):
            print(f"Loading cached data for {subset_name} from {Config.WORKING_DIR}...")
            images = np.load(img_cache_path)
            masks = np.load(mask_cache_path)
            labels = np.load(label_cache_path)
            return images, masks, labels

    print(f"Processing data for {subset_name}...")

    # Filter if sample_size is set (for debugging)
    if sample_size is not None and sample_size < len(df):
        df = df.iloc[:sample_size].copy()

    images = []
    masks = []
    labels = []

    for idx, row in df.iterrows():
        # 1. Read Image
        img_path = os.path.join(Config.INPUT_DIR, row["file_path"])
        img = read_dicom(img_path, size=img_size)
        images.append(img)

        # 2. Create Mask
        # Default empty mask
        mask = np.zeros((img_size, img_size, 1), dtype=np.uint8)

        if "boxes" in row and not pd.isna(row["boxes"]):
            try:
                # We need original dimensions to scale boxes correctly.
                # read_dicom returns resized image, but we need orig dims for mask creation.
                dcm = pydicom.dcmread(img_path, stop_before_pixels=True)
                orig_h, orig_w = dcm.Rows, dcm.Columns

                m = create_mask_from_boxes(row["boxes"], orig_w, orig_h)
                m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(m, axis=-1)
            except Exception as e:
                # Fallback to empty mask if dicom read fails or boxes invalid
                pass

        masks.append(mask)

        # 3. Labels
        if "Negative for Pneumonia" in row:
            # One-hot vector: [Neg, Typ, Ind, Atyp]
            l = [
                row["Negative for Pneumonia"],
                row["Typical Appearance"],
                row["Indeterminate Appearance"],
                row["Atypical Appearance"],
            ]
            labels.append(l)
        else:
            # Dummy label for test set
            labels.append([0, 0, 0, 0])

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)  # (N, H, W, 3)
    masks = np.array(masks, dtype=np.uint8)  # (N, H, W, 1)
    labels = np.array(labels, dtype=np.float32)  # (N, 4)

    # Save to cache
    print(f"Saving processed data to {Config.WORKING_DIR}...")
    np.save(img_cache_path, images)
    np.save(mask_cache_path, masks)
    np.save(label_cache_path, labels)

    return images, masks, labels
