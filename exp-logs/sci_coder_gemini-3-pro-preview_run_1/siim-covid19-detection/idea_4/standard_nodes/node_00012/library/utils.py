import os
import random
import numpy as np
import torch
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom(path, size=None, fix_monochrome=True):
    """
    Reads a DICOM file, handles VOI LUT and Monochrome inversion,
    normalizes to 0-255, resizes, and converts to 3-channel RGB.

    Args:
        path (str): Path to the DICOM file.
        size (int, optional): Target size for resizing (e.g., 512).
        fix_monochrome (bool): Whether to invert MONOCHROME1 images.

    Returns:
        np.ndarray: Image array of shape (size, size, 3) with uint8 values.
    """
    try:
        dicom = pydicom.dcmread(path)

        # Apply VOI LUT if available, otherwise get pixel data
        try:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception:
            data = dicom.pixel_array

        # Handle MONOCHROME1 (where 0 is white) -> convert to MONOCHROME2 (0 is black)
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize to 0-255
        data = data.astype(np.float32)
        data = data - np.min(data)
        max_val = np.max(data)
        if max_val > 0:
            data = data / max_val
        data = (data * 255).astype(np.uint8)

        # Resize
        if size is not None:
            data = cv2.resize(data, (size, size))

        # Convert to 3 channels (RGB)
        data = np.stack([data, data, data], axis=-1)

        return data

    except Exception as e:
        # Fallback for corrupt or unreadable files (return black image)
        print(f"Error reading DICOM {path}: {e}")
        if size is None:
            size = 512
        return np.zeros((size, size, 3), dtype=np.uint8)


def mask2box(mask):
    """
    Converts a binary segmentation mask into a list of bounding boxes.

    Args:
        mask (np.ndarray): Binary mask (H, W).

    Returns:
        list: List of bounding boxes [xmin, ymin, xmax, ymax].
    """
    boxes = []
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter small artifacts if necessary, but generally keep all
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])

    return boxes


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    Boxes are [xmin, ymin, xmax, ymax].
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


def calculate_map(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
    """
    Calculates the PASCAL VOC 2010 mean Average Precision (mAP) for a single class (Opacity).

    Args:
        pred_boxes (list): List of lists, where pred_boxes[i] is a list of [xmin, ymin, xmax, ymax] for image i.
        pred_scores (list): List of lists, where pred_scores[i] is a list of floats for image i.
        true_boxes (list): List of lists, where true_boxes[i] is a list of [xmin, ymin, xmax, ymax] for image i.
        iou_threshold (float): IoU threshold for a match.

    Returns:
        float: The mAP score.
    """
    # Flatten all predictions: (score, image_idx, box_idx)
    detections = []
    for img_idx, (boxes, scores) in enumerate(zip(pred_boxes, pred_scores)):
        for box_idx, (box, score) in enumerate(zip(boxes, scores)):
            detections.append((score, img_idx, box))

    # Sort detections by confidence descending
    detections.sort(key=lambda x: x[0], reverse=True)

    # Track matched ground truths
    # matched_gt[img_idx] is a set of indices of matched true boxes
    matched_gt = {i: set() for i in range(len(true_boxes))}

    # Count total positives
    total_positives = sum(len(boxes) for boxes in true_boxes)

    if total_positives == 0:
        return (
            0.0 if len(detections) > 0 else 1.0
        )  # If no GT and no Preds, perfect. If Preds exist, 0.

    TP = np.zeros(len(detections))
    FP = np.zeros(len(detections))

    for i, (score, img_idx, pred_box) in enumerate(detections):
        gt_boxes_img = true_boxes[img_idx]

        best_iou = 0
        best_gt_idx = -1

        # Find best matching GT box
        for gt_idx, gt_box in enumerate(gt_boxes_img):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Assign detection
        if best_iou >= iou_threshold:
            if best_gt_idx not in matched_gt[img_idx]:
                TP[i] = 1
                matched_gt[img_idx].add(best_gt_idx)
            else:
                FP[i] = 1  # Already matched (duplicate detection)
        else:
            FP[i] = 1

    # Compute cumulative TP and FP
    TP_cum = np.cumsum(TP)
    FP_cum = np.cumsum(FP)

    recalls = TP_cum / total_positives
    precisions = TP_cum / (TP_cum + FP_cum + 1e-6)

    # PASCAL VOC 2010 AP Calculation (Every-point interpolation)
    # Append sentinel values for integration
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision for any recall >= current recall (smoothing)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate area under curve
    # Find indices where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]

    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap
