import os
import random
import numpy as np
import torch
import cv2
import struct
import pandas as pd
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_dicom_image(path):
    """
    Reads a DICOM file using a custom binary parser to bypass missing libraries.
    Performs semantic normalization (inverts MONOCHROME1) and returns original dimensions.

    Args:
        path (str): Path to the .dicom file.

    Returns:
        tuple: (image_array, (original_height, original_width))
               image_array is a float32 numpy array.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        # Robust fallback returning a blank image
        return np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32), (
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        )

    # 1. Check Photometric Interpretation for Inversion (Semantic Normalization)
    # Search in the header (first 2000 bytes)
    header_chunk = data[:2000]
    invert = b"MONOCHROME1" in header_chunk

    # 2. Parse Dimensions (Rows/Cols)
    # Heuristic: Look for Explicit VR 'US' (Unsigned Short) tags for Rows (0028,0010) and Cols (0028,0011)
    # Pattern: Tag(4) + VR(2) + Length(2) = 8 bytes
    # 0028,0010 (LE) -> b'\x28\x00\x10\x00', VR 'US' -> b'\x55\x53', Len 2 -> b'\x02\x00'

    rows = 0
    cols = 0

    row_sig = b"\x28\x00\x10\x00\x55\x53\x02\x00"
    col_sig = b"\x28\x00\x11\x00\x55\x53\x02\x00"

    row_idx = data.find(row_sig)
    if row_idx != -1:
        rows = struct.unpack("<H", data[row_idx + 8 : row_idx + 10])[0]

    col_idx = data.find(col_sig)
    if col_idx != -1:
        cols = struct.unpack("<H", data[col_idx + 8 : col_idx + 10])[0]

    # 3. Find Pixel Data
    # Tag (7FE0, 0010) -> b'\xe0\x7f\x10\x00'
    pix_idx = data.find(b"\xe0\x7f\x10\x00")

    img = None

    if pix_idx != -1:
        # Determine offset to pixel data based on VR
        # Look at the 2 bytes after the tag
        vr = data[pix_idx + 4 : pix_idx + 6]

        offset = 0
        if vr in [b"OB", b"OW"]:
            # Explicit VR: Tag(4) + VR(2) + Reserved(2) + Length(4) = 12 bytes
            offset = 12
        else:
            # Implicit VR: Tag(4) + Length(4) = 8 bytes
            offset = 8

        pixel_data = data[pix_idx + offset :]

        # Attempt 1: Raw 16-bit Read (Most common for CXR)
        if rows > 0 and cols > 0:
            expected_bytes = rows * cols * 2
            if len(pixel_data) >= expected_bytes:
                try:
                    arr = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint16)
                    img = arr.reshape((rows, cols)).astype(np.float32)
                except Exception:
                    pass

            # Attempt 2: Raw 8-bit Read
            if img is None:
                expected_bytes = rows * cols
                if len(pixel_data) >= expected_bytes:
                    try:
                        arr = np.frombuffer(pixel_data[:expected_bytes], dtype=np.uint8)
                        img = arr.reshape((rows, cols)).astype(np.float32)
                    except Exception:
                        pass

        # Attempt 3: OpenCV Imdecode (Encapsulated/Compressed Data)
        # If raw read failed or dimensions unknown, treat payload as an image stream (e.g. JPEG)
        if img is None:
            try:
                arr = np.frombuffer(pixel_data, dtype=np.uint8)
                decoded = cv2.imdecode(arr, -1)  # -1 = Unchanged
                if decoded is not None:
                    img = decoded.astype(np.float32)
                    if len(img.shape) == 3:
                        img = img[:, :, 0]  # Take first channel if RGB
                    rows, cols = img.shape
            except Exception:
                pass

    # 4. Final Processing & Fallback
    if img is None:
        rows = 1024 if rows == 0 else rows
        cols = 1024 if cols == 0 else cols
        img = np.zeros((rows, cols), dtype=np.float32)

    # Invert if MONOCHROME1 (White=0 -> Black=0)
    if invert:
        max_val = np.max(img)
        if max_val > 0:
            img = max_val - img

    return img, (rows, cols)


def calculate_iou(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes.
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def calculate_ap(recall, precision):
    """
    Compute AP using PASCAL VOC 2010+ method (all points integration).
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute convex hull
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    num_classes=14,
    iou_threshold=0.4,
):
    """
    Calculates mAP @ IoU > iou_threshold for thoracic findings.

    Args:
        pred_boxes (list of np.array): List of predicted boxes for each image.
        pred_scores (list of np.array): List of scores.
        pred_labels (list of np.array): List of labels.
        gt_boxes (list of np.array): List of GT boxes.
        gt_labels (list of np.array): List of GT labels.
        num_classes (int): Number of classes.
        iou_threshold (float): IoU threshold for TP.

    Returns:
        dict: {'mAP': float, 'AP_per_class': dict}
    """
    aps = []
    ap_per_class = {}

    # Evaluate classes 0-13 (Findings). Class 14 (No finding) is excluded from box mAP.
    eval_classes = [c for c in range(num_classes) if c != Config.NO_FINDING_CLASS_ID]

    for cls_id in eval_classes:
        cls_preds = []
        cls_gts = []

        # Aggregate all preds and GTs for this class across all images
        for i in range(len(pred_boxes)):
            # Filter Preds
            p_mask = pred_labels[i] == cls_id
            p_box = pred_boxes[i][p_mask]
            p_score = pred_scores[i][p_mask]

            # Filter GTs
            g_mask = gt_labels[i] == cls_id
            g_box = gt_boxes[i][g_mask]

            # Add image_id index to track which image they belong to
            for b, s in zip(p_box, p_score):
                cls_preds.append({"bbox": b, "score": s, "img_idx": i})

            # Keep track of GTs per image
            cls_gts.append(
                {"bboxes": g_box, "matched": np.zeros(len(g_box), dtype=bool)}
            )

        total_gt = sum([len(x["bboxes"]) for x in cls_gts])
        if total_gt == 0:
            continue

        # Sort predictions by score descending
        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, pred in enumerate(cls_preds):
            img_idx = pred["img_idx"]
            pred_box = pred["bbox"]

            gt_data = cls_gts[img_idx]
            gt_boxes_img = gt_data["bboxes"]
            gt_matched = gt_data["matched"]

            if len(gt_boxes_img) > 0:
                # Compute IoU with all GTs in this image
                ious = calculate_iou(pred_box[None, :], gt_boxes_img)[0]
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]

                if max_iou > iou_threshold:
                    if not gt_matched[max_iou_idx]:
                        tp[i] = 1.0
                        gt_matched[max_iou_idx] = True
                    else:
                        fp[i] = 1.0  # Duplicate detection
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute Precision/Recall
        acc_tp = np.cumsum(tp)
        acc_fp = np.cumsum(fp)

        recall = acc_tp / total_gt
        precision = acc_tp / (acc_tp + acc_fp + 1e-6)

        ap = calculate_ap(recall, precision)
        aps.append(ap)
        ap_per_class[Config.CLASS_ID_TO_NAME.get(cls_id, str(cls_id))] = ap

    mAP = np.mean(aps) if aps else 0.0
    return {"mAP": mAP, "AP_per_class": ap_per_class}
