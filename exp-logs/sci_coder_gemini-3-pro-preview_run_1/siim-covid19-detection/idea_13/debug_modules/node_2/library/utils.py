import cv2
import numpy as np
import torch
from library.config import seed_everything


def mask2bbox(mask, threshold=0.5):
    """
    Post-processing: Converts a predicted probability mask into bounding boxes.

    Args:
        mask (np.ndarray or torch.Tensor): Output mask from the model.
                                           Shape (H, W) or (1, H, W).
                                           Values should be probabilities [0, 1].
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of bounding boxes in format [xmin, ymin, xmax, ymax].
    """
    # Convert tensor to numpy if necessary
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Remove channel dimension if present
    if mask.ndim == 3:
        mask = mask[0]

    # Binarize mask
    mask_bin = (mask > threshold).astype(np.uint8)

    # Find contours (OpenCV 4.x returns contours, hierarchy)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out extremely small noise artifacts
        if w > 0 and h > 0:
            # Format: xmin, ymin, xmax, ymax
            boxes.append([x, y, x + w, y + h])

    return boxes


def calculate_ap(rec, prec):
    """
    Compute VOC AP given precision and recall using the 2010 metric (all-point interpolation).

    Args:
        rec (np.array): Recall array.
        prec (np.array): Precision array.

    Returns:
        float: Average Precision.
    """
    # Append sentinel values at both ends
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve
    # Find indices where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (Delta Recall) * (Interpolated Precision)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(
    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5
):
    """
    Calculate Mean Average Precision (mAP) at a specific IoU threshold (PASCAL VOC 2010).

    Args:
        pred_boxes (list): List of lists of boxes for each image [[x1, y1, x2, y2], ...].
        pred_scores (list): List of lists of scores for each image.
        pred_labels (list): List of lists of labels for each image.
        gt_boxes (list): List of lists of GT boxes for each image.
        gt_labels (list): List of lists of GT labels for each image.
        iou_threshold (float): IoU threshold for considering a detection positive.

    Returns:
        float: mAP score.
    """
    # Identify all unique classes in the ground truth
    unique_classes = set()
    for labels in gt_labels:
        unique_classes.update(labels)

    if not unique_classes:
        return 0.0

    aps = []

    for cls in unique_classes:
        dect_boxes = []
        dect_scores = []
        dect_img_ids = []

        npos = 0
        gts = {}  # Map image_index -> numpy array of boxes for this class

        # 1. Collect GTs and Predictions for this specific class
        for i in range(len(gt_boxes)):
            # Filter GTs for this class
            cls_gt_boxes = [b for b, l in zip(gt_boxes[i], gt_labels[i]) if l == cls]
            gts[i] = np.array(cls_gt_boxes)
            npos += len(cls_gt_boxes)

            # Filter Predictions for this class
            cls_pred_boxes = [
                b for b, l in zip(pred_boxes[i], pred_labels[i]) if l == cls
            ]
            cls_pred_scores = [
                s for s, l in zip(pred_scores[i], pred_labels[i]) if l == cls
            ]

            for b, s in zip(cls_pred_boxes, cls_pred_scores):
                dect_boxes.append(b)
                dect_scores.append(s)
                dect_img_ids.append(i)

        if npos == 0:
            continue

        if not dect_boxes:
            aps.append(0.0)
            continue

        # Convert to numpy for vectorized operations
        dect_boxes = np.array(dect_boxes)
        dect_scores = np.array(dect_scores)
        dect_img_ids = np.array(dect_img_ids)

        # 2. Sort predictions by confidence score (descending)
        sorted_indices = np.argsort(-dect_scores)
        dect_boxes = dect_boxes[sorted_indices]
        dect_img_ids = dect_img_ids[sorted_indices]

        nd = len(dect_boxes)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # Track which GTs have been detected to handle multiple detections of same object
        det_gt_status = {i: np.zeros(len(gts[i])) for i in gts}

        # 3. Match predictions to GTs
        for d in range(nd):
            img_id = dect_img_ids[d]
            bb = dect_boxes[d]

            gt_bbs = gts[img_id]

            ovmax = -np.inf
            jmax = -1

            if len(gt_bbs) > 0:
                # Calculate IoU between bb and all gt_bbs
                ixmin = np.maximum(gt_bbs[:, 0], bb[0])
                iymin = np.maximum(gt_bbs[:, 1], bb[1])
                ixmax = np.minimum(gt_bbs[:, 2], bb[2])
                iymax = np.minimum(gt_bbs[:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin, 0.0)
                ih = np.maximum(iymax - iymin, 0.0)
                inters = iw * ih

                uni = (
                    (bb[2] - bb[0]) * (bb[3] - bb[1])
                    + (gt_bbs[:, 2] - gt_bbs[:, 0]) * (gt_bbs[:, 3] - gt_bbs[:, 1])
                    - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            # Assign detection
            if ovmax > iou_threshold:
                if det_gt_status[img_id][jmax] == 0:
                    tp[d] = 1.0
                    det_gt_status[img_id][jmax] = 1.0
                else:
                    fp[d] = 1.0  # Duplicate detection
            else:
                fp[d] = 1.0  # False positive (IoU too low or no GT)

        # 4. Compute Precision and Recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # Avoid divide by zero
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        # 5. Compute AP
        ap = calculate_ap(rec, prec)
        aps.append(ap)

    return np.mean(aps)
