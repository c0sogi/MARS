import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, alpha=1.0, device="cuda"):
    """
    Returns mixed inputs, permutation indices, and lambda constant for MixUp.

    Args:
        x (torch.Tensor): Input batch of images.
        alpha (float): MixUp hyperparameter for Beta distribution.
        device (str): Device to store indices on.

    Returns:
        mixed_x (torch.Tensor): Mixed input images.
        index (torch.Tensor): Permutation indices used to shuffle the batch.
        lam (float): Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the MixUp loss.

    Args:
        criterion (callable): Loss function.
        pred (torch.Tensor): Model predictions.
        y_a (torch.Tensor): Targets for the original batch.
        y_b (torch.Tensor): Targets for the shuffled batch.
        lam (float): Mixing coefficient.

    Returns:
        torch.Tensor: Weighted loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mask2bbox(mask, threshold=0.5):
    """
    Converts a probability mask into bounding boxes using contour extraction.

    Args:
        mask (np.ndarray): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        # cv2.boundingRect returns x, y, w, h
        x, y, w, h = cv2.boundingRect(c)
        # Convert to xmin, ymin, xmax, ymax
        boxes.append([x, y, x + w, y + h])

    return boxes


def _calculate_iou(box1, box2):
    """
    Calculates IoU between two bounding boxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def _compute_ap_voc2010(recall, precision):
    """
    Computes Average Precision using PASCAL VOC 2010 method (all-point interpolation).
    """
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_map_score(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    iou_threshold=0.5,
    num_classes=Config.NUM_CLASSES,
):
    """
    Calculates PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5.

    Args:
        pred_boxes (list of list): List of predicted boxes per image.
        pred_scores (list of list): List of predicted scores per image.
        pred_labels (list of list): List of predicted labels per image.
        gt_boxes (list of list): List of ground truth boxes per image.
        gt_labels (list of list): List of ground truth labels per image.
        iou_threshold (float): IoU threshold for matching.
        num_classes (int): Number of classes.

    Returns:
        float: mAP score.
    """
    average_precisions = []

    # Calculate AP for each class
    for c in range(num_classes):
        dect_boxes = []
        dect_scores = []
        dect_image_ids = []

        gt_class_boxes = {}
        gt_class_counts = 0

        # Flatten predictions and organize GT for the current class
        for i in range(len(gt_boxes)):
            # Ground Truth
            current_gt_boxes = []
            if len(gt_labels[i]) > 0:
                indices = [k for k, label in enumerate(gt_labels[i]) if label == c]
                if indices:
                    current_gt_boxes = [gt_boxes[i][k] for k in indices]

            gt_class_boxes[i] = {
                "boxes": np.array(current_gt_boxes),
                "matched": np.zeros(len(current_gt_boxes), dtype=bool),
            }
            gt_class_counts += len(current_gt_boxes)

            # Predictions
            if len(pred_labels[i]) > 0:
                indices = [k for k, label in enumerate(pred_labels[i]) if label == c]
                if indices:
                    dect_boxes.extend([pred_boxes[i][k] for k in indices])
                    dect_scores.extend([pred_scores[i][k] for k in indices])
                    dect_image_ids.extend([i] * len(indices))

        if gt_class_counts == 0:
            continue

        dect_boxes = np.array(dect_boxes)
        dect_scores = np.array(dect_scores)
        dect_image_ids = np.array(dect_image_ids)

        # Sort detections by score descending
        sorted_indices = np.argsort(-dect_scores)
        dect_boxes = dect_boxes[sorted_indices]
        dect_image_ids = dect_image_ids[sorted_indices]

        tp = np.zeros(len(dect_boxes))
        fp = np.zeros(len(dect_boxes))

        # Match predictions to GT
        for i in range(len(dect_boxes)):
            image_id = dect_image_ids[i]
            pred_box = dect_boxes[i]

            gt_data = gt_class_boxes[image_id]
            gt_boxes_img = gt_data["boxes"]
            matched = gt_data["matched"]

            best_iou = -1.0
            best_gt_idx = -1

            if len(gt_boxes_img) > 0:
                # Calculate IoU with all GT boxes in the image
                # Vectorized IoU calculation could be faster, but loop is clear
                for j, gt_box in enumerate(gt_boxes_img):
                    iou = _calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_iou >= iou_threshold:
                if not matched[best_gt_idx]:
                    tp[i] = 1.0
                    matched[best_gt_idx] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / gt_class_counts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = _compute_ap_voc2010(recalls, precisions)
        average_precisions.append(ap)

    if not average_precisions:
        return 0.0

    return sum(average_precisions) / len(average_precisions)
