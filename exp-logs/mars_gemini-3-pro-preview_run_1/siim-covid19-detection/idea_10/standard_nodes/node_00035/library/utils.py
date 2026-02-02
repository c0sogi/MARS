import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask2bbox(mask, threshold=0.5):
    """
    Converts a probability mask or binary mask into bounding boxes.

    Args:
        mask (np.ndarray or torch.Tensor): The input mask (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: A list of bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Binarize mask
    binary_mask = (mask > threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Convert xywh to xmin, ymin, xmax, ymax
        boxes.append([x, y, x + w, y + h])

    return boxes


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two boxes.

    Args:
        box1 (list/array): [xmin, ymin, xmax, ymax]
        box2 (list/array): [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU value.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculates the Mean Average Precision (mAP) at a specific IoU threshold
    using the PASCAL VOC 2010 method (all-point interpolation).

    This implementation assumes a single detection class ('opacity').

    Args:
        pred_boxes (list): List of lists, where each inner list contains predictions
                           for one image. Each prediction is [xmin, ymin, xmax, ymax, score].
        true_boxes (list): List of lists, where each inner list contains ground truth
                           boxes for one image. Each box is [xmin, ymin, xmax, ymax].
        iou_threshold (float): The IoU threshold for a correct detection.

    Returns:
        float: The Average Precision (AP).
    """
    # Flatten all predictions into a single list of (score, image_idx, box)
    all_preds = []
    for img_idx, boxes in enumerate(pred_boxes):
        for box in boxes:
            # box is expected to be [x1, y1, x2, y2, score]
            if len(box) >= 5:
                all_preds.append((box[4], img_idx, box[:4]))

    # Sort predictions by confidence score (descending)
    all_preds.sort(key=lambda x: x[0], reverse=True)

    # Initialize arrays for True Positives (tp) and False Positives (fp)
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Keep track of which GT boxes have been matched to avoid double counting
    # true_boxes_matched[img_idx] is a boolean array of size len(gt_boxes_in_img)
    true_boxes_matched = {}
    total_positives = 0

    for img_idx, boxes in enumerate(true_boxes):
        true_boxes_matched[img_idx] = np.zeros(len(boxes), dtype=bool)
        total_positives += len(boxes)

    if total_positives == 0:
        # If there are no ground truth objects in the entire dataset
        return 0.0

    # Match predictions to ground truth
    for i, (score, img_idx, pred_box) in enumerate(all_preds):
        gt_boxes = true_boxes[img_idx]
        best_iou = -1.0
        best_gt_idx = -1

        # Find the best matching ground truth box
        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Determine if it's a TP or FP
        if best_iou >= iou_threshold:
            if not true_boxes_matched[img_idx][best_gt_idx]:
                tp[i] = 1
                true_boxes_matched[img_idx][best_gt_idx] = True
            else:
                fp[i] = 1  # Duplicate detection for the same object
        else:
            fp[i] = 1

    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Compute Precision and Recall
    recalls = tp_cumsum / total_positives
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Compute AP using VOC 2010 method (Interpolation)
    # Append sentinel values to handle edge cases
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (maximum precision to the right)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate Area Under Curve
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum rectangular areas
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)
