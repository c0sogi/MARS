import os
import random
import numpy as np
import torch
import cv2
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for reproducibility across various libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_boxes_from_mask(mask, threshold=0.5):
    """
    Extracts bounding boxes from a probability mask using contour detection.

    Args:
        mask (np.ndarray or torch.Tensor): Probability mask of shape (H, W).
        threshold (float): Threshold to binarize the mask.

    Returns:
        list: List of bounding boxes [xmin, ymin, xmax, ymax].
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
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # x, y is top-left, w, h is width, height
        # Convert to xmin, ymin, xmax, ymax
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])

    return boxes


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    Boxes are expected in [xmin, ymin, xmax, ymax] format.
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


def calculate_map(
    pred_boxes, pred_scores, pred_class_ids, gt_boxes, gt_class_ids, iou_threshold=0.5
):
    """
    Calculates PASCAL VOC 2010 mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        pred_boxes (list): List of lists of boxes for each image.
        pred_scores (list): List of lists of scores for each image.
        pred_class_ids (list): List of lists of class IDs for each image.
        gt_boxes (list): List of lists of GT boxes for each image.
        gt_class_ids (list): List of lists of GT class IDs for each image.
        iou_threshold (float): IoU threshold for considering a detection a True Positive.

    Returns:
        float: The mAP score.
    """
    # Identify all unique classes in the dataset
    unique_classes = set()
    for classes in gt_class_ids:
        unique_classes.update(classes)
    for classes in pred_class_ids:
        unique_classes.update(classes)

    if not unique_classes:
        return 0.0

    aps = []

    for cls_id in unique_classes:
        detections = []
        ground_truths = {}
        n_pos = 0

        # 1. Organize detections and ground truths for this class
        for i in range(len(pred_boxes)):
            # Detections
            p_boxes = pred_boxes[i]
            p_scores = pred_scores[i]
            p_ids = pred_class_ids[i]

            for j in range(len(p_boxes)):
                if p_ids[j] == cls_id:
                    detections.append(
                        {
                            "confidence": float(p_scores[j]),
                            "box": p_boxes[j],
                            "image_id": i,
                        }
                    )

            # Ground Truths
            g_boxes = gt_boxes[i]
            g_ids = gt_class_ids[i]
            img_gt = []
            for j in range(len(g_boxes)):
                if g_ids[j] == cls_id:
                    img_gt.append({"box": g_boxes[j], "used": False})

            ground_truths[i] = img_gt
            n_pos += len(img_gt)

        # If there are no ground truth objects for this class, we skip it in the mean calculation
        # (Standard VOC practice: if a class is not in GT, it doesn't contribute to mAP)
        if n_pos == 0:
            continue

        # 2. Sort detections by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        # 3. Match detections to ground truths
        for d in range(len(detections)):
            detection = detections[d]
            image_id = detection["image_id"]
            box = detection["box"]

            best_iou = 0
            best_gt_idx = -1

            gts = ground_truths[image_id]

            # Find best matching GT
            for g in range(len(gts)):
                iou = calculate_iou(box, gts[g]["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g

            # Check threshold and usage
            if best_iou >= iou_threshold:
                if not gts[best_gt_idx]["used"]:
                    TP[d] = 1
                    gts[best_gt_idx]["used"] = True
                else:
                    FP[d] = 1
            else:
                FP[d] = 1

        # 4. Compute Precision and Recall
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)

        rec = acc_TP / n_pos
        prec = acc_TP / (acc_TP + acc_FP + 1e-6)  # Avoid div by zero

        # 5. Compute AP using VOC 2010 (All-point interpolation)
        # Add sentinel values
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Make precision monotonically decreasing
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Calculate Area Under Curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        aps.append(ap)

    if not aps:
        return 0.0

    return np.mean(aps)


def format_prediction_string(labels, confidences, boxes):
    """
    Formats predictions into the submission string format.
    Format: "label confidence xmin ymin xmax ymax ..."

    Args:
        labels (list): List of class labels (strings).
        confidences (list): List of confidence scores (floats).
        boxes (list): List of bounding boxes [xmin, ymin, xmax, ymax].

    Returns:
        str: The formatted prediction string.
    """
    pred_strings = []
    for label, conf, box in zip(labels, confidences, boxes):
        if box is None:
            # Fallback for empty box if passed explicitly (e.g. for study labels)
            b_str = "0 0 1 1"
        else:
            # Ensure coordinates are standard
            b_str = f"{box[0]} {box[1]} {box[2]} {box[3]}"

        pred_strings.append(f"{label} {conf} {b_str}")

    return " ".join(pred_strings)
