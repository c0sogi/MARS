import numpy as np
import torch
from torchvision.ops import box_iou
from library.config import Config


def compute_ap(recalls, precisions):
    """
    Compute Average Precision using the PASCAL VOC 2010 method (interpolated precision).

    Args:
        recalls (np.array): Array of recall values.
        precisions (np.array): Array of precision values.

    Returns:
        float: The Average Precision (AP).
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (make it monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve
    # Look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_map(predictions, targets, iou_threshold=Config.IOU_THRESHOLD):
    """
    Calculates the mean Average Precision (mAP) at a specific IoU threshold.

    Args:
        predictions (list): List of dicts, each containing:
            - 'boxes': Tensor of shape (N, 4)
            - 'scores': Tensor of shape (N,)
            - 'labels': Tensor of shape (N,)
        targets (list): List of dicts, each containing:
            - 'boxes': Tensor of shape (M, 4)
            - 'labels': Tensor of shape (M,)
        iou_threshold (float): IoU threshold for considering a positive match.

    Returns:
        dict: Dictionary containing 'map' (float) and 'class_aps' (dict of class_id -> ap).
    """
    # Identify unique classes to evaluate. We exclude background (0).
    # Config.CLASS_MAPPING maps names to IDs (1, 2, 3).
    valid_classes = sorted([v for k, v in Config.CLASS_MAPPING.items() if v != 0])

    class_aps = {}

    for class_id in valid_classes:
        pred_boxes_list = []
        pred_scores_list = []
        pred_img_indices = []

        gt_boxes_map = {}
        gt_matched_map = {}
        num_gt_instances = 0

        # Organize data by image index to facilitate matching
        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            # 1. Process Ground Truth
            # Move to CPU for processing
            tgt_labels = target["labels"].cpu()
            tgt_boxes = target["boxes"].cpu()

            # Filter for current class
            gt_mask = tgt_labels == class_id
            class_gt_boxes = tgt_boxes[gt_mask]

            gt_boxes_map[img_idx] = class_gt_boxes
            gt_matched_map[img_idx] = np.zeros(len(class_gt_boxes), dtype=bool)
            num_gt_instances += len(class_gt_boxes)

            # 2. Process Predictions
            p_labels = pred["labels"].cpu()
            p_boxes = pred["boxes"].cpu()
            p_scores = pred["scores"].cpu()

            # Filter for current class
            pred_mask = p_labels == class_id
            class_pred_boxes = p_boxes[pred_mask]
            class_pred_scores = p_scores[pred_mask]

            if len(class_pred_boxes) > 0:
                pred_boxes_list.append(class_pred_boxes)
                pred_scores_list.append(class_pred_scores)
                pred_img_indices.extend([img_idx] * len(class_pred_boxes))

        # Handle Edge Cases
        if num_gt_instances == 0:
            # If no ground truth exists for this class
            if len(pred_boxes_list) > 0:
                # Predictions exist but no GT -> Precision is 0, AP is 0
                class_aps[class_id] = 0.0
            else:
                # No GT and no Predictions -> Class not present in this set
                # We skip it to avoid skewing the mean with undefined values
                pass
            continue

        if len(pred_boxes_list) == 0:
            # GT exists but no predictions -> AP is 0
            class_aps[class_id] = 0.0
            continue

        # Concatenate all predictions for this class across all images
        all_pred_boxes = torch.cat(pred_boxes_list, dim=0)
        all_pred_scores = torch.cat(pred_scores_list, dim=0)
        all_pred_img_indices = np.array(pred_img_indices)

        # Sort by confidence score (descending)
        sort_indices = torch.argsort(all_pred_scores, descending=True).numpy()
        sorted_boxes = all_pred_boxes[sort_indices]
        sorted_img_indices = all_pred_img_indices[sort_indices]

        tp = np.zeros(len(sorted_boxes))
        fp = np.zeros(len(sorted_boxes))

        # Match predictions to GT
        for i in range(len(sorted_boxes)):
            box = sorted_boxes[i].unsqueeze(0)  # (1, 4)
            img_idx = sorted_img_indices[i]

            gt_boxes = gt_boxes_map[img_idx]

            if len(gt_boxes) == 0:
                fp[i] = 1
                continue

            # Compute IoU between this prediction and all GT boxes in the image
            ious = box_iou(box, gt_boxes).squeeze(0)  # (num_gt,)

            # Find the best matching GT
            max_iou, max_idx = torch.max(ious, dim=0)
            max_iou = max_iou.item()
            max_idx = max_idx.item()

            if max_iou >= iou_threshold:
                if not gt_matched_map[img_idx][max_idx]:
                    tp[i] = 1
                    gt_matched_map[img_idx][max_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection (already matched)
            else:
                fp[i] = 1  # IoU too low

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt_instances
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        ap = compute_ap(recalls, precisions)
        class_aps[class_id] = ap

    # Compute mean AP
    if class_aps:
        m_ap = sum(class_aps.values()) / len(class_aps)
    else:
        m_ap = 0.0

    return {"map": m_ap, "class_aps": class_aps}
