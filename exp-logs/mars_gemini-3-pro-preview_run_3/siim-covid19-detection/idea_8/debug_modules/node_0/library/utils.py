import torch
import numpy as np
import random
import os
from library.config import Config


class AverageMeter:
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


def seed_everything(seed=Config.SEED):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """
    Collate function for DETR-style training.

    Args:
        batch: List of tuples (image, target).
               image: Tensor of shape (C, H, W).
               target: Dict containing 'boxes', 'labels', etc.

    Returns:
        images: Stacked tensor of images (N, C, H, W).
        targets: List of target dictionaries.
    """
    # Filter out None values if any
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None

    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images (assuming they are resized to same dimensions via Letterbox)
    images = torch.stack(images, dim=0)

    return images, targets


def box_cxcywh_to_xyxy(x):
    """Converts bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Converts bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes.
    Boxes are in (x1, y1, x2, y2) format.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Computes the intersection over union of two sets of boxes.
    The boxes are expected in (x1, y1, x2, y2) format.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): The NxM matrix containing the pairwise IoU values.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def calculate_map(
    pred_boxes,
    pred_scores,
    pred_labels,
    gt_boxes,
    gt_labels,
    num_classes=1,
    iou_threshold=0.5,
):
    """
    Calculate mAP @ IoU > 0.5 using PASCAL VOC 2010 method (all-point interpolation).

    Args:
        pred_boxes (list of Tensor): List of predicted boxes (xyxy) for each image.
        pred_scores (list of Tensor): List of predicted scores for each image.
        pred_labels (list of Tensor): List of predicted labels for each image.
        gt_boxes (list of Tensor): List of GT boxes (xyxy) for each image.
        gt_labels (list of Tensor): List of GT labels for each image.
        num_classes (int): Number of object classes.
        iou_threshold (float): IoU threshold for matching.

    Returns:
        float: The mean Average Precision (mAP) across all classes.
    """
    average_precisions = []

    for c in range(num_classes):
        # 1. Collect all predictions and targets for this class
        all_preds = []
        total_gt = 0

        # We need to track which GT boxes have been matched to avoid double counting
        # Structure: list of boolean tensors, one per image
        matched_gt_masks = []

        for i in range(len(pred_boxes)):
            # Targets for class c in image i
            t_mask = gt_labels[i] == c
            n_gt = t_mask.sum().item()
            total_gt += n_gt
            matched_gt_masks.append(torch.zeros(n_gt, dtype=torch.bool))

            # Predictions for class c in image i
            p_mask = pred_labels[i] == c
            if p_mask.sum() > 0:
                p_boxes = pred_boxes[i][p_mask]
                p_scores = pred_scores[i][p_mask]

                # Store: [x1, y1, x2, y2, score, image_index]
                img_indices = torch.full_like(p_scores, i)
                preds_concat = torch.cat(
                    [p_boxes, p_scores.unsqueeze(1), img_indices.unsqueeze(1)], dim=1
                )
                all_preds.append(preds_concat)

        if total_gt == 0:
            continue

        if len(all_preds) == 0:
            average_precisions.append(0.0)
            continue

        # Concatenate all predictions and sort by score descending
        all_preds = torch.cat(all_preds, dim=0)
        sort_idx = torch.argsort(all_preds[:, 4], descending=True)
        all_preds = all_preds[sort_idx]

        tp = torch.zeros(len(all_preds))
        fp = torch.zeros(len(all_preds))

        # 2. Match predictions to GT
        for i in range(len(all_preds)):
            pred_box = all_preds[i, :4]
            img_idx = int(all_preds[i, 5].item())

            # Get GT boxes for this image and class
            gt_mask = gt_labels[img_idx] == c
            img_gt_boxes = gt_boxes[img_idx][gt_mask]

            if len(img_gt_boxes) == 0:
                fp[i] = 1
                continue

            # Calculate IoU between this prediction and all GT boxes in the image
            # pred_box: [4], img_gt_boxes: [M, 4]
            ious = box_iou(pred_box.unsqueeze(0), img_gt_boxes).squeeze(0)

            max_iou, max_idx = torch.max(ious, dim=0)

            if max_iou > iou_threshold:
                if not matched_gt_masks[img_idx][max_idx]:
                    tp[i] = 1
                    matched_gt_masks[img_idx][max_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection
            else:
                fp[i] = 1  # Poor localization

        # 3. Compute Precision and Recall
        tp_cumsum = torch.cumsum(tp, dim=0).cpu().numpy()
        fp_cumsum = torch.cumsum(fp, dim=0).cpu().numpy()

        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 4. PASCAL VOC 2010 AP Calculation
        # Add sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute precision envelope (monotonically decreasing)
        for j in range(mpre.size - 1, 0, -1):
            mpre[j - 1] = np.maximum(mpre[j - 1], mpre[j])

        # Integrate area under curve where recall changes
        i_indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1])

        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return 0.0

    return sum(average_precisions) / len(average_precisions)
