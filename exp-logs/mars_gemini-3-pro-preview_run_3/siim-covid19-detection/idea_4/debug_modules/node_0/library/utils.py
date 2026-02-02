import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
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
    Custom collate function for object detection.

    Args:
        batch: List of tuples (image, target, image_id)
            - image: Tensor of shape (C, H, W)
            - target: Dict containing 'boxes', 'labels', 'study_labels'
            - image_id: String identifier

    Returns:
        images: Stacked tensor of shape (B, C, H, W)
        targets: List of target dictionaries (one per image)
        image_ids: List of image IDs
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    image_ids = [item[2] for item in batch]

    images = torch.stack(images, 0)

    return images, targets, image_ids


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

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


def box_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) of two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format

    Returns:
        iou: Tensor of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou


class MeanAveragePrecision:
    """
    Calculates the Mean Average Precision (mAP) at a specific IoU threshold
    using the PASCAL VOC 2010 method (all-points interpolation).
    """

    def __init__(self, num_classes=1, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        # Store predictions and ground truths
        # Format: list of dicts or lists
        self.predictions = []  # List of (boxes, scores, labels)
        self.ground_truths = []  # List of (boxes, labels)

    def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
        """
        Update state with a batch of predictions and ground truths.
        All inputs should be lists of tensors (one tensor per image).
        """
        for i in range(len(pred_boxes)):
            self.predictions.append(
                {
                    "boxes": pred_boxes[i].cpu(),
                    "scores": pred_scores[i].cpu(),
                    "labels": pred_labels[i].cpu(),
                }
            )
            self.ground_truths.append(
                {"boxes": gt_boxes[i].cpu(), "labels": gt_labels[i].cpu()}
            )

    def compute(self):
        """
        Compute mAP based on accumulated data.
        Returns:
            mAP: float
        """
        average_precisions = []

        # Iterate over each class
        for class_id in range(1, self.num_classes + 1):
            detections = []
            ground_truths = []

            # Collect all detections and GTs for this class
            for idx, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
                # Filter predictions by class
                mask_p = pred["labels"] == class_id
                if mask_p.sum() > 0:
                    d_boxes = pred["boxes"][mask_p]
                    d_scores = pred["scores"][mask_p]
                    for b, s in zip(d_boxes, d_scores):
                        detections.append((s.item(), b, idx))  # score, box, image_index

                # Filter ground truths by class
                mask_g = gt["labels"] == class_id
                if mask_g.sum() > 0:
                    g_boxes = gt["boxes"][mask_g]
                    # Keep track of which GTs have been detected
                    visited = torch.zeros(g_boxes.shape[0], dtype=torch.bool)
                    ground_truths.append(
                        {"boxes": g_boxes, "visited": visited, "image_index": idx}
                    )
                else:
                    # No GT for this class in this image
                    ground_truths.append(
                        {
                            "boxes": torch.empty((0, 4)),
                            "visited": torch.tensor([]),
                            "image_index": idx,
                        }
                    )

            # If no ground truths for this class, AP is 0 (unless no detections either, but usually 0)
            total_gt = sum([len(g["boxes"]) for g in ground_truths])
            if total_gt == 0:
                continue

            # Sort detections by score descending
            detections.sort(key=lambda x: x[0], reverse=True)

            TP = torch.zeros(len(detections))
            FP = torch.zeros(len(detections))

            # Create a map from image_index to the ground_truth object for O(1) access
            gt_map = {g["image_index"]: g for g in ground_truths}

            for i, (score, pred_box, img_idx) in enumerate(detections):
                gt_data = gt_map.get(img_idx)

                best_iou = 0
                best_gt_idx = -1

                if gt_data is not None and len(gt_data["boxes"]) > 0:
                    # Calculate IoU with all GT boxes in this image
                    # pred_box: (4,), gt_boxes: (M, 4)
                    # Expand pred_box to (1, 4) for box_iou
                    ious = box_iou(pred_box.unsqueeze(0), gt_data["boxes"]).squeeze(0)

                    if len(ious) > 0:
                        best_iou, best_gt_idx = torch.max(ious, 0)
                        best_iou = best_iou.item()
                        best_gt_idx = best_gt_idx.item()

                if best_iou >= self.iou_threshold:
                    if not gt_data["visited"][best_gt_idx]:
                        TP[i] = 1
                        gt_data["visited"][best_gt_idx] = True
                    else:
                        FP[i] = 1  # Duplicate detection
                else:
                    FP[i] = 1

            # Compute cumulative sums
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)

            recalls = TP_cumsum / (total_gt + 1e-6)
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

            # Add sentinel values for integration (0,0) and (1,0) are implicitly handled
            # by the smoothing algorithm usually, but standard VOC adds 0 and 1 to recall
            recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
            precisions = torch.cat(
                (torch.tensor([0.0]), precisions, torch.tensor([0.0]))
            )

            # Smooth precision (VOC 2010+)
            # Compute max precision to the right
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = torch.max(precisions[i], precisions[i + 1])

            # Integrate area under curve
            # Find indices where recall changes
            indices = torch.where(recalls[1:] != recalls[:-1])[0]

            ap = torch.sum(
                (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
            )
            average_precisions.append(ap.item())

        if not average_precisions:
            return 0.0

        return sum(average_precisions) / len(average_precisions)
