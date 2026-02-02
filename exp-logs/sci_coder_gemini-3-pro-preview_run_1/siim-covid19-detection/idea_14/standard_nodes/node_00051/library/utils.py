import os
import random
import numpy as np
import torch
import torchvision


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking losses and metrics during training.
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


def calculate_map(preds, targets, iou_threshold=0.5, num_classes=1):
    """
    Calculates the mean Average Precision (mAP) according to PASCAL VOC 2010 standards.

    Args:
        preds (List[Dict]): List of prediction dictionaries for each image.
                            Each dict must contain:
                            - 'boxes': Tensor of shape (N, 4)
                            - 'scores': Tensor of shape (N,)
                            - 'labels': Tensor of shape (N,)
        targets (List[Dict]): List of ground truth dictionaries for each image.
                              Each dict must contain:
                              - 'boxes': Tensor of shape (M, 4)
                              - 'labels': Tensor of shape (M,)
        iou_threshold (float): IoU threshold for considering a prediction a True Positive.
        num_classes (int): Number of classes to calculate AP for.

    Returns:
        float: The mean Average Precision across all classes.
    """
    average_precisions = []

    # Iterate over each class
    for class_idx in range(num_classes):
        class_preds = []
        class_targets = {}
        total_gt_boxes = 0

        # 1. Collect all predictions and targets for this class across all images
        for image_idx, (pred, target) in enumerate(zip(preds, targets)):
            # Process Targets
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            # Filter GT for current class
            mask_gt = gt_labels == class_idx
            gt_boxes_c = gt_boxes[mask_gt]

            # Store GT boxes and a 'visited' flag for matching
            class_targets[image_idx] = {
                "boxes": gt_boxes_c,
                "visited": torch.zeros(len(gt_boxes_c), dtype=torch.bool),
            }
            total_gt_boxes += len(gt_boxes_c)

            # Process Predictions
            p_boxes = pred["boxes"]
            p_scores = pred["scores"]
            p_labels = pred["labels"]

            # Filter Preds for current class
            mask_p = p_labels == class_idx
            p_boxes_c = p_boxes[mask_p]
            p_scores_c = p_scores[mask_p]

            # Add to list: [score, image_idx, box_tensor]
            for i in range(len(p_boxes_c)):
                class_preds.append([float(p_scores_c[i]), image_idx, p_boxes_c[i]])

        # If no ground truth exists for this class, AP is 0.0 (or undefined, but effectively 0 contribution)
        if total_gt_boxes == 0:
            average_precisions.append(0.0)
            continue

        # 2. Sort predictions by confidence score (descending)
        class_preds.sort(key=lambda x: x[0], reverse=True)

        # 3. Calculate TP and FP
        tps = torch.zeros(len(class_preds))
        fps = torch.zeros(len(class_preds))

        for i, (score, image_idx, pred_box) in enumerate(class_preds):
            gt_data = class_targets[image_idx]
            gt_boxes = gt_data["boxes"]
            visited = gt_data["visited"]

            # If no GT boxes for this image, it's a False Positive
            if len(gt_boxes) == 0:
                fps[i] = 1
                continue

            # Calculate IoU between this prediction and all GT boxes in the image
            # pred_box needs to be (1, 4) for box_iou
            ious = torchvision.ops.box_iou(pred_box.unsqueeze(0), gt_boxes)

            # Find the best matching GT box
            best_iou, best_idx = torch.max(ious, dim=1)
            best_iou = float(best_iou)
            best_idx = int(best_idx)

            if best_iou > iou_threshold:
                if not visited[best_idx]:
                    tps[i] = 1
                    visited[best_idx] = True
                else:
                    # Already matched this GT box (duplicate detection)
                    fps[i] = 1
            else:
                # IoU too low
                fps[i] = 1

        # 4. Compute Precision and Recall
        tp_cumsum = torch.cumsum(tps, dim=0)
        fp_cumsum = torch.cumsum(fps, dim=0)

        recalls = tp_cumsum / total_gt_boxes
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # 5. Compute AP using VOC 2010+ method (All-point interpolation)
        # Add sentinel values to simplify integration
        recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
        precisions = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))

        # Ensure precision is monotonically decreasing (envelope)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = torch.max(precisions[i], precisions[i + 1])

        # Integrate area under the curve
        # Find points where recall changes
        indices = torch.where(recalls[1:] != recalls[:-1])[0]

        ap = torch.sum(
            (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
        )
        average_precisions.append(float(ap))

    # Return mean AP across all classes
    return (
        sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    )
