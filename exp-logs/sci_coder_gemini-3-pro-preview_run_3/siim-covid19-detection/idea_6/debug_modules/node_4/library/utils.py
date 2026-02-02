import os
import random
import numpy as np
import torch
from collections import defaultdict
from library.config import Config


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


def collate_fn(batch):
    """
    Custom collate function for object detection DataLoader.

    Args:
        batch: List of tuples (image, target, image_id) from Dataset.__getitem__

    Returns:
        images: torch.Tensor of shape (B, C, H, W)
        targets: List of dictionaries containing boxes and labels
        image_ids: List of image ID strings
    """
    images = []
    targets = []
    image_ids = []

    for item in batch:
        img, target, img_id = item
        images.append(img)
        targets.append(target)
        image_ids.append(img_id)

    # Stack images into a single tensor (assumes all are resized to Config.IMG_SIZE)
    images = torch.stack(images, dim=0)

    return images, targets, image_ids


def calculate_iou(boxes1, boxes2):
    """
    Calculates the Intersection over Union (IoU) between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) [xmin, ymin, xmax, ymax]
        boxes2: Tensor of shape (M, 4) [xmin, ymin, xmax, ymax]

    Returns:
        iou: Tensor of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    Calculates Mean Average Precision (mAP) at a specific IoU threshold (default 0.5).
    Specifically designed for the 'opacity' class (class ID 1).

    Args:
        predictions: List of dicts {'boxes': tensor, 'scores': tensor, 'labels': tensor}
        targets: List of dicts {'boxes': tensor, 'labels': tensor}
        iou_threshold: Float, IoU threshold for a match (default 0.5)

    Returns:
        ap: Float, Average Precision for the opacity class.
    """
    # Target class ID for Opacity is 1 (based on Config)
    target_class_id = 1

    all_preds_scores = []
    all_preds_boxes = []
    all_preds_image_indices = []

    all_gt_boxes = defaultdict(list)  # image_index -> boxes
    num_gt_total = 0

    # 1. Aggregate Predictions and Ground Truths
    for idx, (pred, target) in enumerate(zip(predictions, targets)):
        # Filter predictions for opacity class
        p_mask = pred["labels"] == target_class_id
        p_boxes = pred["boxes"][p_mask]
        p_scores = pred["scores"][p_mask]

        if len(p_boxes) > 0:
            all_preds_boxes.append(p_boxes)
            all_preds_scores.append(p_scores)
            all_preds_image_indices.extend([idx] * len(p_boxes))

        # Filter targets for opacity class
        t_mask = target["labels"] == target_class_id
        t_boxes = target["boxes"][t_mask]

        if len(t_boxes) > 0:
            all_gt_boxes[idx] = t_boxes
            num_gt_total += len(t_boxes)

    if num_gt_total == 0:
        return 0.0

    if not all_preds_scores:
        return 0.0

    all_preds_scores = torch.cat(all_preds_scores)
    all_preds_boxes = torch.cat(all_preds_boxes)
    all_preds_image_indices = torch.tensor(all_preds_image_indices)

    # 2. Sort predictions by confidence score
    sort_inds = torch.argsort(all_preds_scores, descending=True)
    all_preds_boxes = all_preds_boxes[sort_inds]
    all_preds_image_indices = all_preds_image_indices[sort_inds]

    tp = torch.zeros(len(all_preds_boxes))
    fp = torch.zeros(len(all_preds_boxes))

    # Track matched GTs: image_idx -> tensor of bools
    gt_matched = {
        k: torch.zeros(len(v), dtype=torch.bool) for k, v in all_gt_boxes.items()
    }

    # 3. Match predictions to ground truth
    for i in range(len(all_preds_boxes)):
        img_idx = all_preds_image_indices[i].item()
        box = all_preds_boxes[i]

        if img_idx in all_gt_boxes:
            gt_boxes_img = all_gt_boxes[img_idx]

            # Calculate IoU with all GTs in this image
            # box: [4], gt_boxes_img: [M, 4] -> iou: [M]
            iou = calculate_iou(box.unsqueeze(0), gt_boxes_img).squeeze(0)

            max_iou, max_idx = torch.max(iou, dim=0)

            if max_iou >= iou_threshold:
                if not gt_matched[img_idx][max_idx]:
                    tp[i] = 1
                    gt_matched[img_idx][max_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection
            else:
                fp[i] = 1  # Poor localization
        else:
            fp[i] = 1  # False positive (no GT in image)

    # 4. Calculate Precision and Recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / num_gt_total
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # 5. Calculate AP (PASCAL VOC 2010 style - Area Under Curve with interpolation)
    # Add sentinel values
    recalls = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    precisions = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    # Compute maximum precision for any recall >= r
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])

    # Integrate area under curve
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap.item()


def format_prediction_string(labels, boxes, scores):
    """
    Formats image-level predictions into the submission string format.

    Args:
        labels: List/Tensor of class IDs.
        boxes: List/Tensor of bounding boxes [xmin, ymin, xmax, ymax].
        scores: List/Tensor of confidence scores.

    Returns:
        String in format "class conf xmin ymin xmax ymax ..." or "none 1 0 0 1 1"
    """
    pred_strings = []

    # If inputs are tensors, convert to list/numpy for iteration
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    for i in range(len(labels)):
        label = labels[i]

        # Map label ID to string
        if label == 1:  # Opacity class ID
            label_str = "opacity"
        else:
            label_str = str(label)

        score = scores[i]
        box = boxes[i]

        # Ensure box coordinates are valid
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        pred_strings.append(f"{label_str} {score:.4f} {xmin} {ymin} {xmax} {ymax}")

    if not pred_strings:
        return "none 1 0 0 1 1"

    return " ".join(pred_strings)


def format_study_prediction(class_id, confidence):
    """
    Formats study-level prediction into the submission string format.

    Args:
        class_id: String (e.g., 'negative', 'typical', 'indeterminate', 'atypical')
        confidence: Float confidence score

    Returns:
        String in format "class_id confidence 0 0 1 1"
    """
    return f"{class_id} {confidence:.4f} 0 0 1 1"
