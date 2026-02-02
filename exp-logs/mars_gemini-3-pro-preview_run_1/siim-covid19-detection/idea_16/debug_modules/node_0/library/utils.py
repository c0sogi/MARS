import os
import random
import numpy as np
import torch
import copy
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
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


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model weights.
    This helps in stabilizing the training and often leads to better generalization.
    """

    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(self.device)

    def update(self, model):
        """
        Update the EMA model parameters using the current model parameters.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k in msd:
                if msd[k].dtype.is_floating_point:
                    esd[k].copy_(self.decay * esd[k] + (1.0 - self.decay) * msd[k])


def xyxy_to_normalized(boxes, width, height):
    """
    Converts bounding boxes from [x_min, y_min, x_max, y_max] to normalized [0, 1] coordinates.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.copy()
        boxes[:, 0] /= width
        boxes[:, 1] /= height
        boxes[:, 2] /= width
        boxes[:, 3] /= height
    return boxes


def normalized_to_xyxy(boxes, width, height):
    """
    Converts bounding boxes from normalized [0, 1] coordinates to [x_min, y_min, x_max, y_max].
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.copy()
        boxes[:, 0] *= width
        boxes[:, 1] *= height
        boxes[:, 2] *= width
        boxes[:, 3] *= height
    return boxes


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two sets of boxes.
    Args:
        box1: (N, 4) tensor
        box2: (M, 4) tensor
    Returns:
        iou: (N, M) tensor
    """
    # box1: N x 4
    # box2: M x 4

    # Intersection
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
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
    Calculates Mean Average Precision (mAP) at a specific IoU threshold.
    Implements PASCAL VOC 2010 style AP calculation (Area under Precision-Recall Curve).

    Args:
        pred_boxes: List of tensors, one per image, shape (N, 4) in xyxy
        pred_scores: List of tensors, one per image, shape (N,)
        pred_labels: List of tensors, one per image, shape (N,)
        gt_boxes: List of tensors, one per image, shape (M, 4) in xyxy
        gt_labels: List of tensors, one per image, shape (M,)
        num_classes: Number of classes (excluding background)
        iou_threshold: IoU threshold for a match

    Returns:
        mAP: Mean Average Precision
    """
    average_precisions = []

    # Move everything to CPU for calculation
    pred_boxes = [b.cpu() for b in pred_boxes]
    pred_scores = [s.cpu() for s in pred_scores]
    pred_labels = [l.cpu() for l in pred_labels]
    gt_boxes = [b.cpu() for b in gt_boxes]
    gt_labels = [l.cpu() for l in gt_labels]

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Aggregate all detections and ground truths for this class
        for i in range(len(pred_boxes)):
            # Filter predictions for class c
            mask_pred = pred_labels[i] == c
            if mask_pred.sum() > 0:
                p_boxes = pred_boxes[i][mask_pred]
                p_scores = pred_scores[i][mask_pred]
                for j in range(len(p_boxes)):
                    detections.append(
                        (p_scores[j].item(), p_boxes[j], i)
                    )  # score, box, image_idx

            # Filter ground truths for class c
            mask_gt = gt_labels[i] == c
            if mask_gt.sum() > 0:
                g_boxes = gt_boxes[i][mask_gt]
                for j in range(len(g_boxes)):
                    ground_truths.append((g_boxes[j], i, False))  # box, image_idx, used

        # If no ground truths for this class
        if len(ground_truths) == 0:
            # If there are detections, precision is 0. If no detections, undefined (usually 0 or ignored).
            # In PASCAL VOC, if a class is not present in the test set, it's usually excluded or 0.
            # Here we assume if it's a valid class it should be handled.
            # If we have FPs but no TPs, AP is 0.
            average_precisions.append(0.0)
            continue

        # Sort detections by score descending
        detections.sort(key=lambda x: x[0], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        # Keep track of which GTs have been matched in each image
        # ground_truths is a list of tuples. We need to be able to mark them as used.
        # Let's organize GTs by image_idx for faster lookup
        gt_by_image = {}
        for idx, (box, img_id, used) in enumerate(ground_truths):
            if img_id not in gt_by_image:
                gt_by_image[img_id] = []
            gt_by_image[img_id].append({"box": box, "used": False, "orig_idx": idx})

        for d_idx, (score, d_box, img_id) in enumerate(detections):
            if img_id not in gt_by_image:
                FP[d_idx] = 1
                continue

            gts = gt_by_image[img_id]
            best_iou = 0
            best_gt_idx = -1

            # Find best matching GT
            for g_idx, gt_item in enumerate(gts):
                # Calculate IoU
                # d_box: (4,), gt_item['box']: (4,)
                # unsqueeze to make (1,4)
                iou = calculate_iou(
                    d_box.unsqueeze(0), gt_item["box"].unsqueeze(0)
                ).item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou > iou_threshold:
                if not gts[best_gt_idx]["used"]:
                    TP[d_idx] = 1
                    gts[best_gt_idx]["used"] = True
                else:
                    FP[d_idx] = 1  # Duplicate detection
            else:
                FP[d_idx] = 1

        # Compute cumulative precision and recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / len(ground_truths)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

        # PASCAL VOC 2010 AP Calculation (Integration)
        # Add sentinel values for integration
        precisions = torch.cat((torch.tensor([0.0]), precisions, torch.tensor([0.0])))
        recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))

        # Ensure precision is monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = torch.max(precisions[i], precisions[i + 1])

        # Calculate Area Under Curve
        # Indices where recall changes
        i_list = torch.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = torch.sum((recalls[i_list] - recalls[i_list - 1]) * precisions[i_list])

        average_precisions.append(ap.item())

    return sum(average_precisions) / (len(average_precisions) + 1e-6)


def calculate_classification_ap(y_true, y_scores):
    """
    Calculates Average Precision for multi-label classification.
    y_true: (N, NumClasses)
    y_scores: (N, NumClasses)
    Returns list of APs per class.
    """
    from sklearn.metrics import average_precision_score

    # Ensure numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_scores, torch.Tensor):
        y_scores = y_scores.cpu().numpy()

    aps = []
    # Handle case where a class might not be present in the batch
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:
            aps.append(0.0)
        else:
            aps.append(average_precision_score(y_true[:, i], y_scores[:, i]))

    return aps
