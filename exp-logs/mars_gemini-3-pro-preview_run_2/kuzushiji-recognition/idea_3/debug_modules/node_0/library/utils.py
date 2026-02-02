import os
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
from library.config import Config

# Global cache for label map
_UNICODE_TO_ID = None
_ID_TO_UNICODE = None


def get_label_map(config=None):
    """
    Loads and caches the mapping between Unicode characters and Integer IDs.
    ID 0 is reserved for the background.
    IDs 1..N are for characters.
    """
    global _UNICODE_TO_ID, _ID_TO_UNICODE

    if _UNICODE_TO_ID is not None:
        return _UNICODE_TO_ID, _ID_TO_UNICODE

    if config is None:
        config = Config()

    # Load unicode translation file
    if not os.path.exists(config.UNICODE_MAP):
        raise FileNotFoundError(f"Unicode map not found at {config.UNICODE_MAP}")

    df = pd.read_csv(config.UNICODE_MAP)

    # Sort unicode values to ensure deterministic mapping
    # Assuming the CSV has a 'Unicode' column
    chars = sorted(df["Unicode"].unique())

    # Create mappings
    # 0 is reserved for background
    _UNICODE_TO_ID = {c: i + 1 for i, c in enumerate(chars)}
    _ID_TO_UNICODE = {i + 1: c for i, c in enumerate(chars)}

    return _UNICODE_TO_ID, _ID_TO_UNICODE


def parse_labels(label_str, label_map):
    """
    Parses a label string into bounding boxes and label indices.

    Args:
        label_str (str): Space-separated string "Unicode X Y W H ..."
        label_map (dict): Dictionary mapping Unicode strings to Integer IDs.

    Returns:
        boxes (Tensor): FloatTensor of shape (N, 4) in [x1, y1, x2, y2] format.
        labels (Tensor): Int64Tensor of shape (N,).
    """
    if pd.isna(label_str) or not label_str:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.int64
        )

    parts = label_str.split()
    boxes = []
    labels = []

    # Each label is 5 parts: Char, X, Y, W, H
    step = 5
    for i in range(0, len(parts), step):
        if i + 4 >= len(parts):
            break

        char = parts[i]
        try:
            x = int(parts[i + 1])
            y = int(parts[i + 2])
            w = int(parts[i + 3])
            h = int(parts[i + 4])

            if char in label_map:
                labels.append(label_map[char])
                # Convert xywh to x1y1x2y2
                boxes.append([x, y, x + w, y + h])
        except ValueError:
            continue

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), torch.empty(
            (0,), dtype=torch.int64
        )

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.int64
    )


def get_transforms(train=False):
    """
    Returns the data transformation pipeline.

    Args:
        train (bool): Whether to apply training augmentations.
    """
    transforms = []

    if train:
        # Photometric Augmentation
        # Randomly change brightness, contrast, and saturation
        transforms.append(
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
        )

    # Convert PIL image to Tensor [C, H, W] in range [0, 1]
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def collate_fn(batch):
    """
    Custom collate function for object detection.
    Batch is a list of tuples (image, target).
    """
    return tuple(zip(*batch))


class F1Score:
    """
    Modified F1 Score for Kuzushiji Character Recognition.
    Metric: Center point within Ground Truth bounding box + Matching Label.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds, targets):
        """
        Update metric with a batch of predictions and targets.

        Args:
            preds (list[dict]): List of prediction dictionaries.
                Each dict contains:
                    'boxes': Tensor of shape (N, 4) [x1, y1, x2, y2]
                    'labels': Tensor of shape (N,)
                    'scores': Tensor of shape (N,)
            targets (list[dict]): List of ground truth dictionaries.
                Each dict contains:
                    'boxes': Tensor of shape (M, 4) [x1, y1, x2, y2]
                    'labels': Tensor of shape (M,)
        """
        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"].cpu()
            pred_labels = pred["labels"].cpu()

            gt_boxes = target["boxes"].cpu()
            gt_labels = target["labels"].cpu()

            # Calculate centers of predictions
            # x_center = (x1 + x2) / 2
            # y_center = (y1 + y2) / 2
            pred_centers_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            pred_centers_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2

            matched_gt_indices = set()

            # Iterate over predictions to find matches
            # We assume predictions are already filtered by score threshold if necessary,
            # or we can process all. Usually, low confidence predictions increase FP.

            for i in range(len(pred_labels)):
                p_label = pred_labels[i].item()
                px = pred_centers_x[i].item()
                py = pred_centers_y[i].item()

                match_found = False

                # Check against all GTs
                for j in range(len(gt_labels)):
                    # If this GT is already matched, skip it (one-to-one match for TP)
                    if j in matched_gt_indices:
                        continue

                    g_label = gt_labels[j].item()

                    # Check label match
                    if p_label == g_label:
                        gx1, gy1, gx2, gy2 = gt_boxes[j].tolist()

                        # Check spatial match (center inside box)
                        if gx1 <= px <= gx2 and gy1 <= py <= gy2:
                            self.tp += 1
                            matched_gt_indices.add(j)
                            match_found = True
                            break

                if not match_found:
                    self.fp += 1

            # False Negatives: GTs that were not matched by any prediction
            self.fn += len(gt_labels) - len(matched_gt_indices)

    def compute(self):
        """
        Compute the F1 Score, Precision, and Recall.
        """
        epsilon = 1e-7
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }
