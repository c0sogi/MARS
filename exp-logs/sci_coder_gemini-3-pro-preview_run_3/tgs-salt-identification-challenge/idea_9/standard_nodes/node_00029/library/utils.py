import numpy as np
import torch
from library.config import seed_everything, DataConfig


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


def rle_encode(img):
    """
    Run-length encoding for a binary mask.

    Args:
        img (np.array): Binary mask of shape (H, W), where 1 - mask, 0 - background.

    Returns:
        str: Run-length encoded string formatted as 'start length start length ...'.
             Uses 1-based indexing and Fortran (column-major) flattening order.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(DataConfig.ORIG_H, DataConfig.ORIG_W)):
    """
    Decodes a run-length encoded string to a binary mask.

    Args:
        mask_rle (str): Run-length encoded string.
        shape (tuple): Target shape (height, width) of the mask.

    Returns:
        np.array: Binary mask of shape `shape`.
    """
    if str(mask_rle) == "nan" or mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calculate_iou_map(preds, labels, threshold=0.5):
    """
    Calculates the Mean Average Precision at IoU thresholds ranging from 0.5 to 0.95
    with a step size of 0.05.

    Args:
        preds (np.array or torch.Tensor): Predicted masks (N, H, W) or (N, 1, H, W).
                                          Can be probabilities [0, 1] or binary.
        labels (np.array or torch.Tensor): Ground truth masks (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        float: The mean average precision over the batch.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Binarize predictions and labels
    preds = (preds > threshold).astype(np.uint8)
    labels = (labels > 0.5).astype(np.uint8)

    batch_size = preds.shape[0]
    metric = []

    # Define thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    for i in range(batch_size):
        p = preds[i].flatten()
        t = labels[i].flatten()

        # Calculate Intersection over Union
        if np.sum(p) == 0 and np.sum(t) == 0:
            # Both empty: Perfect match
            iou = 1.0
        elif np.sum(p) > 0 and np.sum(t) > 0:
            intersection = np.logical_and(p, t).sum()
            union = np.logical_or(p, t).sum()
            iou = intersection / union
        else:
            # One empty, one not: No match
            iou = 0.0

        # Calculate Average Precision for this image
        # At each threshold t, precision is 1 if IoU > t, else 0
        matches = iou > iou_thresholds
        score = np.mean(matches)
        metric.append(score)

    return np.mean(metric)
