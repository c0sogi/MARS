import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask to Run-Length Encoding (RLE).
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited string of RLE pairs (start length).
    """
    # Flatten column-wise (Fortran style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")
    # Pad with zeros to detect changes at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Convert end indices to lengths
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or pd.isna(mask_rle):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # Convert to 0-indexed
    ends = starts + lengths

    # Create flat array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise
    return img.reshape(shape, order="F")


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss and metrics during training.
    """

    def __init__(self, name="Metric"):
        self.name = name
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

    def __str__(self):
        return f"{self.name}: {self.val:.6f} (Avg: {self.avg:.6f})"


def save_numpy_cache(data, filename):
    """
    Saves a numpy array to the configured cache directory.

    Args:
        data (np.ndarray): Data to save.
        filename (str): Filename (e.g. 'data.npy').
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    path = os.path.join(Config.CACHE_DIR, filename)
    np.save(path, data)


def load_numpy_cache(filename):
    """
    Loads a numpy array from the configured cache directory.

    Args:
        filename (str): Filename to load.

    Returns:
        np.ndarray or None: Loaded data or None if not found.
    """
    path = os.path.join(Config.CACHE_DIR, filename)
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None


def compute_salt_metric(pred_mask, gt_mask):
    """
    Computes the mean Average Precision at IoU thresholds [0.5, ..., 0.95].

    Args:
        pred_mask (np.ndarray): Predicted binary mask (0 or 1).
        gt_mask (np.ndarray): Ground truth binary mask (0 or 1).

    Returns:
        float: The average precision score for this image.
    """
    # Ensure inputs are binary uint8
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Handle empty mask cases
    gt_empty = gt_mask.sum() == 0
    pred_empty = pred_mask.sum() == 0

    if gt_empty:
        return 1.0 if pred_empty else 0.0

    if pred_empty:
        return 0.0

    # Calculate IoU
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    # Calculate Precision at each threshold
    # For a single object task:
    # If IoU > threshold: TP=1, FP=0, FN=0 -> Precision = 1
    # If IoU <= threshold: TP=0, FP=1, FN=1 -> Precision = 0
    thresholds = Config.IOU_THRESHOLDS
    matches = [1 if iou > t else 0 for t in thresholds]

    return np.mean(matches)
