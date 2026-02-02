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
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE) format.
    The competition specifies pixels are numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 indicates salt, 0 indicates sediment.

    Returns:
        str: Space-delimited string of RLE pairs (start length).
    """
    # Flatten in column-major order (Fortran-style) to match (top->bottom, left->right) indexing
    pixels = mask.flatten(order="F")

    # Prepend and append 0 to detect runs at the start/end
    pixels = np.concatenate([[0], pixels, [0]])

    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited string of RLE pairs.
        shape (tuple): The shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W) with dtype uint8.
    """
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Adjust 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape back to image dimensions using column-major order
    return img.reshape(shape, order="F")


def calc_iou_metric(
    y_pred, y_true, thresholds=Config.IOU_THRESHOLDS, binarization_threshold=0.5
):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    The metric sweeps over a range of IoU thresholds (0.5 to 0.95 with step 0.05).
    For each threshold, a prediction is a "hit" if IoU > threshold.
    The score for an image is the average precision across all thresholds.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted output. Shape (B, H, W) or (B, 1, H, W).
        y_true (torch.Tensor or np.ndarray): Ground truth. Shape (B, H, W) or (B, 1, H, W).
        thresholds (list): List of IoU thresholds to evaluate.
        binarization_threshold (float): Threshold to convert probabilities to binary mask.

    Returns:
        float: The mean average precision over the batch.
    """
    # Convert tensors to numpy
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    # Remove channel dimension if present
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(1)
    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)

    # Binarize predictions and ensure ground truth is binary
    y_pred = (y_pred > binarization_threshold).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)

    batch_size = y_true.shape[0]
    precisions = []

    for i in range(batch_size):
        t = y_true[i].flatten()
        p = y_pred[i].flatten()

        sum_t = np.sum(t)
        sum_p = np.sum(p)

        # Calculate IoU for the single image
        if sum_t == 0 and sum_p == 0:
            # Both empty: Perfect match
            iou = 1.0
        elif sum_t == 0 or sum_p == 0:
            # One empty, one not: No overlap
            iou = 0.0
        else:
            intersection = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            iou = intersection / union

        # Calculate score: fraction of thresholds that the IoU exceeds
        # Note: The metric says "hit" if IoU > threshold.
        matches = iou > np.array(thresholds)
        score = np.mean(matches)
        precisions.append(score)

    return np.mean(precisions)
