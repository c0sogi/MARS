import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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
        str: Space-delimited RLE string.
    """
    # Flatten in Fortran order (column-major) to match top-to-bottom, left-to-right indexing
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
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Output shape (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if str(mask_rle) == "nan" or mask_rle is None or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    # Convert 1-based indexing to 0-based
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape in Fortran order
    return img.reshape(shape, order="F")


def get_iou_vector(y_pred_bin, y_true):
    """
    Calculates the IoU for a batch of binary masks.

    Args:
        y_pred_bin (np.ndarray): Predicted binary masks (N, H, W) or (N, H*W).
        y_true (np.ndarray): Ground truth binary masks (N, H, W) or (N, H*W).

    Returns:
        np.ndarray: Vector of IoU scores of shape (N,).
    """
    y_pred_bin = y_pred_bin.reshape(y_pred_bin.shape[0], -1).astype(bool)
    y_true = y_true.reshape(y_true.shape[0], -1).astype(bool)

    intersection = np.logical_and(y_pred_bin, y_true).sum(axis=1)
    union = np.logical_or(y_pred_bin, y_true).sum(axis=1)

    # If union is 0, it means both masks are empty -> IoU is 1.0
    iou = np.where(union == 0, 1.0, intersection / union)
    return iou


def do_kaggle_metric(predict_prob, truth, threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    Args:
        predict_prob (np.ndarray): Predicted probabilities (N, H, W).
        truth (np.ndarray): Ground truth masks (N, H, W).
        threshold (float): Threshold to binarize the predicted probabilities.

    Returns:
        float: The mean average precision score.
    """
    # Binarize predictions
    y_pred_bin = (predict_prob > threshold).astype(np.uint8)

    # Calculate IoU for each image in batch
    ious = get_iou_vector(y_pred_bin, truth)

    # Metric sweeps over IoU thresholds: 0.5, 0.55, ..., 0.95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Compare calculated IoUs with metric thresholds
    # ious: (N, 1), thresholds: (1, 10) -> broadcast to (N, 10)
    matches = ious[:, None] > iou_thresholds[None, :]

    # Average over thresholds (axis 1) to get score per image
    image_scores = np.mean(matches, axis=1)

    # Return mean score over the batch
    return np.mean(image_scores)


def get_best_threshold(y_true, y_pred_prob, start=0.2, end=0.8, step=0.05):
    """
    Finds the best probability threshold for binarization that maximizes the Kaggle metric.

    Args:
        y_true (np.ndarray): Ground truth masks.
        y_pred_prob (np.ndarray): Predicted probabilities.
        start (float): Start of threshold search range.
        end (float): End of threshold search range.
        step (float): Step size for search.

    Returns:
        tuple: (best_threshold, best_score)
    """
    best_threshold = 0.5
    best_score = -1.0

    # Generate range of thresholds to test
    # Add small epsilon to end to include it
    thresholds = np.arange(start, end + 0.001, step)

    for t in thresholds:
        score = do_kaggle_metric(y_pred_prob, y_true, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score
