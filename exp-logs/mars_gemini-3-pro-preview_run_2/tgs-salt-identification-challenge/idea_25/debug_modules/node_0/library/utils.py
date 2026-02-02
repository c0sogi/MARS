import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    Enforces deterministic behavior in cuDNN.

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


def do_length_decode(rle_string):
    """
    Helper to parse an RLE string into a list of integers.

    Args:
        rle_string (str): Space-delimited RLE string.

    Returns:
        list: List of integers [start, length, start, length, ...].
    """
    if pd.isna(rle_string) or rle_string == "":
        return []
    return [int(x) for x in rle_string.split()]


def do_length_encode(rle_list):
    """
    Helper to format a list of run-length integers into a space-delimited string.

    Args:
        rle_list (list): List of integers [start, length, ...].

    Returns:
        str: Space-delimited RLE string.
    """
    return " ".join(str(x) for x in rle_list)


def rle_decode(mask_rle, shape=(Config.ORIG_SIZE, Config.ORIG_SIZE)):
    """
    Decodes a run-length encoded string to a binary mask.
    The encoding uses 1-based indexing and column-major order.

    Args:
        mask_rle (str): RLE string (start length start length ...).
        shape (tuple): (height, width) of the mask.

    Returns:
        np.ndarray: Binary mask of shape `shape` (uint8).
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or mask_rle == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(mask):
    """
    Encodes a binary mask to a run-length encoded string.
    The encoding uses 1-based indexing and column-major order.

    Args:
        mask (np.ndarray): Binary mask (0 or 1).

    Returns:
        str: RLE string.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def calculate_iou_batch(y_pred, y_true, threshold=0.5):
    """
    Calculates the Intersection over Union (IoU) for a batch of predictions.

    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted probabilities or masks.
                                             Shape: (N, H, W) or (N, 1, H, W).
        y_true (torch.Tensor or np.ndarray): Ground truth masks.
                                             Shape: (N, H, W) or (N, 1, H, W).
        threshold (float): Threshold to binarize predictions if they are probabilities.

    Returns:
        np.ndarray: Array of IoU scores for each image in the batch.
    """
    # Convert tensors to numpy
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Squeeze channel dimension if present (N, 1, H, W) -> (N, H, W)
    if y_pred.ndim == 4:
        y_pred = y_pred.squeeze(1)
    if y_true.ndim == 4:
        y_true = y_true.squeeze(1)

    # Binarize predictions and targets
    pred_mask = (y_pred > threshold).astype(np.uint8)
    true_mask = (y_true > 0.5).astype(np.uint8)

    # Flatten spatial dimensions for batch calculation: (N, H*W)
    pred_flat = pred_mask.reshape(pred_mask.shape[0], -1)
    true_flat = true_mask.reshape(true_mask.shape[0], -1)

    intersection = (pred_flat * true_flat).sum(axis=1)
    union = pred_flat.sum(axis=1) + true_flat.sum(axis=1) - intersection

    # Handle division by zero (empty union means both empty -> IoU = 1.0)
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    return iou
