import numpy as np
import torch
import os
import random


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

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
    os.environ["PYTHONHASHSEED"] = str(seed)


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The pixels are one-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: RLE string "start length start length ..."
    """
    # Flatten column-wise (Fortran style) to match top-to-bottom, left-to-right indexing
    pixels = mask.flatten(order="F")
    # Prepend and append 0 to detect runs at start/end
    pixels = np.concatenate([[0], pixels, [0]])
    # Find indices where values change
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths: runs[1::2] are ends, runs[::2] are starts
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoding (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calculate_iou_map(y_true, y_pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    Args:
        y_true (np.ndarray): Ground truth masks, shape (N, H, W).
        y_pred (np.ndarray): Predicted masks (binary), shape (N, H, W).
        thresholds (np.ndarray): Array of IoU thresholds.

    Returns:
        float: The mean average precision over the dataset.
    """
    # Ensure inputs are boolean/binary
    y_true = y_true > 0
    y_pred = y_pred > 0

    # Flatten spatial dimensions to (N, -1)
    if y_true.ndim > 2:
        y_true = y_true.reshape(y_true.shape[0], -1)
    if y_pred.ndim > 2:
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

    ious = []
    for i in range(len(y_true)):
        t = y_true[i]
        p = y_pred[i]

        t_sum = t.sum()
        p_sum = p.sum()

        # Handle empty mask cases
        if t_sum == 0 and p_sum == 0:
            ious.append(1.0)
        elif t_sum == 0 and p_sum > 0:
            ious.append(0.0)
        elif t_sum > 0 and p_sum == 0:
            ious.append(0.0)
        else:
            intersection = np.logical_and(t, p).sum()
            union = np.logical_or(t, p).sum()
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

    ious = np.array(ious)

    # Calculate precision at each threshold for each image
    # For a single image, precision is 1 if IoU > threshold, else 0
    precisions = []
    for t in thresholds:
        tp = ious > t
        precisions.append(tp)

    # precisions shape: (num_thresholds, N)
    # Average over thresholds to get AP per image
    ap_per_image = np.mean(precisions, axis=0)

    # Mean over dataset
    map_score = np.mean(ap_per_image)

    return map_score
