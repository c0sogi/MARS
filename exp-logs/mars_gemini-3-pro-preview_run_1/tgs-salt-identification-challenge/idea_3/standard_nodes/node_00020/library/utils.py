import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rle_encode(mask):
    """
    Encodes a binary mask into Run-Length Encoding (RLE).
    The pixels are 1-indexed and numbered from top to bottom, then left to right.

    Args:
        mask (np.ndarray): Binary mask of shape (H, W).

    Returns:
        str: Space-delimited list of pairs (start, length).
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded string into a binary mask.

    Args:
        mask_rle (str): Space-delimited list of pairs (start, length).
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


def calculate_iou(pred_mask, true_mask):
    """
    Calculates the Intersection over Union (IoU) for a single pair of masks.
    Handles the empty-empty case as 1.0.
    """
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    if not true_mask.any():
        if not pred_mask.any():
            return 1.0
        else:
            return 0.0

    intersection = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()

    if union == 0:
        return 1.0

    return intersection / union


def calculate_map_score(preds, truths, thresholds=None):
    """
    Calculates the Mean Average Precision (mAP) at specified IoU thresholds.

    Args:
        preds (list or np.ndarray): List of predicted masks.
        truths (list or np.ndarray): List of ground truth masks.
        thresholds (list, optional): List of IoU thresholds. Defaults to [0.5, 0.55, ..., 0.95].

    Returns:
        float: The mean average precision score.
    """
    if thresholds is None:
        thresholds = np.arange(0.5, 0.96, 0.05)

    ious = []
    for pred, true in zip(preds, truths):
        ious.append(calculate_iou(pred, true))

    ious = np.array(ious)

    # Calculate precision for each threshold
    # For a single image, precision is 1 if IoU > t, else 0
    # Average precision per image is mean over thresholds

    # Shape: (num_images, num_thresholds)
    # Metric definition: "greater than" the threshold
    matches = ious[:, None] > thresholds[None, :]

    # Mean over thresholds for each image -> AP per image
    ap_per_image = matches.mean(axis=1)

    # Mean over all images -> mAP
    map_score = ap_per_image.mean()

    return map_score
