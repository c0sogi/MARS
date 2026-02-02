import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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

    Args:
        mask (np.ndarray): Binary mask of shape (H, W). 1 - mask, 0 - background.

    Returns:
        str: Space-delimited list of pairs (start, length).
             Pixels are 1-indexed and numbered from top to bottom, then left to right.
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    """
    Decodes a Run-Length Encoded (RLE) string into a binary mask.

    Args:
        mask_rle (str): RLE string.
        shape (tuple): Shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def calc_map(predictions, targets, pixel_threshold=0.5):
    """
    Calculates the Mean Average Precision at different IoU thresholds (0.5 to 0.95).

    The metric sweeps over a range of IoU thresholds:
    (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).

    Args:
        predictions (np.ndarray or torch.Tensor): Predicted masks or probabilities (N, H, W).
        targets (np.ndarray or torch.Tensor): Ground truth masks (N, H, W).
        pixel_threshold (float): Threshold to convert probability maps to binary masks.

    Returns:
        float: The mean average precision over the batch and thresholds.
    """
    # Thresholds for the metric
    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    # Ensure inputs are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Binarize predictions and targets
    predictions = (predictions > pixel_threshold).astype(np.uint8)
    targets = (targets > 0.5).astype(np.uint8)

    # Flatten spatial dimensions for easier processing: (N, H*W)
    preds_flat = predictions.reshape(predictions.shape[0], -1)
    targs_flat = targets.reshape(targets.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (preds_flat & targs_flat).sum(axis=1)
    union = (preds_flat | targs_flat).sum(axis=1)

    # Calculate IoU
    # Handle division by zero (empty union means both empty -> IoU = 1)
    # If union is 0, intersection must be 0.
    iou = np.ones_like(intersection, dtype=np.float32)
    non_empty = union > 0
    iou[non_empty] = intersection[non_empty] / union[non_empty]

    # Compare IoU against thresholds
    # iou: (N,), thresholds: (10,)
    # Result: (N, 10) boolean matrix
    matches = iou[:, None] > iou_thresholds[None, :]

    # Calculate precision per image (mean over thresholds)
    # Since TP=1 if match else 0, and we only have 1 object, precision is just the boolean match.
    precisions = matches.mean(axis=1)

    # Return mean over the batch
    return precisions.mean()
