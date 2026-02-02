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
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        str: Space delimited string of 'start length' pairs.
    """
    # Flatten column-wise (Fortran-style)
    pixels = mask.flatten(order="F")
    # Pad with 0s at start and end to detect all runs
    pixels = np.concatenate([[0], pixels, [0]])
    # Find where the value changes
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # Calculate lengths (end - start)
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(Config.ORIG_HEIGHT, Config.ORIG_WIDTH)):
    """
    Decodes a Run-Length Encoded string to a binary mask.

    Args:
        mask_rle (str): Space delimited string of 'start length' pairs.
        shape (tuple): The shape of the output mask (H, W).

    Returns:
        np.ndarray: Binary mask of shape (H, W).
    """
    if (
        not isinstance(mask_rle, str)
        or mask_rle.strip() == ""
        or str(mask_rle) == "nan"
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    # Parse starts and lengths
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # Convert 1-indexed to 0-indexed
    starts -= 1
    ends = starts + lengths

    # Create flattened array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    # Reshape column-wise (Fortran-style)
    return img.reshape(shape, order="F")


def compute_map_score(preds, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Calculates the Mean Average Precision at different IoU thresholds.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions (N, H, W) or (N, 1, H, W).
                                            Can be probabilities or binary.
        targets (np.ndarray or torch.Tensor): Ground truth (N, H, W) or (N, 1, H, W).
        thresholds (np.ndarray): Array of IoU thresholds to evaluate.

    Returns:
        float: The mean average precision score.
    """
    # Convert torch tensors to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Handle channel dimension if present
    if preds.ndim == 4:
        preds = preds.squeeze(1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)

    # Binarize predictions if they are probabilities (assuming 0.5 threshold)
    if preds.dtype == float or preds.dtype == np.float32 or preds.dtype == np.float64:
        preds = (preds > 0.5).astype(np.uint8)
    else:
        preds = preds.astype(np.uint8)

    targets = targets.astype(np.uint8)

    # Flatten spatial dimensions: (N, H, W) -> (N, H*W)
    preds_flat = preds.reshape(preds.shape[0], -1)
    targets_flat = targets.reshape(targets.shape[0], -1)

    # Calculate Intersection and Union
    intersection = (preds_flat * targets_flat).sum(axis=1)
    union = (preds_flat + targets_flat).astype(bool).astype(int).sum(axis=1)

    # Calculate IoU
    # Initialize IoU as 1.0 (perfect match) for cases where union is 0 (both empty)
    iou = np.ones(len(preds_flat), dtype=float)

    # Only calculate IoU where union > 0
    mask_union_positive = union > 0
    iou[mask_union_positive] = (
        intersection[mask_union_positive] / union[mask_union_positive]
    )

    # Calculate precision at each threshold
    # iou: (N,)
    # thresholds: (T,)
    # matches: (N, T) - True if IoU > threshold
    matches = iou[:, None] > thresholds[None, :]

    # Average precision for each image over all thresholds
    # For a single object task, Precision is 1 if Hit, 0 if Miss.
    # So AP is simply the fraction of thresholds passed.
    image_scores = matches.mean(axis=1)

    # Return mean score over the batch
    return float(image_scores.mean())
